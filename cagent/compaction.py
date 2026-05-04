"""Fold stale tool-result messages from conversation history."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from difflib import SequenceMatcher

from cagent.llm import LLMMessage, ToolCall
from cagent.tracing import get_trace

DEFAULT_KEEP_RECENT = 8
DEFAULT_MIN_SAVINGS_BYTES = 2048
DEFAULT_BUDGET_BYTES = 32_000
DEFAULT_SIMILARITY_THRESHOLD = 0.85

TOMBSTONE_PREFIX = "[compacted:"

_WHITESPACE_RE = re.compile(r"\s+")


def compact_history(
    messages: Sequence[LLMMessage],
    *,
    keep_recent: int = DEFAULT_KEEP_RECENT,
    min_savings_bytes: int = DEFAULT_MIN_SAVINGS_BYTES,
    budget_bytes: int = DEFAULT_BUDGET_BYTES,
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
) -> tuple[LLMMessage, ...]:
    """Return history with stale duplicate tool results replaced by tombstones.

    Preserves the system prompt, the first user message, every assistant message,
    and the last ``keep_recent`` tool-result messages. An older tool result is
    folded when a later pair uses the same tool name and an argument signature
    whose ``difflib.SequenceMatcher.ratio()`` is at least
    ``similarity_threshold``. Folding only happens when the per-message saving
    exceeds ``min_savings_bytes`` and the total history exceeds ``budget_bytes``.
    """

    original = tuple(messages)
    total_before = _history_bytes(original)
    pairs_before = _tool_result_pairs(original)
    stats_before = _history_stats(original, pairs_before)

    # Collect tool_call_ids that are already tombstoned from previous passes.
    tombstoned_call_ids: set[str] = set()
    for p in pairs_before:
        if _is_tombstone(p.result.content):
            tombstoned_call_ids.add(p.call.id)

    with get_trace().span(
        "compaction.run",
        {
            "keep_recent": keep_recent,
            "min_savings_bytes": min_savings_bytes,
            "budget_bytes": budget_bytes,
            "similarity_threshold": similarity_threshold,
            "before": stats_before,
        },
    ) as span:
        if total_before <= budget_bytes:
            _record_noop(span, stats_before, "under_budget", triggered=False)
            return original

        protected_ids = _protected_tool_call_ids(pairs_before, keep_recent)
        pairs = pairs_before
        signatures = [_arg_signature(p.call) for p in pairs]

        folds: list[_Fold] = []
        for i, pair in enumerate(pairs):
            if pair.call.id in protected_ids:
                continue
            if _is_tombstone(pair.result.content):
                continue
            superseder = _find_superseder(i, pairs, signatures, similarity_threshold)
            if superseder is None:
                continue
            savings = len(pair.result.content or "") - _tombstone_size(pair.call)
            if savings < min_savings_bytes:
                continue
            folds.append(
                _Fold(
                    index=pair.index,
                    call=pair.call,
                    superseded_by_index=superseder.pair.index,
                    similarity=superseder.similarity,
                    match_mode=superseder.match_mode,
                )
            )

        if not folds:
            _record_noop(span, stats_before, "no_duplicates", triggered=True)
            return original

        # Determine which assistant messages should have reasoning stripped.
        # The signed reasoning chain is broken once any result for one of the
        # assistant's tool calls is tombstoned, so the signature must go too.
        folded_call_ids = {fold.call.id for fold in folds if fold.call.id}
        assistant_indices_to_strip_reasoning: set[int] = set()
        for idx, m in enumerate(original):
            if (
                m.role == "assistant"
                and (m.reasoning is not None or m.thought_signature is not None)
                and m.tool_calls
            ):
                ids = [tc.id for tc in m.tool_calls if tc.id]
                if not ids:
                    continue
                if any(
                    tc_id in folded_call_ids or tc_id in tombstoned_call_ids
                    for tc_id in ids
                ):
                    assistant_indices_to_strip_reasoning.add(idx)

        folded_messages = list(original)
        for fold in folds:
            old = folded_messages[fold.index]
            folded_messages[fold.index] = LLMMessage(
                role=old.role,
                content=_tombstone_text(fold.call, fold.superseded_by_index),
                tool_calls=old.tool_calls,
                tool_call_id=old.tool_call_id,
                reasoning=old.reasoning,
            )

        # Strip reasoning and the signature bound to it.
        for idx in assistant_indices_to_strip_reasoning:
            old = folded_messages[idx]
            folded_messages[idx] = LLMMessage(
                role=old.role,
                content=old.content,
                tool_calls=old.tool_calls,
                tool_call_id=old.tool_call_id,
                reasoning=None,
                thought_signature=None,
            )

        result = tuple(folded_messages)
        stats_after = _history_stats(result, _tool_result_pairs(result))
        bytes_saved = stats_before["total_bytes"] - stats_after["total_bytes"]
        saved_pct = (
            round(100.0 * bytes_saved / stats_before["total_bytes"], 2)
            if stats_before["total_bytes"]
            else 0.0
        )

        span.set_attribute("triggered", True)
        span.set_attribute("reason", "folded")
        span.set_attribute("folded_count", len(folds))
        span.set_attribute(
            "folded",
            [
                {
                    "index": fold.index,
                    "tool_call_id": fold.call.id,
                    "tool_name": fold.call.name,
                    "superseded_by_index": fold.superseded_by_index,
                    "match_mode": fold.match_mode,
                    "similarity": round(fold.similarity, 3),
                    "bytes_before": len(original[fold.index].content or ""),
                    "bytes_after": len(
                        _tombstone_text(fold.call, fold.superseded_by_index)
                    ),
                }
                for fold in folds
            ],
        )
        span.set_attribute("after", stats_after)
        span.set_attribute(
            "delta",
            {
                "bytes_saved": bytes_saved,
                "bytes_saved_pct": saved_pct,
                "tombstones_added": (
                    stats_after["tombstone_count"] - stats_before["tombstone_count"]
                ),
                "active_tool_results_removed": (
                    stats_before["active_tool_result_count"]
                    - stats_after["active_tool_result_count"]
                ),
            },
        )
        return result


class _Pair:
    __slots__ = ("index", "call", "result")

    def __init__(self, index: int, call: ToolCall, result: LLMMessage) -> None:
        self.index = index
        self.call = call
        self.result = result


class _Fold:
    __slots__ = (
        "index",
        "call",
        "superseded_by_index",
        "similarity",
        "match_mode",
    )

    def __init__(
        self,
        index: int,
        call: ToolCall,
        superseded_by_index: int,
        similarity: float,
        match_mode: str,
    ) -> None:
        self.index = index
        self.call = call
        self.superseded_by_index = superseded_by_index
        self.similarity = similarity
        self.match_mode = match_mode


class _Superseder:
    __slots__ = ("pair", "similarity", "match_mode")

    def __init__(self, pair: _Pair, similarity: float, match_mode: str) -> None:
        self.pair = pair
        self.similarity = similarity
        self.match_mode = match_mode


def _tool_result_pairs(messages: Sequence[LLMMessage]) -> list[_Pair]:
    calls_by_id: dict[str, ToolCall] = {}
    for message in messages:
        if message.role == "assistant":
            for call in message.tool_calls:
                if call.id:
                    calls_by_id[call.id] = call

    pairs: list[_Pair] = []
    for index, message in enumerate(messages):
        if message.role != "tool" or not message.tool_call_id:
            continue
        call = calls_by_id.get(message.tool_call_id)
        if call is None:
            continue
        pairs.append(_Pair(index=index, call=call, result=message))
    return pairs


def _protected_tool_call_ids(pairs: Sequence[_Pair], keep_recent: int) -> set[str]:
    protected: set[str] = set()
    if keep_recent <= 0:
        return protected
    for pair in list(pairs)[-keep_recent:]:
        if pair.call.id:
            protected.add(pair.call.id)
    return protected


def _find_superseder(
    i: int,
    pairs: Sequence[_Pair],
    signatures: Sequence[str],
    threshold: float,
) -> _Superseder | None:
    """Return the latest later pair that matches pairs[i] above the threshold."""

    pair = pairs[i]
    sig = signatures[i]
    best: _Superseder | None = None
    for j in range(i + 1, len(pairs)):
        other = pairs[j]
        if other.call.name != pair.call.name:
            continue
        other_sig = signatures[j]
        if sig == other_sig:
            similarity = 1.0
            mode = "exact"
        else:
            similarity = SequenceMatcher(None, sig, other_sig).ratio()
            if similarity < threshold:
                continue
            mode = "fuzzy"
        # Prefer the latest match so the tombstone points at the freshest data.
        if best is None or other.index > best.pair.index:
            best = _Superseder(pair=other, similarity=similarity, match_mode=mode)
    return best


def _arg_signature(call: ToolCall) -> str:
    """Return a normalized argument string for similarity comparison."""

    normalized = _normalize_args(_jsonable(call.arguments))
    return json.dumps(normalized, sort_keys=True, separators=(",", ":"))


def _normalize_args(value: object) -> object:
    if isinstance(value, dict):
        return {str(k): _normalize_args(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_normalize_args(v) for v in value]
    if isinstance(value, str):
        return _WHITESPACE_RE.sub(" ", value.strip())
    return value


def _jsonable(value: object) -> object:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    if value is None or isinstance(value, str | int | float | bool):
        return value
    return str(value)


def _is_tombstone(content: str | None) -> bool:
    return bool(content) and content.startswith(TOMBSTONE_PREFIX)  # type: ignore[union-attr]


def _tombstone_text(call: ToolCall, superseded_by_index: int) -> str:
    return (
        f"{TOMBSTONE_PREFIX} tool={call.name} output hidden; "
        f"superseded by a newer call with similar arguments later in this "
        f"conversation (message #{superseded_by_index}).]"
    )


def _tombstone_size(call: ToolCall) -> int:
    return len(_tombstone_text(call, 0))


def _history_stats(
    messages: Sequence[LLMMessage],
    pairs: Sequence[_Pair],
) -> dict[str, int]:
    tombstone_count = sum(1 for p in pairs if _is_tombstone(p.result.content))
    tool_result_bytes = sum(len(p.result.content or "") for p in pairs)
    active_tool_result_bytes = sum(
        len(p.result.content or "")
        for p in pairs
        if not _is_tombstone(p.result.content)
    )
    return {
        "message_count": len(messages),
        "total_bytes": _history_bytes(messages),
        "tool_call_count": sum(
            len(m.tool_calls) for m in messages if m.role == "assistant"
        ),
        "tool_result_count": len(pairs),
        "active_tool_result_count": len(pairs) - tombstone_count,
        "tombstone_count": tombstone_count,
        "tool_result_bytes": tool_result_bytes,
        "active_tool_result_bytes": active_tool_result_bytes,
    }


def _record_noop(
    span: object,
    stats_before: dict[str, int],
    reason: str,
    *,
    triggered: bool,
) -> None:
    span.set_attribute("triggered", triggered)  # type: ignore[attr-defined]
    span.set_attribute("reason", reason)  # type: ignore[attr-defined]
    span.set_attribute("folded_count", 0)  # type: ignore[attr-defined]
    span.set_attribute("after", stats_before)  # type: ignore[attr-defined]
    span.set_attribute(  # type: ignore[attr-defined]
        "delta",
        {
            "bytes_saved": 0,
            "bytes_saved_pct": 0.0,
            "tombstones_added": 0,
            "active_tool_results_removed": 0,
        },
    )


def _history_bytes(messages: Sequence[LLMMessage]) -> int:
    total = 0
    for message in messages:
        if message.content:
            total += len(message.content)
        if message.reasoning:
            total += len(message.reasoning)
        for call in message.tool_calls:
            total += len(call.name)
            total += len(json.dumps(_jsonable(call.arguments), separators=(",", ":")))
    return total


__all__ = [
    "DEFAULT_BUDGET_BYTES",
    "DEFAULT_KEEP_RECENT",
    "DEFAULT_MIN_SAVINGS_BYTES",
    "DEFAULT_SIMILARITY_THRESHOLD",
    "TOMBSTONE_PREFIX",
    "compact_history",
]
