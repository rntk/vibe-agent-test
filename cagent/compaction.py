"""Fold stale tool-result messages from conversation history."""

from __future__ import annotations

import json
from collections.abc import Sequence

from cagent.llm import LLMMessage, ToolCall
from cagent.tracing import get_trace

DEFAULT_KEEP_RECENT = 8
DEFAULT_MIN_SAVINGS_BYTES = 2048
DEFAULT_BUDGET_BYTES = 32_000

TOMBSTONE_PREFIX = "[compacted:"


def compact_history(
    messages: Sequence[LLMMessage],
    *,
    keep_recent: int = DEFAULT_KEEP_RECENT,
    min_savings_bytes: int = DEFAULT_MIN_SAVINGS_BYTES,
    budget_bytes: int = DEFAULT_BUDGET_BYTES,
) -> tuple[LLMMessage, ...]:
    """Return history with stale duplicate tool results replaced by tombstones.

    Preserves the system prompt, the first user message, every assistant message,
    and the last ``keep_recent`` tool-result messages. Older tool results whose
    ``(tool_name, arguments)`` have been repeated later in the conversation are
    replaced by a short tombstone, but only when the saving per message exceeds
    ``min_savings_bytes`` and the total history exceeds ``budget_bytes``.
    """

    original = tuple(messages)
    total_before = _history_bytes(original)
    pairs_before = _tool_result_pairs(original)
    stats_before = _history_stats(original, pairs_before)

    with get_trace().span(
        "compaction.run",
        {
            "keep_recent": keep_recent,
            "min_savings_bytes": min_savings_bytes,
            "budget_bytes": budget_bytes,
            "before": stats_before,
        },
    ) as span:
        if total_before <= budget_bytes:
            _record_noop(span, stats_before, "under_budget", triggered=False)
            return original

        protected_ids = _protected_tool_call_ids(pairs_before, keep_recent)
        newest_by_key = _newest_pair_by_key(pairs_before)
        pairs = pairs_before

        folds: list[_Fold] = []
        for pair in pairs:
            if pair.call.id in protected_ids:
                continue
            if _is_tombstone(pair.result.content):
                continue
            key = _pair_key(pair.call)
            newest = newest_by_key.get(key)
            if newest is None or newest.index <= pair.index:
                continue
            savings = len(pair.result.content or "") - _tombstone_size(pair.call)
            if savings < min_savings_bytes:
                continue
            folds.append(
                _Fold(
                    index=pair.index,
                    call=pair.call,
                    superseded_by_index=newest.index,
                )
            )

        if not folds:
            _record_noop(span, stats_before, "no_duplicates", triggered=True)
            return original

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
                    "bytes_before": len(
                        original[fold.index].content or ""
                    ),
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
                    stats_after["tombstone_count"]
                    - stats_before["tombstone_count"]
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
    __slots__ = ("index", "call", "superseded_by_index")

    def __init__(
        self,
        index: int,
        call: ToolCall,
        superseded_by_index: int,
    ) -> None:
        self.index = index
        self.call = call
        self.superseded_by_index = superseded_by_index


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
    for pair in list(pairs)[-keep_recent:]:
        if pair.call.id:
            protected.add(pair.call.id)
    return protected


def _newest_pair_by_key(pairs: Sequence[_Pair]) -> dict[str, _Pair]:
    newest: dict[str, _Pair] = {}
    for pair in pairs:
        key = _pair_key(pair.call)
        existing = newest.get(key)
        if existing is None or pair.index > existing.index:
            newest[key] = pair
    return newest


def _pair_key(call: ToolCall) -> str:
    canonical_args = json.dumps(
        _jsonable(call.arguments),
        sort_keys=True,
        separators=(",", ":"),
    )
    return f"{call.name}\x00{canonical_args}"


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
        f"superseded by a newer call with identical arguments later in this "
        f"conversation (message #{superseded_by_index}).]"
    )


def _tombstone_size(call: ToolCall) -> int:
    return len(_tombstone_text(call, 0))


def _history_stats(
    messages: Sequence[LLMMessage],
    pairs: Sequence[_Pair],
) -> dict[str, int]:
    tombstone_count = sum(
        1 for p in pairs if _is_tombstone(p.result.content)
    )
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
            total += len(
                json.dumps(_jsonable(call.arguments), separators=(",", ":"))
            )
    return total


__all__ = [
    "DEFAULT_BUDGET_BYTES",
    "DEFAULT_KEEP_RECENT",
    "DEFAULT_MIN_SAVINGS_BYTES",
    "TOMBSTONE_PREFIX",
    "compact_history",
]
