from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from cagent.compaction import (
    DEFAULT_BUDGET_BYTES,
    TOMBSTONE_PREFIX,
    _Pair,
    _protected_tool_call_ids,
    compact_history,
)
from cagent.llm import LLMMessage, ToolCall
from cagent.tracing import Trace, reset_trace, set_trace


def _pair(
    index: int, tool: str, args: Mapping[str, Any], content: str
) -> list[LLMMessage]:
    call_id = f"call_{index}"
    return [
        LLMMessage(
            role="assistant",
            content=None,
            tool_calls=(ToolCall(id=call_id, name=tool, arguments=args),),
        ),
        LLMMessage(role="tool", content=content, tool_call_id=call_id),
    ]


def test_skips_when_history_under_budget() -> None:
    messages = [
        LLMMessage(role="system", content="sys"),
        LLMMessage(role="user", content="hello"),
        *_pair(0, "bash", {"command": "ls"}, "short output"),
    ]
    result = compact_history(messages, budget_bytes=10_000)
    assert result == tuple(messages)


def test_folds_duplicate_older_tool_result() -> None:
    big = "X" * 5000
    messages = [
        LLMMessage(role="system", content="sys"),
        LLMMessage(role="user", content="hi"),
        *_pair(0, "read", {"path": "a.py"}, big),
        *_pair(1, "bash", {"command": "ls"}, "a.py"),
        *_pair(2, "read", {"path": "a.py"}, big + "!"),
    ]
    result = compact_history(
        messages,
        keep_recent=1,
        min_savings_bytes=100,
        budget_bytes=1000,
    )
    # The old read result (index 3) should be tombstoned.
    old_tool_msg = result[3]
    assert old_tool_msg.role == "tool"
    assert old_tool_msg.tool_call_id == "call_0"
    assert old_tool_msg.content.startswith(TOMBSTONE_PREFIX)
    # The newest read result (last message) must be untouched.
    assert result[-1].content == big + "!"
    # The unrelated bash call must be untouched.
    assert result[5].content == "a.py"


def test_keeps_recent_duplicates_untouched() -> None:
    big = "Y" * 5000
    # Two identical reads, both within keep_recent window.
    messages = [
        LLMMessage(role="system", content="sys"),
        LLMMessage(role="user", content="hi"),
        *_pair(0, "read", {"path": "a.py"}, big),
        *_pair(1, "read", {"path": "a.py"}, big),
    ]
    result = compact_history(
        messages,
        keep_recent=8,
        min_savings_bytes=100,
        budget_bytes=1000,
    )
    assert result == tuple(messages)


def test_respects_min_savings_threshold() -> None:
    small = "x" * 50
    messages = [
        LLMMessage(role="system", content="sys"),
        LLMMessage(role="user", content="hi"),
        *_pair(0, "read", {"path": "a.py"}, small),
        *_pair(1, "b", {}, "x" * DEFAULT_BUDGET_BYTES),
        *_pair(2, "read", {"path": "a.py"}, small),
    ]
    result = compact_history(
        messages,
        keep_recent=1,
        min_savings_bytes=10_000,
        budget_bytes=100,
    )
    # Both reads are smaller than the threshold -> no fold.
    assert result[3].content == small
    assert result[-1].content == small


def test_does_not_refold_existing_tombstone() -> None:
    big = "Z" * 5000
    messages = [
        LLMMessage(role="system", content="sys"),
        LLMMessage(role="user", content="hi"),
        *_pair(0, "read", {"path": "a.py"}, big),
        *_pair(1, "read", {"path": "a.py"}, big),
    ]
    once = compact_history(
        messages,
        keep_recent=1,
        min_savings_bytes=100,
        budget_bytes=100,
    )
    twice = compact_history(
        list(once),
        keep_recent=1,
        min_savings_bytes=100,
        budget_bytes=100,
    )
    assert once == twice


def test_folds_fuzzy_match_on_bash_command() -> None:
    big_a = "A" * 5000
    big_b = "B" * 5000
    # Two bash calls against near-identical targets; ratio should clear 0.85.
    messages = [
        LLMMessage(role="system", content="sys"),
        LLMMessage(role="user", content="hi"),
        *_pair(0, "bash", {"command": "cat   /app/cagent/main.py"}, big_a),
        *_pair(1, "bash", {"command": "cat /app/cagent/main.py"}, big_b),
    ]
    result = compact_history(
        messages,
        keep_recent=1,
        min_savings_bytes=100,
        budget_bytes=1000,
        similarity_threshold=0.85,
    )
    assert result[3].content.startswith(TOMBSTONE_PREFIX)
    assert result[-1].content == big_b


def test_does_not_fold_dissimilar_commands() -> None:
    big = "C" * 5000
    messages = [
        LLMMessage(role="system", content="sys"),
        LLMMessage(role="user", content="hi"),
        *_pair(0, "bash", {"command": "ls /app"}, big),
        *_pair(1, "bash", {"command": "find / -name '*.py' | head"}, big),
    ]
    result = compact_history(
        messages,
        keep_recent=1,
        min_savings_bytes=100,
        budget_bytes=1000,
        similarity_threshold=0.85,
    )
    # Commands are too different — neither should be folded.
    assert result[3].content == big
    assert result[-1].content == big


def test_does_not_fold_across_different_tool_names() -> None:
    big = "D" * 5000
    messages = [
        LLMMessage(role="system", content="sys"),
        LLMMessage(role="user", content="hi"),
        *_pair(0, "bash", {"command": "cat a.py"}, big),
        *_pair(1, "write_file", {"path": "a.py", "content": "x"}, big),
    ]
    result = compact_history(
        messages,
        keep_recent=1,
        min_savings_bytes=100,
        budget_bytes=1000,
        similarity_threshold=0.5,
    )
    assert result[3].content == big
    assert result[-1].content == big


def test_emits_trace_span() -> None:
    trace = Trace()
    token = set_trace(trace)
    try:
        big = "W" * 5000
        messages = [
            LLMMessage(role="system", content="sys"),
            LLMMessage(role="user", content="hi"),
            *_pair(0, "read", {"path": "a.py"}, big),
            *_pair(1, "read", {"path": "a.py"}, big),
        ]
        compact_history(
            messages,
            keep_recent=1,
            min_savings_bytes=100,
            budget_bytes=100,
        )
    finally:
        reset_trace(token)

    spans = [s for s in trace.spans if s.name == "compaction.run"]
    assert len(spans) == 1
    attrs = spans[0].attributes
    assert attrs["triggered"] is True
    assert attrs["reason"] == "folded"
    assert attrs["folded_count"] == 1
    before = attrs["before"]
    after = attrs["after"]
    delta = attrs["delta"]
    assert before["tool_result_count"] == 2
    assert before["active_tool_result_count"] == 2
    assert before["tombstone_count"] == 0
    assert after["tool_result_count"] == 2
    assert after["tombstone_count"] == 1
    assert after["active_tool_result_count"] == 1
    assert delta["bytes_saved"] > 0
    assert delta["bytes_saved_pct"] > 0
    assert delta["tombstones_added"] == 1
    assert delta["active_tool_results_removed"] == 1
    fold = attrs["folded"][0]
    assert fold["bytes_before"] > fold["bytes_after"]


def _assistant_with_reasoning(
    index: int | str,
    tool: str,
    args: Mapping[str, Any],
) -> LLMMessage:
    """Create an assistant message with reasoning and a single tool call."""
    call_id = f"call_{index}"
    return LLMMessage(
        role="assistant",
        content=None,
        tool_calls=(ToolCall(id=call_id, name=tool, arguments=args),),
        reasoning=f"Thinking about {tool} call",
    )


def _tool_result_msg(index: int | str, content: str) -> LLMMessage:
    return LLMMessage(role="tool", content=content, tool_call_id=f"call_{index}")


def test_strips_reasoning_when_single_tool_call_folded() -> None:
    """Strip reasoning when an assistant's only tool call is folded."""
    big = "X" * 5000
    messages = [
        LLMMessage(role="system", content="sys"),
        LLMMessage(role="user", content="hi"),
        _assistant_with_reasoning(0, "read", {"path": "a.py"}),
        _tool_result_msg(0, big),
        _assistant_with_reasoning(1, "read", {"path": "a.py"}),
        _tool_result_msg(1, big + "!"),  # different enough to not be identical
    ]
    result = compact_history(
        messages,
        keep_recent=1,
        min_savings_bytes=100,
        budget_bytes=1000,
    )

    # The older tool result (index 3) should be tombstoned.
    assert result[3].content.startswith(TOMBSTONE_PREFIX)
    # The assistant message at index 2 (which initiated the folded call)
    # should have its reasoning stripped.
    assert result[2].reasoning is None
    # The newer assistant message (index 4) should keep its reasoning.
    assert result[4].reasoning is not None


def test_strips_reasoning_when_all_multiple_tool_calls_folded() -> None:
    """Strip reasoning when all assistant tool calls are folded."""
    big = "X" * 5000
    messages = [
        LLMMessage(role="system", content="sys"),
        LLMMessage(role="user", content="hi"),
        # Assistant with two tool calls
        LLMMessage(
            role="assistant",
            content=None,
            tool_calls=(
                ToolCall(id="call_a", name="read", arguments={"path": "a.py"}),
                ToolCall(id="call_b", name="bash", arguments={"command": "ls"}),
            ),
            reasoning="Let me read the file and list files.",
        ),
        _tool_result_msg("a", big),
        _tool_result_msg("b", big),  # large enough to make folding worthwhile
        # Later superseding pair
        _assistant_with_reasoning(1, "read", {"path": "a.py"}),
        _tool_result_msg(1, big + "!"),
        _assistant_with_reasoning(2, "bash", {"command": "ls"}),
        _tool_result_msg(2, big + "?"),
    ]
    result = compact_history(
        messages,
        keep_recent=0,
        min_savings_bytes=1,
        budget_bytes=100,
    )

    # Both older tool results should be tombstoned.
    assert result[3].content.startswith(TOMBSTONE_PREFIX)
    assert result[4].content.startswith(TOMBSTONE_PREFIX)
    # The assistant message at index 2 should have reasoning stripped.
    assert result[2].reasoning is None


def test_strips_reasoning_when_any_tool_call_folded() -> None:
    """When any tool call is folded, signed reasoning is stripped."""
    big = "X" * 5000
    messages = [
        LLMMessage(role="system", content="sys"),
        LLMMessage(role="user", content="hi"),
        # Assistant with two tool calls: read (will be folded) and write (unique)
        LLMMessage(
            role="assistant",
            content=None,
            tool_calls=(
                ToolCall(id="call_a", name="read", arguments={"path": "a.py"}),
                ToolCall(
                    id="call_b",
                    name="write_file",
                    arguments={"path": "b.txt", "content": "x"},
                ),
            ),
            reasoning="Read a.py then write b.txt.",
            thought_signature="sig-bound-to-partial-chain",
        ),
        _tool_result_msg("a", big),
        _tool_result_msg("b", big),  # large enough but no superseder for write_file
        # Later superseding read, but no superseder for write_file
        _assistant_with_reasoning(1, "read", {"path": "a.py"}),
        _tool_result_msg(1, big + "!"),
    ]
    result = compact_history(
        messages,
        keep_recent=0,
        min_savings_bytes=1,
        budget_bytes=100,
    )

    # The read tool result is tombstoned, but write_file result remains.
    assert result[3].content.startswith(TOMBSTONE_PREFIX)
    assert result[4].content == big
    # Any tombstoned result breaks the signed reasoning chain.
    assert result[2].reasoning is None
    assert result[2].thought_signature is None


def test_strips_reasoning_with_pre_existing_tombstone() -> None:
    """When a previous pass already tombstoned one result, and this pass
    folds the remaining one, reasoning is stripped."""
    big = "Y" * 5000
    messages = [
        LLMMessage(role="system", content="sys"),
        LLMMessage(role="user", content="hi"),
        # Assistant with two tool calls
        LLMMessage(
            role="assistant",
            content=None,
            tool_calls=(
                ToolCall(id="call_x", name="read", arguments={"path": "a.py"}),
                ToolCall(id="call_y", name="read", arguments={"path": "b.py"}),
            ),
            reasoning="Read both files.",
        ),
        # call_x result is already a tombstone from a previous compaction
        LLMMessage(
            role="tool",
            content=f"{TOMBSTONE_PREFIX} tool=read output hidden; superseded earlier.]",
            tool_call_id="call_x",
        ),
        _tool_result_msg("y", big),
        # Later superseder for call_y (b.py)
        _assistant_with_reasoning(1, "read", {"path": "b.py"}),
        _tool_result_msg(1, big + "!"),
    ]
    result = compact_history(
        messages,
        keep_recent=0,
        min_savings_bytes=1,
        budget_bytes=100,
    )

    # The remaining active tool result (call_y) should now be tombstoned.
    assert result[4].content.startswith(TOMBSTONE_PREFIX)
    # Reasoning should be stripped since both tool calls are now folded.
    assert result[2].reasoning is None


def test_protected_tool_call_ids_keep_recent_zero() -> None:
    """keep_recent=0 must protect nothing (regression: pairs[-0:] == pairs[:])."""
    pairs = [
        _Pair(
            index=i,
            call=ToolCall(id=f"call_{i}", name="read", arguments={}),
            result=LLMMessage(role="tool", content="x", tool_call_id=f"call_{i}"),
        )
        for i in range(3)
    ]
    assert _protected_tool_call_ids(pairs, 0) == set()
    assert _protected_tool_call_ids(pairs, 1) == {"call_2"}


def test_strips_signature_when_reasoning_stripped() -> None:
    """When reasoning is removed, the bound thought_signature must also be cleared."""
    big = "X" * 5000
    messages = [
        LLMMessage(role="system", content="sys"),
        LLMMessage(role="user", content="hi"),
        LLMMessage(
            role="assistant",
            content=None,
            tool_calls=(
                ToolCall(id="call_0", name="read", arguments={"path": "a.py"}),
            ),
            reasoning="Thinking",
            thought_signature="sig-bound-to-reasoning",
        ),
        LLMMessage(role="tool", content=big, tool_call_id="call_0"),
        _assistant_with_reasoning(1, "read", {"path": "a.py"}),
        _tool_result_msg(1, big + "!"),
    ]
    result = compact_history(
        messages,
        keep_recent=1,
        min_savings_bytes=100,
        budget_bytes=1000,
    )
    assert result[3].content.startswith(TOMBSTONE_PREFIX)
    assert result[2].reasoning is None
    assert result[2].thought_signature is None
