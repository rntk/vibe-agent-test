from __future__ import annotations

from cagent.compaction import (
    DEFAULT_BUDGET_BYTES,
    TOMBSTONE_PREFIX,
    compact_history,
)
from cagent.llm import LLMMessage, ToolCall
from cagent.tracing import Trace, reset_trace, set_trace


def _pair(index: int, tool: str, args: dict, content: str) -> list[LLMMessage]:
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
    assert attrs["folded_count"] == 1
    assert attrs["bytes_saved"] > 0
