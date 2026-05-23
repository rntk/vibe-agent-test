from __future__ import annotations

import json
import threading
from http.server import ThreadingHTTPServer
from typing import Any
from urllib.request import urlopen

from cagent.compaction import TOMBSTONE_PREFIX, is_tombstone
from cagent.llm import LLMMessage, ToolCall, ToolDefinition
from cagent.web_mode import (
    WebSession,
    _context_size,
    _make_handler,
    _normalize_for_llm,
)


def test_normalize_for_llm_drops_orphan_tool_messages() -> None:
    messages = [
        LLMMessage(role="user", content="start"),
        LLMMessage(role="tool", content="orphaned", tool_call_id="missing"),
        LLMMessage(
            role="assistant",
            tool_calls=(ToolCall(id="call_1", name="bash", arguments={}),),
        ),
        LLMMessage(role="tool", content="kept", tool_call_id="call_1"),
    ]

    normalized = _normalize_for_llm(messages)

    assert [message.content for message in normalized] == ["start", None, "kept"]


def test_context_size_counts_prompt_tools_and_messages() -> None:
    size = _context_size(
        [LLMMessage(role="user", content="hello world")],
        system_prompt="system",
        tools=[
            ToolDefinition(
                name="example",
                description="Example tool.",
                parameters={"type": "object"},
            )
        ],
    )

    assert size.characters > len("hello world")
    assert size.estimated_tokens == (size.characters + 3) // 4


def test_state_api_includes_status_and_current_context_size() -> None:
    session = WebSession()
    session.add_user_message("current task")
    session.set_status("waiting_for_lm", detail="Agent turn 1")
    handler = _make_handler(
        session,
        system_prompt="system",
        tools=[
            ToolDefinition(
                name="example",
                description="Example tool.",
                parameters={"type": "object"},
            )
        ],
    )
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        host, port = server.server_address
        with urlopen(f"http://{host}:{port}/api/state", timeout=5) as response:
            payload: dict[str, Any] = json.loads(response.read().decode("utf-8"))
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    assert payload["status"] == "waiting_for_lm"
    assert payload["busy"] is True
    assert payload["active_detail"] == "Agent turn 1"
    assert payload["context_size"]["characters"] > len("current task")
    assert payload["context_size"]["estimated_tokens"] > 0


def test_set_status_preserves_detail_when_detail_is_unset() -> None:
    session = WebSession()
    session.set_status("waiting_for_lm", detail="Agent turn 1")

    session.set_status("running_tool")

    assert session.status == "running_tool"
    assert session.active_detail == "Agent turn 1"


def test_pause_request_does_not_hide_active_lm_status() -> None:
    session = WebSession()
    session.set_status("waiting_for_lm", detail="Agent turn 1")

    session.set_paused(True)

    assert session.paused is True
    assert session.status == "waiting_for_lm"
    assert session.busy is True


def test_waiting_status_distinguishes_done_from_initial_input() -> None:
    empty_session = WebSession()
    done_session = WebSession()
    done_session.add_message(LLMMessage(role="assistant", content="Done."))

    with empty_session.cond:
        assert empty_session._waiting_status_locked() == "waiting_for_user"
        assert empty_session._waiting_detail_locked() == (
            "Waiting for initial user input"
        )

    with done_session.cond:
        assert done_session._waiting_status_locked() == "done_waiting_for_user"
        assert done_session._waiting_detail_locked() == (
            "Task done; waiting for user input"
        )


# ---------------------------------------------------------------------------
# Tests for remove_message tombstone behavior
# ---------------------------------------------------------------------------


def _make_tool_conversation() -> WebSession:
    """Return a session with a user message, an assistant with a tool call,
    and a tool result."""
    session = WebSession()
    session.add_user_message("list files")
    session.add_message(
        LLMMessage(
            role="assistant",
            content=None,
            tool_calls=(ToolCall(id="call_1", name="bash", arguments={}),),
        )
    )
    session.add_message(
        LLMMessage(role="tool", content="file1.txt\nfile2.txt", tool_call_id="call_1")
    )
    return session


def _find_stored(session: WebSession, role: str) -> list[Any]:
    """Return stored messages with the given role."""
    return [s for s in session.messages if s.message.role == role]


def test_remove_tool_result_replaces_with_tombstone() -> None:
    """Removing a tool result should replace its content with a tombstone
    instead of deleting the message."""
    session = _make_tool_conversation()
    assert len(session.messages) == 3

    tool_msg = _find_stored(session, "tool")[0]
    removed = session.remove_message(tool_msg.id)
    assert removed is True

    # Message count stays the same (not deleted).
    assert len(session.messages) == 3

    # The tool message is now a tombstone.
    tool_msg_new = _find_stored(session, "tool")[0]
    assert tool_msg_new.message.content is not None
    assert tool_msg_new.message.content.startswith(TOMBSTONE_PREFIX)
    assert is_tombstone(tool_msg_new.message.content)
    assert "bash" in tool_msg_new.message.content
    assert "deleted by user" in tool_msg_new.message.content


def test_remove_user_message_removes_it() -> None:
    """Removing a user message should delete it from the list."""
    session = _make_tool_conversation()
    assert len(session.messages) == 3

    user_msg = _find_stored(session, "user")[0]
    removed = session.remove_message(user_msg.id)
    assert removed is True

    # User message is gone.
    assert len(session.messages) == 2
    assert _find_stored(session, "user") == []


def test_remove_assistant_message_removes_it() -> None:
    """Removing an assistant message should delete it from the list."""
    session = _make_tool_conversation()
    assert len(session.messages) == 3

    assistant_msg = _find_stored(session, "assistant")[0]
    removed = session.remove_message(assistant_msg.id)
    assert removed is True

    # Assistant message is gone.
    assert len(session.messages) == 2
    assert _find_stored(session, "assistant") == []


def test_remove_tool_result_when_assistant_already_removed() -> None:
    """Removing a tool result whose assistant was already removed should
    still tombstone the tool result (orphan tool messages are later
    filtered by _normalize_for_llm)."""
    session = _make_tool_conversation()
    assistant_msg = _find_stored(session, "assistant")[0]
    session.remove_message(assistant_msg.id)

    # Now remove the tool result.
    tool_msg = _find_stored(session, "tool")[0]
    removed = session.remove_message(tool_msg.id)
    assert removed is True

    # Tool result is tombstoned (not deleted).
    tool_msgs = _find_stored(session, "tool")
    assert len(tool_msgs) == 1
    assert is_tombstone(tool_msgs[0].message.content)


def test_normalize_for_llm_keeps_tombstoned_tool_with_matching_assistant() -> None:
    """Tombstoned tool messages that still have a matching assistant
    tool_call should be kept by _normalize_for_llm."""
    session = _make_tool_conversation()
    tool_msg = _find_stored(session, "tool")[0]
    session.remove_message(tool_msg.id)

    normalized = _normalize_for_llm(session.snapshot_messages())

    # Should have 3 messages: user, assistant (with tool_calls), tombstoned tool.
    assert len(normalized) == 3
    tool_msgs = [m for m in normalized if m.role == "tool"]
    assert len(tool_msgs) == 1
    assert is_tombstone(tool_msgs[0].content)


def test_remove_nonexistent_message_returns_false() -> None:
    """Removing a non-existent message ID should return False."""
    session = _make_tool_conversation()
    result = session.remove_message("nonexistent-id")
    assert result is False
    assert len(session.messages) == 3


def test_two_consecutive_tool_removes_tombstone_correctly() -> None:
    """Removing two consecutive tool results should tombstone both."""
    session = WebSession()
    session.add_user_message("do things")
    session.add_message(
        LLMMessage(
            role="assistant",
            content=None,
            tool_calls=(
                ToolCall(id="call_1", name="bash", arguments={}),
                ToolCall(id="call_2", name="read", arguments={"path": "a.py"}),
            ),
        )
    )
    session.add_message(
        LLMMessage(role="tool", content="output1", tool_call_id="call_1")
    )
    session.add_message(
        LLMMessage(role="tool", content="output2", tool_call_id="call_2")
    )

    tool_msgs = _find_stored(session, "tool")
    assert len(tool_msgs) == 2

    # Remove both tool results.
    session.remove_message(tool_msgs[0].id)
    session.remove_message(tool_msgs[1].id)

    # Both should be tombstoned.
    tool_msgs_after = _find_stored(session, "tool")
    assert len(tool_msgs_after) == 2
    assert is_tombstone(tool_msgs_after[0].message.content)
    assert is_tombstone(tool_msgs_after[1].message.content)

    # Assistant message should have its thought_signature stripped
    # (any tombstoned result breaks the chain).
    assistant_msgs = _find_stored(session, "assistant")
    assert assistant_msgs[0].message.thought_signature is None


def test_strip_reasoning_when_all_tool_results_tombstoned() -> None:
    """When all tool results for an assistant message are tombstoned,
    its thought_signature should be cleared."""
    session = WebSession()
    session.add_user_message("do things")
    session.add_message(
        LLMMessage(
            role="assistant",
            content=None,
            reasoning="Let me think about this",
            thought_signature="sig123",
            tool_calls=(
                ToolCall(id="call_1", name="bash", arguments={}),
            ),
        )
    )
    session.add_message(
        LLMMessage(role="tool", content="output1", tool_call_id="call_1")
    )

    tool_msg = _find_stored(session, "tool")[0]
    session.remove_message(tool_msg.id)

    assistant_msgs = _find_stored(session, "assistant")
    assert assistant_msgs[0].message.reasoning == "Let me think about this"
    assert assistant_msgs[0].message.thought_signature is None


def test_strips_signature_when_any_tool_result_tombstoned() -> None:
    """When any tool result for an assistant is tombstoned, the
    thought_signature must be cleared (the chain is broken).
    reasoning is preserved (some providers require it each turn)."""
    session = WebSession()
    session.add_user_message("do things")
    session.add_message(
        LLMMessage(
            role="assistant",
            content=None,
            reasoning="Let me think",
            thought_signature="sig123",
            tool_calls=(
                ToolCall(id="call_1", name="bash", arguments={}),
                ToolCall(id="call_2", name="read", arguments={"path": "a.py"}),
            ),
        )
    )
    session.add_message(
        LLMMessage(role="tool", content="bash output", tool_call_id="call_1")
    )
    session.add_message(
        LLMMessage(role="tool", content="file content", tool_call_id="call_2")
    )

    # Tombstone only one of the two tool results.
    tool_msgs = _find_stored(session, "tool")
    session.remove_message(tool_msgs[0].id)

    # Signature is stripped (chain broken), reasoning is preserved.
    assistant_msgs = _find_stored(session, "assistant")
    assert assistant_msgs[0].message.reasoning == "Let me think"
    assert assistant_msgs[0].message.thought_signature is None


def test_context_size_includes_tombstoned_messages() -> None:
    """Context size calculation should still include tombstoned messages
    (they are counted and have a positive character size)."""
    session = _make_tool_conversation()
    tool_msg = _find_stored(session, "tool")[0]
    session.remove_message(tool_msg.id)
    tombstoned = session.snapshot_messages()
    tombstoned_size = _context_size(tombstoned, system_prompt="sys", tools=[])

    # Tombstoned messages contribute to context size.
    assert tombstoned_size.characters > 0
    assert tombstoned_size.estimated_tokens > 0


def test_remove_tool_result_without_tool_call_id_uses_unknown() -> None:
    """A tool result with no tool_call_id should be tombstoned with
    'unknown' as the tool name."""
    session = WebSession()
    session.add_user_message("hi")
    session.add_message(
        LLMMessage(role="tool", content="some output", tool_call_id=None)
    )

    tool_msg = _find_stored(session, "tool")[0]
    session.remove_message(tool_msg.id)

    tool_msgs = _find_stored(session, "tool")
    assert len(tool_msgs) == 1
    assert is_tombstone(tool_msgs[0].message.content)
    assert "unknown" in tool_msgs[0].message.content
