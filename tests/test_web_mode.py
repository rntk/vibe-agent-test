from __future__ import annotations

import json
import threading
from http.server import ThreadingHTTPServer
from typing import Any
from urllib.request import urlopen

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
