from __future__ import annotations

import json
from pathlib import Path

from cagent.llm import LLMClient, LLMMessage, LLMRequest, LLMResponse, ToolCall
from cagent.tools import run_tool_call
from cagent.tracing import Trace, reset_trace, set_trace, write_trace_html


class TraceFakeLLMClient(LLMClient):
    def _complete(self, request: LLMRequest) -> LLMResponse:
        return LLMResponse(content=f"response to {request.user_prompt}")


def test_trace_collects_nested_llm_and_tool_spans(tmp_path: Path) -> None:
    trace = Trace()
    token = set_trace(trace)
    sample_file = tmp_path / "sample.txt"
    sample_file.write_text("hello\n", encoding="utf-8")

    try:
        with trace.span("root"):
            client = TraceFakeLLMClient()
            response = client.complete("prompt")
            tool_result = run_tool_call(
                ToolCall(name="read_file", arguments={"path": str(sample_file)})
            )
    finally:
        reset_trace(token)

    assert response.content == "response to prompt"
    expected_tool_result = f'<file name="{sample_file}">\n1: hello\n</file>'
    assert tool_result == expected_tool_result
    assert len(trace.roots) == 1
    assert [child.name for child in trace.roots[0].children] == [
        "llm.complete",
        "tool.call",
    ]

    llm_span = trace.roots[0].children[0]
    assert llm_span.attributes["request"]["user_prompt"] == "prompt"
    assert llm_span.attributes["response"]["content"] == "response to prompt"

    tool_span = trace.roots[0].children[1]
    assert tool_span.attributes["tool_name"] == "read_file"
    assert tool_span.attributes["result"] == expected_tool_result
    assert [child.name for child in tool_span.children] == ["tool.run"]
    assert tool_span.children[0].attributes["result"] == expected_tool_result


def test_trace_flush_writes_json_file(tmp_path: Path) -> None:
    trace = Trace()
    output_file = tmp_path / "nested" / "trace.json"

    with trace.span("root"):
        pass
    trace.flush(output_file)

    data = json.loads(output_file.read_text(encoding="utf-8"))
    assert data["span_count"] == 1
    assert data["spans"][0]["name"] == "root"


def test_write_trace_html_renders_conversation_from_trace_file(tmp_path: Path) -> None:
    trace = Trace()
    output_file = tmp_path / "trace.json"

    with trace.span(
        "llm.complete",
        {
            "all_messages": [
                LLMMessage(role="system", content="Be brief."),
                LLMMessage(role="user", content="Hello"),
            ],
            "response": LLMResponse(content="Hi there."),
            "message_count": 2,
        },
    ):
        pass
    trace.flush(output_file)

    html_file = write_trace_html(output_file)
    html = html_file.read_text(encoding="utf-8")

    assert html_file == tmp_path / "trace.html"
    assert "Conversation" in html
    assert "Be brief." in html
    assert "Hello" in html
    assert "Hi there." in html


def test_trace_deduplicates_appended_message_history() -> None:
    trace = Trace()
    first_message = LLMMessage(role="user", content="first prompt")
    second_message = LLMMessage(role="assistant", content="first answer")
    third_message = LLMMessage(role="user", content="continue")

    with trace.span("first", {"all_messages": [first_message, second_message]}):
        pass
    with trace.span(
        "second",
        {"all_messages": [first_message, second_message, third_message]},
    ):
        pass

    first_messages = trace.roots[0].attributes["all_messages"]
    second_messages = trace.roots[1].attributes["all_messages"]

    assert first_messages[0]["content"] == "first prompt"
    assert first_messages[1]["content"] == "first answer"
    assert second_messages[0]["$deduplicated"] is True
    assert second_messages[1]["$deduplicated"] is True
    assert second_messages[0]["$ref"] != second_messages[1]["$ref"]
    assert second_messages[2]["content"] == "continue"
