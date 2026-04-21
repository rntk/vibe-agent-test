from __future__ import annotations

import json
from pathlib import Path

from cagent.llm import LLMClient, LLMRequest, LLMResponse, ToolCall
from cagent.tools import run_tool_call
from cagent.tracing import Trace, reset_trace, set_trace


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
