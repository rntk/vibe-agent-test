from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from cagent.llm import LLMClient, LLMRequest, LLMResponse, ToolCall
from cagent.llm.anthropic import AnthropicClient
from cagent.llm.openai import OpenAIChatCompletionsClient
from cagent.main import (
    EchoLLMClient,
    create_fast_api_client,
    create_smart_api_client,
    main,
    run_implementation_mode,
    run_plan_mode,
)


class FakeLLMClient(LLMClient):
    """Fake LLM client that returns pre-configured responses."""

    def __init__(self, responses: list[LLMResponse]) -> None:
        self.responses = responses
        self.call_index = 0
        self.requests: list[LLMRequest] = []

    def _complete(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        response = self.responses[self.call_index]
        self.call_index += 1
        return response


def test_run_implementation_mode_prints_final_content(
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    prompt_file = tmp_path / "prompt.md"
    prompt_file.write_text("Implement something.", encoding="utf-8")

    client = FakeLLMClient([LLMResponse(content="All done.")])

    with (
        patch("cagent.main.create_fast_api_client", return_value=client),
        patch("cagent.main.create_smart_api_client", return_value=None),
    ):
        run_implementation_mode(str(prompt_file))

    captured = capsys.readouterr()
    assert captured.out.strip() == "All done."
    assert client.call_index == 1
    assert client.requests[0].tools


def test_run_implementation_mode_calls_tools_and_loops(
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    prompt_file = tmp_path / "prompt.md"
    prompt_file.write_text("Write a file.", encoding="utf-8")

    target_file = tmp_path / "output.txt"

    responses = [
        LLMResponse(
            content=None,
            tool_calls=[
                ToolCall(
                    id="call_1",
                    name="write_file",
                    arguments={"path": str(target_file), "content": "hello"},
                )
            ],
        ),
        LLMResponse(content="File written successfully."),
    ]
    client = FakeLLMClient(responses)

    with (
        patch("cagent.main.create_fast_api_client", return_value=client),
        patch("cagent.main.create_smart_api_client", return_value=None),
    ):
        run_implementation_mode(str(prompt_file))

    captured = capsys.readouterr()
    assert captured.out.strip() == "File written successfully."
    assert client.call_index == 2
    assert target_file.read_text(encoding="utf-8") == "hello"

    # Verify conversation history includes tool result
    second_request = client.requests[1]
    roles = [msg.role for msg in second_request.all_messages()]
    assert "tool" in roles


def test_run_implementation_mode_passes_advisor_tool_to_smart_api(
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    prompt_file = tmp_path / "prompt.md"
    prompt_file.write_text("Implement something difficult.", encoding="utf-8")

    fast_client = FakeLLMClient(
        [
            LLMResponse(
                tool_calls=[
                    ToolCall(
                        id="call_1",
                        name="advisor",
                        arguments={"prompt": "How should I approach this?"},
                    )
                ],
            ),
            LLMResponse(content="Used advisor guidance."),
        ]
    )
    smart_client = FakeLLMClient([LLMResponse(content="Prefer the existing helper.")])

    with (
        patch("cagent.main.create_fast_api_client", return_value=fast_client),
        patch("cagent.main.create_smart_api_client", return_value=smart_client),
    ):
        run_implementation_mode(str(prompt_file))

    captured = capsys.readouterr()
    assert captured.out.strip() == "Used advisor guidance."
    assert smart_client.requests[0].user_prompt == "How should I approach this?"
    assert smart_client.requests[0].system_prompt

    second_request = fast_client.requests[1]
    tool_messages = [
        message for message in second_request.all_messages() if message.role == "tool"
    ]
    assert tool_messages[0].content == "Prefer the existing helper."


def test_run_implementation_mode_exits_when_no_client_configured(
    tmp_path: Path,
) -> None:
    prompt_file = tmp_path / "prompt.md"
    prompt_file.write_text("Implement something.", encoding="utf-8")

    with (
        patch("cagent.main.create_fast_api_client", return_value=None),
        pytest.raises(SystemExit) as exc_info,
    ):
        run_implementation_mode(str(prompt_file))

    assert exc_info.value.code == 1


def test_run_implementation_mode_exits_on_max_iterations(tmp_path: Path) -> None:
    prompt_file = tmp_path / "prompt.md"
    prompt_file.write_text("Loop forever.", encoding="utf-8")

    # Always return a tool call so it never finishes
    responses = [
        LLMResponse(
            content=None,
            tool_calls=[
                ToolCall(
                    id=f"call_{i}",
                    name="bash",
                    arguments={"command": f"echo {i}"},
                )
            ],
        )
        for i in range(25)
    ]
    client = FakeLLMClient(responses)

    with (
        patch("cagent.main.create_fast_api_client", return_value=client),
        patch("cagent.main.create_smart_api_client", return_value=None),
        patch("cagent.main.precheck_tool_call", return_value=None),
        pytest.raises(SystemExit) as exc_info,
    ):
        run_implementation_mode(str(prompt_file))

    assert exc_info.value.code == 1
    assert client.call_index == 20


def test_echo_llm_client_returns_user_prompt() -> None:
    client = EchoLLMClient()
    response = client.complete("hello")
    assert response.content == "hello"


def test_create_fast_api_client_allows_openai_without_host(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FAST_API_TYPE", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.delenv("FAST_API_HOST", raising=False)
    monkeypatch.delenv("FAST_API_TOKEN", raising=False)

    client = create_fast_api_client()

    assert isinstance(client, OpenAIChatCompletionsClient)
    assert client.client.api_key == "test-openai-key"


def test_create_smart_api_client_allows_anthropic_without_host(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SMART_API_TYPE", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic-key")
    monkeypatch.delenv("SMART_API_HOST", raising=False)
    monkeypatch.delenv("SMART_API_TOKEN", raising=False)

    client = create_smart_api_client()

    assert isinstance(client, AnthropicClient)


def test_main_writes_trace_to_non_empty_trace_path(
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    trace_file = tmp_path / "trace" / "run.json"

    with (
        patch("sys.argv", ["cagent", "--trace", str(trace_file)]),
        patch("cagent.main.create_smart_api_client", return_value=None),
        patch("cagent.main.create_fast_api_client", return_value=None),
    ):
        main()

    captured = capsys.readouterr()
    assert captured.out.strip() == "Hello, World!"

    data = json.loads(trace_file.read_text(encoding="utf-8"))
    assert data["span_count"] == 2
    assert data["spans"][0]["name"] == "cagent.main"
    assert data["spans"][0]["attributes"]["trace_file"] == str(trace_file)
    html = trace_file.with_suffix(".html").read_text(encoding="utf-8")
    assert "Conversation" in html
    assert "Hello, World!" in html


def test_main_does_not_write_trace_for_empty_trace_path(
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    with (
        patch("sys.argv", ["cagent", "--trace", ""]),
        patch("cagent.main.create_smart_api_client", return_value=None),
        patch("cagent.main.create_fast_api_client", return_value=None),
        patch("pathlib.Path.write_text") as write_text,
    ):
        main()

    captured = capsys.readouterr()
    assert captured.out.strip() == "Hello, World!"
    assert list(tmp_path.iterdir()) == []
    write_text.assert_not_called()


def test_run_plan_mode_saves_plan(
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    import os

    task_file = tmp_path / "task.txt"
    task_file.write_text("test task", encoding="utf-8")

    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()
    (prompts_dir / "plan.md").write_text("Task: {task}", encoding="utf-8")

    client = FakeLLMClient([LLMResponse(content="# Plan\n1. Step one.")])

    original_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        with (
            patch("cagent.main.create_fast_api_client", return_value=client),
            patch("cagent.main.create_smart_api_client", return_value=None),
        ):
            run_plan_mode(str(task_file))
    finally:
        os.chdir(original_cwd)

    captured = capsys.readouterr()
    assert "Plan saved to" in captured.out
    assert (tmp_path / "plans").is_dir()


def test_run_plan_mode_calls_tools_and_loops(
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    import os

    task_file = tmp_path / "task.txt"
    task_file.write_text("read and plan", encoding="utf-8")

    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()
    (prompts_dir / "plan.md").write_text("Task: {task}", encoding="utf-8")

    responses = [
        LLMResponse(
            content=None,
            tool_calls=[
                ToolCall(
                    id="call_1",
                    name="bash",
                    arguments={"command": f"cat {task_file}"},
                )
            ],
        ),
        LLMResponse(content="# Plan\n1. Read the file.\n2. Done."),
    ]
    client = FakeLLMClient(responses)

    original_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        with (
            patch("cagent.main.create_fast_api_client", return_value=client),
            patch("cagent.main.create_smart_api_client", return_value=None),
            patch("cagent.main.precheck_tool_call", return_value=None),
        ):
            run_plan_mode(str(task_file))
    finally:
        os.chdir(original_cwd)

    captured = capsys.readouterr()
    assert "Plan saved to" in captured.out
    assert client.call_index == 2

    # Verify conversation history includes tool result
    second_request = client.requests[1]
    roles = [msg.role for msg in second_request.all_messages()]
    assert "tool" in roles


def test_run_plan_mode_exits_on_max_iterations(
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    import os

    task_file = tmp_path / "task.txt"
    task_file.write_text("loop forever", encoding="utf-8")

    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()
    (prompts_dir / "plan.md").write_text("Task: {task}", encoding="utf-8")

    # Always return a tool call so it never finishes
    responses = [
        LLMResponse(
            content=None,
            tool_calls=[
                ToolCall(
                    id=f"call_{i}",
                    name="bash",
                    arguments={"command": f"echo {i}"},
                )
            ],
        )
        for i in range(25)
    ]
    client = FakeLLMClient(responses)

    original_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        with (
            patch("cagent.main.create_fast_api_client", return_value=client),
            patch("cagent.main.create_smart_api_client", return_value=None),
            patch("cagent.main.precheck_tool_call", return_value=None),
            pytest.raises(SystemExit) as exc_info,
        ):
            run_plan_mode(str(task_file))
    finally:
        os.chdir(original_cwd)

    assert exc_info.value.code == 1
    assert client.call_index == 20


def test_run_plan_mode_exits_when_no_client_configured(tmp_path: Path) -> None:
    task_file = tmp_path / "task.txt"
    task_file.write_text("test task", encoding="utf-8")

    with (
        patch("cagent.main.create_fast_api_client", return_value=None),
        pytest.raises(SystemExit) as exc_info,
    ):
        run_plan_mode(str(task_file))

    assert exc_info.value.code == 1
