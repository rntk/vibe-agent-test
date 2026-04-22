from __future__ import annotations

from pathlib import Path

import pytest

from cagent.llm import (
    BASH_TOOL,
    WRITE_FILE_TOOL,
    LLMClient,
    LLMRequest,
    LLMResponse,
    ToolCall,
)
from cagent.tools import (
    ADVISOR_TOOL,
    BUILTIN_TOOLS,
    IMPLEMENTATION_TOOLS,
    PLAN_TOOLS,
    _format_bash_output,
    advisor,
    bash,
    run_tool,
    run_tool_call,
    write_file,
)


class FakeLLMClient(LLMClient):
    """Fake LLM client that records requests and returns one response."""

    def __init__(self, response: LLMResponse) -> None:
        self.response = response
        self.requests: list[LLMRequest] = []

    def _complete(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        return self.response


def test_bash_returns_exit_code_stdout_and_stderr() -> None:
    result = bash("printf normal; printf error >&2; exit 7")

    assert "exit_code: 7\n" in result
    assert "stdout:\nnormal\n" in result
    assert "stderr:\nerror" in result


def test_bash_skips_empty_stdout() -> None:
    result = bash("printf error >&2; exit 1")

    assert result == "exit_code: 1\nstderr:\nerror"


def test_bash_skips_empty_stderr() -> None:
    result = bash("printf normal; exit 0")

    assert result == "exit_code: 0\nstdout:\nnormal"


def test_bash_skips_empty_stdout_and_stderr() -> None:
    result = bash("exit 0")

    assert result == "exit_code: 0"


def test_run_tool_dispatches_bash_with_cwd(tmp_path: Path) -> None:
    result = run_tool(
        "bash",
        {"command": "pwd", "cwd": str(tmp_path), "timeout_seconds": 5},
    )

    assert "exit_code: 0\n" in result
    assert f"stdout:\n{tmp_path}\n" in result


def test_bash_output_not_truncated_when_within_limit() -> None:
    stdout = "\n".join(f"line {i}" for i in range(998))
    result = _format_bash_output(exit_code=0, stdout=stdout, stderr="")

    assert "[WARNING]" not in result
    assert result.count("\n") == 999


def test_bash_output_truncated_when_exceeds_limit() -> None:
    stdout = "\n".join(f"line {i}" for i in range(999))
    result = _format_bash_output(exit_code=0, stdout=stdout, stderr="")

    lines = result.splitlines()
    assert lines[0].startswith("[WARNING]")
    assert "truncated" in lines[0].lower()
    assert "1000 lines" in lines[0]
    assert len(lines) == 1002  # warning + blank + 1000 truncated lines


def test_bash_output_truncated_when_exceeds_byte_size() -> None:
    # Few lines but each line is large enough to push total over 50 KB
    long_line = "x" * 1024
    stdout = "\n".join(long_line for _ in range(60))
    result = _format_bash_output(exit_code=0, stdout=stdout, stderr="")

    lines = result.splitlines()
    assert lines[0].startswith("[WARNING]")
    assert "50 KB" in lines[0]
    assert len(result.encode("utf-8")) <= 50 * 1024 + len(lines[0].encode("utf-8")) + 2


def test_write_file_creates_file_and_directories(tmp_path: Path) -> None:
    file_path = tmp_path / "nested" / "file.txt"

    result = write_file(str(file_path), "hello world")

    assert result == f"Wrote to {file_path}"
    assert file_path.read_text(encoding="utf-8") == "hello world"


def test_write_file_appends_content(tmp_path: Path) -> None:
    file_path = tmp_path / "file.txt"
    file_path.write_text("first ", encoding="utf-8")

    result = write_file(str(file_path), "second", append=True)

    assert result == f"Appended to {file_path}"
    assert file_path.read_text(encoding="utf-8") == "first second"


def test_write_file_replaces_line_range(tmp_path: Path) -> None:
    file_path = tmp_path / "file.txt"
    file_path.write_text("alpha\nbeta\ngamma\ndelta\n", encoding="utf-8")

    result = write_file(str(file_path), "BETA\nGAMMA\n", start_line=2, end_line=3)

    assert result == f"Updated lines 2-3 in {file_path}"
    assert file_path.read_text(encoding="utf-8") == "alpha\nBETA\nGAMMA\ndelta\n"


def test_write_file_replaces_open_ended_line_ranges(tmp_path: Path) -> None:
    file_path = tmp_path / "file.txt"
    file_path.write_text("alpha\nbeta\ngamma\n", encoding="utf-8")

    first_result = write_file(str(file_path), "ALPHA\nBETA\n", end_line=2)
    second_result = write_file(str(file_path), "GAMMA", start_line=3)

    assert first_result == f"Updated lines 1-2 in {file_path}"
    assert second_result == f"Updated lines 3-3 in {file_path}"
    assert file_path.read_text(encoding="utf-8") == "ALPHA\nBETA\nGAMMA"


def test_write_file_rejects_invalid_line_updates(tmp_path: Path) -> None:
    file_path = tmp_path / "file.txt"
    file_path.write_text("alpha\nbeta\n", encoding="utf-8")

    with pytest.raises(ValueError, match="append"):
        write_file(str(file_path), "new", append=True, start_line=1)

    with pytest.raises(ValueError, match="start_line"):
        write_file(str(file_path), "new", start_line=3)

    with pytest.raises(ValueError, match="end_line"):
        write_file(str(file_path), "new", end_line=3)


def test_run_tool_dispatches_write_file(tmp_path: Path) -> None:
    file_path = tmp_path / "file.txt"

    result = run_tool(
        "write_file",
        {"path": str(file_path), "content": "via tool"},
    )

    assert result == f"Wrote to {file_path}"
    assert file_path.read_text(encoding="utf-8") == "via tool"


def test_run_tool_dispatches_write_file_line_update(tmp_path: Path) -> None:
    file_path = tmp_path / "file.txt"
    file_path.write_text("alpha\nbeta\ngamma\n", encoding="utf-8")

    result = run_tool(
        "write_file",
        {
            "path": str(file_path),
            "content": "BETA",
            "start_line": 2,
            "end_line": 2,
        },
    )

    assert result == f"Updated lines 2-2 in {file_path}"
    assert file_path.read_text(encoding="utf-8") == "alpha\nBETA\ngamma\n"


def test_run_tool_call_write_file(tmp_path: Path) -> None:
    file_path = tmp_path / "file.txt"

    result = run_tool_call(
        ToolCall(
            name="write_file",
            arguments={"path": str(file_path), "content": "tool call"},
        )
    )

    assert result == f"Wrote to {file_path}"
    assert file_path.read_text(encoding="utf-8") == "tool call"


def test_advisor_uses_client_with_prompt() -> None:
    client = FakeLLMClient(LLMResponse(content="Check the import path."))

    result = advisor("I am stuck on an import error.", client)

    assert result == "Check the import path."
    assert client.requests[0].user_prompt == "I am stuck on an import error."
    assert client.requests[0].system_prompt


def test_run_tool_dispatches_advisor_with_smart_client() -> None:
    client = FakeLLMClient(LLMResponse(content="Use the local helper."))

    result = run_tool(
        "advisor",
        {"prompt": "What should I do next?"},
        advisor_client=client,
    )

    assert result == "Use the local helper."
    assert client.requests[0].user_prompt == "What should I do next?"


def test_advisor_reports_unavailable_without_client() -> None:
    result = advisor("Need help.", None)

    assert "SMART_API is not configured" in result


def test_bash_tool_is_registered() -> None:
    assert BASH_TOOL in BUILTIN_TOOLS
    assert ADVISOR_TOOL in BUILTIN_TOOLS
    assert WRITE_FILE_TOOL in BUILTIN_TOOLS
    assert BASH_TOOL.parameters["required"] == ["command"]
    assert ADVISOR_TOOL.parameters["required"] == ["prompt"]
    assert WRITE_FILE_TOOL.parameters["required"] == ["path", "content"]


def test_plan_tools_excludes_write_file() -> None:
    assert BASH_TOOL in PLAN_TOOLS
    assert ADVISOR_TOOL in PLAN_TOOLS
    assert WRITE_FILE_TOOL not in PLAN_TOOLS


def test_implementation_tools_includes_write_file() -> None:
    assert BASH_TOOL in IMPLEMENTATION_TOOLS
    assert ADVISOR_TOOL in IMPLEMENTATION_TOOLS
    assert WRITE_FILE_TOOL in IMPLEMENTATION_TOOLS
