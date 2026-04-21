from __future__ import annotations

from pathlib import Path

import pytest

from cagent.llm import BASH_TOOL, READ_FILE_TOOL, ToolCall
from cagent.tools import BUILTIN_TOOLS, bash, read_file, run_tool, run_tool_call


def test_read_file_reads_full_file(tmp_path: Path) -> None:
    file_path = tmp_path / "sample.txt"
    file_path.write_text("alpha\nbeta\ngamma\n", encoding="utf-8")

    assert read_file(str(file_path)) == "alpha\nbeta\ngamma\n"


def test_read_file_reads_line_range(tmp_path: Path) -> None:
    file_path = tmp_path / "sample.txt"
    file_path.write_text("alpha\nbeta\ngamma\n", encoding="utf-8")

    assert read_file(str(file_path), start_line=2, end_line=3) == "beta\ngamma\n"
    assert (
        run_tool(
            "read_file",
            {"path": str(file_path), "start_line": 2, "end_line": 2},
        )
        == "beta\n"
    )


def test_read_file_accepts_open_ended_ranges(tmp_path: Path) -> None:
    file_path = tmp_path / "sample.txt"
    file_path.write_text("alpha\nbeta\ngamma\n", encoding="utf-8")

    assert read_file(str(file_path), end_line=2) == "alpha\nbeta\n"
    assert read_file(str(file_path), start_line=3) == "gamma\n"


def test_run_tool_call_reads_file_range(tmp_path: Path) -> None:
    file_path = tmp_path / "sample.txt"
    file_path.write_text("alpha\nbeta\ngamma\n", encoding="utf-8")

    result = run_tool_call(
        ToolCall(
            name="read_file",
            arguments={"path": str(file_path), "start_line": 1, "end_line": 1},
        )
    )

    assert result == "alpha\n"


def test_bash_returns_exit_code_stdout_and_stderr() -> None:
    result = bash("printf normal; printf error >&2; exit 7")

    assert "exit_code: 7\n" in result
    assert "stdout:\nnormal\n" in result
    assert "stderr:\nerror" in result


def test_run_tool_dispatches_bash_with_cwd(tmp_path: Path) -> None:
    result = run_tool(
        "bash",
        {"command": "pwd", "cwd": str(tmp_path), "timeout_seconds": 5},
    )

    assert "exit_code: 0\n" in result
    assert f"stdout:\n{tmp_path}\n" in result


def test_read_file_rejects_invalid_ranges(tmp_path: Path) -> None:
    file_path = tmp_path / "sample.txt"
    file_path.write_text("alpha\nbeta\n", encoding="utf-8")

    with pytest.raises(ValueError, match="start_line"):
        read_file(str(file_path), start_line=0)

    with pytest.raises(ValueError, match="end_line"):
        read_file(str(file_path), start_line=2, end_line=1)


def test_read_file_tool_is_registered() -> None:
    assert READ_FILE_TOOL in BUILTIN_TOOLS
    assert BASH_TOOL in BUILTIN_TOOLS
    assert READ_FILE_TOOL.parameters["required"] == ["path"]
    assert BASH_TOOL.parameters["required"] == ["command"]
