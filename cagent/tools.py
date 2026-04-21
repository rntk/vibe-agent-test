"""Built-in tool implementations."""

from __future__ import annotations

import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from cagent.llm.base import BASH_TOOL, READ_FILE_TOOL, WRITE_FILE_TOOL, ToolCall, ToolDefinition
from cagent.tracing import get_trace

BUILTIN_TOOLS: tuple[ToolDefinition, ...] = (READ_FILE_TOOL, BASH_TOOL, WRITE_FILE_TOOL)
__all__ = ["BUILTIN_TOOLS", "bash", "read_file", "run_tool", "run_tool_call", "write_file"]


def _optional_int(arguments: Mapping[str, Any], key: str) -> int | None:
    value = arguments.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        msg = f"{key} must be an integer."
        raise TypeError(msg)
    return value


def read_file(
    path: str,
    *,
    start_line: int | None = None,
    end_line: int | None = None,
    encoding: str = "utf-8",
) -> str:
    """Read a whole text file or a 1-based inclusive line range."""

    if start_line is not None and start_line < 1:
        msg = "start_line must be greater than or equal to 1."
        raise ValueError(msg)
    if end_line is not None and end_line < 1:
        msg = "end_line must be greater than or equal to 1."
        raise ValueError(msg)
    if start_line is not None and end_line is not None and end_line < start_line:
        msg = "end_line must be greater than or equal to start_line."
        raise ValueError(msg)

    file_path = Path(path)
    if not file_path.is_file():
        msg = f"File not found: {path}"
        raise FileNotFoundError(msg)

    content = file_path.read_text(encoding=encoding)
    if start_line is None and end_line is None:
        return content

    lines = content.splitlines(keepends=True)
    start_index = 0 if start_line is None else start_line - 1
    end_index = end_line if end_line is not None else len(lines)
    return "".join(lines[start_index:end_index])


def write_file(
    path: str,
    content: str,
    *,
    append: bool = False,
    encoding: str = "utf-8",
) -> str:
    """Write content to a text file, creating parent directories if needed."""

    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with file_path.open(mode, encoding=encoding) as f:
        f.write(content)
    return f"{'Appended to' if append else 'Wrote to'} {path}"


def bash(
    command: str,
    *,
    cwd: str | None = None,
    timeout_seconds: int | None = None,
) -> str:
    """Run a bash command and return exit code, stdout, and stderr."""

    if timeout_seconds is not None and timeout_seconds < 1:
        msg = "timeout_seconds must be greater than or equal to 1."
        raise ValueError(msg)
    if cwd is not None and not Path(cwd).is_dir():
        msg = f"Working directory not found: {cwd}"
        raise FileNotFoundError(msg)

    try:
        result = subprocess.run(
            ["bash", "-lc", command],
            capture_output=True,
            cwd=cwd,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        return _format_bash_output(
            exit_code=None,
            stdout=stdout,
            stderr=f"{stderr}Command timed out after {timeout_seconds} seconds.",
        )

    return _format_bash_output(
        exit_code=result.returncode,
        stdout=result.stdout,
        stderr=result.stderr,
    )


def _format_bash_output(
    *,
    exit_code: int | None,
    stdout: str,
    stderr: str,
) -> str:
    exit_code_text = "timeout" if exit_code is None else str(exit_code)
    return f"exit_code: {exit_code_text}\nstdout:\n{stdout}\nstderr:\n{stderr}"


def run_tool(name: str, arguments: Mapping[str, Any]) -> str:
    """Run a built-in tool by name with provider-decoded arguments."""

    with get_trace().span(
        "tool.run",
        {
            "tool_name": name,
            "arguments": arguments,
        },
    ) as span:
        result = _run_tool(name, arguments)
        span.set_attribute("result", result)
        return result


def _run_tool(name: str, arguments: Mapping[str, Any]) -> str:
    """Run a built-in tool without adding a trace span."""

    if name == READ_FILE_TOOL.name:
        path = arguments.get("path")
        if not isinstance(path, str):
            msg = "path must be a string."
            raise TypeError(msg)
        return read_file(
            path,
            start_line=_optional_int(arguments, "start_line"),
            end_line=_optional_int(arguments, "end_line"),
        )
    if name == BASH_TOOL.name:
        command = arguments.get("command")
        if not isinstance(command, str):
            msg = "command must be a string."
            raise TypeError(msg)
        cwd = arguments.get("cwd")
        if cwd is not None and not isinstance(cwd, str):
            msg = "cwd must be a string."
            raise TypeError(msg)
        return bash(
            command,
            cwd=cwd,
            timeout_seconds=_optional_int(arguments, "timeout_seconds"),
        )
    if name == WRITE_FILE_TOOL.name:
        path = arguments.get("path")
        if not isinstance(path, str):
            msg = "path must be a string."
            raise TypeError(msg)
        content = arguments.get("content")
        if not isinstance(content, str):
            msg = "content must be a string."
            raise TypeError(msg)
        append = arguments.get("append")
        if append is not None and not isinstance(append, bool):
            msg = "append must be a boolean."
            raise TypeError(msg)
        return write_file(path, content, append=bool(append))

    msg = f"Unknown tool: {name}"
    raise ValueError(msg)


def run_tool_call(tool_call: ToolCall) -> str:
    """Run a provider-neutral built-in tool call."""

    with get_trace().span(
        "tool.call",
        {
            "tool_call": tool_call,
            "tool_name": tool_call.name,
            "tool_call_id": tool_call.id,
            "arguments": tool_call.arguments,
        },
    ) as span:
        result = run_tool(tool_call.name, tool_call.arguments)
        span.set_attribute("result", result)
        return result
