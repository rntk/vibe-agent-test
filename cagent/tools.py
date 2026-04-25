"""Built-in tool implementations."""

from __future__ import annotations

import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cagent.llm.base import (
    BASH_TOOL,
    SEARCH_AND_REPLACE_TOOL,
    WRITE_FILE_TOOL,
    LLMClient,
    ToolCall,
    ToolDefinition,
)
from cagent.tracing import get_trace

ADVISOR_TOOL = ToolDefinition(
    name="advisor",
    description=(
        "Ask a smarter, more experienced coding advisor for concise guidance. "
        "Use this when you are stuck, uncertain about an approach, or need help "
        "debugging confusing code. Provide a prompt that describes the goal, "
        "relevant context, what you tried, and the specific confusion point."
    ),
    parameters={
        "type": "object",
        "properties": {
            "prompt": {
                "type": "string",
                "description": (
                    "Question and context for the coding advisor. Include the "
                    "goal, current findings, attempted approach, and the exact "
                    "point where guidance is needed."
                ),
            },
        },
        "required": ["prompt"],
        "additionalProperties": False,
    },
)

ADVISOR_TOOL_SYSTEM_PROMPT = (
    "You are a senior coding advisor for another software engineering agent.\n"
    "Your job is to help when the agent is stuck, uncertain, or needs a second "
    "opinion on implementation or debugging.\n"
    "Be brief and concise. Prefer direct guidance, key risks, and the next "
    "useful action. Do not write long responses, large code blocks, or full "
    "implementations unless absolutely necessary."
)

BUILTIN_TOOLS: tuple[ToolDefinition, ...] = (
    BASH_TOOL,
    ADVISOR_TOOL,
    WRITE_FILE_TOOL,
    SEARCH_AND_REPLACE_TOOL,
)
PLAN_TOOLS: tuple[ToolDefinition, ...] = (BASH_TOOL, ADVISOR_TOOL)
IMPLEMENTATION_TOOLS: tuple[ToolDefinition, ...] = (
    BASH_TOOL,
    ADVISOR_TOOL,
    WRITE_FILE_TOOL,
    SEARCH_AND_REPLACE_TOOL,
)
__all__ = [
    "ADVISOR_TOOL",
    "ADVISOR_TOOL_SYSTEM_PROMPT",
    "BUILTIN_TOOLS",
    "PLAN_TOOLS",
    "IMPLEMENTATION_TOOLS",
    "AdvisorInput",
    "ToolResult",
    "advisor",
    "bash",
    "run_tool",
    "run_tool_call",
    "search_and_replace",
    "write_file",
]


@dataclass(frozen=True, slots=True)
class AdvisorInput:
    """Context passed to the advisor when a tool call looks like it failed."""

    tool_name: str
    command: str
    exit_code: int | None
    stdout: str
    stderr: str


@dataclass(frozen=True, slots=True)
class ToolResult:
    """Result of a built-in tool invocation, checkable for advisor follow-up."""

    output: str
    advisor_input: AdvisorInput | None = None

    def __str__(self) -> str:
        """Return the user-visible tool output."""

        return self.output

    def __contains__(self, item: object) -> bool:
        """Support substring checks against the tool output."""

        if not isinstance(item, str):
            return False
        return item in self.output

    def __eq__(self, other: object) -> bool:
        """Compare directly with strings for compatibility."""

        if isinstance(other, str):
            return self.output == other
        if isinstance(other, ToolResult):
            return (
                self.output == other.output
                and self.advisor_input == other.advisor_input
            )
        return NotImplemented

    def with_advice(self, advice: str | None) -> ToolResult:
        """Return a new ToolResult with an advisor note prepended to output."""

        if not advice:
            return ToolResult(output=self.output)
        note = f"[ADVISOR NOTE] {advice}\n\n"
        return ToolResult(output=note + self.output)


def _optional_int(arguments: Mapping[str, Any], key: str) -> int | None:
    value = arguments.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        msg = f"{key} must be an integer."
        raise TypeError(msg)
    return value


def write_file(
    path: str,
    content: str,
    *,
    append: bool = False,
    start_line: int | None = None,
    end_line: int | None = None,
    encoding: str = "utf-8",
) -> str:
    """Write, append, or replace a 1-based inclusive line range in a text file."""

    if start_line is not None and start_line < 1:
        msg = "start_line must be greater than or equal to 1."
        raise ValueError(msg)
    if end_line is not None and end_line < 1:
        msg = "end_line must be greater than or equal to 1."
        raise ValueError(msg)
    if start_line is not None and end_line is not None and end_line < start_line:
        msg = "end_line must be greater than or equal to start_line."
        raise ValueError(msg)
    if append and (start_line is not None or end_line is not None):
        msg = "append cannot be combined with start_line or end_line."
        raise ValueError(msg)

    file_path = Path(path)

    if start_line is not None or end_line is not None:
        if not file_path.is_file():
            msg = f"File not found: {path}"
            raise FileNotFoundError(msg)
        existing_content = file_path.read_text(encoding=encoding)
        lines = existing_content.splitlines(keepends=True)
        line_count = len(lines)
        range_start = 1 if start_line is None else start_line
        range_end = line_count if end_line is None else end_line
        if range_start > line_count:
            msg = f"start_line must be less than or equal to {line_count}."
            raise ValueError(msg)
        if range_end > line_count:
            msg = f"end_line must be less than or equal to {line_count}."
            raise ValueError(msg)

        replacement = _normalize_line_replacement(
            content,
            has_following_lines=range_end < line_count,
        )
        updated_content = "".join(
            [*lines[: range_start - 1], replacement, *lines[range_end:]]
        )
        file_path.write_text(updated_content, encoding=encoding)
        return f"Updated lines {range_start}-{range_end} in {path}"

    file_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with file_path.open(mode, encoding=encoding) as f:
        f.write(content)
    return f"{'Appended to' if append else 'Wrote to'} {path}"


def search_and_replace(
    path: str,
    old_text: str,
    new_text: str,
    *,
    encoding: str = "utf-8",
) -> str:
    """Replace an exact substring in a text file."""

    if not old_text:
        msg = "old_text must not be empty."
        raise ValueError(msg)

    file_path = Path(path)
    if not file_path.is_file():
        msg = f"File not found: {path}"
        raise FileNotFoundError(msg)

    content = file_path.read_text(encoding=encoding)
    occurrences = content.count(old_text)
    if occurrences == 0:
        msg = f"old_text not found in {path}"
        raise ValueError(msg)
    if occurrences > 1:
        msg = (
            f"old_text appears {occurrences} times in {path}; "
            "provide more context so it matches exactly once."
        )
        raise ValueError(msg)

    updated_content = content.replace(old_text, new_text, 1)
    file_path.write_text(updated_content, encoding=encoding)
    return f"Replaced 1 occurrence in {path}"


def advisor(prompt: str, client: LLMClient | None) -> ToolResult:
    """Ask the SMART_API advisor client for concise coding guidance."""

    if not prompt.strip():
        msg = "prompt must not be empty."
        raise ValueError(msg)
    if client is None:
        return ToolResult(
            output=(
                "Advisor unavailable: SMART_API is not configured. Configure "
                "SMART_API_TYPE/HOST/TOKEN/MODEL to use the advisor tool."
            )
        )

    with get_trace().span("advisor_tool.request", {"prompt": prompt}) as span:
        response = client.complete(
            prompt,
            system_prompt=ADVISOR_TOOL_SYSTEM_PROMPT,
            trace_name="llm.complete.advisor_tool",
            trace_attributes={
                "llm_purpose": "advisor_tool",
            },
        )
        content = (response.content or "").strip()
        output = content or "Advisor returned no guidance."
        span.set_attribute("response", output)
        return ToolResult(output=output)


def _normalize_line_replacement(content: str, *, has_following_lines: bool) -> str:
    if content and has_following_lines and not content.endswith(("\n", "\r")):
        return f"{content}\n"
    return content


def bash(
    command: str,
    *,
    cwd: str | None = None,
    timeout_seconds: int | None = None,
) -> ToolResult:
    """Run a bash command and return a ToolResult with optional advisor context."""

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
        stderr_text = (
            f"{exc.stderr or ''}"
            f"Command timed out after {timeout_seconds} seconds."
        )
        output = _format_bash_output(
            exit_code=None,
            stdout=stdout,
            stderr=stderr_text,
        )
        return ToolResult(
            output=output,
            advisor_input=AdvisorInput(
                tool_name=BASH_TOOL.name,
                command=command,
                exit_code=None,
                stdout=stdout,
                stderr=stderr_text,
            ),
        )

    output = _format_bash_output(
        exit_code=result.returncode,
        stdout=result.stdout,
        stderr=result.stderr,
    )
    advisor_input: AdvisorInput | None = None
    if result.returncode != 0 or result.stderr:
        advisor_input = AdvisorInput(
            tool_name=BASH_TOOL.name,
            command=command,
            exit_code=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
        )
    return ToolResult(output=output, advisor_input=advisor_input)


_MAX_OUTPUT_LINES = 1000
_MAX_OUTPUT_BYTES = 50 * 1024


def _format_bash_output(
    *,
    exit_code: int | None,
    stdout: str,
    stderr: str,
) -> str:
    exit_code_text = "timeout" if exit_code is None else str(exit_code)
    parts = [f"exit_code: {exit_code_text}"]
    if stdout:
        parts.append(f"stdout:\n{stdout}")
    if stderr:
        parts.append(f"stderr:\n{stderr}")
    result = "\n".join(parts)

    truncated_by_lines = False
    lines = result.splitlines()
    if len(lines) > _MAX_OUTPUT_LINES:
        result = "\n".join(lines[:_MAX_OUTPUT_LINES])
        truncated_by_lines = True

    truncated_by_bytes = False
    encoded = result.encode("utf-8")
    if len(encoded) > _MAX_OUTPUT_BYTES:
        result = encoded[:_MAX_OUTPUT_BYTES].decode("utf-8", errors="ignore")
        truncated_by_bytes = True

    if truncated_by_lines or truncated_by_bytes:
        reasons = []
        if truncated_by_lines:
            reasons.append(f"exceeds {_MAX_OUTPUT_LINES} lines")
        if truncated_by_bytes:
            reasons.append(f"exceeds {_MAX_OUTPUT_BYTES // 1024} KB")
        warning = (
            f"[WARNING] The output was truncated because it {' and '.join(reasons)}."
            " Try to make the command more precise to reduce the output size.\n\n"
        )
        return warning + result

    return result


def run_tool(
    name: str,
    arguments: Mapping[str, Any],
    *,
    advisor_client: LLMClient | None = None,
) -> ToolResult:
    """Run a built-in tool by name with provider-decoded arguments."""

    with get_trace().span(
        "tool.run",
        {
            "tool_name": name,
            "arguments": arguments,
        },
    ) as span:
        result = _run_tool(name, arguments, advisor_client=advisor_client)
        span.set_attribute("result", result.output)
        if result.advisor_input is not None:
            span.set_attribute("advisor_requested", True)
        return result


def _run_tool(
    name: str,
    arguments: Mapping[str, Any],
    *,
    advisor_client: LLMClient | None = None,
) -> ToolResult:
    """Run a built-in tool without adding a trace span."""

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
    if name == ADVISOR_TOOL.name:
        prompt = arguments.get("prompt")
        if not isinstance(prompt, str):
            msg = "prompt must be a string."
            raise TypeError(msg)
        return advisor(prompt, advisor_client)
    if name == SEARCH_AND_REPLACE_TOOL.name:
        path = arguments.get("path")
        if not isinstance(path, str):
            msg = "path must be a string."
            raise TypeError(msg)
        old_text = arguments.get("old_text")
        if not isinstance(old_text, str):
            msg = "old_text must be a string."
            raise TypeError(msg)
        new_text = arguments.get("new_text")
        if not isinstance(new_text, str):
            msg = "new_text must be a string."
            raise TypeError(msg)
        output = search_and_replace(path, old_text, new_text)
        return ToolResult(output=output)
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
        output = write_file(
            path,
            content,
            append=bool(append),
            start_line=_optional_int(arguments, "start_line"),
            end_line=_optional_int(arguments, "end_line"),
        )
        return ToolResult(output=output)

    msg = f"Unknown tool: {name}"
    raise ValueError(msg)


def run_tool_call(
    tool_call: ToolCall,
    *,
    advisor_client: LLMClient | None = None,
) -> ToolResult:
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
        result = run_tool(
            tool_call.name,
            tool_call.arguments,
            advisor_client=advisor_client,
        )
        span.set_attribute("result", result.output)
        return result
