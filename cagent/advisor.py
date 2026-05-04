"""Advisor that asks an LLM for a hint when a tool call looks like it failed."""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

from cagent.llm.base import (
    BASH_TOOL,
    SEARCH_AND_REPLACE_TOOL,
    WRITE_FILE_TOOL,
    LLMClient,
    LLMMessage,
    ToolCall,
)
from cagent.tools import AdvisorInput, ToolResult
from cagent.tracing import get_trace

__all__ = [
    "ADVISOR_SYSTEM_PROMPT",
    "EDIT_PRECHECK_SYSTEM_PROMPT",
    "PRECHECK_SYSTEM_PROMPT",
    "apply_advisor",
    "precheck_tool_call",
    "request_advice",
    "request_edit_precheck",
    "request_precheck",
]

ADVISOR_SYSTEM_PROMPT = (
    "You are a brief error-diagnosis advisor for a coding agent.\n"
    "You see a single failed tool invocation (command, exit code, stdout, stderr) "
    "with NO conversation history or goal.\n\n"
    "Be extremely concise. One short sentence, high-level. "
    "Point at the likely cause and hint at a direction to try. "
    "Do NOT rewrite the command, do NOT produce fixed code, do NOT list steps, "
    "do NOT go into details or explain the error text. "
    "The agent will decide the exact fix itself.\n"
    "If you have nothing to add beyond what the error text already says, "
    "reply with exactly the single word: None"
)

PRECHECK_SYSTEM_PROMPT = (
    "You are a brief pre-execution reviewer for a coding agent's bash tool.\n"
    "You see one bash command about to run, with NO conversation history or goal.\n\n"
    "Only flag concrete, high-level risks: broken syntax, likely to hang "
    "(interactive prompt, long-running server, waits on stdin/network), or "
    "obviously destructive (broad rm -rf, force-push to main, dropping databases, "
    "wiping disks, disabling security checks, exfiltrating secrets).\n\n"
    "If the command looks fine, reply with exactly the single word: None\n"
    "Otherwise answer in ONE short sentence naming the risk at a high level. "
    "Do NOT rewrite the command, do NOT produce a replacement command, "
    "do NOT list steps or go into details. The agent will decide the exact fix itself."
)

EDIT_PRECHECK_SYSTEM_PROMPT = (
    "You review one pending file edit before a fast agent applies it.\n"
    "Use `bash` to inspect context (neighbors, helpers, existing patterns).\n\n"
    "Flag ONLY clear, evidence-backed architectural problems:\n"
    "- Reimplementing logic that already exists in the codebase\n"
    "- Breaking an established pattern visible in sibling/neighbor code\n"
    "- An obviously unmaintainable shape (e.g. god-function with tangled "
    "responsibilities, magic values that belong in config)\n"
    "- Editing the wrong place for the stated task\n\n"
    "Do NOT block for style, naming, formatting, missing tests, or anything "
    "you cannot prove with file evidence.\n"
    "Default to approving — block only when convinced.\n\n"
    "When done, reply with ONLY a JSON object (no prose, no fences):\n"
    '  {"block": false}                   — let the edit proceed\n'
    '  {"block": true, "comment": "..."}  — block with one short sentence\n\n'
    "`block` must be boolean (default false). Omit `comment` when not blocking."
)

_NONE_MARKERS = {"none", "none.", "n/a", "n/a."}
_EDIT_PRECHECK_MAX_ITERATIONS = 5
_EDIT_PRECHECK_TIME_BUDGET_SECONDS = 60.0
_MAX_FILE_LINES_FOR_PRECHECK = 1000
_MAX_FILE_BYTES_FOR_PRECHECK = 50 * 1024


def request_advice(
    advisor_input: AdvisorInput,
    client: LLMClient,
) -> str | None:
    """Ask the LLM for a short diagnosis; return None if no useful advice."""

    prompt = _format_prompt(advisor_input)
    with get_trace().span(
        "advisor.request",
        {
            "tool_name": advisor_input.tool_name,
            "command": advisor_input.command,
            "exit_code": advisor_input.exit_code,
            "span_context": "advisor",
        },
    ) as span:
        response = client.complete(
            prompt,
            system_prompt=ADVISOR_SYSTEM_PROMPT,
            trace_name="llm.complete.tool_failure_advice",
            trace_attributes={
                "llm_purpose": "tool_failure_advice",
                "tool_name": advisor_input.tool_name,
                "exit_code": advisor_input.exit_code,
                "span_context": "advisor",
            },
        )
        content = (response.content or "").strip()
        advice = _parse_advice(content)
        span.set_attribute("advice", advice)
        return advice


def apply_advisor(
    result: ToolResult,
    client: LLMClient | None,
) -> ToolResult:
    """Run the advisor on a tool result if needed and return the (possibly) annotated result."""

    if result.advisor_input is None or client is None:
        return ToolResult(output=result.output)
    advice = request_advice(result.advisor_input, client)
    return result.with_advice(advice)


def request_precheck(
    command: str,
    client: LLMClient,
) -> str | None:
    """Ask the LLM to review a bash command before it runs; return critical note or None."""

    prompt = f"Command:\n{command}\n"
    with get_trace().span(
        "advisor.precheck",
        {"tool_name": BASH_TOOL.name, "command": command, "span_context": "advisor"},
    ) as span:
        response = client.complete(
            prompt,
            system_prompt=PRECHECK_SYSTEM_PROMPT,
            trace_name="llm.complete.bash_precheck",
            trace_attributes={
                "llm_purpose": "bash_precheck",
                "tool_name": BASH_TOOL.name,
                "span_context": "advisor",
            },
        )
        content = (response.content or "").strip()
        advice = _parse_advice(content)
        span.set_attribute("advice", advice)
        return advice


def precheck_tool_call(
    tool_call: ToolCall,
    client: LLMClient | None,
    *,
    smart_client: LLMClient | None = None,
    task_summary: str | None = None,
) -> ToolResult | None:
    """Pre-execution advisor: block risky tool calls with a critical note, else return None."""

    if tool_call.name == BASH_TOOL.name:
        if client is None:
            return None
        command = tool_call.arguments.get("command")
        if not isinstance(command, str) or not command.strip():
            return None
        advice = request_precheck(command, client)
        if not advice:
            return None
        output = (
            f"[ADVISOR BLOCKED] The pre-execution advisor has a critical comment "
            f"about this bash command and it was NOT executed. Reconsider the "
            f"command (fix syntax, narrow its scope, or pick a safer alternative) "
            f"before retrying.\n\n"
            f"Advisor note: {advice}\n\n"
            f"Command:\n{command}\n"
        )
        return ToolResult(output=output)

    if tool_call.name in (WRITE_FILE_TOOL.name, SEARCH_AND_REPLACE_TOOL.name):
        if smart_client is None:
            return None
        advice = request_edit_precheck(tool_call, task_summary, smart_client)
        if not advice:
            return None
        output = (
            f"[ADVISOR BLOCKED] The pre-edit advisor has a critical architectural "
            f"comment about this edit and it was NOT applied. Reconsider the "
            f"approach before retrying.\n\n"
            f"Advisor note: {advice}\n\n"
            f"Pending edit:\n{_format_edit_for_review(tool_call)}\n"
        )
        return ToolResult(output=output)

    return None


def request_edit_precheck(
    tool_call: ToolCall,
    task_summary: str | None,
    client: LLMClient,
    *,
    max_iterations: int = _EDIT_PRECHECK_MAX_ITERATIONS,
    time_budget_seconds: float = _EDIT_PRECHECK_TIME_BUDGET_SECONDS,
) -> str | None:
    """Run a tool-using advisor loop to review a pending edit; return critical note or None."""

    from cagent.tools import run_tool

    edit_summary = _format_edit_for_review(tool_call)
    task_line = (
        task_summary.strip()
        if task_summary and task_summary.strip()
        else "(no task summary provided)"
    )
    edit_path = _get_edit_path(tool_call)
    file_content = _read_file_for_precheck(edit_path)

    file_section = ""
    if file_content is not None:
        file_section = (
            f'<target_file path="{edit_path}">\n'
            f"{file_content}\n"
            f"</target_file>\n\n"
        )

    user_prompt = (
        f"<task>\n{task_line}\n</task>\n\n"
        f"<pending_edit>\n{edit_summary}\n</pending_edit>\n\n"
        f"{file_section}"
        f"<instructions>\n"
        f"Budget: up to {max_iterations} tool calls and "
        f"{int(time_budget_seconds)} seconds of wall clock. After that your "
        f"answer is discarded and the edit proceeds. When done, reply with no "
        f"tool calls: exactly `None` if acceptable or uncertain, otherwise one "
        f"short sentence naming the concrete problem.\n"
        f"</instructions>"
    )

    messages: list[LLMMessage] = []
    start = time.monotonic()

    with get_trace().span(
        "advisor.edit_precheck",
        {
            "tool_name": tool_call.name,
            "arguments": tool_call.arguments,
            "max_iterations": max_iterations,
            "time_budget_seconds": time_budget_seconds,
            "span_context": "advisor",
        },
    ) as span:
        for iteration in range(max_iterations):
            if time.monotonic() - start > time_budget_seconds:
                span.set_attribute("timeout", True)
                return None

            if iteration == 0:
                messages.append(LLMMessage(role="user", content=user_prompt))

            response = client.complete(
                "",
                system_prompt=EDIT_PRECHECK_SYSTEM_PROMPT,
                tools=(BASH_TOOL,),
                messages=tuple(messages),
                trace_name="llm.complete.edit_precheck",
                trace_attributes={
                    "llm_purpose": "edit_precheck",
                    "reviewed_tool_name": tool_call.name,
                    "reviewed_tool_arguments": tool_call.arguments,
                    "iteration": iteration,
                    "span_context": "advisor",
                },
            )

            if not response.tool_calls:
                content = (response.content or "").strip()
                advice, parse_status = _parse_block_decision(content)
                span.set_attribute("advice", advice)
                span.set_attribute("decision_parse_status", parse_status)
                span.set_attribute("raw_response", content)
                span.set_attribute("iterations_used", iteration + 1)
                return advice

            normalized: list[ToolCall] = []
            for idx, tc in enumerate(response.tool_calls):
                if tc.id:
                    normalized.append(tc)
                else:
                    normalized.append(
                        ToolCall(
                            id=f"edit_precheck_{iteration}_{idx}",
                            name=tc.name,
                            arguments=tc.arguments,
                        )
                    )
            messages.append(
                LLMMessage(
                    role="assistant",
                    content=response.content,
                    tool_calls=tuple(normalized),
                    reasoning=response.reasoning,
                    thought_signature=response.thought_signature,
                )
            )
            for tc in normalized:
                try:
                    result = run_tool(tc.name, tc.arguments)
                    output = result.output
                except Exception as exc:  # noqa: BLE001
                    output = f"Error: {exc}"
                messages.append(
                    LLMMessage(
                        role="tool",
                        content=output,
                        tool_call_id=tc.id or "",
                    )
                )

        span.set_attribute("iterations_exhausted", True)
        return None


def _get_edit_path(tool_call: ToolCall) -> str:
    return tool_call.arguments.get("path", "") or ""


def _read_file_for_precheck(path: str) -> str | None:
    if not path:
        return None
    file_path = Path(path)
    if not file_path.is_file():
        return None
    try:
        content = file_path.read_text(encoding="utf-8")
    except Exception:
        return None

    lines = content.splitlines()
    truncated_by_lines = False
    truncated_by_bytes = False

    if len(lines) > _MAX_FILE_LINES_FOR_PRECHECK:
        content = "\n".join(lines[:_MAX_FILE_LINES_FOR_PRECHECK])
        truncated_by_lines = True

    encoded = content.encode("utf-8")
    if len(encoded) > _MAX_FILE_BYTES_FOR_PRECHECK:
        content = encoded[:_MAX_FILE_BYTES_FOR_PRECHECK].decode(
            "utf-8", errors="ignore"
        )
        truncated_by_bytes = True

    if truncated_by_lines or truncated_by_bytes:
        reasons = []
        if truncated_by_lines:
            reasons.append(f"exceeds {_MAX_FILE_LINES_FOR_PRECHECK} lines")
        if truncated_by_bytes:
            reasons.append(f"exceeds {_MAX_FILE_BYTES_FOR_PRECHECK // 1024} KB")
        warning = (
            f"[WARNING] File content was truncated because it "
            f"{' and '.join(reasons)}.\n\n"
        )
        content = warning + content

    return content


def _format_edit_for_review(tool_call: ToolCall) -> str:
    name = tool_call.name
    args = tool_call.arguments
    if name == WRITE_FILE_TOOL.name:
        path = args.get("path", "")
        content = args.get("content", "")
        append = bool(args.get("append"))
        start_line = args.get("start_line")
        end_line = args.get("end_line")
        if append:
            mode = "append"
        elif start_line is not None or end_line is not None:
            mode = f"replace lines {start_line or '?'}-{end_line or '?'}"
        else:
            mode = "overwrite"
        return (
            f'<edit tool="write_file">\n'
            f"  <path>{path}</path>\n"
            f"  <mode>{mode}</mode>\n"
            f"  <content>\n{content}\n  </content>\n"
            f"</edit>"
        )
    if name == SEARCH_AND_REPLACE_TOOL.name:
        path = args.get("path", "")
        old_text = args.get("old_text", "")
        new_text = args.get("new_text", "")
        return (
            f'<edit tool="search_and_replace">\n'
            f"  <path>{path}</path>\n"
            f"  <old_text>\n{old_text}\n  </old_text>\n"
            f"  <new_text>\n{new_text}\n  </new_text>\n"
            f"</edit>"
        )
    return (
        f'<edit tool="{name}">\n'
        f"  <args>{args}</args>\n"
        f"</edit>"
    )


def _format_prompt(advisor_input: AdvisorInput) -> str:
    exit_code = (
        "timeout" if advisor_input.exit_code is None else str(advisor_input.exit_code)
    )
    return (
        f"Tool: {advisor_input.tool_name}\n"
        f"Exit code: {exit_code}\n\n"
        f"Command:\n{advisor_input.command}\n\n"
        f"STDOUT:\n{advisor_input.stdout}\n\n"
        f"STDERR:\n{advisor_input.stderr}\n"
    )


def _parse_advice(content: str) -> str | None:
    if not content:
        return None
    normalized = content.strip().lower()
    if normalized in _NONE_MARKERS:
        return None
    return content


def _parse_block_decision(content: str) -> tuple[str | None, str]:
    """Parse a JSON {"block": bool, "comment": str} reply from the edit precheck.

    Returns (advice, parse_status). `advice` is the comment string when the
    model explicitly asks to block, otherwise None. The parser is intentionally
    lenient and biased toward NOT blocking: any malformed, missing, or
    ambiguous reply returns (None, <status>) so the edit proceeds.
    """

    if not content:
        return None, "empty"

    payload = _extract_json_object(content)
    if payload is None:
        return None, "no_json_found"

    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        return None, "json_decode_error"

    if not isinstance(data, dict):
        return None, "not_an_object"

    block = data.get("block")
    if isinstance(block, str):
        block = block.strip().lower() in {"true", "yes", "1"}
    if not isinstance(block, bool):
        return None, "missing_block_flag"

    if not block:
        return None, "ok_no_block"

    comment_raw = data.get("comment")
    comment = comment_raw.strip() if isinstance(comment_raw, str) else ""
    if not comment:
        return None, "block_without_comment"

    return comment, "ok_block"


_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json_object(content: str) -> str | None:
    """Return the first balanced JSON object substring, stripping code fences."""

    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        text = text.strip()

    if text.startswith("{") and text.endswith("}"):
        return text

    match = _JSON_OBJECT_RE.search(text)
    return match.group(0) if match else None
