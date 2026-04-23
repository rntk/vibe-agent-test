"""Advisor that asks an LLM for a hint when a tool call looks like it failed."""

from __future__ import annotations

import time

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
    "You are a senior code-review gate before another, faster agent applies a "
    "file edit. You see ONE pending edit (write_file or search_and_replace) and "
    "a brief description of the overall task. You have NO conversation history "
    "and no knowledge of what the fast agent already tried.\n\n"
    "You may call the `bash` tool to gather context: read the target file with "
    "`cat -n`, `grep`/`rg` for duplicated logic or existing helpers, inspect "
    "neighboring modules, check how similar things are done elsewhere. You are "
    "the slow, careful overseer — the fast agent is quick but can miss "
    "architectural context.\n\n"
    "Flag ONLY clear, evidence-backed architectural problems, such as:\n"
    "- Reimplementing logic that already exists in the codebase\n"
    "- Breaking an established pattern visible in sibling/neighbor code\n"
    "- An obviously unmaintainable shape (e.g. a god-function, tangled "
    "responsibilities, magic values that belong in config based on neighbors)\n"
    "- Editing the wrong place for the stated task\n\n"
    "Do NOT flag style, naming, formatting, missing tests/docs, minor "
    "inefficiencies, or anything you cannot back with evidence from files you "
    "actually read. Quick-and-dirty is acceptable when the task is a quick fix; "
    "bias strongly toward approving.\n\n"
    "When you are done investigating (or have seen enough), reply with a final "
    "message that contains NO tool calls:\n"
    "- If the edit looks acceptable OR you are uncertain, reply with exactly "
    "the single word: None\n"
    "- Otherwise reply with ONE short sentence naming the concrete problem and "
    "the evidence. Do NOT rewrite the code, do NOT produce replacement code, "
    "do NOT list steps. The fast agent will decide the exact fix itself."
)

_NONE_MARKERS = {"none", "none.", "n/a", "n/a."}
_EDIT_PRECHECK_MAX_ITERATIONS = 5
_EDIT_PRECHECK_TIME_BUDGET_SECONDS = 60.0


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
        },
    ) as span:
        response = client.complete(
            prompt,
            system_prompt=ADVISOR_SYSTEM_PROMPT,
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
        {"tool_name": BASH_TOOL.name, "command": command},
    ) as span:
        response = client.complete(
            prompt,
            system_prompt=PRECHECK_SYSTEM_PROMPT,
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
    user_prompt = (
        f"Overall task:\n{task_line}\n\n"
        f"Pending edit to review:\n{edit_summary}\n\n"
        f"Budget: up to {max_iterations} tool calls and "
        f"{int(time_budget_seconds)} seconds of wall clock. After that your "
        f"answer is discarded and the edit proceeds. When done, reply with no "
        f"tool calls: exactly `None` if acceptable or uncertain, otherwise one "
        f"short sentence naming the concrete problem."
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
            )

            if not response.tool_calls:
                advice = _parse_advice((response.content or "").strip())
                span.set_attribute("advice", advice)
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
            f"Tool: write_file\nPath: {path}\nMode: {mode}\n"
            f"Content:\n{content}"
        )
    if name == SEARCH_AND_REPLACE_TOOL.name:
        path = args.get("path", "")
        old_text = args.get("old_text", "")
        new_text = args.get("new_text", "")
        return (
            f"Tool: search_and_replace\nPath: {path}\n"
            f"OLD TEXT:\n{old_text}\n\nNEW TEXT:\n{new_text}"
        )
    return f"Tool: {name}\nArgs: {args}"


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
