"""Advisor that asks an LLM for a hint when a tool call looks like it failed."""

from __future__ import annotations

from cagent.llm.base import BASH_TOOL, LLMClient, ToolCall
from cagent.tools import AdvisorInput, ToolResult
from cagent.tracing import get_trace

__all__ = [
    "ADVISOR_SYSTEM_PROMPT",
    "PRECHECK_SYSTEM_PROMPT",
    "apply_advisor",
    "precheck_tool_call",
    "request_advice",
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

_NONE_MARKERS = {"none", "none.", "n/a", "n/a."}


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
) -> ToolResult | None:
    """Pre-execution advisor: block bash calls with a critical note, else return None."""

    if client is None or tool_call.name != BASH_TOOL.name:
        return None
    command = tool_call.arguments.get("command")
    if not isinstance(command, str) or not command.strip():
        return None
    advice = request_precheck(command, client)
    if not advice:
        return None
    output = (
        f"[ADVISOR BLOCKED] The pre-execution advisor has a critical comment about "
        f"this bash command and it was NOT executed. Reconsider the command (fix "
        f"syntax, narrow its scope, or pick a safer alternative) before retrying.\n\n"
        f"Advisor note: {advice}\n\n"
        f"Command:\n{command}\n"
    )
    return ToolResult(output=output)


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
