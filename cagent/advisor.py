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
    "You are an error-diagnosis advisor for a coding agent.\n"
    "You will see a single tool invocation that failed or produced stderr output: "
    "the command, its exit code, and the captured stdout/stderr. "
    "You do NOT see any conversation history or surrounding goal.\n\n"
    "If you can confidently identify the likely cause and suggest a concrete fix, "
    "answer in 1-3 short sentences. Be specific and actionable.\n"
    "If you cannot add anything helpful beyond what the error text already says, "
    "reply with exactly the single word: None"
)

PRECHECK_SYSTEM_PROMPT = (
    "You are a pre-execution safety reviewer for a coding agent's bash tool.\n"
    "You will see a single bash command that is ABOUT TO RUN. "
    "You do NOT see any conversation history or surrounding goal.\n\n"
    "Judge only on the command itself: is the syntax correct, is it safe to run, "
    "is it likely to hang for a long time (interactive prompts, long-running servers, "
    "infinite loops, waiting on network/stdin), or is it obviously destructive "
    "(rm -rf on broad paths, force-push to main, dropping databases, wiping disks, "
    "disabling security checks, exfiltrating secrets)?\n\n"
    "If the command looks fine — normal syntax, reasonable scope, will terminate — "
    "reply with exactly the single word: None\n"
    "Only raise a CRITICAL note when there is a concrete problem the caller should "
    "reconsider. In that case answer in 1-3 short sentences naming the specific "
    "issue and a safer alternative. Do not nitpick style."
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
