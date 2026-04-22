"""Advisor that asks an LLM for a hint when a tool call looks like it failed."""

from __future__ import annotations

from cagent.llm.base import LLMClient
from cagent.tools import AdvisorInput, ToolResult
from cagent.tracing import get_trace

__all__ = ["ADVISOR_SYSTEM_PROMPT", "request_advice", "apply_advisor"]

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
