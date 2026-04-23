"""Checkpoint formatting logic."""

from __future__ import annotations

import json
from collections.abc import Sequence

from cagent.llm import LLMClient, ToolCall


_CHECKPOINT_SYSTEM_PROMPT = (
    "<role>\n"
    "You compress an agent's step into a terse checkpoint so the agent can "
    "track progress across iterations, like a checklist entry.\n"
    "</role>\n"
    "<input>\n"
    "You will receive the agent's reasoning, the tool call it made, and the "
    "tool result, each wrapped in its own tag.\n"
    "</input>\n"
    "<output_format>\n"
    "Output exactly three short lines, nothing else:\n"
    "intention: <one short sentence describing what the agent wanted>\n"
    "action: <very brief description of the action taken>\n"
    "arguments>\n"
    "result: <very brief phrase: success/failure and at most one key fact>\n"
    "</output_format>\n"
    "<rules>\n"
    "- Be extremely brief on action and result. The agent already has the "
    "full tool call and full tool result in its context; do not restate them.\n"
    "- For action, do not quote full arguments or file contents. Example: "
    "for `cat -n /app/file.py` write `action: read /app/file.py`.\n"
    "- For result, do not repeat the output. Just say whether it succeeded "
    "and, if useful, a tiny hint. Examples: `result: file read successfully`, "
    "`result: command failed (file not found)`, `result: grep found 3 "
    "matches in src/`.\n"
    "- No preamble, no bullets, no code fences, no extra lines.\n"
    "</rules>"
)


def _format_checkpoint(
    client: LLMClient,
    reasoning: str | None,
    steps: Sequence[tuple[ToolCall, str]],
) -> str:
    """Ask the LLM to summarize reasoning + tool calls + results as a checkpoint."""

    raw_lines: list[str] = []
    reasoning_text = (reasoning or "").strip()
    raw_lines.append(f"<reasoning>\n{reasoning_text}\n</reasoning>")
    for tool_call, result in steps:
        try:
            args_str = json.dumps(tool_call.arguments or {}, ensure_ascii=False)
        except (TypeError, ValueError):
            args_str = str(tool_call.arguments)
        raw_lines.append(
            f"<tool_call name=\"{tool_call.name}\">\n{args_str}\n</tool_call>"
        )
        raw_lines.append(
            f"<tool_result>\n{(result or '').strip()}\n</tool_result>"
        )
    raw = "\n".join(raw_lines)

    try:
        response = client.complete(raw, system_prompt=_CHECKPOINT_SYSTEM_PROMPT)
        summary = (response.content or "").strip()
    except Exception as exc:
        summary = f"(checkpoint summarization failed: {exc})\n{raw}"

    return summary or raw


__all__ = ["_format_checkpoint"]
