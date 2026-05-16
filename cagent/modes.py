"""Mode runners for planning and implementation."""

from __future__ import annotations

import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.resources import files
from pathlib import Path
from typing import Any

from cagent.advisor import apply_advisor, precheck_tool_call
from cagent.checkpoints import _format_checkpoint
from cagent.clients import create_fast_api_client, create_smart_api_client
from cagent.compaction import compact_history
from cagent.llm import LLMClient, LLMMessage, ToolCall
from cagent.system_prompts import PLAN_SYSTEM_PROMPT, implementation_system_prompt
from cagent.tools import IMPLEMENTATION_TOOLS, PLAN_TOOLS, ToolDefinition, run_tool_call

MAX_ITERATIONS = 20

BashAdvisor = str  # "off" | "fast" | "smart"


def _tool_calls_with_ids(
    tool_calls: Sequence[ToolCall],
    iteration: int,
) -> tuple[ToolCall, ...]:
    """Return tool calls with stable IDs for history linkage."""

    normalized: list[ToolCall] = []
    for index, tool_call in enumerate(tool_calls):
        if tool_call.id:
            normalized.append(tool_call)
            continue
        normalized.append(
            ToolCall(
                id=f"call_{iteration}_{index}",
                name=tool_call.name,
                arguments=tool_call.arguments,
            )
        )
    return tuple(normalized)


def _resolve_bash_client(
    bash_advisor: BashAdvisor,
    fast_client: LLMClient,
    smart_client: LLMClient | None,
) -> LLMClient | None:
    if bash_advisor == "off":
        return None
    if bash_advisor == "fast":
        return fast_client
    if bash_advisor == "smart":
        return smart_client
    raise ValueError(f"Unknown bash_advisor mode: {bash_advisor!r}")


def _dispatch_tool_call(
    tool_call: ToolCall,
    *,
    bash_client: LLMClient | None,
    smart_client: LLMClient | None,
    fast_client: LLMClient,
    task_summary: str,
) -> str:
    """Run precheck + tool + advisor for one call. Returns the tool result content."""

    try:
        blocked = precheck_tool_call(
            tool_call,
            bash_client,
            smart_client=smart_client,
            task_summary=task_summary,
        )
    except Exception as exc:
        return f"Error: precheck failed: {exc}"

    if blocked is not None:
        return blocked.output

    try:
        tool_result = run_tool_call(tool_call, advisor_client=smart_client)
    except Exception as exc:
        return f"Error: {exc}"

    try:
        tool_result = apply_advisor(tool_result, fast_client)
    except Exception as exc:
        # Advisor is non-essential: log to stderr, return raw tool output.
        print(f"Warning: advisor failed: {exc}", file=sys.stderr)

    return tool_result.output


@dataclass(frozen=True)
class ModeConfig:
    """Per-mode configuration for the shared agent loop."""

    tools: Sequence[ToolDefinition]
    initial_user_content: str
    task_summary: str
    on_final: Callable[[str], None]
    trace_name: str
    trace_attributes: dict[str, Any]
    system_prompt: str | None = None
    max_iterations: int = MAX_ITERATIONS


def _run_agent_loop(
    config: ModeConfig,
    *,
    fast_client: LLMClient,
    smart_client: LLMClient | None,
    bash_client: LLMClient | None,
    tool_summary: bool,
) -> None:
    """Shared agent loop for plan/implementation modes."""

    messages: list[LLMMessage] = [
        LLMMessage(role="user", content=config.initial_user_content)
    ]

    for iteration in range(config.max_iterations):
        response = fast_client.complete(
            "",
            system_prompt=config.system_prompt,
            tools=config.tools,
            messages=messages,
            trace_name=config.trace_name,
            trace_attributes={**config.trace_attributes, "iteration": iteration},
        )

        if not response.tool_calls:
            content = response.content or ""
            if not content.strip():
                print(
                    "Error: model returned empty response with no tool calls.",
                    file=sys.stderr,
                )
                sys.exit(1)
            config.on_final(content)
            return

        tool_calls = _tool_calls_with_ids(response.tool_calls, iteration)
        messages.append(
            LLMMessage(
                role="assistant",
                content=response.content,
                tool_calls=tool_calls,
                reasoning=response.reasoning,
                thought_signature=response.thought_signature,
            )
        )

        steps: list[tuple[ToolCall, str]] = []
        for tool_call in tool_calls:
            content = _dispatch_tool_call(
                tool_call,
                bash_client=bash_client,
                smart_client=smart_client,
                fast_client=fast_client,
                task_summary=config.task_summary,
            )
            messages.append(
                LLMMessage(
                    role="tool",
                    content=content,
                    tool_call_id=tool_call.id,
                )
            )
            steps.append((tool_call, content))

        # Compact once per turn, after all tool results are appended.
        messages = list(compact_history(messages))

        if tool_summary:
            summary = _format_checkpoint(fast_client, response.content, steps)
            if summary.strip():
                messages.append(LLMMessage(role="user", content=summary))

    print(
        "Error: Maximum iterations reached without final result.",
        file=sys.stderr,
    )
    sys.exit(1)


def _make_clients(
    bash_advisor: BashAdvisor,
) -> tuple[LLMClient, LLMClient | None, LLMClient | None]:
    """Build fast/smart/bash clients or exit if fast is unavailable."""

    fast_client = create_fast_api_client()
    if fast_client is None:
        print("Error: FAST_API_HOST is not configured.", file=sys.stderr)
        sys.exit(1)
    smart_client = create_smart_api_client()
    bash_client = _resolve_bash_client(bash_advisor, fast_client, smart_client)
    return fast_client, smart_client, bash_client


def _save_plan(content: str) -> None:
    plans_dir = Path("plans")
    plans_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
    plan_file = plans_dir / f"plan_{timestamp}.md"
    plan_file.write_text(content, encoding="utf-8")
    print(f"Plan saved to {plan_file}")


def run_plan_mode(
    file_path: str,
    bash_advisor: BashAdvisor = "off",
    tool_summary: bool = False,
) -> None:
    """Run planning mode using FAST_API and save the result."""

    fast_client, smart_client, bash_client = _make_clients(bash_advisor)

    task_content = Path(file_path).read_text(encoding="utf-8")
    plan_template = (
        files("cagent").joinpath("prompts/plan.md").read_text(encoding="utf-8")
    )
    prompt = plan_template.replace("{task}", task_content)

    config = ModeConfig(
        tools=PLAN_TOOLS,
        initial_user_content=prompt,
        task_summary=prompt,
        on_final=_save_plan,
        trace_name="llm.complete.plan.agent_turn",
        trace_attributes={
            "llm_purpose": "agent_turn",
            "agent_mode": "plan",
            "history_kind": "plan_chat_history",
            "span_context": "agent",
        },
        system_prompt=PLAN_SYSTEM_PROMPT,
    )

    _run_agent_loop(
        config,
        fast_client=fast_client,
        smart_client=smart_client,
        bash_client=bash_client,
        tool_summary=tool_summary,
    )


def run_implementation_mode(
    file_path: str,
    bash_advisor: BashAdvisor = "off",
    tool_summary: bool = False,
) -> None:
    """Run implementation mode using FAST_API with read, bash, and write tools."""

    fast_client, smart_client, bash_client = _make_clients(bash_advisor)

    task_content = Path(file_path).read_text(encoding="utf-8")

    config = ModeConfig(
        tools=IMPLEMENTATION_TOOLS,
        initial_user_content=task_content,
        task_summary=task_content,
        on_final=lambda content: print(content),
        trace_name="llm.complete.implementation.agent_turn",
        trace_attributes={
            "llm_purpose": "agent_turn",
            "agent_mode": "implementation",
            "history_kind": "implementation_chat_history",
            "span_context": "agent",
        },
        system_prompt=implementation_system_prompt(),
    )

    _run_agent_loop(
        config,
        fast_client=fast_client,
        smart_client=smart_client,
        bash_client=bash_client,
        tool_summary=tool_summary,
    )


__all__ = ["run_plan_mode", "run_implementation_mode", "_tool_calls_with_ids"]
