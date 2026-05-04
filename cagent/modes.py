"""Mode runners for planning and implementation."""

from __future__ import annotations

import sys
from datetime import UTC, datetime
from importlib.resources import files
from pathlib import Path
from collections.abc import Sequence

from cagent.advisor import apply_advisor, precheck_tool_call
from cagent.clients import create_fast_api_client, create_smart_api_client
from cagent.checkpoints import _format_checkpoint
from cagent.compaction import compact_history
from cagent.llm import LLMMessage, ToolCall
from cagent.tools import IMPLEMENTATION_TOOLS, PLAN_TOOLS, run_tool_call


def _tool_calls_with_ids(
    tool_calls: Sequence[ToolCall],
    iteration: int,
) -> tuple[ToolCall, ...]:
    """Return tool calls with stable IDs for history linkage."""

    normalized_tool_calls: list[ToolCall] = []
    for index, tool_call in enumerate(tool_calls):
        if tool_call.id:
            normalized_tool_calls.append(tool_call)
            continue
        normalized_tool_calls.append(
            ToolCall(
                id=f"call_{iteration}_{index}",
                name=tool_call.name,
                arguments=tool_call.arguments,
            )
        )
    return tuple(normalized_tool_calls)


def run_plan_mode(
    file_path: str,
    bash_advisor: str = "off",
    tool_summary: bool = False,
) -> None:
    """Run planning mode using FAST_API and save the result."""
    fast_client = create_fast_api_client()
    if fast_client is None:
        print("Error: FAST_API_HOST is not configured.", file=sys.stderr)
        sys.exit(1)
    smart_client = create_smart_api_client()

    bash_client = None
    if bash_advisor == "fast":
        bash_client = fast_client
    elif bash_advisor == "smart":
        bash_client = smart_client

    task_content = Path(file_path).read_text(encoding="utf-8")
    plan_template = files("cagent").joinpath("prompts/plan.md").read_text(
        encoding="utf-8"
    )
    prompt = plan_template.replace("{task}", task_content)

    messages: list[LLMMessage] = []
    max_iterations = 20
    next_user_prompt = ""

    for iteration in range(max_iterations):
        if iteration == 0:
            user_prompt = ""
            user_message = LLMMessage(role="user", content=prompt)
            messages.append(user_message)
        else:
            user_prompt = next_user_prompt
            next_user_prompt = ""
        response = fast_client.complete(
            user_prompt,
            tools=PLAN_TOOLS,
            messages=messages,
            trace_name="llm.complete.plan.agent_turn",
            trace_attributes={
                "llm_purpose": "agent_turn",
                "agent_mode": "plan",
                "iteration": iteration,
                "history_kind": "plan_chat_history",
                "span_context": "agent",
            },
        )

        if response.tool_calls:
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
                try:
                    blocked = precheck_tool_call(
                        tool_call,
                        bash_client,
                        smart_client=smart_client,
                        task_summary=prompt,
                    )
                    if blocked is not None:
                        tool_result = blocked
                    else:
                        tool_result = run_tool_call(
                            tool_call,
                            advisor_client=smart_client,
                        )
                        tool_result = apply_advisor(tool_result, fast_client)
                    content = tool_result.output
                except Exception as exc:
                    content = f"Error: {exc}"
                messages.append(
                    LLMMessage(
                        role="tool",
                        content=content,
                        tool_call_id=tool_call.id or "",
                    )
                )
                steps.append((tool_call, content))
                messages = list(compact_history(messages))
            if tool_summary:
                next_user_prompt = _format_checkpoint(
                    fast_client, response.content, steps
                )
            else:
                next_user_prompt = ""
        else:
            plans_dir = Path("plans")
            plans_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
            plan_file = plans_dir / f"plan_{timestamp}.md"
            plan_file.write_text(response.content or "", encoding="utf-8")

            print(f"Plan saved to {plan_file}")
            return

    print(
        "Error: Maximum iterations reached without final result.",
        file=sys.stderr,
    )
    sys.exit(1)


def run_implementation_mode(
    file_path: str,
    bash_advisor: str = "off",
    tool_summary: bool = False,
) -> None:
    """Run implementation mode using FAST_API with read, bash, and write tools."""
    fast_client = create_fast_api_client()
    if fast_client is None:
        print("Error: FAST_API_HOST is not configured.", file=sys.stderr)
        sys.exit(1)
    smart_client = create_smart_api_client()

    bash_client = None
    if bash_advisor == "fast":
        bash_client = fast_client
    elif bash_advisor == "smart":
        bash_client = smart_client

    task_content = Path(file_path).read_text(encoding="utf-8")

    system_prompt = (
        "You are a software engineering assistant. "
        "Use the available tools to research the current project. "
        "If the users task already implemented in the codebase, find the "
        "relevant code, explain it to the user and finish. "
        "Otherwise, research how to implement the users request using the "
        "available tools and information in the codebase. "
        f"Current directory: {Path.cwd()}"
    )

    messages: list[LLMMessage] = []
    max_iterations = 20
    next_user_prompt = ""

    for iteration in range(max_iterations):
        if iteration == 0:
            user_prompt = ""
            user_message = LLMMessage(role="user", content=task_content)
            messages.append(user_message)
        else:
            user_prompt = next_user_prompt
            next_user_prompt = ""
        response = fast_client.complete(
            user_prompt,
            system_prompt=system_prompt,
            tools=IMPLEMENTATION_TOOLS,
            messages=messages,
            trace_name="llm.complete.implementation.agent_turn",
            trace_attributes={
                "llm_purpose": "agent_turn",
                "agent_mode": "implementation",
                "iteration": iteration,
                "history_kind": "implementation_chat_history",
                "span_context": "agent",
            },
        )

        if response.tool_calls:
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
                try:
                    blocked = precheck_tool_call(
                        tool_call,
                        bash_client,
                        smart_client=smart_client,
                        task_summary=task_content,
                    )
                    if blocked is not None:
                        tool_result = blocked
                    else:
                        tool_result = run_tool_call(
                            tool_call,
                            advisor_client=smart_client,
                        )
                        tool_result = apply_advisor(tool_result, fast_client)
                    content = tool_result.output
                except Exception as exc:
                    content = f"Error: {exc}"
                messages.append(
                    LLMMessage(
                        role="tool",
                        content=content,
                        tool_call_id=tool_call.id or "",
                    )
                )
                steps.append((tool_call, content))
                messages = list(compact_history(messages))
            if tool_summary:
                next_user_prompt = _format_checkpoint(
                    fast_client, response.content, steps
                )
            else:
                next_user_prompt = ""
        else:
            print(response.content or "")
            return

    print(
        "Error: Maximum iterations reached without final result.",
        file=sys.stderr,
    )
    sys.exit(1)


__all__ = ["run_plan_mode", "run_implementation_mode", "_tool_calls_with_ids"]
