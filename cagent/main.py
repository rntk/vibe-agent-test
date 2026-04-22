"""Application entry point."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from cagent.advisor import apply_advisor, precheck_tool_call
from cagent.config import ProviderConfig, load_fast_api_config, load_smart_api_config
from cagent.llm import LLMClient, LLMMessage, LLMRequest, LLMResponse, ToolCall
from cagent.llm.anthropic import AnthropicClient
from cagent.llm.llamacpp import LLamaCPP
from cagent.llm.openai import OpenAIChatCompletionsClient
from cagent.tools import IMPLEMENTATION_TOOLS, PLAN_TOOLS, run_tool_call
from cagent.tracing import Trace, reset_trace, set_trace, write_trace_html


class EchoLLMClient(LLMClient):
    """Local provider used by the skeleton entry point."""

    def _complete(self, request: LLMRequest) -> LLMResponse:
        return LLMResponse(content=request.user_prompt)


def create_fast_api_client() -> LLMClient | None:
    """Create a FAST_API LLM client when provider configuration is set."""
    config = load_fast_api_config()
    if not _has_usable_provider_config(config):
        return None
    return _create_client_from_config(config)


def create_smart_api_client() -> LLMClient | None:
    """Create a SMART_API LLM client when provider configuration is set."""
    config = load_smart_api_config()
    if not _has_usable_provider_config(config):
        return None
    return _create_client_from_config(config)


def _has_usable_provider_config(config: ProviderConfig) -> bool:
    """Return whether a provider config is sufficient to create a client."""
    provider_type = (config.type or "llamacpp").lower()
    if provider_type == "llamacpp":
        return bool(config.host)
    return bool(config.type)


def _create_client_from_config(config: ProviderConfig) -> LLMClient:
    """Instantiate the correct LLM client based on provider configuration."""
    provider_type = (config.type or "llamacpp").lower()

    if provider_type == "llamacpp":
        if not config.host:
            raise ValueError("HOST is required for TYPE=llamacpp.")
        return LLamaCPP(
            host=config.host,
            token=config.token,
            model=config.model or "moonshotai/Kimi-K2.5",
        )

    if provider_type == "openai":
        try:
            import openai as openai_sdk
        except ImportError as exc:
            raise ImportError(
                "The 'openai' package is required for TYPE=openai. "
                "Install it with: pip install openai"
            ) from exc
        openai_kwargs: dict[str, str] = {}
        if config.host:
            openai_kwargs["base_url"] = config.host
        if config.token:
            openai_kwargs["api_key"] = config.token
        client = openai_sdk.OpenAI(**openai_kwargs)
        return OpenAIChatCompletionsClient(
            client=client,
            default_model=config.model or "gpt-4o",
        )

    if provider_type == "anthropic":
        try:
            import anthropic as anthropic_sdk
        except ImportError as exc:
            raise ImportError(
                "The 'anthropic' package is required for TYPE=anthropic. "
                "Install it with: pip install anthropic"
            ) from exc
        anthropic_kwargs: dict[str, str] = {}
        if config.host:
            anthropic_kwargs["base_url"] = config.host
        if config.token:
            anthropic_kwargs["api_key"] = config.token
        client = anthropic_sdk.Anthropic(**anthropic_kwargs)
        return AnthropicClient(
            client=client,
            model=config.model or "claude-3-5-sonnet-20241022",
        )

    raise ValueError(f"Unsupported LLM provider type: {config.type}")


def run_plan_mode(file_path: str) -> None:
    """Run planning mode using FAST_API and save the result."""
    fast_client = create_fast_api_client()
    if fast_client is None:
        print("Error: FAST_API_HOST is not configured.", file=sys.stderr)
        sys.exit(1)

    task_content = Path(file_path).read_text(encoding="utf-8")
    plan_template = Path("prompts/plan.md").read_text(encoding="utf-8")
    prompt = plan_template.replace("{task}", task_content)

    messages: list[LLMMessage] = []
    max_iterations = 20

    for iteration in range(max_iterations):
        if iteration == 0:
            user_prompt = ""
            user_message = LLMMessage(role="user", content=prompt)
            messages.append(user_message)
        else:
            user_prompt = ""
        response = fast_client.complete(
            user_prompt,
            tools=PLAN_TOOLS,
            messages=messages,
        )

        if response.tool_calls:
            tool_calls = _tool_calls_with_ids(response.tool_calls, iteration)
            messages.append(
                LLMMessage(
                    role="assistant",
                    content=response.content,
                    tool_calls=tool_calls,
                )
            )
            for tool_call in tool_calls:
                try:
                    blocked = precheck_tool_call(tool_call, fast_client)
                    if blocked is not None:
                        tool_result = blocked
                    else:
                        tool_result = run_tool_call(tool_call)
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


def run_implementation_mode(file_path: str) -> None:
    """Run implementation mode using FAST_API with read, bash, and write tools."""
    fast_client = create_fast_api_client()
    if fast_client is None:
        print("Error: FAST_API_HOST is not configured.", file=sys.stderr)
        sys.exit(1)

    task_content = Path(file_path).read_text(encoding="utf-8")

    system_prompt = (
        "You are a software engineering assistant. "
        "Use the available tools to research the current project. "
        "If the users task already implemented in the codebase, find the "
        "relevant code, explain it to the user and finish. "
        "Otherwise, research how to implement the users request using the "
        "available tools and information in the codebase. "
        "Current directory: /app"
    )

    messages: list[LLMMessage] = []
    max_iterations = 20

    for iteration in range(max_iterations):
        if iteration == 0:
            user_prompt = ""
            user_message = LLMMessage(role="user", content=task_content)
            messages.append(user_message)
        else:
            user_prompt = ""
        response = fast_client.complete(
            user_prompt,
            system_prompt=system_prompt,
            tools=IMPLEMENTATION_TOOLS,
            messages=messages,
        )

        if response.tool_calls:
            tool_calls = _tool_calls_with_ids(response.tool_calls, iteration)
            messages.append(
                LLMMessage(
                    role="assistant",
                    content=response.content,
                    tool_calls=tool_calls,
                )
            )
            for tool_call in tool_calls:
                try:
                    blocked = precheck_tool_call(tool_call, fast_client)
                    if blocked is not None:
                        tool_result = blocked
                    else:
                        tool_result = run_tool_call(tool_call)
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
        else:
            print(response.content or "")
            return

    print(
        "Error: Maximum iterations reached without final result.",
        file=sys.stderr,
    )
    sys.exit(1)


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


def main() -> None:
    parser = argparse.ArgumentParser(description="cagent CLI")
    parser.add_argument(
        "-p",
        "--plan",
        metavar="FILE",
        help="Path to a task file for plan mode",
    )
    parser.add_argument(
        "-i",
        "--implementation",
        metavar="FILE",
        help="Path to a plan/prompt file for implementation mode with write tools",
    )
    parser.add_argument(
        "--trace",
        nargs="?",
        const="",
        default=None,
        metavar="FILE",
        help="Enable debug tracing and optionally write JSON trace output to FILE",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logging.info(
        "Start params: plan=%s, implementation=%s, trace=%s",
        args.plan,
        args.implementation,
        args.trace,
    )

    if args.trace is not None:
        trace = Trace()
        token = set_trace(trace)
        trace_file = _trace_file_from_arg(args.trace)
        trace_html_file = _trace_html_file_from_trace_file(trace_file)
        try:
            with trace.span(
                "cagent.main",
                {
                    "plan": args.plan,
                    "implementation": args.implementation,
                    "trace_file": trace_file,
                    "trace_html_file": trace_html_file,
                },
            ):
                _run_args(args)
        finally:
            trace.flush(trace_file)
            if trace_file and trace_html_file:
                write_trace_html(trace_file, trace_html_file)
            reset_trace(token)
        return

    _run_args(args)


def _run_args(args: argparse.Namespace) -> None:
    """Run the CLI mode selected by parsed arguments."""

    if args.plan:
        run_plan_mode(args.plan)
        return

    if args.implementation:
        run_implementation_mode(args.implementation)
        return

    fast_client = create_fast_api_client()
    smart_client = create_smart_api_client()
    llm = smart_client or fast_client or EchoLLMClient()
    response = llm.complete(
        "Hello, World!",
        system_prompt="Repeat the user prompt.",
    )
    print(response.content)


def _trace_file_from_arg(trace_arg: str | None) -> str | None:
    """Return the trace output path only when the CLI argument is non-empty."""

    if trace_arg is None or trace_arg == "":
        return None
    return trace_arg


def _trace_html_file_from_trace_file(trace_file: str | None) -> str | None:
    """Return the sidecar HTML trace path for a JSON trace output path."""

    if trace_file is None:
        return None
    return str(Path(trace_file).with_suffix(".html"))


__all__ = [
    "EchoLLMClient",
    "create_fast_api_client",
    "create_smart_api_client",
    "main",
    "run_implementation_mode",
    "run_plan_mode",
    "_run_args",
    "_trace_file_from_arg",
    "_trace_html_file_from_trace_file",
]


if __name__ == "__main__":
    main()
