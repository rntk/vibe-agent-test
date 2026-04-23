"""CLI entry point and orchestration."""

from __future__ import annotations

import argparse
import logging

from cagent.clients import EchoLLMClient, create_fast_api_client, create_smart_api_client
from cagent.modes import run_implementation_mode, run_plan_mode
from cagent.tracing import Trace, reset_trace, set_trace, write_trace_html
from pathlib import Path


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
    "main",
    "_run_args",
    "_trace_file_from_arg",
    "_trace_html_file_from_trace_file",
]
