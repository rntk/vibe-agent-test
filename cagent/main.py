"""Application entry point shim."""

from __future__ import annotations

from cagent.clients import (
    EchoLLMClient,
    create_fast_api_client,
    create_smart_api_client,
)
from cagent.cli import (
    _run_args,
    _trace_file_from_arg,
    _trace_html_file_from_trace_file,
    main,
)
from cagent.modes import (
    _tool_calls_with_ids,
    run_implementation_mode,
    run_plan_mode,
)

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
