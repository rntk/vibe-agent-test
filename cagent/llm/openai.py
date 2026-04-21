"""OpenAI-compatible LLM client adapter.

This module intentionally accepts an already-created SDK/API client. Keeping SDK
construction outside the adapter lets the core interface avoid external package
dependencies and lets callers choose an SDK, HTTP client, or test double.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

from cagent.llm.base import (
    LLMClient,
    LLMMessage,
    LLMRequest,
    LLMResponse,
    ToolCall,
    ToolDefinition,
)


@dataclass(frozen=True, slots=True)
class OpenAIChatCompletionsClient(LLMClient):
    """Adapter for OpenAI-compatible chat-completions clients."""

    client: Any
    default_model: str

    def _complete(self, request: LLMRequest) -> LLMResponse:
        """Run one LLM turn through an OpenAI-compatible client."""

        payload: dict[str, Any] = {
            "model": request.model or self.default_model,
            "messages": self.to_provider_messages(request.all_messages()),
        }
        if request.tools:
            payload["tools"] = self.to_provider_tools(request.tools)
        if request.temperature is not None:
            payload["temperature"] = request.temperature

        completion = self.client.chat.completions.create(**payload)
        return self.from_provider_response(completion)

    @staticmethod
    def to_provider_messages(messages: Sequence[LLMMessage]) -> list[dict[str, Any]]:
        """Convert provider-neutral messages to OpenAI chat-completion messages."""

        return [
            OpenAIChatCompletionsClient.to_provider_message(msg) for msg in messages
        ]

    @staticmethod
    def to_provider_message(message: LLMMessage) -> dict[str, Any]:
        """Convert one provider-neutral message to an OpenAI chat message."""

        output: dict[str, Any] = {"role": message.role, "content": message.content}
        if message.role == "tool":
            output["tool_call_id"] = message.tool_call_id
        if message.tool_calls:
            output["tool_calls"] = [
                {
                    "id": tool_call.id,
                    "type": "function",
                    "function": {
                        "name": tool_call.name,
                        "arguments": json.dumps(dict(tool_call.arguments)),
                    },
                }
                for tool_call in message.tool_calls
            ]
        return output

    @staticmethod
    def to_provider_tools(tools: Sequence[ToolDefinition]) -> list[dict[str, Any]]:
        """Convert provider-neutral tools to OpenAI chat-completion tools."""

        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": dict(tool.parameters),
                },
            }
            for tool in tools
        ]

    @staticmethod
    def from_provider_tool_calls(tool_calls: Any) -> tuple[ToolCall, ...]:
        """Convert OpenAI chat-completion tool calls to neutral tool calls."""

        if not tool_calls:
            return ()

        parsed_calls: list[ToolCall] = []
        for tool_call in tool_calls:
            function = OpenAIChatCompletionsClient.get_value(
                tool_call,
                "function",
                tool_call,
            )
            parsed_calls.append(
                ToolCall(
                    id=OpenAIChatCompletionsClient.get_value(
                        tool_call,
                        "id",
                        OpenAIChatCompletionsClient.get_value(tool_call, "call_id"),
                    ),
                    name=cast(
                        str,
                        OpenAIChatCompletionsClient.get_value(function, "name"),
                    ),
                    arguments=OpenAIChatCompletionsClient.parse_arguments(
                        OpenAIChatCompletionsClient.get_value(
                            function,
                            "arguments",
                            {},
                        )
                    ),
                )
            )
        return tuple(parsed_calls)

    @staticmethod
    def from_provider_response(response: Any) -> LLMResponse:
        """Convert an OpenAI-compatible response to a neutral response."""

        choices = OpenAIChatCompletionsClient.get_value(response, "choices", ())
        first_choice = choices[0] if choices else None
        message = OpenAIChatCompletionsClient.get_value(first_choice, "message", {})
        return LLMResponse(
            content=OpenAIChatCompletionsClient.get_value(message, "content"),
            tool_calls=OpenAIChatCompletionsClient.from_provider_tool_calls(
                OpenAIChatCompletionsClient.get_value(message, "tool_calls")
            ),
            raw=response,
        )

    @staticmethod
    def parse_arguments(arguments: Any) -> Mapping[str, Any]:
        """Parse provider tool-call arguments into a mapping."""

        if arguments is None:
            return {}
        if isinstance(arguments, str):
            decoded = json.loads(arguments or "{}")
            if not isinstance(decoded, dict):
                msg = "Tool-call arguments must decode to a JSON object."
                raise ValueError(msg)
            return decoded
        if isinstance(arguments, Mapping):
            return arguments

        msg = "Tool-call arguments must be a JSON object string or mapping."
        raise ValueError(msg)

    @staticmethod
    def get_value(value: Any, key: str, default: Any = None) -> Any:
        """Read a provider value from either a mapping or an SDK object."""

        if isinstance(value, Mapping):
            return value.get(key, default)
        return getattr(value, key, default)
