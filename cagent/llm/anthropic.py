"""Anthropic LLM client adapter."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

from anthropic import transform_schema
from anthropic.types import (
    MessageParam,
    OutputConfigParam,
    TextBlockParam,
    ToolParam,
    ToolResultBlockParam,
    ToolUnionParam,
    ToolUseBlockParam,
)

from cagent.llm.base import (
    LLMClient,
    LLMMessage,
    LLMRequest,
    LLMResponse,
    ToolCall,
    ToolDefinition,
)
from cagent.tracing import get_trace


@dataclass(frozen=True, slots=True)
class AnthropicClient(LLMClient):
    """Adapter for Anthropic Messages API."""

    client: Any
    model: str
    max_tokens: int = 4096

    def _complete(self, request: LLMRequest) -> LLMResponse:
        """Run one LLM turn through the Anthropic Messages API."""

        system_texts: list[str] = []
        provider_messages: list[MessageParam] = []

        for msg in request.all_messages():
            if msg.role == "system":
                if msg.content:
                    system_texts.append(msg.content)
                continue

            provider_message = AnthropicClient.to_provider_message(msg)
            if (
                provider_messages
                and provider_messages[-1]["role"] == provider_message["role"]
            ):
                content = cast(list[dict[str, Any]], provider_messages[-1]["content"])
                content.extend(cast(list[dict[str, Any]], provider_message["content"]))
            else:
                provider_messages.append(provider_message)

        kwargs: dict[str, Any] = {
            "model": request.model or self.model,
            "max_tokens": self.max_tokens,
            "messages": provider_messages,
        }
        if system_texts:
            kwargs["system"] = "\n\n".join(system_texts)
        if request.tools:
            kwargs["tools"] = AnthropicClient.to_provider_tools(request.tools)
        if request.output_schema:
            kwargs["output_config"] = AnthropicClient.to_output_config(
                request.output_schema
            )

        with get_trace().span(
            "llm.anthropic.complete",
            {
                "model": kwargs["model"],
                "max_tokens": self.max_tokens,
                "tool_count": len(request.tools),
            },
        ) as span:
            response = self.client.messages.create(**kwargs)
            usage = AnthropicClient.get_value(response, "usage")
            if usage is not None:
                span.set_attribute("usage", usage)
            stop_reason = AnthropicClient.get_value(response, "stop_reason")
            if stop_reason is not None:
                span.set_attribute("stop_reason", stop_reason)
            span.set_attribute(
                "response_model", AnthropicClient.get_value(response, "model")
            )
            result = AnthropicClient.from_provider_response(response)
            if result.reasoning is not None:
                span.set_attribute("response_reasoning", result.reasoning)
            return result

    @staticmethod
    def to_provider_message(message: LLMMessage) -> MessageParam:
        """Convert one provider-neutral message to an Anthropic message."""

        blocks: list[TextBlockParam | ToolUseBlockParam | ToolResultBlockParam] = []
        if message.role != "tool" and message.content:
            blocks.append({"type": "text", "text": message.content})

        if message.role == "assistant" and message.tool_calls:
            for tool_call in message.tool_calls:
                blocks.append(
                    {
                        "type": "tool_use",
                        "id": tool_call.id or "",
                        "name": tool_call.name,
                        "input": dict(tool_call.arguments),
                    }
                )
        elif message.role == "tool":
            blocks.append(
                {
                    "type": "tool_result",
                    "tool_use_id": message.tool_call_id or "",
                    "content": message.content or "",
                }
            )

        role = "user" if message.role == "tool" else message.role
        return {"role": role, "content": blocks}

    @staticmethod
    def to_provider_tools(tools: Sequence[ToolDefinition]) -> list[ToolUnionParam]:
        """Convert provider-neutral tools to Anthropic Messages API tools."""

        provider_tools: list[ToolUnionParam] = []
        for tool in tools:
            provider_tool: ToolParam = {
                "name": tool.name,
                "description": tool.description,
                "input_schema": dict(tool.parameters),
            }
            if tool.strict is not None:
                provider_tool["strict"] = tool.strict
            provider_tools.append(provider_tool)
        return provider_tools

    @staticmethod
    def to_output_config(output_schema: Mapping[str, Any]) -> OutputConfigParam:
        """Build an SDK-shaped Anthropic structured output config."""

        return {
            "format": {
                "type": "json_schema",
                "schema": transform_schema(dict(output_schema)),
            }
        }

    @staticmethod
    def from_provider_response(response: Any) -> LLMResponse:
        """Convert an Anthropic response to a neutral response."""

        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        tool_calls: list[ToolCall] = []

        blocks = AnthropicClient.get_value(response, "content", ())
        for block in blocks:
            block_type = AnthropicClient.get_value(block, "type")
            if block_type == "thinking":
                thinking = AnthropicClient.get_value(block, "thinking")
                if thinking:
                    reasoning_parts.append(cast(str, thinking))
            elif block_type == "text":
                text = AnthropicClient.get_value(block, "text")
                if text:
                    content_parts.append(cast(str, text))
            elif block_type == "tool_use":
                tool_calls.append(
                    ToolCall(
                        id=AnthropicClient.get_value(block, "id") or "",
                        name=cast(
                            str,
                            AnthropicClient.get_value(block, "name"),
                        ),
                        arguments=AnthropicClient.parse_arguments(
                            AnthropicClient.get_value(block, "input", {})
                        ),
                    )
                )

        return LLMResponse(
            content="\n".join(content_parts) if content_parts else None,
            reasoning="\n\n".join(reasoning_parts) if reasoning_parts else None,
            tool_calls=tuple(tool_calls),
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
