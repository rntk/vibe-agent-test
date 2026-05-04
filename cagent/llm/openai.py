"""OpenAI-compatible LLM client adapter.

This module imports the ``openai`` SDK and uses its native types for request
serialization and response parsing. The adapter still accepts an already-created
``openai.OpenAI`` client so callers can configure authentication, base URL, and
HTTP options externally.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

import openai
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionMessageCustomToolCall,
    ChatCompletionMessageFunctionToolCall,
)
from openai.types.responses import (
    Response,
    ResponseCustomToolCall,
    ResponseFunctionToolCall,
    ResponseOutputMessage,
    ResponseOutputText,
    ResponseReasoningItem,
)

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

    client: openai.OpenAI
    default_model: str
    deepseek_thinking: bool = False

    @classmethod
    def from_config(
        cls,
        *,
        api_key: str,
        base_url: str | None = None,
        default_model: str = "gpt-4o",
        deepseek_thinking: bool = False,
    ) -> OpenAIChatCompletionsClient:
        """Create a client from configuration values."""

        return cls(
            client=openai.OpenAI(
                api_key=api_key,
                base_url=base_url,
            ),
            default_model=default_model,
            deepseek_thinking=deepseek_thinking,
        )

    def _complete(self, request: LLMRequest) -> LLMResponse:
        """Run one LLM turn through an OpenAI-compatible client."""

        payload: dict[str, Any] = {
            "model": request.model or self.default_model,
            "messages": self.to_provider_messages(request.all_messages()),
        }
        if request.tools:
            payload["tools"] = self.to_provider_tools(request.tools)
        if request.tool_choice is not None:
            payload["tool_choice"] = request.tool_choice
        if request.parallel_tool_calls is not None:
            payload["parallel_tool_calls"] = request.parallel_tool_calls
        if request.temperature is not None and not self.deepseek_thinking:
            # DeepSeek thinking mode ignores temperature; omit to keep payload clean.
            payload["temperature"] = request.temperature
        if request.reasoning_effort is not None:
            payload["reasoning_effort"] = request.reasoning_effort
        if self.deepseek_thinking:
            payload["extra_body"] = {"thinking": {"type": "enabled"}}

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
        if message.role == "assistant" and message.reasoning:
            output["reasoning_content"] = message.reasoning
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

        result: list[dict[str, Any]] = []
        for tool in tools:
            tool_def: dict[str, Any] = {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": dict(tool.parameters),
                },
            }
            if tool.strict is not None:
                tool_def["function"]["strict"] = tool.strict
            result.append(tool_def)
        return result

    @staticmethod
    def from_provider_tool_calls(
        tool_calls: Sequence[ChatCompletionMessageFunctionToolCall]
        | Sequence[ChatCompletionMessageCustomToolCall]
        | None,
    ) -> tuple[ToolCall, ...]:
        """Convert OpenAI chat-completion tool calls to neutral tool calls."""

        if not tool_calls:
            return ()

        parsed_calls: list[ToolCall] = []
        for tool_call in tool_calls:
            if tool_call.type == "function":
                parsed_calls.append(
                    ToolCall(
                        id=tool_call.id,
                        name=tool_call.function.name,
                        arguments=OpenAIChatCompletionsClient.parse_arguments(
                            tool_call.function.arguments
                        ),
                    )
                )
            elif tool_call.type == "custom":
                parsed_calls.append(
                    ToolCall(
                        id=tool_call.id,
                        name=tool_call.custom.name,
                        arguments={"input": tool_call.custom.input},
                    )
                )
        return tuple(parsed_calls)

    @staticmethod
    def from_provider_response(response: ChatCompletion) -> LLMResponse:
        """Convert an OpenAI-compatible response to a neutral response."""

        first_choice = response.choices[0] if response.choices else None
        message = first_choice.message if first_choice else None
        reasoning = getattr(message, "reasoning_content", None) if message else None
        return LLMResponse(
            content=message.content if message else None,
            reasoning=reasoning,
            tool_calls=OpenAIChatCompletionsClient.from_provider_tool_calls(
                message.tool_calls if message else None
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


@dataclass(frozen=True, slots=True)
class OpenAIResponsesClient(LLMClient):
    """Adapter for the OpenAI Responses API.

    The Responses API uses a different message format than chat completions:
    conversation history is passed as an ``input`` array of items, and model
    outputs are read from ``response.output``.
    """

    client: openai.OpenAI
    default_model: str

    @classmethod
    def from_config(
        cls,
        *,
        api_key: str,
        base_url: str | None = None,
        default_model: str = "gpt-4o",
    ) -> OpenAIResponsesClient:
        """Create a client from configuration values."""

        return cls(
            client=openai.OpenAI(
                api_key=api_key,
                base_url=base_url,
            ),
            default_model=default_model,
        )

    def _complete(self, request: LLMRequest) -> LLMResponse:
        """Run one LLM turn through the OpenAI Responses API."""

        input_items: list[dict[str, Any]] = []
        instructions: str | None = request.system_prompt

        for msg in request.all_messages():
            if msg.role == "system":
                if instructions:
                    instructions = f"{instructions}\n\n{msg.content}"
                else:
                    instructions = msg.content or ""
                continue

            if msg.role == "user":
                input_items.append(
                    {"role": "user", "content": msg.content or ""}
                )
            elif msg.role == "assistant":
                output: list[dict[str, Any]] = []
                if msg.reasoning:
                    output.append(
                        {
                            "type": "reasoning",
                            "summary": [
                                {
                                    "type": "summary_text",
                                    "text": msg.reasoning,
                                }
                            ],
                        }
                    )
                for tc in msg.tool_calls:
                    output.append(
                        {
                            "type": "function_call",
                            "call_id": tc.id or "",
                            "name": tc.name,
                            "arguments": json.dumps(dict(tc.arguments)),
                        }
                    )
                if msg.content:
                    output.append(
                        {"type": "output_text", "text": msg.content}
                    )
                input_items.append({"role": "assistant", "output": output})
            elif msg.role == "tool":
                input_items.append(
                    {
                        "type": "function_call_output",
                        "call_id": msg.tool_call_id or "",
                        "output": msg.content or "",
                    }
                )

        kwargs: dict[str, Any] = {
            "model": request.model or self.default_model,
            "input": input_items,
        }
        if instructions:
            kwargs["instructions"] = instructions
        if request.tools:
            kwargs["tools"] = self.to_provider_tools(request.tools)
        if request.tool_choice is not None:
            kwargs["tool_choice"] = request.tool_choice
        if request.parallel_tool_calls is not None:
            kwargs["parallel_tool_calls"] = request.parallel_tool_calls
        if request.temperature is not None:
            kwargs["temperature"] = request.temperature
        if request.reasoning_effort is not None:
            kwargs["reasoning"] = {"effort": request.reasoning_effort}

        response = self.client.responses.create(**kwargs)
        return self.from_provider_response(response)

    @staticmethod
    def to_provider_tools(tools: Sequence[ToolDefinition]) -> list[dict[str, Any]]:
        """Convert provider-neutral tools to OpenAI Responses API tools."""

        result: list[dict[str, Any]] = []
        for tool in tools:
            tool_def: dict[str, Any] = {
                "type": "function",
                "name": tool.name,
                "description": tool.description,
                "parameters": dict(tool.parameters),
            }
            if tool.strict is not None:
                tool_def["strict"] = tool.strict
            result.append(tool_def)
        return result

    @staticmethod
    def from_provider_response(response: Response) -> LLMResponse:
        """Convert an OpenAI Responses API response to a neutral response."""

        content_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        reasoning_parts: list[str] = []

        for item in response.output:
            if isinstance(item, ResponseOutputMessage):
                for sub in item.content:
                    if isinstance(sub, ResponseOutputText):
                        content_parts.append(sub.text)
                    elif isinstance(sub, ResponseReasoningItem):
                        for s in sub.summary:
                            reasoning_parts.append(s.text)
            elif isinstance(item, ResponseFunctionToolCall):
                tool_calls.append(
                    ToolCall(
                        id=item.call_id,
                        name=item.name,
                        arguments=OpenAIResponsesClient.parse_arguments(
                            item.arguments
                        ),
                    )
                )
            elif isinstance(item, ResponseCustomToolCall):
                tool_calls.append(
                    ToolCall(
                        id=item.call_id,
                        name=item.name,
                        arguments={"input": item.input},
                    )
                )
            elif isinstance(item, ResponseReasoningItem):
                for s in item.summary:
                    reasoning_parts.append(s.text)

        return LLMResponse(
            content="\n".join(content_parts) if content_parts else None,
            reasoning="\n".join(reasoning_parts) if reasoning_parts else None,
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
