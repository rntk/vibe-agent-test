"""Gemini LLM client adapter."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

from google import genai
from google.genai import types

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
class GeminiClient(LLMClient):
    """Adapter for Google Gemini API via google-genai SDK."""

    client: genai.Client
    model: str

    def _complete(self, request: LLMRequest) -> LLMResponse:
        """Run one LLM turn through the Gemini API."""

        system_instruction: types.Content | None = None
        contents: list[types.Content] = []

        for msg in request.all_messages():
            if msg.role == "system":
                if msg.content:
                    if system_instruction is None:
                        system_instruction = types.Content(
                            role="system",
                            parts=[types.Part(text=msg.content)],
                        )
                    else:
                        # Append to existing system instruction
                        cast(list[types.Part], system_instruction.parts).append(
                            types.Part(text=f"\n\n{msg.content}")
                        )
                continue

            content = self.to_provider_content(msg)
            # Gemini expects alternating user/model roles. 
            # If we have consecutive roles, we might need to merge or wrap.
            if contents and contents[-1].role == content.role:
                cast(list[types.Part], contents[-1].parts).extend(
                    cast(list[types.Part], content.parts)
                )
            else:
                contents.append(content)

        config_kwargs: dict[str, Any] = {}
        if system_instruction:
            config_kwargs["system_instruction"] = system_instruction
        if request.tools:
            config_kwargs["tools"] = self.to_provider_tools(request.tools)
        if request.temperature is not None:
            config_kwargs["temperature"] = request.temperature
        if request.output_schema:
            config_kwargs["response_mime_type"] = "application/json"
            config_kwargs["response_schema"] = request.output_schema

        # Gemini 2.0 Thinking/Reasoning support
        # Note: Depending on the model, reasoning might be enabled differently.
        # For gemini-2.0-flash-thinking-preview, it's often automatic.
        
        config = types.GenerateContentConfig(**config_kwargs)

        with get_trace().span(
            "llm.gemini.complete",
            {
                "model": request.model or self.model,
                "tool_count": len(request.tools),
            },
        ) as span:
            response = self.client.models.generate_content(
                model=request.model or self.model,
                contents=contents,
                config=config,
            )
            
            # Trace usage if available
            if response.usage_metadata:
                span.set_attribute("usage", {
                    "prompt_token_count": response.usage_metadata.prompt_token_count,
                    "candidates_token_count": response.usage_metadata.candidates_token_count,
                    "total_token_count": response.usage_metadata.total_token_count,
                })

            result = self.from_provider_response(response)
            if result.reasoning:
                span.set_attribute("response_reasoning", result.reasoning)
            return result

    @staticmethod
    def to_provider_content(message: LLMMessage) -> types.Content:
        """Convert one provider-neutral message to a Gemini Content object."""

        parts: list[types.Part] = []
        
        if message.role == "tool":
            # Gemini uses function_response for tool results
            parts.append(
                types.Part(
                    function_response=types.FunctionResponse(
                        name=message.tool_call_id or "unknown",
                        response={"result": message.content},
                    )
                )
            )
            return types.Content(role="user", parts=parts)

        if message.content:
            parts.append(types.Part(text=message.content))

        if message.role == "assistant" and message.tool_calls:
            for tool_call in message.tool_calls:
                parts.append(
                    types.Part(
                        function_call=types.FunctionCall(
                            name=tool_call.name,
                            args=dict(tool_call.arguments),
                        )
                    )
                )

        role = "user" if message.role == "user" else "model"
        return types.Content(role=role, parts=parts)

    @staticmethod
    def to_provider_tools(tools: Sequence[ToolDefinition]) -> list[types.Tool]:
        """Convert provider-neutral tools to Gemini API tools."""

        function_declarations = []
        for tool in tools:
            # Gemini expects Schema object or dict that matches.
            # python-genai usually accepts dicts for parameters.
            function_declarations.append(
                types.FunctionDeclaration(
                    name=tool.name,
                    description=tool.description,
                    parameters=tool.parameters,
                )
            )
        return [types.Tool(function_declarations=function_declarations)]

    @staticmethod
    def from_provider_response(response: types.GenerateContentResponse) -> LLMResponse:
        """Convert a Gemini response to a neutral response."""

        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        tool_calls: list[ToolCall] = []

        if not response.candidates:
            return LLMResponse(raw=response)

        candidate = response.candidates[0]
        if candidate.content and candidate.content.parts:
            for part in candidate.content.parts:
                if getattr(part, "thought", False):
                    reasoning_parts.append(part.text or "")
                elif part.text:
                    content_parts.append(part.text)

                if part.function_call:
                    tool_calls.append(
                        ToolCall(
                            id=part.function_call.name, # Gemini doesn't have call IDs like OpenAI/Anthropic
                            name=part.function_call.name,
                            arguments=part.function_call.args or {},
                        )
                    )

        return LLMResponse(
            content="".join(content_parts) if content_parts else None,
            reasoning="\n\n".join(reasoning_parts) if reasoning_parts else None,
            tool_calls=tuple(tool_calls),
            raw=response,
        )
