"""Gemini LLM client adapter."""

from __future__ import annotations

import base64
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

        if message.role == "assistant":
            if message.reasoning:
                parts.append(types.Part(text=message.reasoning, thought=True))

            if message.tool_calls:
                for index, tool_call in enumerate(message.tool_calls):
                    part_kwargs: dict[str, Any] = {
                        "function_call": types.FunctionCall(
                            name=tool_call.name,
                            args=dict(tool_call.arguments),
                        )
                    }
                    # According to Gemini API docs, thought_signature should be 
                    # included with the function call (typically the first one 
                    # if parallel).
                    if index == 0 and message.thought_signature:
                        try:
                            part_kwargs["thought_signature"] = base64.b64decode(message.thought_signature)
                        except Exception:
                            part_kwargs["thought_signature"] = message.thought_signature
                    
                    parts.append(types.Part(**part_kwargs))
            elif message.thought_signature:
                try:
                    ts_bytes = base64.b64decode(message.thought_signature)
                except Exception:
                    ts_bytes = message.thought_signature  # type: ignore

                if parts:
                    last_part = parts[-1]
                    if last_part.text:
                        parts[-1] = types.Part(
                            text=last_part.text,
                            thought_signature=ts_bytes
                        )
                else:
                    parts.append(types.Part(thought_signature=ts_bytes))

        role = "user" if message.role == "user" else "model"
        return types.Content(role=role, parts=parts)

    @staticmethod
    def to_provider_tools(tools: Sequence[ToolDefinition]) -> list[types.Tool]:
        """Convert provider-neutral tools to Gemini API tools."""

        function_declarations = []
        for tool in tools:
            # Gemini is strict about JSON schema and often rejects 'additionalProperties'.
            # We recursively strip it from the parameters.
            cleaned_parameters = GeminiClient._strip_additional_properties(
                dict(tool.parameters)
            )
            function_declarations.append(
                types.FunctionDeclaration(
                    name=tool.name,
                    description=tool.description,
                    parameters=cleaned_parameters,
                )
            )
        return [types.Tool(function_declarations=function_declarations)]

    @staticmethod
    def _strip_additional_properties(schema: Any) -> Any:
        """Recursively remove additionalProperties from a JSON schema."""
        if not isinstance(schema, dict):
            return schema

        # Create a copy to avoid mutating the original
        cleaned = {k: GeminiClient._strip_additional_properties(v) for k, v in schema.items() 
                   if k not in ("additionalProperties", "additional_properties")}
        return cleaned

    @staticmethod
    def from_provider_response(response: types.GenerateContentResponse) -> LLMResponse:
        """Convert a Gemini response to a neutral response."""

        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        thought_signature: str | None = None

        if not response.candidates:
            return LLMResponse(raw=response)

        candidate = response.candidates[0]
        if candidate.content and candidate.content.parts:
            for part in candidate.content.parts:
                if part.thought_signature:
                    if isinstance(part.thought_signature, bytes):
                        thought_signature = base64.b64encode(part.thought_signature).decode("utf-8")
                    else:
                        thought_signature = str(part.thought_signature)

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
            thought_signature=thought_signature,
            tool_calls=tuple(tool_calls),
            raw=response,
        )
