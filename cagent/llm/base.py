"""Provider-neutral LLM interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from cagent.tracing import get_trace

MessageRole = Literal["system", "user", "assistant", "tool"]


@dataclass(frozen=True, slots=True)
class ToolDefinition:
    """Provider-neutral function/tool schema."""

    name: str
    description: str
    parameters: Mapping[str, Any] = field(default_factory=dict)
    strict: bool | None = None


BASH_TOOL = ToolDefinition(
    name="bash",
    description=(
        "Run a bash command and return the exit code, standard output, and "
        "standard error."
    ),
    parameters={
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "Bash command to run.",
            },
            "cwd": {
                "type": "string",
                "description": "Optional working directory for the command.",
            },
            "timeout_seconds": {
                "type": "integer",
                "minimum": 1,
                "description": "Optional command timeout in seconds.",
            },
        },
        "required": ["command"],
        "additionalProperties": False,
    },
)

SEARCH_AND_REPLACE_TOOL = ToolDefinition(
    name="search_and_replace",
    description=(
        "Replace an exact substring in a text file. Provide the file path, "
        "the existing text to find, and the new text to put in its place. "
        "The old text must match exactly (including whitespace) and must "
        "appear exactly once in the file."
    ),
    parameters={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to the text file to modify.",
            },
            "old_text": {
                "type": "string",
                "description": "Exact text to search for in the file.",
            },
            "new_text": {
                "type": "string",
                "description": "Replacement text.",
            },
        },
        "required": ["path", "old_text", "new_text"],
        "additionalProperties": False,
    },
)


WRITE_FILE_TOOL = ToolDefinition(
    name="write_file",
    description=(
        "Use this tool to create, overwrite, or surgically update a text file. "
        "You can replace the entire file, append to it, or replace a specific "
        "1-based inclusive line range by providing start_line and/or end_line. "
        "This is the preferred way to edit files. To make a small change, use "
        "bash to read the file with cat -n first, note the line numbers you want "
        "to change, then call write_file with start_line and end_line set to "
        "those numbers and content set to the replacement text. Creates parent "
        "directories for new files."
    ),
    parameters={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to the text file to write.",
            },
            "content": {
                "type": "string",
                "description": "Content to write to the file.",
            },
            "append": {
                "type": "boolean",
                "description": "If true, append content instead of overwriting.",
            },
            "start_line": {
                "type": "integer",
                "minimum": 1,
                "description": (
                    "Optional 1-based first line to replace. If omitted while "
                    "end_line is set, replacement starts at line 1."
                ),
            },
            "end_line": {
                "type": "integer",
                "minimum": 1,
                "description": (
                    "Optional 1-based last line to replace, inclusive. If "
                    "omitted while start_line is set, replacement continues "
                    "through the end of the file."
                ),
            },
        },
        "required": ["path", "content"],
        "additionalProperties": False,
    },
)


@dataclass(frozen=True, slots=True)
class ToolCall:
    """Provider-neutral tool invocation emitted by an LLM."""

    name: str
    arguments: Mapping[str, Any]
    id: str | None = None


@dataclass(frozen=True, slots=True)
class LLMMessage:
    """Provider-neutral chat message."""

    role: MessageRole
    content: str | None = None
    tool_calls: Sequence[ToolCall] = field(default_factory=tuple)
    tool_call_id: str | None = None
    reasoning: str | None = None
    thought_signature: str | None = None


@dataclass(frozen=True, slots=True)
class LLMRequest:
    """Provider-neutral request for a single model turn."""

    user_prompt: str
    system_prompt: str | None = None
    tools: Sequence[ToolDefinition] = field(default_factory=tuple)
    model: str | None = None
    temperature: float | None = None
    messages: Sequence[LLMMessage] = field(default_factory=tuple)
    tool_choice: str | dict[str, Any] | None = None
    parallel_tool_calls: bool | None = None
    output_schema: Mapping[str, Any] | None = None
    reasoning_effort: str | None = None

    def all_messages(self) -> tuple[LLMMessage, ...]:
        """Return request messages with system and user prompts applied."""

        messages: list[LLMMessage] = []
        if self.system_prompt:
            messages.append(LLMMessage(role="system", content=self.system_prompt))
        messages.extend(_reasoning_before_tool_call_messages(self.messages))
        if self.user_prompt:
            messages.append(LLMMessage(role="user", content=self.user_prompt))
        return tuple(messages)


@dataclass(frozen=True, slots=True)
class LLMResponse:
    """Provider-neutral response for a single model turn."""

    content: str | None = None
    reasoning: str | None = None
    thought_signature: str | None = None
    tool_calls: Sequence[ToolCall] = field(default_factory=tuple)
    raw: Any | None = None


def _reasoning_before_tool_call_messages(
    messages: Sequence[LLMMessage],
) -> tuple[LLMMessage, ...]:
    """Split assistant reasoning into a message before its tool calls."""

    normalized: list[LLMMessage] = []
    for message in messages:
        if (
            message.role != "assistant"
            or message.content is not None
            or not message.reasoning
            or not message.tool_calls
        ):
            normalized.append(message)
            continue

        normalized.append(
            LLMMessage(
                role="assistant",
                content=message.reasoning,
                reasoning=message.reasoning,
                thought_signature=message.thought_signature,
            )
        )
        normalized.append(
            LLMMessage(
                role="assistant",
                content=None,
                reasoning=message.reasoning,
                tool_calls=tuple(message.tool_calls),
            )
        )

    return tuple(normalized)


class LLMClient(ABC):
    """Base class for provider-specific LLM clients."""

    @abstractmethod
    def _complete(self, request: LLMRequest) -> LLMResponse:
        """Run one provider-specific LLM turn."""

    def complete(
        self,
        user_prompt: str,
        *,
        system_prompt: str | None = None,
        tools: Sequence[ToolDefinition] = (),
        model: str | None = None,
        temperature: float | None = None,
        messages: Sequence[LLMMessage] = (),
        tool_choice: str | dict[str, Any] | None = None,
        parallel_tool_calls: bool | None = None,
        output_schema: Mapping[str, Any] | None = None,
        reasoning_effort: str | None = None,
        trace_name: str | None = None,
        trace_attributes: Mapping[str, Any] | None = None,
    ) -> LLMResponse:
        """Run one provider-neutral LLM turn."""

        if not messages and system_prompt is None:
            system_prompt = (
                "You are a software engineering assistant. "
                "Use the available tools to research the current project. "
                "If the users task already implemented in the codebase, find "
                "the relevant code, explain it to the user and finish. "
                "Otherwise, research how to implement the users request using "
                "the available tools and information in the codebase. "
                f"Current directory: {Path.cwd()}"
            )

        request = LLMRequest(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            tools=tools,
            model=model,
            temperature=temperature,
            messages=messages,
            tool_choice=tool_choice,
            parallel_tool_calls=parallel_tool_calls,
            output_schema=output_schema,
            reasoning_effort=reasoning_effort,
        )
        all_msgs = request.all_messages()
        attributes: dict[str, Any] = {
            "request": request,
            "all_messages": all_msgs,
            "message_count": len(all_msgs),
            "tool_names": [tool.name for tool in tools],
        }
        if trace_attributes:
            attributes.update(trace_attributes)
        with get_trace().span(
            trace_name or "llm.complete",
            attributes,
        ) as span:
            response = self._complete(request)
            span.set_attribute("response", response)
            span.set_attribute("response_content", response.content)
            span.set_attribute("response_reasoning", response.reasoning)
            span.set_attribute("response_tool_calls", response.tool_calls)
            return response
