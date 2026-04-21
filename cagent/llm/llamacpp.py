import json
import logging
import os
import re
from collections.abc import Mapping, Sequence
from http.client import HTTPConnection, HTTPSConnection
from typing import Any, cast
from urllib.parse import urlparse

from cagent.llm import LLMClient
from cagent.llm.base import (
    LLMMessage,
    LLMRequest,
    LLMResponse,
    ToolCall,
    ToolDefinition,
)

_THINK_TAG_RE = re.compile(
    r"<think\b[^>]*>(.*?)</think>",
    flags=re.DOTALL | re.IGNORECASE,
)


class LLamaCPP(LLMClient):
    def __init__(
        self,
        host: str,
        model: str = "moonshotai/Kimi-K2.5",
        max_context_tokens: int = 11000,
        token: str | None = None,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        temperature: float = 0.8,
        min_p: float = 0.05,
        repeat_penalty: float = 1.1,
        repeat_last_n: int = 64,
        dry_multiplier: float = 0.8,
        dry_base: float = 1.75,
        dry_allowed_length: int = 2,
        stop: list[str] | None = None,
        provider_name: str = "LlamaCPP",
        provider_key: str = "llamacpp",
    ) -> None:
        super().__init__()
        u = urlparse(host)
        self.__host = u.netloc
        self.__is_https = u.scheme.lower() == "https"
        self.__model = model
        # Token can be passed in explicitly or read from the environment variable TOKEN
        self.__token = token or os.getenv("TOKEN")
        self.__temperature = temperature
        self.__min_p = min_p
        self.__repeat_penalty = repeat_penalty
        self.__repeat_last_n = repeat_last_n
        self.__dry_multiplier = dry_multiplier
        self.__dry_base = dry_base
        self.__dry_allowed_length = dry_allowed_length
        self.__stop = stop or ["User:", "\n\n"]
        self.__provider_name = provider_name
        self.__provider_key = provider_key

    @property
    def provider_name(self) -> str:
        return self.__provider_name

    @property
    def provider_key(self) -> str:
        return self.__provider_key

    @property
    def model_name(self) -> str:
        return self.__model

    def _extract_reasoning_and_content(
        self,
        response_payload: dict[str, Any],
    ) -> tuple[str | None, str | None]:
        choices = response_payload.get("choices")
        first_choice = choices[0] if isinstance(choices, list) and choices else {}
        message = first_choice.get("message") if isinstance(first_choice, dict) else {}
        if not isinstance(message, dict):
            message = {}

        raw_content = message.get("content")
        content = raw_content if isinstance(raw_content, str) else ""

        reasoning_parts: list[str] = []
        for key in ("reasoning", "reasoning_content", "thinking"):
            value = message.get(key)
            if isinstance(value, str):
                stripped = value.strip()
                if stripped:
                    reasoning_parts.append(stripped)

        for think_match in _THINK_TAG_RE.findall(content):
            stripped = think_match.strip()
            if stripped:
                reasoning_parts.append(stripped)

        reasoning = "\n\n".join(reasoning_parts).strip() or None
        cleaned_content = _THINK_TAG_RE.sub("", content).strip() or None
        return reasoning, cleaned_content

    @staticmethod
    def to_provider_tools(tools: Sequence[ToolDefinition]) -> list[dict[str, Any]]:
        """Convert provider-neutral tools to llama.cpp chat-completion tools."""
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
    def to_provider_messages(messages: Sequence[LLMMessage]) -> list[dict[str, Any]]:
        """Convert provider-neutral messages to llama.cpp chat-completion messages."""
        return [LLamaCPP.to_provider_message(msg) for msg in messages]

    @staticmethod
    def to_provider_message(message: LLMMessage) -> dict[str, Any]:
        """Convert one provider-neutral message to a llama.cpp chat message."""
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
                        "arguments": (
                            json.dumps(dict(tool_call.arguments))
                            if isinstance(tool_call.arguments, Mapping)
                            else str(tool_call.arguments)
                        ),
                    },
                }
                for tool_call in message.tool_calls
            ]
        return output

    @staticmethod
    def from_provider_tool_calls(tool_calls: Any) -> tuple[ToolCall, ...]:
        """Convert llama.cpp tool calls to neutral tool calls."""
        if not tool_calls:
            return ()

        parsed_calls: list[ToolCall] = []
        for tool_call in tool_calls:
            function = LLamaCPP.get_value(tool_call, "function", tool_call)

            parsed_calls.append(
                ToolCall(
                    id=LLamaCPP.get_value(
                        tool_call,
                        "id",
                        LLamaCPP.get_value(tool_call, "call_id"),
                    ),
                    name=cast(str, LLamaCPP.get_value(function, "name")),
                    arguments=LLamaCPP.parse_arguments(
                        LLamaCPP.get_value(function, "arguments", "{}")
                    ),
                )
            )
        return tuple(parsed_calls)

    @staticmethod
    def from_provider_response(response: Any) -> LLMResponse:
        """Convert a llama.cpp chat-completion response to a neutral response."""

        choices = LLamaCPP.get_value(response, "choices", ())
        first_choice = choices[0] if choices else {}
        message = LLamaCPP.get_value(first_choice, "message", {})
        if not isinstance(message, Mapping):
            message = {}

        content = LLamaCPP.get_value(message, "content")
        tool_calls = LLamaCPP.from_provider_tool_calls(
            LLamaCPP.get_value(message, "tool_calls")
        )
        return LLMResponse(content=content, tool_calls=tool_calls, raw=response)

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

    def _call_single(
        self,
        messages: Sequence[LLMMessage],
        temperature: float,
        tools: Sequence[ToolDefinition] = (),
    ) -> LLMResponse:
        """Single attempt to call the LLM without retry logic."""
        conn = self.get_connection()
        try:
            prompt_content = messages[-1].content if messages else ""
            logging.info(f"LLM request: {prompt_content}")

            payload: dict[str, Any] = {
                "model": self.__model,
                "messages": self.to_provider_messages(messages),
                "temperature": temperature,
                "cache_prompt": True,
                "min_p": self.__min_p,
                "repeat_penalty": self.__repeat_penalty,
                "repeat_last_n": self.__repeat_last_n,
                "dry_multiplier": self.__dry_multiplier,
                "dry_base": self.__dry_base,
                "dry_allowed_length": self.__dry_allowed_length,
                # "stop": self.__stop,
            }
            if tools:
                payload["tools"] = self.to_provider_tools(tools)

            body = json.dumps(payload)
            headers = {"Content-type": "application/json"}
            if self.__token:
                headers["Authorization"] = f"Bearer {self.__token}"
            conn.request("POST", "/v1/chat/completions", body, headers)
            res = conn.getresponse()
            resp_body = res.read()
            if res.status != 200:
                err_msg = f"{res.status} - {res.reason} - {resp_body}"
                logging.error(err_msg)
                raise RuntimeError(f"LLM API error: {res.status} {res.reason}")

            resp = json.loads(resp_body)

            reasoning, content = self._extract_reasoning_and_content(resp)

            # Extract tool calls from response
            choices = resp.get("choices")
            first_choice = choices[0] if isinstance(choices, list) and choices else {}
            message = (
                first_choice.get("message") if isinstance(first_choice, dict) else {}
            )
            if not isinstance(message, dict):
                message = {}

            tool_calls = self.from_provider_tool_calls(message.get("tool_calls"))

            if content is None and not tool_calls:
                logging.error(
                    "LLM response missing 'choices[0].message.content' and tool_calls"
                )
                logging.error(f"Full response: {resp}")
                raise RuntimeError("LLM returned empty response")
            if reasoning:
                logging.info(f"LLM reasoning: {reasoning}")
            if content:
                logging.info(f"LLM response: {content}")
            if tool_calls:
                logging.info(f"LLM tool calls: {[tc.name for tc in tool_calls]}")

            return LLMResponse(content=content, tool_calls=tool_calls, raw=resp)
        except json.JSONDecodeError as e:
            err_msg = f"JSON decode error: {e}"
            logging.error(err_msg)
            raise RuntimeError(f"Invalid JSON response from LLM: {e}") from e
        except RuntimeError:
            raise
        except Exception as e:
            err_msg = f"LLM call exception: {type(e).__name__}: {e}"
            logging.error(err_msg)
            raise RuntimeError(f"LLM call failed: {e}") from e
        finally:
            conn.close()

    def _complete(self, request: LLMRequest) -> LLMResponse:
        temperature = (
            request.temperature
            if request.temperature is not None
            else self.__temperature
        )
        return self._call_single(
            messages=request.all_messages(),
            temperature=temperature,
            tools=request.tools,
        )

    def get_connection(self) -> HTTPConnection | HTTPSConnection:
        if self.__is_https:
            return HTTPSConnection(self.__host)
        else:
            return HTTPConnection(self.__host)
