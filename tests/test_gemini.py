"""Tests for Gemini LLM client adapter."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

from google.genai import types

from cagent.llm import LLMMessage, ToolCall, ToolDefinition
from cagent.llm.gemini import GeminiClient


class FakeGeminiModels:
    def __init__(self, response: types.GenerateContentResponse) -> None:
        self.payload: dict[str, Any] = {}
        self.response = response

    def generate_content(self, **kwargs: Any) -> types.GenerateContentResponse:
        self.payload = kwargs
        return self.response


def test_gemini_complete_basic() -> None:
    candidate = types.Candidate(
        content=types.Content(
            role="model",
            parts=[types.Part(text="Hello world")],
        )
    )
    response = types.GenerateContentResponse(candidates=[candidate])

    fake_client = MagicMock()
    fake_client.models = FakeGeminiModels(response)

    client = GeminiClient(client=fake_client, model="gemini-2.0-flash")

    resp = client.complete(user_prompt="Hi")

    assert resp.content == "Hello world"
    assert fake_client.models.payload["contents"][0].parts[0].text == "Hi"
    assert fake_client.models.payload["model"] == "gemini-2.0-flash"


def test_gemini_complete_with_reasoning() -> None:
    candidate = types.Candidate(
        content=types.Content(
            role="model",
            parts=[
                types.Part(text="Thinking...", thought=True),
                types.Part(text="Final answer"),
            ],
        )
    )
    response = types.GenerateContentResponse(candidates=[candidate])

    fake_client = MagicMock()
    fake_client.models = FakeGeminiModels(response)

    client = GeminiClient(client=fake_client, model="gemini-2.0-flash")

    resp = client.complete(user_prompt="Explain quantum physics")

    assert resp.content == "Final answer"
    assert resp.reasoning == "Thinking..."


def test_gemini_tool_calls() -> None:
    candidate = types.Candidate(
        content=types.Content(
            role="model",
            parts=[
                types.Part(
                    function_call=types.FunctionCall(
                        name="get_weather",
                        args={"location": "San Francisco"},
                    )
                )
            ],
        )
    )
    response = types.GenerateContentResponse(candidates=[candidate])

    fake_client = MagicMock()
    fake_client.models = FakeGeminiModels(response)

    client = GeminiClient(client=fake_client, model="gemini-2.0-flash")

    tool = ToolDefinition(
        name="get_weather",
        description="Get weather",
        parameters={"type": "object", "properties": {"location": {"type": "string"}}},
    )

    resp = client.complete(user_prompt="Weather in SF?", tools=[tool])

    assert len(resp.tool_calls) == 1
    assert resp.tool_calls[0].name == "get_weather"
    assert resp.tool_calls[0].arguments == {"location": "San Francisco"}

    # Verify tool conversion
    tools_payload = fake_client.models.payload["config"].tools
    assert len(tools_payload) == 1
    assert len(tools_payload[0].function_declarations) == 1
    assert tools_payload[0].function_declarations[0].name == "get_weather"


def test_gemini_tool_response() -> None:
    fake_client = MagicMock()
    # We just want to check the payload sent to the API
    response = types.GenerateContentResponse(
        candidates=[
            types.Candidate(
                content=types.Content(role="model", parts=[types.Part(text="OK")])
            )
        ]
    )
    fake_client.models = FakeGeminiModels(response)

    client = GeminiClient(client=fake_client, model="gemini-2.0-flash")

    messages = [
        LLMMessage(role="user", content="What is the weather?"),
        LLMMessage(
            role="assistant",
            tool_calls=[
                ToolCall(
                    name="get_weather", arguments={"location": "SF"}, id="get_weather"
                )
            ],
        ),
        LLMMessage(role="tool", content="Sunny", tool_call_id="get_weather"),
    ]

    client.complete(user_prompt="And now?", messages=messages)

    contents = fake_client.models.payload["contents"]
    # messages include system prompt (default), plus 3 in 'messages', plus user_prompt
    # 1. system (moved to system_instruction)
    # 2. user "What is the weather?"
    # 3. assistant (model) tool call
    # 4. tool (user) function response
    # 5. user "And now?"

    # contents should have:
    # 0: user "What is the weather?"
    # 1: model tool call
    # 2: user function response + text "And now?" (merged because consecutive user roles)

    assert len(contents) == 3
    assert contents[0].role == "user"
    assert contents[1].role == "model"
    assert contents[2].role == "user"

    # Check function response
    parts = contents[2].parts
    assert parts[0].function_response is not None
    assert parts[0].function_response.name == "get_weather"
    assert parts[0].function_response.response == {"result": "Sunny"}
    assert parts[1].text == "And now?"
