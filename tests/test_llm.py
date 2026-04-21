from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from cagent.llm import (
    LLMClient,
    LLMMessage,
    LLMRequest,
    LLMResponse,
    ToolCall,
    ToolDefinition,
)
from cagent.llm.llamacpp import LLamaCPP
from cagent.llm.openai import OpenAIChatCompletionsClient


class FakeClient(LLMClient):
    def __init__(self) -> None:
        self.request: LLMRequest | None = None

    def _complete(self, request: LLMRequest) -> LLMResponse:
        self.request = request
        return LLMResponse(content="ok")


class FakeOpenAIClient:
    def __init__(self, message: Any | None = None) -> None:
        self.payload: dict[str, Any] | None = None
        self.message = message
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self.create_completion)
        )

    def create_completion(self, **payload: Any) -> Any:
        self.payload = payload
        message = self.message or SimpleNamespace(
            content=None,
            tool_calls=[
                SimpleNamespace(
                    id="call_1",
                    function=SimpleNamespace(
                        name="search_docs",
                        arguments='{"query": "llm adapters"}',
                    ),
                )
            ],
        )
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def test_llm_client_base_passes_system_user_and_tools_to_subclass() -> None:
    client = FakeClient()
    tool = ToolDefinition(
        name="search_docs",
        description="Search docs.",
        parameters={"type": "object"},
    )

    response = client.complete(
        "Find adapters",
        system_prompt="You are concise.",
        tools=[tool],
        model="test-model",
        temperature=0.2,
    )

    assert response.content == "ok"
    assert client.request is not None
    assert client.request.system_prompt == "You are concise."
    assert client.request.user_prompt == "Find adapters"
    assert client.request.tools == [tool]
    assert client.request.model == "test-model"
    assert client.request.temperature == 0.2


def test_request_all_messages_includes_system_and_user_prompt() -> None:
    request = LLMRequest(
        system_prompt="Use tools when helpful.",
        user_prompt="What is next?",
    )

    assert [message.role for message in request.all_messages()] == ["system", "user"]
    assert [message.content for message in request.all_messages()] == [
        "Use tools when helpful.",
        "What is next?",
    ]


def test_openai_adapter_transforms_text_response() -> None:
    sdk_client = FakeOpenAIClient(
        message=SimpleNamespace(content="plain answer", tool_calls=None),
    )
    client = OpenAIChatCompletionsClient(
        client=sdk_client,
        default_model="gpt-test",
    )

    response = client.complete("Say hi.")

    assert response.content == "plain answer"
    assert response.tool_calls == ()


def test_openai_adapter_serializes_tool_calls_and_results() -> None:
    assistant_tool_call = ToolCall(
        id="call_1",
        name="search_docs",
        arguments={"query": "llm adapters"},
    )
    messages = [
        LLMMessage(
            role="assistant",
            content=None,
            tool_calls=[assistant_tool_call],
        ),
        LLMMessage(
            role="tool",
            content="Adapter docs result",
            tool_call_id="call_1",
        ),
    ]

    provider_messages = OpenAIChatCompletionsClient.to_provider_messages(messages)

    assert provider_messages == [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "search_docs",
                        "arguments": '{"query": "llm adapters"}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "content": "Adapter docs result",
            "tool_call_id": "call_1",
        },
    ]


def test_openai_adapter_parses_dict_tool_calls() -> None:
    response = OpenAIChatCompletionsClient.from_provider_response(
        {
            "choices": [
                {
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "function": {
                                    "name": "search_docs",
                                    "arguments": {"query": "llm adapters"},
                                },
                            }
                        ],
                    }
                }
            ]
        }
    )

    assert response.tool_calls == (
        ToolCall(
            id="call_1",
            name="search_docs",
            arguments={"query": "llm adapters"},
        ),
    )


def test_openai_adapter_parses_flat_tool_calls() -> None:
    tool_calls = OpenAIChatCompletionsClient.from_provider_tool_calls(
        [
            {
                "call_id": "call_1",
                "name": "search_docs",
                "arguments": None,
            }
        ]
    )

    assert tool_calls == (
        ToolCall(
            id="call_1",
            name="search_docs",
            arguments={},
        ),
    )


def test_llamacpp_adapter_serializes_tool_calls_and_results() -> None:
    assistant_tool_call = ToolCall(
        id="call_1",
        name="search_docs",
        arguments={"query": "llm adapters"},
    )
    messages = [
        LLMMessage(
            role="assistant",
            content=None,
            tool_calls=[assistant_tool_call],
        ),
        LLMMessage(
            role="tool",
            content="Adapter docs result",
            tool_call_id="call_1",
        ),
    ]

    provider_messages = LLamaCPP.to_provider_messages(messages)

    assert provider_messages == [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "search_docs",
                        "arguments": '{"query": "llm adapters"}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "content": "Adapter docs result",
            "tool_call_id": "call_1",
        },
    ]


def test_llamacpp_adapter_parses_text_and_tool_call_responses() -> None:
    text_response = LLamaCPP.from_provider_response(
        {"choices": [{"message": {"content": "plain answer"}}]}
    )
    tool_response = LLamaCPP.from_provider_response(
        {
            "choices": [
                {
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "function": {
                                    "name": "search_docs",
                                    "arguments": '{"query": "llm adapters"}',
                                },
                            }
                        ],
                    }
                }
            ]
        }
    )

    assert text_response.content == "plain answer"
    assert text_response.tool_calls == ()
    assert tool_response.content is None
    assert tool_response.tool_calls == (
        ToolCall(
            id="call_1",
            name="search_docs",
            arguments={"query": "llm adapters"},
        ),
    )


def test_llamacpp_adapter_extracts_reasoning_from_response_fields() -> None:
    response = LLamaCPP.from_provider_response(
        {
            "choices": [
                {
                    "message": {
                        "reasoning_content": "I should answer directly.",
                        "content": "plain answer",
                    }
                }
            ]
        }
    )

    assert response.reasoning == "I should answer directly."
    assert response.content == "plain answer"


def test_llamacpp_adapter_extracts_reasoning_from_think_tags() -> None:
    response = LLamaCPP.from_provider_response(
        {
            "choices": [
                {
                    "message": {
                        "content": "<think>Check inputs first.</think>\nplain answer",
                    }
                }
            ]
        }
    )

    assert response.reasoning == "Check inputs first."
    assert response.content == "plain answer"


def test_openai_adapter_transforms_request_and_tool_calls() -> None:
    sdk_client = FakeOpenAIClient()
    client = OpenAIChatCompletionsClient(
        client=sdk_client,
        default_model="gpt-test",
    )
    tool = ToolDefinition(
        name="search_docs",
        description="Search docs.",
        parameters={
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    )

    response = client.complete(
        "Find the adapter docs.",
        system_prompt="Answer with sources.",
        tools=[tool],
        temperature=0.1,
    )

    assert sdk_client.payload == {
        "model": "gpt-test",
        "messages": [
            {"role": "system", "content": "Answer with sources."},
            {"role": "user", "content": "Find the adapter docs."},
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "search_docs",
                    "description": "Search docs.",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"],
                    },
                },
            }
        ],
        "temperature": 0.1,
    }
    assert response.tool_calls == (
        ToolCall(
            id="call_1",
            name="search_docs",
            arguments={"query": "llm adapters"},
        ),
    )
