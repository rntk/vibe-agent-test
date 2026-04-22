from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

from openai.types.chat import ChatCompletion, ChatCompletionMessageFunctionToolCall
from openai.types.responses import (
    Response,
    ResponseCustomToolCall,
    ResponseFunctionToolCall,
    ResponseOutputMessage,
    ResponseOutputText,
    ResponseReasoningItem,
)

from cagent.llm import (
    LLMClient,
    LLMMessage,
    LLMRequest,
    LLMResponse,
    ToolCall,
    ToolDefinition,
)
from cagent.llm.anthropic import AnthropicClient
from cagent.llm.llamacpp import LLamaCPP
from cagent.llm.openai import OpenAIChatCompletionsClient, OpenAIResponsesClient
from cagent.tracing import Trace, reset_trace, set_trace


class FakeClient(LLMClient):
    def __init__(self) -> None:
        self.request: LLMRequest | None = None

    def _complete(self, request: LLMRequest) -> LLMResponse:
        self.request = request
        return LLMResponse(content="ok")


def _make_chat_completion_tool_call(
    id: str,
    name: str,
    arguments: str,
) -> ChatCompletionMessageFunctionToolCall:
    tc = MagicMock(spec=ChatCompletionMessageFunctionToolCall)
    tc.type = "function"
    tc.id = id
    tc.function = MagicMock()
    tc.function.name = name
    tc.function.arguments = arguments
    return tc


def _make_chat_completion(
    content: str | None = None,
    tool_calls: list[ChatCompletionMessageFunctionToolCall] | None = None,
    reasoning_content: str | None = None,
) -> ChatCompletion:
    msg = MagicMock()
    msg.content = content
    msg.tool_calls = tool_calls
    msg.reasoning_content = reasoning_content
    choice = MagicMock()
    choice.message = msg
    completion = MagicMock(spec=ChatCompletion)
    completion.choices = [choice]
    return completion


class FakeOpenAIClient:
    def __init__(self, completion: ChatCompletion | None = None) -> None:
        self.payload: dict[str, Any] | None = None
        self.completion = completion
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self.create_completion)
        )

    def create_completion(self, **payload: Any) -> Any:
        self.payload = payload
        completion = self.completion or _make_chat_completion(
            tool_calls=[
                _make_chat_completion_tool_call(
                    id="call_1",
                    name="search_docs",
                    arguments='{"query": "llm adapters"}',
                )
            ],
        )
        return completion


class FakeAnthropicMessages:
    def __init__(self) -> None:
        self.payload: dict[str, Any] | None = None

    def create(self, **payload: Any) -> Any:
        self.payload = payload
        return SimpleNamespace(content=[SimpleNamespace(type="text", text="{}")])


class FakeAnthropicClient:
    def __init__(self) -> None:
        self.messages = FakeAnthropicMessages()


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
        completion=_make_chat_completion(content="plain answer"),
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
    tc = _make_chat_completion_tool_call(
        id="call_1",
        name="search_docs",
        arguments='{"query": "llm adapters"}',
    )
    response = OpenAIChatCompletionsClient.from_provider_response(
        _make_chat_completion(tool_calls=[tc])
    )

    assert response.tool_calls == (
        ToolCall(
            id="call_1",
            name="search_docs",
            arguments={"query": "llm adapters"},
        ),
    )


def test_openai_adapter_parses_flat_tool_calls() -> None:
    tc = MagicMock(spec=ChatCompletionMessageFunctionToolCall)
    tc.type = "function"
    tc.id = "call_1"
    tc.function = MagicMock()
    tc.function.name = "search_docs"
    tc.function.arguments = None

    tool_calls = OpenAIChatCompletionsClient.from_provider_tool_calls([tc])

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


def test_anthropic_adapter_serializes_structured_outputs_and_strict_tools() -> None:
    sdk_client = FakeAnthropicClient()
    client = AnthropicClient(
        client=sdk_client,
        model="claude-test",
        max_tokens=1024,
    )
    tool = ToolDefinition(
        name="search_docs",
        description="Search docs.",
        parameters={
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
            "additionalProperties": False,
        },
        strict=True,
    )
    output_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "summary": {"type": "string"},
            "confidence": {"type": "number", "minimum": 0},
            "next_steps": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["summary", "confidence", "next_steps"],
        "additionalProperties": False,
    }

    response = client.complete(
        "Find the adapter docs.",
        system_prompt="Answer with sources.",
        tools=[tool],
        output_schema=output_schema,
    )

    assert sdk_client.messages.payload == {
        "model": "claude-test",
        "max_tokens": 1024,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Find the adapter docs."},
                ],
            }
        ],
        "system": "Answer with sources.",
        "tools": [
            {
                "name": "search_docs",
                "description": "Search docs.",
                "input_schema": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                    "additionalProperties": False,
                },
                "strict": True,
            }
        ],
        "output_config": {
            "format": {
                "type": "json_schema",
                "schema": {
                    "type": "object",
                    "properties": {
                        "summary": {"type": "string"},
                        "confidence": {
                            "type": "number",
                            "description": "{minimum: 0}",
                        },
                        "next_steps": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    },
                    "additionalProperties": False,
                    "required": ["summary", "confidence", "next_steps"],
                },
            }
        },
    }
    assert response.content == "{}"


def test_anthropic_adapter_serializes_tool_results_as_user_blocks() -> None:
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

    provider_messages = [
        AnthropicClient.to_provider_message(message) for message in messages
    ]

    assert provider_messages == [
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "call_1",
                    "name": "search_docs",
                    "input": {"query": "llm adapters"},
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "call_1",
                    "content": "Adapter docs result",
                }
            ],
        },
    ]


def _make_responses_function_tool_call(
    call_id: str,
    name: str,
    arguments: str,
) -> ResponseFunctionToolCall:
    tc = MagicMock(spec=ResponseFunctionToolCall)
    tc.call_id = call_id
    tc.name = name
    tc.arguments = arguments
    return tc


def _make_responses_custom_tool_call(
    call_id: str,
    name: str,
    input_text: str,
) -> ResponseCustomToolCall:
    tc = MagicMock(spec=ResponseCustomToolCall)
    tc.call_id = call_id
    tc.name = name
    tc.input = input_text
    return tc


def _make_responses_output_text(text: str) -> ResponseOutputText:
    ot = MagicMock(spec=ResponseOutputText)
    ot.text = text
    return ot


def _make_responses_reasoning_item(text: str) -> ResponseReasoningItem:
    summary = MagicMock()
    summary.text = text
    ri = MagicMock(spec=ResponseReasoningItem)
    ri.summary = [summary]
    return ri


def _make_responses_output_message(
    content: list[Any],
) -> ResponseOutputMessage:
    msg = MagicMock(spec=ResponseOutputMessage)
    msg.content = content
    return msg


def _make_response(output: list[Any]) -> Response:
    r = MagicMock(spec=Response)
    r.output = output
    return r


class FakeOpenAIResponsesClient:
    def __init__(self, response: Response | None = None) -> None:
        self.payload: dict[str, Any] | None = None
        self.response = response
        self.responses = SimpleNamespace(create=self.create_response)

    def create_response(self, **payload: Any) -> Any:
        self.payload = payload
        response = self.response or _make_response(
            output=[
                _make_responses_function_tool_call(
                    call_id="call_1",
                    name="search_docs",
                    arguments='{"query": "llm adapters"}',
                )
            ]
        )
        return response


def test_openai_chat_completions_passes_strict_in_tools() -> None:
    sdk_client = FakeOpenAIClient(
        completion=_make_chat_completion(content="ok"),
    )
    client = OpenAIChatCompletionsClient(
        client=sdk_client,
        default_model="gpt-test",
    )
    tool = ToolDefinition(
        name="get_weather",
        description="Get weather.",
        parameters={"type": "object"},
        strict=True,
    )

    client.complete("What's the weather?", tools=[tool])

    assert sdk_client.payload is not None
    assert sdk_client.payload["tools"][0]["function"]["strict"] is True


def test_openai_chat_completions_passes_tool_choice_and_parallel_tool_calls() -> None:
    sdk_client = FakeOpenAIClient(
        completion=_make_chat_completion(content="ok"),
    )
    client = OpenAIChatCompletionsClient(
        client=sdk_client,
        default_model="gpt-test",
    )

    client.complete(
        "What's the weather?",
        tool_choice="required",
        parallel_tool_calls=False,
    )

    assert sdk_client.payload is not None
    assert sdk_client.payload["tool_choice"] == "required"
    assert sdk_client.payload["parallel_tool_calls"] is False


def test_openai_chat_completions_extracts_reasoning() -> None:
    response = OpenAIChatCompletionsClient.from_provider_response(
        _make_chat_completion(
            content="plain answer",
            reasoning_content="Step by step.",
        )
    )

    assert response.reasoning == "Step by step."
    assert response.content == "plain answer"


def test_openai_responses_transforms_text_response() -> None:
    sdk_client = FakeOpenAIResponsesClient(
        response=_make_response(
            output=[
                _make_responses_output_message(
                    content=[
                        _make_responses_output_text("plain answer"),
                    ]
                )
            ]
        )
    )
    client = OpenAIResponsesClient(
        client=sdk_client,
        default_model="gpt-test",
    )

    response = client.complete("Say hi.")

    assert response.content == "plain answer"
    assert response.tool_calls == ()
    assert response.reasoning is None


def test_openai_responses_parses_function_call() -> None:
    sdk_client = FakeOpenAIResponsesClient(
        response=_make_response(
            output=[
                _make_responses_function_tool_call(
                    call_id="call_1",
                    name="search_docs",
                    arguments='{"query": "llm adapters"}',
                )
            ]
        )
    )
    client = OpenAIResponsesClient(
        client=sdk_client,
        default_model="gpt-test",
    )

    response = client.complete("Find docs.")

    assert response.content is None
    assert response.tool_calls == (
        ToolCall(
            id="call_1",
            name="search_docs",
            arguments={"query": "llm adapters"},
        ),
    )


def test_openai_responses_parses_custom_tool_call() -> None:
    sdk_client = FakeOpenAIResponsesClient(
        response=_make_response(
            output=[
                _make_responses_custom_tool_call(
                    call_id="call_2",
                    name="code_exec",
                    input_text='print("hello world")',
                )
            ]
        )
    )
    client = OpenAIResponsesClient(
        client=sdk_client,
        default_model="gpt-test",
    )

    response = client.complete("Run code.")

    assert response.tool_calls == (
        ToolCall(
            id="call_2",
            name="code_exec",
            arguments={"input": 'print("hello world")'},
        ),
    )


def test_openai_responses_extracts_reasoning() -> None:
    sdk_client = FakeOpenAIResponsesClient(
        response=_make_response(
            output=[
                _make_responses_reasoning_item("Thinking..."),
                _make_responses_output_message(
                    content=[_make_responses_output_text("done")]
                ),
            ]
        )
    )
    client = OpenAIResponsesClient(
        client=sdk_client,
        default_model="gpt-test",
    )

    response = client.complete("Think.")

    assert response.reasoning == "Thinking..."
    assert response.content == "done"


def test_openai_responses_serializes_conversation_history() -> None:
    sdk_client = FakeOpenAIResponsesClient(
        response=_make_response(
            output=[
                _make_responses_output_message(
                    content=[_make_responses_output_text("ok")]
                )
            ]
        )
    )
    client = OpenAIResponsesClient(
        client=sdk_client,
        default_model="gpt-test",
    )
    messages = [
        LLMMessage(role="user", content="Hello"),
        LLMMessage(
            role="assistant",
            content="Let me search.",
            reasoning="I need to search.",
            tool_calls=[
                ToolCall(
                    id="call_1",
                    name="search_docs",
                    arguments={"query": "test"},
                )
            ],
        ),
        LLMMessage(
            role="tool",
            content="results",
            tool_call_id="call_1",
        ),
    ]

    client.complete("Continue.", messages=messages)

    assert sdk_client.payload is not None
    assert sdk_client.payload["input"] == [
        {"role": "user", "content": "Hello"},
        {
            "role": "assistant",
            "output": [
                {
                    "type": "reasoning",
                    "summary": [
                        {"type": "summary_text", "text": "I need to search."}
                    ],
                },
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "search_docs",
                    "arguments": '{"query": "test"}',
                },
                {"type": "output_text", "text": "Let me search."},
            ],
        },
        {
            "type": "function_call_output",
            "call_id": "call_1",
            "output": "results",
        },
        {"role": "user", "content": "Continue."},
    ]


def test_openai_responses_passes_tool_choice_and_parallel_tool_calls() -> None:
    sdk_client = FakeOpenAIResponsesClient(
        response=_make_response(
            output=[
                _make_responses_output_message(
                    content=[_make_responses_output_text("ok")]
                )
            ]
        )
    )
    client = OpenAIResponsesClient(
        client=sdk_client,
        default_model="gpt-test",
    )
    tool = ToolDefinition(
        name="get_weather",
        description="Get weather.",
        parameters={"type": "object"},
        strict=True,
    )

    client.complete(
        "What's the weather?",
        tools=[tool],
        tool_choice={"type": "function", "name": "get_weather"},
        parallel_tool_calls=False,
    )

    assert sdk_client.payload is not None
    assert sdk_client.payload["tools"][0]["strict"] is True
    assert sdk_client.payload["tool_choice"] == {
        "type": "function",
        "name": "get_weather",
    }
    assert sdk_client.payload["parallel_tool_calls"] is False


def test_openai_responses_uses_default_model() -> None:
    sdk_client = FakeOpenAIResponsesClient()
    client = OpenAIResponsesClient(
        client=sdk_client,
        default_model="default-model",
    )

    client.complete("Hi.")

    assert sdk_client.payload is not None
    assert sdk_client.payload["model"] == "default-model"


def test_openai_responses_overrides_model() -> None:
    sdk_client = FakeOpenAIResponsesClient()
    client = OpenAIResponsesClient(
        client=sdk_client,
        default_model="default-model",
    )

    client.complete("Hi.", model="override-model")

    assert sdk_client.payload is not None
    assert sdk_client.payload["model"] == "override-model"


def test_openai_chat_completions_creates_trace_span() -> None:
    trace = Trace()
    token = set_trace(trace)
    sdk_client = FakeOpenAIClient(
        completion=_make_chat_completion(content="traced answer"),
    )
    client = OpenAIChatCompletionsClient(
        client=sdk_client,
        default_model="gpt-test",
    )
    try:
        response = client.complete("Trace me.", system_prompt="You are helpful.")
    finally:
        reset_trace(token)

    assert response.content == "traced answer"
    assert len(trace.roots) == 1
    span = trace.roots[0]
    assert span.name == "llm.complete"
    assert span.attributes["request"]["user_prompt"] == "Trace me."
    assert span.attributes["response_content"] == "traced answer"
    assert span.attributes["message_count"] == 2


def test_openai_chat_completions_reasoning_in_trace() -> None:
    trace = Trace()
    token = set_trace(trace)
    sdk_client = FakeOpenAIClient(
        completion=_make_chat_completion(
            content="answer",
            reasoning_content="Thinking step by step.",
        ),
    )
    client = OpenAIChatCompletionsClient(
        client=sdk_client,
        default_model="gpt-test",
    )
    try:
        response = client.complete("Solve this.", reasoning_effort="high")
    finally:
        reset_trace(token)

    assert response.reasoning == "Thinking step by step."
    assert sdk_client.payload is not None
    assert sdk_client.payload["reasoning_effort"] == "high"
    assert len(trace.roots) == 1
    span = trace.roots[0]
    assert span.attributes["response_reasoning"] == "Thinking step by step."


def test_openai_responses_creates_trace_span() -> None:
    trace = Trace()
    token = set_trace(trace)
    sdk_client = FakeOpenAIResponsesClient(
        response=_make_response(
            output=[
                _make_responses_output_message(
                    content=[_make_responses_output_text("traced response")]
                )
            ]
        )
    )
    client = OpenAIResponsesClient(
        client=sdk_client,
        default_model="gpt-test",
    )
    try:
        response = client.complete("Trace me too.")
    finally:
        reset_trace(token)

    assert response.content == "traced response"
    assert len(trace.roots) == 1
    span = trace.roots[0]
    assert span.name == "llm.complete"
    assert span.attributes["request"]["user_prompt"] == "Trace me too."
    assert span.attributes["response_content"] == "traced response"


def test_openai_responses_reasoning_in_trace() -> None:
    trace = Trace()
    token = set_trace(trace)
    sdk_client = FakeOpenAIResponsesClient(
        response=_make_response(
            output=[
                _make_responses_reasoning_item("Thinking..."),
                _make_responses_output_message(
                    content=[_make_responses_output_text("done")]
                ),
            ]
        )
    )
    client = OpenAIResponsesClient(
        client=sdk_client,
        default_model="gpt-test",
    )
    try:
        response = client.complete("Think.", reasoning_effort="medium")
    finally:
        reset_trace(token)

    assert response.reasoning == "Thinking..."
    assert sdk_client.payload is not None
    assert sdk_client.payload["reasoning"] == {"effort": "medium"}
    assert len(trace.roots) == 1
    span = trace.roots[0]
    assert span.attributes["response_reasoning"] == "Thinking..."
