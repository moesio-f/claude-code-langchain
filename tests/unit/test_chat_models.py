import operator
from collections.abc import AsyncIterator
from functools import reduce
from typing import Any

import pytest
from claude_agent_sdk import (
    AssistantMessage,
    ResultMessage,
    StreamEvent,
    TextBlock,
    ThinkingBlock,
    ToolUseBlock,
)
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from pydantic import BaseModel

from claude_code_langchain import ChatClaudeCode, ClaudeTool
from claude_code_langchain.chat_models import (
    TOOL_PREFIX,
    convert_messages,
    convert_to_claude_tool,
)

USAGE = {
    "input_tokens": 10,
    "output_tokens": 5,
    "cache_read_input_tokens": 2,
    "cache_creation_input_tokens": 1,
}


@tool
def get_weather(city: str) -> str:
    """Get the weather for a city."""
    return "sunny"


class Person(BaseModel):
    """A person."""

    name: str
    age: int


def assistant(*blocks: Any) -> AssistantMessage:
    return AssistantMessage(
        content=list(blocks),
        model="claude-opus-5",
        usage=dict(USAGE),
        session_id="sess-1",
    )


def result(**kwargs: Any) -> ResultMessage:
    defaults: dict[str, Any] = {
        "subtype": "success",
        "duration_ms": 12,
        "duration_api_ms": 10,
        "is_error": False,
        "num_turns": 1,
        "session_id": "sess-1",
        "usage": dict(USAGE),
        "stop_reason": "end_turn",
        "total_cost_usd": 0.01,
    }
    defaults.update(kwargs)
    return ResultMessage(**defaults)


def fake_query(messages: list[Any], captured: dict[str, Any]) -> Any:
    """Build a stand-in for ``claude_agent_sdk.query``."""

    async def _query(*, prompt: Any, options: Any, **_: Any) -> AsyncIterator[Any]:
        captured["prompt"] = prompt
        captured["options"] = options
        for message in messages:
            yield message

    return _query


@pytest.fixture
def captured(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Patch the SDK query with a canned transcript; expose what was sent."""
    state: dict[str, Any] = {"messages": [assistant(TextBlock("hi")), result()]}

    def patch() -> None:
        monkeypatch.setattr(
            "claude_code_langchain.chat_models.query",
            fake_query(state["messages"], state),
        )

    state["patch"] = patch
    patch()
    return state


# ── Message conversion ───────────────────────────────────────────────────────


def test_system_message_becomes_system_prompt() -> None:
    system, content = convert_messages([SystemMessage("be terse"), HumanMessage("hi")])
    assert system == "be terse"
    assert content == [{"type": "text", "text": "hi"}]


def test_conversation_is_flattened_into_a_transcript() -> None:
    _, content = convert_messages(
        [
            HumanMessage("hi"),
            AIMessage("", tool_calls=[{"name": "f", "args": {"a": 1}, "id": "1"}]),
            ToolMessage("42", tool_call_id="1"),
        ]
    )
    text = content[0]["text"]
    assert "Human: hi" in text
    assert 'called tool f with {"a": 1}' in text
    assert "Tool result for 1:\n42" in text


@pytest.mark.parametrize(
    "block",
    [
        {"type": "image", "base64": "abc", "mime_type": "image/png"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
    ],
)
def test_images_become_anthropic_blocks(block: dict[str, Any]) -> None:
    _, content = convert_messages(
        [HumanMessage(content=[{"type": "text", "text": "what?"}, block])]
    )
    assert content[0] == {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/png", "data": "abc"},
    }
    assert content[1]["text"] == "what?"


def test_tool_schema_conversion() -> None:
    schema = convert_to_claude_tool(get_weather)
    assert schema["name"] == "get_weather"
    assert schema["input_schema"]["properties"]["city"]["type"] == "string"
    assert convert_to_claude_tool(schema) is schema


# ── Options ──────────────────────────────────────────────────────────────────


def test_builtin_tools_are_disabled_by_default(captured: dict[str, Any]) -> None:
    ChatClaudeCode().invoke("hi")
    assert captured["options"].tools == []
    assert captured["options"].max_turns == 1
    assert captured["prompt"] == "hi"


def test_builtin_tool_configuration(captured: dict[str, Any]) -> None:
    llm = ChatClaudeCode(
        builtin_tools=[ClaudeTool.READ, "Glob"],
        allowed_tools=[ClaudeTool.READ],
        disallowed_tools=[ClaudeTool.BASH],
        max_turns=5,
        session_id="abc",
    )
    llm.invoke("hi")
    options = captured["options"]
    assert options.tools == ["Read", "Glob"]
    assert options.allowed_tools == ["Read"]
    assert options.disallowed_tools == ["Bash"]
    assert options.max_turns == 5
    assert options.resume == "abc"


def test_claude_code_preset(captured: dict[str, Any]) -> None:
    ChatClaudeCode(builtin_tools="claude_code").invoke("hi")
    assert captured["options"].tools == {"type": "preset", "preset": "claude_code"}


def test_bound_tools_are_served_over_mcp(captured: dict[str, Any]) -> None:
    ChatClaudeCode(max_turns=8).bind_tools([get_weather]).invoke("hi")
    options = captured["options"]
    assert f"{TOOL_PREFIX}get_weather" in options.allowed_tools
    assert "langchain" in options.mcp_servers
    # The caller runs the tool, so the CLI must not take another turn.
    assert options.max_turns == 1


def test_tool_choice_adds_a_system_instruction(captured: dict[str, Any]) -> None:
    ChatClaudeCode().bind_tools([get_weather], tool_choice="get_weather").invoke("hi")
    assert "must call the tool `get_weather`" in captured["options"].system_prompt


# ── Responses ────────────────────────────────────────────────────────────────


def test_invoke_returns_text_usage_and_metadata(captured: dict[str, Any]) -> None:
    message = ChatClaudeCode().invoke("hi")
    assert message.content == "hi"
    assert message.usage_metadata == {
        "input_tokens": 13,
        "output_tokens": 5,
        "total_tokens": 18,
        "input_token_details": {"cache_read": 2, "cache_creation": 1},
    }
    assert message.response_metadata["session_id"] == "sess-1"
    assert message.response_metadata["total_cost_usd"] == 0.01
    assert message.response_metadata["model_name"] == "claude-opus-5"


def test_thinking_becomes_a_reasoning_block(captured: dict[str, Any]) -> None:
    captured["messages"][:] = [
        assistant(ThinkingBlock(thinking="hmm", signature="sig"), TextBlock("hi")),
        result(),
    ]
    message = ChatClaudeCode().invoke("hi")
    assert message.content == [
        {"type": "reasoning", "reasoning": "hmm"},
        {"type": "text", "text": "hi"},
    ]
    assert message.text == "hi"


def test_tool_use_blocks_become_tool_calls(captured: dict[str, Any]) -> None:
    captured["messages"][:] = [
        assistant(
            ToolUseBlock(
                id="toolu_1",
                name=f"{TOOL_PREFIX}get_weather",
                input={"city": "Paris"},
            )
        ),
        result(),
    ]
    message = ChatClaudeCode().bind_tools([get_weather]).invoke("hi")
    assert message.tool_calls == [
        {
            "name": "get_weather",
            "args": {"city": "Paris"},
            "id": "toolu_1",
            "type": "tool_call",
        }
    ]


def test_builtin_tool_use_is_not_reported_as_a_tool_call(
    captured: dict[str, Any],
) -> None:
    captured["messages"][:] = [
        assistant(ToolUseBlock(id="t1", name="Read", input={"file_path": "x"})),
        assistant(TextBlock("done")),
        result(),
    ]
    message = ChatClaudeCode(builtin_tools=[ClaudeTool.READ]).invoke("hi")
    assert message.tool_calls == []
    assert message.content == "done"


async def test_ainvoke(captured: dict[str, Any]) -> None:
    message = await ChatClaudeCode().ainvoke("hi")
    assert message.content == "hi"


# ── Streaming ────────────────────────────────────────────────────────────────


def _delta(text: str) -> StreamEvent:
    return StreamEvent(
        uuid="u",
        session_id="sess-1",
        event={
            "type": "content_block_delta",
            "delta": {"type": "text_delta", "text": text},
        },
    )


def test_stream_yields_text_then_usage(captured: dict[str, Any]) -> None:
    captured["messages"][:] = [_delta("he"), _delta("llo"), result()]
    chunks = list(ChatClaudeCode().stream("hi"))
    merged = reduce(operator.add, chunks)
    assert merged.text == "hello"
    assert merged.usage_metadata["output_tokens"] == 5
    assert captured["options"].include_partial_messages is True


async def test_astream(captured: dict[str, Any]) -> None:
    captured["messages"][:] = [_delta("hi"), result()]
    chunks = [chunk async for chunk in ChatClaudeCode().astream("hi")]
    assert "".join(chunk.text for chunk in chunks) == "hi"


def test_stream_with_tools_emits_a_single_chunk(captured: dict[str, Any]) -> None:
    captured["messages"][:] = [
        assistant(
            ToolUseBlock(
                id="toolu_1",
                name=f"{TOOL_PREFIX}get_weather",
                input={"city": "Paris"},
            )
        ),
        result(),
    ]
    chunks = list(ChatClaudeCode().bind_tools([get_weather]).stream("hi"))
    merged = reduce(operator.add, chunks)
    assert merged.tool_calls[0]["name"] == "get_weather"


# ── Structured output ────────────────────────────────────────────────────────


def test_structured_output_uses_the_cli_json_schema(captured: dict[str, Any]) -> None:
    captured["messages"][:] = [
        assistant(TextBlock('{"name": "Bob", "age": 42}')),
        result(structured_output={"name": "Bob", "age": 42}),
    ]
    person = ChatClaudeCode().with_structured_output(Person).invoke("Bob is 42.")
    assert person == Person(name="Bob", age=42)
    output_format = captured["options"].output_format
    assert output_format["type"] == "json_schema"
    assert output_format["schema"]["properties"]["age"]["type"] == "integer"


def test_structured_output_falls_back_to_parsing_the_text(
    captured: dict[str, Any],
) -> None:
    captured["messages"][:] = [
        assistant(TextBlock('{"name": "Bob", "age": 42}')),
        result(),
    ]
    data = (
        ChatClaudeCode()
        .with_structured_output(
            {"type": "object", "properties": {"name": {"type": "string"}}}
        )
        .invoke("Bob is 42.")
    )
    assert data == {"name": "Bob", "age": 42}


def test_structured_output_include_raw(captured: dict[str, Any]) -> None:
    captured["messages"][:] = [
        assistant(TextBlock("not json")),
        result(),
    ]
    output = (
        ChatClaudeCode()
        .with_structured_output(Person, include_raw=True)
        .invoke("Bob is 42.")
    )
    assert output["parsed"] is None
    assert output["parsing_error"] is not None
    assert isinstance(output["raw"], AIMessage)
