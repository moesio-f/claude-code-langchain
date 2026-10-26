import os

import pytest
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from pydantic import BaseModel

from claude_code_langchain import ChatClaudeCode

pytestmark = pytest.mark.skipif(
    os.environ.get("CLAUDE_CODE_LANGCHAIN_INTEGRATION") != "1",
    reason="set CLAUDE_CODE_LANGCHAIN_INTEGRATION=1 to run against the claude CLI",
)

MODEL = os.environ.get("CLAUDE_CODE_LANGCHAIN_TEST_MODEL", "claude-haiku-4-5")


@tool
def get_weather(city: str) -> str:
    """Get the weather for a city."""
    return "12C and sunny"


class Person(BaseModel):
    """A person."""

    name: str
    age: int


@pytest.fixture
def llm() -> ChatClaudeCode:
    return ChatClaudeCode(model=MODEL)


def test_invoke(llm: ChatClaudeCode) -> None:
    message = llm.invoke(
        [SystemMessage("Answer in one word."), HumanMessage("Capital of France?")]
    )
    assert "paris" in str(message.text).lower()
    assert message.usage_metadata["output_tokens"] > 0
    assert message.response_metadata["session_id"]


def test_stream(llm: ChatClaudeCode) -> None:
    chunks = list(llm.stream("Count from 1 to 5, digits only."))
    assert "5" in "".join(str(chunk.text) for chunk in chunks)


async def test_ainvoke(llm: ChatClaudeCode) -> None:
    message = await llm.ainvoke("Reply with the single word: OK")
    assert "ok" in str(message.text).lower()


def test_tool_calling_round_trip(llm: ChatClaudeCode) -> None:
    bound = llm.bind_tools([get_weather])
    messages = [HumanMessage("What is the weather in Paris?")]
    ai = bound.invoke(messages)
    assert ai.tool_calls[0]["name"] == "get_weather"

    call = ai.tool_calls[0]
    messages += [
        ai,
        ToolMessage(get_weather.invoke(call["args"]), tool_call_id=call["id"]),
    ]
    final = bound.invoke(messages)
    assert "12" in str(final.text)


def test_structured_output(llm: ChatClaudeCode) -> None:
    person = llm.with_structured_output(Person).invoke("Bob is 42 years old.")
    assert person == Person(name="Bob", age=42)


def test_image_input(llm: ChatClaudeCode) -> None:
    png = (
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAE"
        "hQGAhKmMIQAAAABJRU5ErkJggg=="
    )
    message = llm.invoke(
        [
            HumanMessage(
                content=[
                    {"type": "text", "text": "Describe this image in three words."},
                    {"type": "image", "base64": png, "mime_type": "image/png"},
                ]
            )
        ]
    )
    assert str(message.text).strip()
