"""Integration tests for ChatBrainiall.

These tests make real API calls and require a valid BRAINIALL_API_KEY
environment variable. They are skipped by default.

Run with:
    BRAINIALL_API_KEY=your-key pytest tests/integration_tests/ -m integration
"""

import os

import pytest

from langchain_brainiall import ChatBrainiall

# Skip all tests in this module if no API key is set
pytestmark = pytest.mark.skipif(
    not os.environ.get("BRAINIALL_API_KEY"),
    reason="BRAINIALL_API_KEY not set",
)


@pytest.mark.integration
def test_invoke_basic() -> None:
    """Basic invoke returns a response with content."""
    llm = ChatBrainiall(model="nova-micro", temperature=0, max_tokens=50)
    response = llm.invoke("Say 'hello' and nothing else.")
    assert response.content is not None
    assert len(response.content) > 0


@pytest.mark.integration
def test_invoke_with_system_message() -> None:
    """Invoke with system + human messages."""
    llm = ChatBrainiall(model="nova-micro", temperature=0, max_tokens=50)
    messages = [
        ("system", "You are a helpful assistant. Be very brief."),
        ("human", "What is 2+2?"),
    ]
    response = llm.invoke(messages)
    assert "4" in response.content


@pytest.mark.integration
def test_streaming() -> None:
    """Streaming returns chunks with content."""
    llm = ChatBrainiall(model="nova-micro", temperature=0, max_tokens=50)
    chunks = list(llm.stream("Say 'hello world'"))
    assert len(chunks) > 0
    full_content = "".join(c.content for c in chunks if c.content)
    assert len(full_content) > 0


@pytest.mark.integration
async def test_ainvoke() -> None:
    """Async invoke returns a response."""
    llm = ChatBrainiall(model="nova-micro", temperature=0, max_tokens=50)
    response = await llm.ainvoke("Say 'test'")
    assert response.content is not None
    assert len(response.content) > 0


@pytest.mark.integration
def test_structured_output() -> None:
    """Structured output with Pydantic model."""
    from pydantic import BaseModel

    class MathAnswer(BaseModel):
        result: int

    llm = ChatBrainiall(model="claude-haiku-4-5", temperature=0)
    structured = llm.with_structured_output(MathAnswer)
    answer = structured.invoke("What is 10 + 5?")
    assert isinstance(answer, MathAnswer)
    assert answer.result == 15


@pytest.mark.integration
def test_tool_calling() -> None:
    """Tool calling with a simple function schema."""
    from pydantic import BaseModel, Field

    class GetWeather(BaseModel):
        """Get the weather for a location."""

        location: str = Field(description="City name")

    llm = ChatBrainiall(model="claude-haiku-4-5", temperature=0)
    llm_with_tools = llm.bind_tools([GetWeather])
    response = llm_with_tools.invoke("What's the weather in Paris?")
    assert response.tool_calls is not None
    assert len(response.tool_calls) > 0
    assert response.tool_calls[0]["name"] == "GetWeather"


@pytest.mark.integration
def test_multiple_models() -> None:
    """Different models can be used with the same class."""
    for model in ["nova-micro", "claude-haiku-4-5"]:
        llm = ChatBrainiall(model=model, temperature=0, max_tokens=20)
        response = llm.invoke("Say 'ok'")
        assert response.content is not None
