"""Brainiall LLM Gateway chat model integration for LangChain.

Thin wrapper around ChatOpenAI that pre-configures the Brainiall endpoint,
giving access to 113+ models from 17 providers via a single API.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional, Union

from langchain_openai import ChatOpenAI
from pydantic import Field, SecretStr, model_validator
from typing_extensions import Self

BRAINIALL_API_BASE = "https://apim-ai-apis.azure-api.net/v1"
DEFAULT_MODEL = "claude-sonnet-4-6"

# fmt: off
BRAINIALL_MODELS: Dict[str, Dict[str, Any]] = {
    # Anthropic Claude
    "claude-opus-4-6":          {"context": 200000, "max_output": 64000},
    "claude-opus-4-6-1m":       {"context": 1000000, "max_output": 64000},
    "claude-opus-4-5":          {"context": 200000, "max_output": 32000},
    "claude-sonnet-4-6":        {"context": 200000, "max_output": 64000},
    "claude-sonnet-4-6-1m":     {"context": 1000000, "max_output": 64000},
    "claude-haiku-4-5":         {"context": 200000, "max_output": 16000},
    "claude-3-opus":            {"context": 200000, "max_output": 4096},
    # DeepSeek
    "deepseek-r1":              {"context": 128000, "max_output": 64000},
    "deepseek-v3":              {"context": 128000, "max_output": 16000},
    # Meta Llama
    "llama-3.3-70b":            {"context": 128000, "max_output": 4096},
    "llama-4-scout-17b":        {"context": 1048576, "max_output": 16384},
    "llama-4-maverick-17b":     {"context": 1048576, "max_output": 16384},
    # Qwen
    "qwen-3-235b":              {"context": 128000, "max_output": 16000},
    "qwen-3-32b":               {"context": 128000, "max_output": 16000},
    "qwen-3-8b":                {"context": 128000, "max_output": 16000},
    "qwen-3-80b":               {"context": 128000, "max_output": 16000},
    # Mistral
    "mistral-large-3":          {"context": 128000, "max_output": 16000},
    "mistral-small-3":          {"context": 128000, "max_output": 16000},
    # Amazon Nova
    "nova-pro":                 {"context": 300000, "max_output": 5120},
    "nova-lite":                {"context": 300000, "max_output": 5120},
    "nova-micro":               {"context": 128000, "max_output": 5120},
    # MiniMax
    "minimax-m2":               {"context": 1000000, "max_output": 128000},
    # Others
    "nemotron-ultra-253b":      {"context": 128000, "max_output": 16000},
    "kimi-k2.5":                {"context": 131072, "max_output": 16384},
}
# fmt: on


class ChatBrainiall(ChatOpenAI):
    """Chat model for the Brainiall LLM Gateway.

    Brainiall provides a unified OpenAI-compatible API for 113+ models from
    17 providers (Anthropic, DeepSeek, Meta, Qwen, Mistral, Amazon, and more),
    powered by AWS Bedrock with built-in cost optimization.

    This class is a thin wrapper around ``ChatOpenAI`` that sets the correct
    base URL and default model. All ``ChatOpenAI`` features are supported:
    streaming, tool calling, structured output, multi-modal input, etc.

    Setup:
        Install the package and set your API key:

        .. code-block:: bash

            pip install langchain-brainiall
            export BRAINIALL_API_KEY="your-api-key"

        Get an API key at https://brainiall.com

    Key init args - completion params:
        model: str
            Name of the model to use. Defaults to ``"claude-sonnet-4-6"``.
        temperature: Optional[float]
            Sampling temperature (0-2).
        max_tokens: Optional[int]
            Max number of tokens to generate.

    Key init args - client params:
        api_key: Optional[SecretStr]
            Brainiall API key. If not provided, reads from
            ``BRAINIALL_API_KEY`` env var.
        base_url: Optional[str]
            Base URL for the API. Defaults to the Brainiall gateway.
        max_retries: int
            Max retries on failure. Defaults to 2.
        timeout: Optional[float]
            Request timeout in seconds.

    Instantiate:
        .. code-block:: python

            from langchain_brainiall import ChatBrainiall

            llm = ChatBrainiall(
                model="claude-sonnet-4-6",
                temperature=0,
                # api_key="...",  # or set BRAINIALL_API_KEY env var
            )

    Invoke:
        .. code-block:: python

            messages = [
                ("system", "You are a helpful assistant."),
                ("human", "Explain quantum computing in one sentence."),
            ]
            response = llm.invoke(messages)
            print(response.content)

    Stream:
        .. code-block:: python

            for chunk in llm.stream("Tell me a joke"):
                print(chunk.content, end="", flush=True)

    Tool calling:
        .. code-block:: python

            from pydantic import BaseModel

            class GetWeather(BaseModel):
                location: str

            llm_with_tools = llm.bind_tools([GetWeather])
            response = llm_with_tools.invoke("What's the weather in Tokyo?")

    Structured output:
        .. code-block:: python

            from pydantic import BaseModel

            class Answer(BaseModel):
                answer: str
                confidence: float

            structured_llm = llm.with_structured_output(Answer)
            result = structured_llm.invoke("What is 2+2?")

    Multi-model chains:
        .. code-block:: python

            fast = ChatBrainiall(model="nova-micro", temperature=0)
            smart = ChatBrainiall(model="claude-opus-4-6", temperature=0)

            draft = fast.invoke("Draft an email about the meeting")
            final = smart.invoke(f"Improve this email: {draft.content}")
    """

    model_name: str = Field(default=DEFAULT_MODEL, alias="model")
    """Model name to use. See BRAINIALL_MODELS for available models."""

    openai_api_base: Optional[str] = Field(
        default=None,
        alias="base_url",
    )
    """Base URL for the Brainiall API."""

    openai_api_key: Optional[Union[SecretStr, str]] = Field(
        default=None,
        alias="api_key",
    )
    """API key for authentication. Falls back to BRAINIALL_API_KEY env var."""

    @model_validator(mode="before")
    @classmethod
    def set_brainiall_defaults(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Set Brainiall-specific defaults before standard OpenAI validation.

        Resolves the API key from BRAINIALL_API_KEY env var and sets the
        default base URL if not explicitly provided.
        """
        # Resolve API key: explicit > BRAINIALL_API_KEY > OPENAI_API_KEY
        api_key = (
            values.get("api_key")
            or values.get("openai_api_key")
            or os.environ.get("BRAINIALL_API_KEY")
        )
        if api_key:
            values["openai_api_key"] = api_key

        # Set default base URL if not provided
        base_url = (
            values.get("base_url")
            or values.get("openai_api_base")
            or os.environ.get("BRAINIALL_API_BASE")
        )
        if not base_url:
            base_url = BRAINIALL_API_BASE
        values["openai_api_base"] = base_url

        return values

    @property
    def _llm_type(self) -> str:
        """Return identifier for this LLM type."""
        return "brainiall-chat"

    @property
    def lc_secrets(self) -> dict[str, str]:
        """Map of secret attribute names to env var names."""
        return {
            "openai_api_key": "BRAINIALL_API_KEY",
        }

    @classmethod
    def get_available_models(cls) -> list[str]:
        """Return list of available model names."""
        return sorted(BRAINIALL_MODELS.keys())

    @classmethod
    def get_model_info(cls, model: str) -> Optional[Dict[str, Any]]:
        """Return context window and max output info for a model.

        Args:
            model: Model name string.

        Returns:
            Dict with 'context' and 'max_output' keys, or None if unknown.
        """
        return BRAINIALL_MODELS.get(model)
