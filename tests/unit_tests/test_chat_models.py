"""Unit tests for ChatBrainiall.

These tests validate instantiation, defaults, and class hierarchy
without making any actual API calls.
"""

import os
from unittest.mock import patch

import pytest
from langchain_openai import ChatOpenAI

from langchain_brainiall.chat_models import (
    BRAINIALL_API_BASE,
    BRAINIALL_MODELS,
    DEFAULT_MODEL,
    ChatBrainiall,
)


class TestChatBrainiallInstantiation:
    """Test basic instantiation and defaults."""

    def test_default_model(self) -> None:
        """Default model is claude-sonnet-4-6."""
        llm = ChatBrainiall(api_key="test-key")
        assert llm.model_name == DEFAULT_MODEL
        assert llm.model_name == "claude-sonnet-4-6"

    def test_custom_model(self) -> None:
        """Custom model can be set."""
        llm = ChatBrainiall(model="deepseek-r1", api_key="test-key")
        assert llm.model_name == "deepseek-r1"

    def test_default_base_url(self) -> None:
        """Default base URL points to Brainiall gateway."""
        llm = ChatBrainiall(api_key="test-key")
        assert llm.openai_api_base == BRAINIALL_API_BASE
        assert "apim-ai-apis.azure-api.net" in llm.openai_api_base

    def test_custom_base_url(self) -> None:
        """Base URL can be overridden."""
        custom_url = "https://custom.example.com/v1"
        llm = ChatBrainiall(api_key="test-key", base_url=custom_url)
        assert llm.openai_api_base == custom_url

    def test_api_key_direct(self) -> None:
        """API key can be passed directly."""
        llm = ChatBrainiall(api_key="my-secret-key")
        assert llm.openai_api_key is not None

    def test_api_key_from_env(self) -> None:
        """API key falls back to BRAINIALL_API_KEY env var."""
        with patch.dict(os.environ, {"BRAINIALL_API_KEY": "env-key"}, clear=False):
            llm = ChatBrainiall()
            assert llm.openai_api_key is not None

    def test_temperature(self) -> None:
        """Temperature parameter is forwarded."""
        llm = ChatBrainiall(api_key="test-key", temperature=0.5)
        assert llm.temperature == 0.5

    def test_max_tokens(self) -> None:
        """Max tokens parameter is forwarded."""
        llm = ChatBrainiall(api_key="test-key", max_tokens=1024)
        assert llm.max_tokens == 1024

    def test_streaming_flag(self) -> None:
        """Streaming can be enabled."""
        llm = ChatBrainiall(api_key="test-key", streaming=True)
        assert llm.streaming is True


class TestChatBrainiallClassHierarchy:
    """Test that ChatBrainiall properly inherits from ChatOpenAI."""

    def test_is_subclass_of_chat_openai(self) -> None:
        """ChatBrainiall is a subclass of ChatOpenAI."""
        assert issubclass(ChatBrainiall, ChatOpenAI)

    def test_instance_of_chat_openai(self) -> None:
        """ChatBrainiall instances are also ChatOpenAI instances."""
        llm = ChatBrainiall(api_key="test-key")
        assert isinstance(llm, ChatOpenAI)

    def test_llm_type(self) -> None:
        """LLM type identifier is brainiall-specific."""
        llm = ChatBrainiall(api_key="test-key")
        assert llm._llm_type == "brainiall-chat"


class TestChatBrainiallHelpers:
    """Test helper class methods."""

    def test_get_available_models(self) -> None:
        """get_available_models returns a sorted list of model names."""
        models = ChatBrainiall.get_available_models()
        assert isinstance(models, list)
        assert len(models) > 0
        assert models == sorted(models)
        assert "claude-sonnet-4-6" in models
        assert "deepseek-r1" in models

    def test_get_model_info_known(self) -> None:
        """get_model_info returns info for known models."""
        info = ChatBrainiall.get_model_info("claude-opus-4-6")
        assert info is not None
        assert "context" in info
        assert "max_output" in info
        assert info["context"] == 200000
        assert info["max_output"] == 64000

    def test_get_model_info_unknown(self) -> None:
        """get_model_info returns None for unknown models."""
        info = ChatBrainiall.get_model_info("nonexistent-model")
        assert info is None

    def test_model_catalog_not_empty(self) -> None:
        """The model catalog has entries."""
        assert len(BRAINIALL_MODELS) > 20


class TestChatBrainiallEnvVars:
    """Test environment variable handling."""

    def test_brainiall_api_base_env(self) -> None:
        """BRAINIALL_API_BASE env var overrides default URL."""
        custom_url = "https://custom.brainiall.com/v1"
        with patch.dict(
            os.environ,
            {"BRAINIALL_API_BASE": custom_url, "BRAINIALL_API_KEY": "key"},
            clear=False,
        ):
            llm = ChatBrainiall()
            assert llm.openai_api_base == custom_url

    def test_explicit_base_url_beats_env(self) -> None:
        """Explicit base_url takes precedence over env var."""
        explicit_url = "https://explicit.example.com/v1"
        with patch.dict(
            os.environ,
            {
                "BRAINIALL_API_BASE": "https://env.example.com/v1",
                "BRAINIALL_API_KEY": "key",
            },
            clear=False,
        ):
            llm = ChatBrainiall(base_url=explicit_url)
            assert llm.openai_api_base == explicit_url

    def test_lc_secrets_mapping(self) -> None:
        """lc_secrets maps to BRAINIALL_API_KEY."""
        llm = ChatBrainiall(api_key="test-key")
        secrets = llm.lc_secrets
        assert "openai_api_key" in secrets
        assert secrets["openai_api_key"] == "BRAINIALL_API_KEY"
