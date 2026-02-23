"""Unit tests for BrainiallEmbeddings.

Validates instantiation and defaults without making API calls.
"""

import os
from unittest.mock import patch

from langchain_openai import OpenAIEmbeddings

from langchain_brainiall.embeddings import (
    BRAINIALL_API_BASE,
    BRAINIALL_EMBEDDING_MODELS,
    DEFAULT_EMBEDDING_MODEL,
    BrainiallEmbeddings,
)


class TestBrainiallEmbeddingsInstantiation:
    """Test basic instantiation and defaults."""

    def test_default_model(self) -> None:
        """Default embedding model is bge-m3."""
        emb = BrainiallEmbeddings(api_key="test-key")
        assert emb.model == DEFAULT_EMBEDDING_MODEL
        assert emb.model == "bge-m3"

    def test_custom_model(self) -> None:
        """Custom embedding model can be set."""
        emb = BrainiallEmbeddings(model="cohere-embed-v3", api_key="test-key")
        assert emb.model == "cohere-embed-v3"

    def test_default_base_url(self) -> None:
        """Default base URL points to Brainiall gateway."""
        emb = BrainiallEmbeddings(api_key="test-key")
        assert emb.openai_api_base == BRAINIALL_API_BASE

    def test_api_key_from_env(self) -> None:
        """API key falls back to BRAINIALL_API_KEY env var."""
        with patch.dict(os.environ, {"BRAINIALL_API_KEY": "env-key"}, clear=False):
            emb = BrainiallEmbeddings()
            assert emb.openai_api_key is not None


class TestBrainiallEmbeddingsClassHierarchy:
    """Test class hierarchy."""

    def test_is_subclass_of_openai_embeddings(self) -> None:
        """BrainiallEmbeddings is a subclass of OpenAIEmbeddings."""
        assert issubclass(BrainiallEmbeddings, OpenAIEmbeddings)

    def test_instance_of_openai_embeddings(self) -> None:
        """BrainiallEmbeddings instances are also OpenAIEmbeddings instances."""
        emb = BrainiallEmbeddings(api_key="test-key")
        assert isinstance(emb, OpenAIEmbeddings)


class TestBrainiallEmbeddingsHelpers:
    """Test helper methods."""

    def test_get_available_models(self) -> None:
        """get_available_models returns sorted list."""
        models = BrainiallEmbeddings.get_available_models()
        assert isinstance(models, list)
        assert len(models) > 0
        assert models == sorted(models)
        assert "bge-m3" in models

    def test_model_catalog_not_empty(self) -> None:
        """The embedding model catalog has entries."""
        assert len(BRAINIALL_EMBEDDING_MODELS) > 0
