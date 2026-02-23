"""Brainiall LLM Gateway embeddings integration for LangChain.

Thin wrapper around OpenAIEmbeddings that pre-configures the Brainiall
endpoint for embedding model access.
"""

from __future__ import annotations

import os
from typing import Any, Dict

from langchain_openai import OpenAIEmbeddings
from pydantic import model_validator

BRAINIALL_API_BASE = "https://apim-ai-apis.azure-api.net/v1"
DEFAULT_EMBEDDING_MODEL = "bge-m3"

# fmt: off
BRAINIALL_EMBEDDING_MODELS: Dict[str, Dict[str, Any]] = {
    "bge-m3":               {"dimensions": 1024, "max_tokens": 8192},
    "bge-large-en-v1.5":    {"dimensions": 1024, "max_tokens": 512},
    "cohere-embed-v3":      {"dimensions": 1024, "max_tokens": 512},
    "titan-embed-v2":       {"dimensions": 1024, "max_tokens": 8192},
}
# fmt: on


class BrainiallEmbeddings(OpenAIEmbeddings):
    """Embeddings model for the Brainiall LLM Gateway.

    Brainiall provides access to multiple embedding models through a
    unified OpenAI-compatible API.

    Setup:
        .. code-block:: bash

            pip install langchain-brainiall
            export BRAINIALL_API_KEY="your-api-key"

    Instantiate:
        .. code-block:: python

            from langchain_brainiall import BrainiallEmbeddings

            embeddings = BrainiallEmbeddings(
                model="bge-m3",
                # api_key="...",  # or set BRAINIALL_API_KEY env var
            )

    Embed single text:
        .. code-block:: python

            vector = embeddings.embed_query("Hello world")
            print(len(vector))

    Embed multiple texts:
        .. code-block:: python

            vectors = embeddings.embed_documents(["Hello", "World"])
            print(len(vectors), len(vectors[0]))
    """

    @model_validator(mode="before")
    @classmethod
    def set_brainiall_defaults(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Set Brainiall-specific defaults before standard OpenAI validation.

        Resolves the API key from BRAINIALL_API_KEY env var, sets the default
        base URL, and sets the default embedding model.
        """
        # Set default model if not provided
        if not values.get("model"):
            values["model"] = DEFAULT_EMBEDDING_MODEL

        # Also set deployment to match model (OpenAIEmbeddings uses it)
        if not values.get("deployment"):
            values["deployment"] = values.get("model", DEFAULT_EMBEDDING_MODEL)

        # Resolve API key: explicit > BRAINIALL_API_KEY > OPENAI_API_KEY
        api_key = (
            values.get("api_key")
            or values.get("openai_api_key")
            or os.environ.get("BRAINIALL_API_KEY")
        )
        if api_key:
            values["openai_api_key"] = api_key
            # Remove api_key to avoid alias conflict
            values.pop("api_key", None)

        # Set default base URL if not provided
        base_url = (
            values.get("base_url")
            or values.get("openai_api_base")
            or os.environ.get("BRAINIALL_API_BASE")
        )
        if not base_url:
            base_url = BRAINIALL_API_BASE
        values["openai_api_base"] = base_url
        # Remove base_url to avoid alias conflict
        values.pop("base_url", None)

        return values

    @classmethod
    def get_available_models(cls) -> list[str]:
        """Return list of available embedding model names."""
        return sorted(BRAINIALL_EMBEDDING_MODELS.keys())
