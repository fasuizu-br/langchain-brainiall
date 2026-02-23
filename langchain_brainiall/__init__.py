"""LangChain integration for Brainiall LLM Gateway.

Provides access to 113+ AI models from 17 providers (Claude, DeepSeek, Llama,
Qwen, Mistral, Nova, and more) through a single OpenAI-compatible endpoint.

Example:
    >>> from langchain_brainiall import ChatBrainiall
    >>> llm = ChatBrainiall(api_key="your-key")
    >>> llm.invoke("Hello!")
"""

from langchain_brainiall.chat_models import ChatBrainiall
from langchain_brainiall.embeddings import BrainiallEmbeddings

__all__ = [
    "ChatBrainiall",
    "BrainiallEmbeddings",
]

__version__ = "0.1.0"
