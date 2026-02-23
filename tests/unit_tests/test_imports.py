"""Test that all public imports work correctly."""


def test_chat_model_import() -> None:
    """ChatBrainiall can be imported from the top-level package."""
    from langchain_brainiall import ChatBrainiall  # noqa: F401

    assert ChatBrainiall is not None


def test_embeddings_import() -> None:
    """BrainiallEmbeddings can be imported from the top-level package."""
    from langchain_brainiall import BrainiallEmbeddings  # noqa: F401

    assert BrainiallEmbeddings is not None


def test_version_import() -> None:
    """Package version is accessible."""
    import langchain_brainiall

    assert hasattr(langchain_brainiall, "__version__")
    assert langchain_brainiall.__version__ == "0.1.0"


def test_all_exports() -> None:
    """__all__ lists expected exports."""
    import langchain_brainiall

    assert "ChatBrainiall" in langchain_brainiall.__all__
    assert "BrainiallEmbeddings" in langchain_brainiall.__all__
