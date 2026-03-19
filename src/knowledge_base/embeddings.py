"""Local Sentence Transformers embeddings through LangChain."""

from langchain_core.embeddings import Embeddings

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def create_embeddings() -> Embeddings:
    """Load the CPU model on demand; the first use downloads model weights."""
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name=MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
