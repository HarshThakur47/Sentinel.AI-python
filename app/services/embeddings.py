"""
Embeddings — Hugging Face sentence-transformers.

Direct replacement for `generateEmbedding()` in the Node vectorDbService.js,
which called Gemini's embedding API. Here we use a local, free,
open-source model instead — no API key required.
"""

from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings

from app.config import settings


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    """
    Cached singleton — the model is loaded into memory once and reused,
    same pattern as keeping a single genAI client alive in the Node version.
    """
    return HuggingFaceEmbeddings(
        model_name=settings.HF_EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
