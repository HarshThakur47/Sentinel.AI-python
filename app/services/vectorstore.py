"""
Vector store — LangChain + Chroma (local) or Pinecone (prod).

Direct replacement for backend/services/vectorDbService.js. Same mock-document
fallback behavior is preserved so the service is demoable with zero API keys.
"""

from functools import lru_cache
from typing import List

from langchain_core.documents import Document
from langchain_chroma import Chroma

from app.config import settings
from app.services.embeddings import get_embeddings

# ─── Same 15 mock documents as the Node version's MOCK_DOCUMENTS ──────────────
# (kept short here — see backend/services/vectorDbService.js for the original
#  full-text set this was ported from)
MOCK_DOCUMENTS: List[Document] = [
    Document(
        page_content=(
            "Metformin is a first-line medication for type 2 diabetes. Common side "
            "effects include nausea, vomiting, diarrhea, and stomach pain. A rare but "
            "serious side effect is lactic acidosis. Source: WHO Essential Medicines List 2023."
        ),
        metadata={"id": "mock-001", "title": "Medical Reference: Metformin Side Effects"},
    ),
    Document(
        page_content=(
            "Docker is an open-source containerization platform that packages "
            "applications and their dependencies into lightweight, portable containers. "
            "Key components include Docker Engine, Docker Hub, Dockerfile, and Docker Compose."
        ),
        metadata={"id": "mock-003", "title": "Docker: Containerization Platform Overview"},
    ),
    Document(
        page_content=(
            "Retrieval-Augmented Generation (RAG) combines information retrieval with "
            "language generation to produce grounded responses. The pipeline consists of: "
            "(1) query embedding, (2) vector similarity search, (3) context injection, "
            "and (4) grounded response generation. RAG reduces hallucinations by anchoring "
            "answers in retrieved documents."
        ),
        metadata={"id": "mock-008", "title": "RAG Architecture: Retrieval-Augmented Generation"},
    ),
    Document(
        page_content=(
            "REST is an architectural style for designing networked applications. Key "
            "principles include statelessness, uniform interface, resource-based URLs, "
            "and standard HTTP methods. Authentication is commonly handled via JWT or OAuth 2.0."
        ),
        metadata={"id": "mock-012", "title": "REST API Design Principles"},
    ),
    Document(
        page_content=(
            "AI hallucination occurs when language models generate plausible-sounding but "
            "factually incorrect information. Mitigation strategies include RAG pipelines, "
            "multi-agent verification, confidence scoring, and source citation requirements. "
            "Studies show multi-agent systems reduce hallucination rates from 35% to under 10%."
        ),
        metadata={"id": "mock-013", "title": "AI Hallucination: Causes and Mitigation"},
    ),
]


@lru_cache(maxsize=1)
def get_vectorstore() -> Chroma:
    """
    Cached singleton Chroma store, seeded with mock docs on first run.
    Swap this function's body for a Pinecone-backed store in production
    (see commented block below) without touching any calling code.
    """
    embeddings = get_embeddings()
    store = Chroma(
        collection_name=settings.CHROMA_COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=settings.CHROMA_PERSIST_DIR,
    )

    # Seed with mock docs only if the collection is empty (first run)
    if store._collection.count() == 0:
        store.add_documents(MOCK_DOCUMENTS)

    return store

    # ── Production alternative: Pinecone ─────────────────────
    # from langchain_pinecone import PineconeVectorStore
    # return PineconeVectorStore(
    #     index_name=settings.PINECONE_INDEX_NAME,
    #     embedding=embeddings,
    #     pinecone_api_key=settings.PINECONE_API_KEY,
    # )


def get_retriever(top_k: int = 15):
    """Equivalent to retrieveDocuments(userQuery, 15) in the Node pipeline."""
    return get_vectorstore().as_retriever(search_kwargs={"k": top_k})


def upsert_documents(chunks: List[str], source_filename: str) -> None:
    """Equivalent to upsertDocuments() in vectorDbService.js — used by the
    ingestion/document-upload endpoint."""
    docs = [
        Document(
            page_content=chunk,
            metadata={"title": source_filename, "pageNumber": i + 1},
        )
        for i, chunk in enumerate(chunks)
    ]
    get_vectorstore().add_documents(docs)
