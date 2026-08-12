"""
Filter Chain (Tier 1) — LangChain LCEL equivalent of backend/agents/filterAgent.js

Uses the SAME system prompt and JSON contract as the original, so the two
implementations are directly comparable.
"""

import json
import logging
from typing import List

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from app.config import settings
from app.services.llm import get_llm

logger = logging.getLogger("sentinel.filter_chain")

SYSTEM_PROMPT = """You are a precision document filter for a RAG system called Sentinel.AI.
Your job is to review a set of retrieved documents and a user query, then select ONLY the documents that are:
1. Directly relevant to answering the user's query.
2. Factually safe (no misleading, outdated, or harmful content).
3. Non-redundant — remove near-duplicate passages.

You MUST respond in valid JSON with this exact structure:
{{
  "filteredDocuments": [
    {{"id": "<original doc id>", "title": "<original doc title>", "content": "<original doc content>", "relevanceReason": "<one sentence why this doc matters>"}}
  ],
  "removedCount": <integer>,
  "filterSummary": "<brief summary of what you filtered and why>"
}}

Do NOT include irrelevant, duplicate, or dangerous documents. Be strict."""

_prompt = ChatPromptTemplate.from_messages(
    [("system", SYSTEM_PROMPT), ("user", "{user_prompt}")]
)


def _build_user_prompt(user_query: str, documents: List[Document]) -> str:
    doc_blocks = "\n".join(
        f"[Doc {i + 1}]\nID: {d.metadata.get('id', f'doc-{i}')}\n"
        f"Title: {d.metadata.get('title', 'Untitled')}\nContent: {d.page_content}\n---"
        for i, d in enumerate(documents)
    )
    return (
        f'User Query: "{user_query}"\n\nRetrieved Documents:\n{doc_blocks}\n\n'
        "Filter these documents. Return only the relevant ones in the required JSON format."
    )


async def run_filter_chain(user_query: str, documents: List[Document]) -> dict:
    logger.info(f"[FilterChain] Starting. Input docs: {len(documents)}")

    if not documents:
        return {"filteredDocuments": [], "filterSummary": "No documents retrieved.", "removedCount": 0}

    llm = get_llm(settings.FILTER_AGENT_MODEL, temperature=0.1, max_tokens=2048, json_mode=True)
    chain = _prompt | llm | StrOutputParser()

    raw = await chain.ainvoke({"user_prompt": _build_user_prompt(user_query, documents)})

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        logger.error("[FilterChain] Malformed JSON — returning all docs.")
        return {
            "filteredDocuments": [
                {"id": d.metadata.get("id", ""), "title": d.metadata.get("title", ""), "content": d.page_content}
                for d in documents
            ],
            "filterSummary": "Filter chain returned malformed JSON — returning all docs.",
            "removedCount": 0,
        }

    filtered = parsed.get("filteredDocuments", [])

    # Safety net — never return zero documents (mirrors Node's fallback-to-top-2 logic)
    if not filtered:
        logger.warning("[FilterChain] All docs filtered out — falling back to top 2.")
        fallback = documents[:2]
        filtered = [
            {"id": d.metadata.get("id", ""), "title": d.metadata.get("title", ""), "content": d.page_content}
            for d in fallback
        ]
        parsed["filterSummary"] = parsed.get("filterSummary", "") + " (Fallback: kept top 2 documents.)"

    return {
        "filteredDocuments": filtered,
        "filterSummary": parsed.get("filterSummary", ""),
        "removedCount": parsed.get("removedCount", len(documents) - len(filtered)),
    }
