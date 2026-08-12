"""
Generator Chain (Tier 2) — LangChain LCEL equivalent of backend/agents/generatorAgent.js
"""

import logging
from typing import List

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from app.config import settings
from app.services.llm import get_llm

logger = logging.getLogger("sentinel.generator_chain")

SYSTEM_PROMPT = """You are Sentinel.AI — an intelligent and helpful AI assistant.

Your rules:
1. If relevant documents are provided, prioritize them and cite inline using [Doc N] notation.
2. If documents are insufficient or not provided, use your own knowledge to give a complete answer.
3. Always provide thorough, accurate, and helpful answers — never refuse unnecessarily.
4. Clearly label answers not backed by documents with: "[General Knowledge]" at the start.
5. Be professional, clear, and detailed. Never truncate your answer.
6. Never fabricate citations — only use [Doc N] if that document was actually provided."""

_prompt = ChatPromptTemplate.from_messages(
    [("system", SYSTEM_PROMPT), ("user", "{user_prompt}")]
)


async def run_generator_chain(user_query: str, filtered_documents: List[dict]) -> str:
    logger.info(f"[GeneratorChain] Starting. Using {len(filtered_documents)} documents.")

    has_documents = len(filtered_documents) > 0
    if has_documents:
        context_block = "\n\n---\n\n".join(
            f"[Doc {i + 1}] — {d['title']}\n{d['content']}" for i, d in enumerate(filtered_documents)
        )
        user_prompt = (
            f'Context Documents:\n{context_block}\n\n---\n\nUser Query: "{user_query}"\n\n'
            "Using the above documents as your primary source, provide a comprehensive and "
            "accurate answer. Cite documents inline where relevant."
        )
    else:
        user_prompt = (
            f'User Query: "{user_query}"\n\nNo documents were found in the knowledge base for '
            "this query. Answer using your general knowledge. Be thorough and helpful."
        )

    llm = get_llm(settings.GENERATOR_AGENT_MODEL, temperature=0.4, max_tokens=4096, json_mode=False)
    chain = _prompt | llm | StrOutputParser()

    answer = await chain.ainvoke({"user_prompt": user_prompt})
    logger.info(f"[GeneratorChain] Done. Answer length: {len(answer)} chars.")
    return answer
