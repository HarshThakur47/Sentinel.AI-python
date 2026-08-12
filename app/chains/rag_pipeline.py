"""
RAG Pipeline Orchestrator — equivalent of backend/services/ragPipelineService.js

Wires together: Chroma+HuggingFace retrieval -> Filter chain -> Generator
chain -> Evaluator chain. Same 4-step flow and same graceful-degradation
behavior (falls back to general knowledge / raw docs / default score on
each step's failure) as the original Node pipeline.
"""

import logging
import time

from app.chains.evaluator_chain import run_evaluator_chain
from app.chains.filter_chain import run_filter_chain
from app.chains.generator_chain import run_generator_chain
from app.services.vectorstore import get_retriever

logger = logging.getLogger("sentinel.rag_pipeline")


async def run_rag_pipeline(user_query: str) -> dict:
    start = time.time()

    # ── Step 1: Retrieval ────────────────────────────────────
    try:
        retriever = get_retriever(top_k=15)
        raw_documents = await retriever.ainvoke(user_query)
    except Exception as err:  # noqa: BLE001
        logger.warning(f"[Pipeline] Retrieval failed — falling back to general knowledge. {err}")
        raw_documents = []

    has_documents = len(raw_documents) > 0

    # ── Step 2: Filter Chain ─────────────────────────────────
    filtered_documents, filter_summary, removed_count = [], "No documents retrieved — using general knowledge.", 0
    if has_documents:
        try:
            filter_result = await run_filter_chain(user_query, raw_documents)
            filtered_documents = filter_result["filteredDocuments"]
            filter_summary = filter_result["filterSummary"]
            removed_count = filter_result["removedCount"]
        except Exception as err:  # noqa: BLE001
            logger.warning(f"[Pipeline] Filter chain failed — using all raw docs. {err}")
            filtered_documents = [
                {"id": d.metadata.get("id", ""), "title": d.metadata.get("title", ""), "content": d.page_content}
                for d in raw_documents
            ]
            filter_summary = "Filter skipped."

    # ── Step 3: Generator Chain ──────────────────────────────
    try:
        generated_answer = await run_generator_chain(user_query, filtered_documents)
    except Exception as err:  # noqa: BLE001
        raise RuntimeError(f"Generator chain failed: {err}") from err

    # ── Step 4: Evaluator Chain ──────────────────────────────
    try:
        eval_result = await run_evaluator_chain(user_query, generated_answer, filtered_documents)
    except Exception as err:  # noqa: BLE001
        logger.warning(f"[Pipeline] Evaluator failed — using default score. {err}")
        eval_result = {
            "confidenceScore": 7,
            "isAccurate": True,
            "refinementNeeded": False,
            "evaluationSummary": "Evaluator unavailable.",
            "finalAnswer": generated_answer,
        }

    duration_ms = int((time.time() - start) * 1000)
    sources = [{"id": d.get("id", ""), "title": d.get("title", "")} for d in filtered_documents]

    logger.info(f"[Pipeline] Completed in {duration_ms}ms. Score: {eval_result['confidenceScore']}/10")

    return {
        "answer": eval_result["finalAnswer"],
        "confidenceScore": eval_result["confidenceScore"],
        "sources": sources,
        "status": "success",
        "metadata": {
            "durationMs": duration_ms,
            "filterSummary": filter_summary,
            "removedCount": removed_count,
            "evaluationSummary": eval_result["evaluationSummary"],
            "isAccurate": eval_result["isAccurate"],
            "refinementNeeded": eval_result["refinementNeeded"],
            "rawDocCount": len(raw_documents),
            "filteredDocCount": len(filtered_documents),
            "usedGeneralKnowledge": len(filtered_documents) == 0,
        },
    }
