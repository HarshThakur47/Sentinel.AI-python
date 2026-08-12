"""
Evaluator Chain (Tier 3) — LangChain LCEL equivalent of backend/agents/evaluatorAgent.js
"""

import json
import logging
from typing import List

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from app.config import settings
from app.services.llm import get_llm

logger = logging.getLogger("sentinel.evaluator_chain")

SYSTEM_PROMPT = """You are the Evaluator Agent in Sentinel.AI — an independent fact-checker and quality assurance system.

Your job is to evaluate a generated answer against its source documents.

Scoring rubric for confidenceScore (integer 0-10):
- 10 : Perfectly grounded, every claim is directly supported by sources.
- 8-9: Mostly grounded with minor interpretation gaps.
- 6-7: Generally accurate but with some unsupported claims.
- 4-5: Mixed accuracy — significant unsupported claims.
- 2-3: Mostly hallucinated or unsupported.
- 0-1: Completely hallucinated or dangerous misinformation.

You MUST respond in valid JSON with this exact structure:
{{
  "confidenceScore": <integer 0-10>,
  "isAccurate": <true | false>,
  "refinementNeeded": <true | false>,
  "evaluationSummary": "<2-3 sentence assessment>",
  "refinedAnswer": "<if refinementNeeded is true, provide a corrected answer; otherwise null>"
}}"""

_prompt = ChatPromptTemplate.from_messages(
    [("system", SYSTEM_PROMPT), ("user", "{user_prompt}")]
)


async def run_evaluator_chain(user_query: str, generated_answer: str, source_documents: List[dict]) -> dict:
    logger.info("[EvaluatorChain] Starting evaluation...")

    context_block = "\n---\n".join(f"[Doc {i + 1}] {d['title']}: {d['content']}" for i, d in enumerate(source_documents))
    user_prompt = (
        f'Original User Query: "{user_query}"\n\nSource Documents Used:\n{context_block}\n\n'
        f'Generated Answer to Evaluate:\n"""\n{generated_answer}\n"""\n\n'
        "Evaluate the generated answer against the source documents. Return the required JSON."
    )

    llm = get_llm(settings.EVALUATOR_AGENT_MODEL, temperature=0.1, max_tokens=1024, json_mode=True)
    chain = _prompt | llm | StrOutputParser()

    raw = await chain.ainvoke({"user_prompt": user_prompt})

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        logger.error("[EvaluatorChain] Malformed JSON — using generator output as-is.")
        return {
            "confidenceScore": 5,
            "isAccurate": True,
            "refinementNeeded": False,
            "evaluationSummary": "Evaluator returned malformed response.",
            "finalAnswer": generated_answer,
        }

    refinement_needed = parsed.get("refinementNeeded", False)
    refined_answer = parsed.get("refinedAnswer")
    final_answer = refined_answer if (refinement_needed and refined_answer) else generated_answer

    score = parsed.get("confidenceScore", 5)
    score = max(0, min(10, int(score) if str(score).isdigit() else 5))

    return {
        "confidenceScore": score,
        "isAccurate": parsed.get("isAccurate", True),
        "refinementNeeded": refinement_needed,
        "evaluationSummary": parsed.get("evaluationSummary", ""),
        "finalAnswer": final_answer,
    }
