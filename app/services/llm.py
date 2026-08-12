"""
LLM client — LangChain ChatOpenAI pointed at OpenRouter.

Direct replacement for backend/utils/llmClient.js. OpenRouter exposes an
OpenAI-compatible /chat/completions endpoint, so LangChain's ChatOpenAI
class works directly by overriding base_url — no custom HTTP code needed
(the Node version hand-rolled this with axios).
"""

from functools import lru_cache

from langchain_openai import ChatOpenAI

from app.config import settings


@lru_cache(maxsize=8)
def get_llm(model: str, temperature: float = 0.2, max_tokens: int = 4096, json_mode: bool = False) -> ChatOpenAI:
    kwargs = {}
    if json_mode:
        kwargs["model_kwargs"] = {"response_format": {"type": "json_object"}}

    return ChatOpenAI(
        model=model,
        api_key=settings.OPENROUTER_API_KEY,
        base_url=settings.OPENROUTER_BASE_URL,
        temperature=temperature,
        max_tokens=max_tokens,
        default_headers={
            "HTTP-Referer": "https://sentinel-ai-langchain.local",
            "X-Title": "Sentinel.AI (LangChain Edition)",
        },
        **kwargs,
    )
