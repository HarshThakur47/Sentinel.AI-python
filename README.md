# Sentinel.AI — LangChain Edition

A reimplementation of [Sentinel.AI](../Sentinel-AI) (the original Node.js/Express
multi-agent RAG system) using **FastAPI + LangChain + Hugging Face**.

Built to explore framework-based RAG orchestration vs. the original hand-rolled
multi-agent architecture — same 4-step pipeline (retrieve → filter → generate →
evaluate), same OpenRouter-backed LLM agents, reimplemented with LangChain's
abstractions and a Hugging Face embedding model instead of Gemini.

## Architecture mapping (original → this service)

| Original (Node) | This service (Python) | What changed |
|---|---|---|
| `backend/server.js` + `app.js` | `app/main.py` | Express → FastAPI |
| `backend/config/config.js` | `app/config.py` | Same env vars, pydantic-settings |
| `backend/utils/llmClient.js` (hand-rolled axios) | `app/services/llm.py` | LangChain's `ChatOpenAI` pointed at OpenRouter's OpenAI-compatible endpoint — no custom HTTP client needed |
| `backend/services/vectorDbService.js` (Gemini embeddings + Pinecone REST) | `app/services/embeddings.py` + `app/services/vectorstore.py` | Hugging Face `sentence-transformers` (local, free, no API key) + LangChain's `Chroma` wrapper (swappable for `PineconeVectorStore` in prod) |
| `backend/agents/filterAgent.js` | `app/chains/filter_chain.py` | Manual prompt-building + `JSON.parse` → LangChain `ChatPromptTemplate` + LCEL chain (`prompt \| llm \| parser`) |
| `backend/agents/generatorAgent.js` | `app/chains/generator_chain.py` | Same LCEL pattern |
| `backend/agents/evaluatorAgent.js` | `app/chains/evaluator_chain.py` | Same LCEL pattern |
| `backend/services/ragPipelineService.js` | `app/chains/rag_pipeline.py` | Same orchestration logic and graceful-degradation fallbacks, ported 1:1 |
| `backend/controllers/queryController.js` (manual `validateQuery()`) | `app/schemas.py` (`QueryRequest`) | Manual validation → declarative Pydantic validation |
| *(no auth existed)* | `app/security.py` + `app/routers/auth.py` | **New**: JWT-protected endpoints via `OAuth2PasswordBearer` |
| *(SSE streaming controller)* | *(not yet ported)* | See "What's not included" below |

## What's genuinely different (not just relabeled)

1. **Embeddings**: swapped Gemini's hosted embedding API for a local, open-source
   Hugging Face model (`all-MiniLM-L6-v2`) — no API key needed to run retrieval.
2. **Vector store**: Chroma for local dev (zero setup) instead of requiring a live
   Pinecone index; swappable to `PineconeVectorStore` by changing one function.
3. **Validation**: Pydantic models replace hand-written `if` checks — FastAPI
   auto-rejects malformed requests before your code even runs.
4. **Auth**: added a JWT layer that didn't exist in the original — demonstrates
   `OAuth2PasswordBearer` + protected routes.
5. **Docs**: FastAPI auto-generates interactive Swagger docs at `/docs` — nothing
   to write by hand.

## What's not included (scope was 2 days)

- SSE streaming (`/query/stream` in the original) — the standard `/query`
  endpoint is ported; streaming would be a follow-up using FastAPI's
  `StreamingResponse`.
- Document ingestion endpoint (`documentController.js`) — `upsert_documents()`
  exists in `vectorstore.py` as the building block, but no route calls it yet.
- Rate limiting is wired via `slowapi` but not yet attached to specific routes
  (the Node version used `express-rate-limit` on all routes by default).

## Setup

```bash
# 1. Create virtual environment
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env
# Edit .env and add your OPENROUTER_API_KEY (get one free at openrouter.ai)
# Chroma + Hugging Face need NO API key — they run locally

# 4. Run the server
uvicorn app.main:app --reload --port 8000
```

Visit **http://localhost:8000/docs** for interactive Swagger UI.

## Try it

```bash
# 1. Get a JWT (demo credentials in .env.example: admin / changeme123)
curl -X POST http://localhost:8000/api/v1/auth/token \
  -H "Content-Type: application/json" \
  -d '{"username": "admin", "password": "changeme123"}'

# Response: { "access_token": "eyJ...", "token_type": "bearer" }

# 2. Ask a question (replace <TOKEN> with the access_token above)
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <TOKEN>" \
  -d '{"userQuery": "How does RAG reduce hallucinations?"}'
```

The first request will be slow (~10-20s) while the Hugging Face embedding
model downloads and loads into memory. Subsequent requests are fast.

## Tech stack

FastAPI · LangChain (LCEL) · Hugging Face `sentence-transformers` · Chroma ·
OpenRouter (Claude 3 Haiku, Llama 3.3 70B, Gemini 2.0 Flash — same models as
the original) · Pydantic · JWT (`python-jose`)
