# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Backend
```bash
# Activate virtual environment
source venv/bin/activate

# Run backend server
uvicorn api:app --reload --port 8000

# Run with custom env
GROQ_API_KEY=... uvicorn api:app --reload --port 8000
```

### Frontend
```bash
cd chatgit-react/frontend
npm install        # Install dependencies
npm run dev        # Dev server at http://localhost:5173
npm run build      # Production build
npm run lint       # ESLint
```

### Evaluation
```bash
# Full benchmark suite
python -m evaluation.run_benchmark

# ConvCodeBench evaluation
python -m evaluation.run_convcodebench

# Ablation study (tests each of the 5 novelties)
python -m evaluation.ablation

# Individual evaluation components
python -m evaluation.eval_retrieval
python -m evaluation.eval_generation
python -m evaluation.eval_faithfulness
```

## Architecture

ChatGIT is an AI-powered code Q&A system. A user loads a GitHub repo URL; the backend clones it, builds a semantic index, analyzes the call graph and git history, then answers natural language questions about the code using a retrieval-augmented generation (RAG) pipeline.

### Backend (`chatgit/`)

**Entry point**: `api.py` → `chatgit/api/app.py` (FastAPI, ~940 lines)

**`ServerContext`** is the global singleton holding all session state: the active repo path, ChromaDB vector index, AST data, graph analyzer, LLM client, and all 5 novelty components.

**Repository loading pipeline** (`POST /api/load_repo`):
1. `git clone` via GitPython
2. AST parsing (`core/ast_parser.py`) — extracts functions/classes per file for Python, JS/TS, Java, Swift, C/C++
3. Token-aware chunking (`core/chunker.py`) — max 512 tokens, 64-token overlap, ~6 chunks/node
4. Embedding (`core/embeddings.py`) — BGE-small-en-v1.5 via HuggingFace/LangChain
5. Vector indexing — ChromaDB persistent store (`~/.chatgit_cache/chroma_db`)
6. PageRank analysis (`core/graph/pagerank.py`) — builds file/function/import graphs via NetworkX
7. Git volatility analysis (`core/git_analyzer.py`) — 500 commit lookback

**Chat pipeline** (`POST /api/chat`) integrates 5 research novelties in sequence:

| Step | Component | Purpose |
|------|-----------|---------|
| 1 | `session_memory.py` (N3) | Co-reference resolution from prior turns |
| 2 | `intent_classifier.py` (N4) | Classify query → LOCATE/EXPLAIN/SUMMARIZE/DEBUG → per-intent `RetrievalConfig` |
| 3 | ChromaDB | Vector search with intent-based `top_k` |
| 4 | `graph/hybrid_importance.py` (N2) | Query-conditioned PageRank + embedding importance scoring |
| 5 | `git_analyzer.py` (N1) | Time-decay weighting from git history (frequency 50%, recency 30%, authors 20%) |
| 6 | `session_memory.py` (N3) | Redundancy penalty + session-zone coherence bonus |
| 7 | `reranker.py` | Cross-encoder reranking (ms-marco-MiniLM-L-6-v2) |
| 8 | N5 (inline) | Bidirectional call-graph neighborhood (callers + callees) |
| 9 | Groq API | Llama 3.1-8B inference |
| 10 | `snippets.py` | Precise line-number annotation on code references |

**Models are lazy-loaded** on first request — embedding model, reranker, and LLM client are initialized on demand.

### Frontend (`chatgit-react/frontend/src/`)

React + Vite SPA. Key components:
- **`App.jsx`** — global state (active repo, metrics, chat log), loading overlay with step animation, session persistence via `localStorage`
- **`Chat.jsx`** — markdown + syntax-highlighted responses, per-message code blocks with line numbers
- **`Dashboard.jsx`** — displays PageRank top-files/functions, HITS hubs/authorities
- **`CallGraph.jsx`** — vis-network graph of function call relationships
- **`StructureExplorer.jsx`** — file/function hierarchy browser
- **`config.js`** — API base URL (`http://localhost:8000`)

### Key Design Decisions

- **ChromaDB** is used for vector persistence; the collection is keyed by repo path to support caching across restarts
- **`top_k` is intent-driven**: LOCATE queries use `top_k=25` with tight per-file diversity; SUMMARIZE uses `top_k=12` with wide diversity
- **Session memory penalties**: same-turn chunks penalized 30%, last-turn 60%, older 85% — prevents repetitive context
- **Call-graph neighborhood (N5)** is injected inline in the prompt, not as retrieved chunks, to avoid token budget interference
- **`api.py`** at root is a thin wrapper kept for backward compatibility; all logic lives in `chatgit/api/app.py`

### Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `GROQ_API_KEY` | — | **Required**. Groq API key |
| `MODEL_NAME` | `llama-3.1-8b-instant` | Groq model |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | HuggingFace embedding model |
| `WORKSPACE_PATH` | `~/Documents/github_repos` | Where repos are cloned |
| `CHROMA_DIR` | `~/.chatgit_cache/chroma_db` | Vector DB persistence path |
