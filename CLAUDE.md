# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Backend
```bash
source venv/bin/activate
uvicorn api:app --reload --port 8000
```

### Frontend
```bash
cd chatgit-react/frontend
npm install
npm run dev        # http://localhost:5173
npm run build
npm run lint
```

### Evaluation

Evaluation scripts expect bench repos cloned locally. Set `CHATGIT_REPO_BASE` to a shared parent directory, or override individual repos:
```bash
export CHATGIT_REPO_BASE=/tmp   # expects /tmp/flask_bench, /tmp/requests_bench, etc.
# or per-repo:
export CHATGIT_REPO_FLASK=/path/to/flask_bench
```

```bash
# Main ConvCodeBench evaluation (10 repos, all conversations)
python -m evaluation.run_convcodebench

# Full benchmark suite (retrieval + generation + faithfulness + latency)
python -m evaluation.run_benchmark

# Ablation study (tests each novelty N1–N5 independently)
python -m evaluation.ablation

# LLM-as-judge quality evaluation
python -m evaluation.llm_judge_eval

# Cross-repo generalization evaluation
python -m evaluation.generate_cross_repo_bench   # generate dataset
python -m evaluation.run_full_eval               # run full suite

# Individual components
python -m evaluation.eval_retrieval
python -m evaluation.eval_generation
python -m evaluation.eval_faithfulness
python -m evaluation.eval_latency

# (Re)train the ML intent classifier from labeled conversation data
python -m evaluation.train_intent_classifier
# Output: chatgit/core/intent_clf.pkl
```

### Syntax check (no server needed)
```bash
python -m py_compile chatgit/core/git_analyzer.py chatgit/core/session_memory.py \
  chatgit/core/intent_classifier.py chatgit/core/chunker.py chatgit/api/app.py && echo OK
```

## Architecture

ChatGIT is an AI-powered code Q&A system. A user loads a GitHub repo URL; the backend clones it, builds a semantic index, analyzes the call graph and git history, then answers natural language questions using RAG.

### Backend (`chatgit/`)

**Entry point**: `api.py` (root) → `chatgit/api/app.py` (FastAPI, ~940 lines). The root `api.py` is a thin compatibility shim; all logic lives in `chatgit/api/app.py`.

**`ServerContext`** is the global singleton (defined in `app.py`) holding all session state: active repo path, ChromaDB vector index, AST data, graph analyzer, LLM client, and all 5 novelty components. Models are lazy-loaded on first request.

**Repository loading pipeline** (`POST /api/load_repo`):
1. `git clone` via GitPython
2. AST parsing (`core/ast_parser.py`) — extracts functions/classes per file for Python, JS/TS, Java, Swift, C/C++
3. Token-aware chunking (`core/chunker.py`) — max 512 tokens, 64-token overlap
4. Embedding (`core/embeddings.py`) — BGE-small-en-v1.5 via HuggingFace/LangChain
5. Vector indexing — ChromaDB persistent store (`~/.chatgit_cache/chroma_db`, keyed by repo path)
6. PageRank analysis (`core/graph/pagerank.py`) — builds file/function/import graphs via NetworkX
7. Git volatility analysis (`core/git_analyzer.py`) — 500 commit lookback

**Chat pipeline** (`POST /api/chat`) — the 5 research novelties run in sequence:

| Step | Module | Novelty | Purpose |
|------|--------|---------|---------|
| 1 | `session_memory.py` | N3 | Co-reference resolution (pronoun/temporal expansion) |
| 2 | `intent_classifier.py` | N4 | Classify → LOCATE/EXPLAIN/SUMMARIZE/DEBUG → `RetrievalConfig` |
| 3 | ChromaDB | — | Vector search with intent-driven `top_k` |
| 4 | `graph/hybrid_importance.py` | N2 | Query-conditioned PageRank + embedding importance |
| 5 | `git_analyzer.py` | N1 | Time-decay weighting (frequency 50%, recency 30%, authors 20%) |
| 6 | `session_memory.py` | N3 | Redundancy penalty + session-zone coherence bonus |
| 7 | `reranker.py` | — | Cross-encoder reranking (ms-marco-MiniLM-L-6-v2) |
| 8 | inline in `app.py` | N5 | Bidirectional call-graph neighborhood (injected into prompt, not retrieved chunks) |
| 9 | Groq API | — | Llama 3.1-8B inference |
| 10 | `snippets.py` | — | Precise line-number annotation on code references |

**Intent classifier (N4)**: `intent_classifier.py` uses keyword scoring by default. A trained ML model (`chatgit/core/intent_clf.pkl`, LinearSVC + TF-IDF) is also available and can be regenerated with `python -m evaluation.train_intent_classifier` using labeled turns from `data/convcodebench/`.

**Key retrieval parameters by intent**:
- LOCATE: `top_k=25`, `max_per_file=2`, no call-graph neighborhood
- EXPLAIN: `top_k=20`, `max_per_file=3`, call-graph neighborhood ON
- SUMMARIZE: `top_k=12`, `max_per_file=5`, no call-graph neighborhood
- DEBUG: `top_k=30`, `max_per_file=4`, call-graph neighborhood ON

**Session memory penalties** (N3): same-turn chunks ×0.30, last-turn ×0.60, older ×0.85. Exempt: `module_summary` chunks, same-referent follow-ups (EXPLAIN/DEBUG on already-discussed functions), class chunks under SUMMARIZE intent.

### Frontend (`chatgit-react/frontend/src/`)

React + Vite SPA. API base URL is configured in `config.js` (`http://localhost:8000`). Key components: `App.jsx` (global state, localStorage persistence), `Chat.jsx` (markdown + syntax-highlighted responses), `Dashboard.jsx` (PageRank/HITS metrics), `CallGraph.jsx` (vis-network), `StructureExplorer.jsx`.

### Evaluation Data & Results

- `data/convcodebench/` — benchmark conversation datasets (JSONL)
  - `sample_conversations.jsonl` — labeled training/eval conversations
  - `eval_conversations.jsonl` — held-out evaluation set
  - `cross_repo_conversations.jsonl` — cross-repo generalization test
  - `human_queries.jsonl` — human-authored test queries
- `results/` — JSON output files from evaluation runs
- `paper/chatgit_springer.tex` — the research paper (Springer format)
- `CHANGE_LOG.md` — detailed record of research decisions, fixes, and evaluation methodology notes (important context for paper-related work)

### Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `GROQ_API_KEY` | — | **Required**. Groq API key |
| `MODEL_NAME` | `llama-3.1-8b-instant` | Groq model |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | HuggingFace embedding model |
| `WORKSPACE_PATH` | `~/Documents/github_repos` | Where repos are cloned |
| `CHROMA_DIR` | `~/.chatgit_cache/chroma_db` | Vector DB persistence path |
| `CHATGIT_REPO_BASE` | `/tmp` | Parent directory for bench repos |
| `CHATGIT_REPO_<NAME>` | — | Override path for individual bench repos (FLASK, REQUESTS, CLICK, FASTAPI, CELERY, TORNADO, SCRAPY, DJANGO, SQLALCHEMY, PYTEST) |
| `CHATGIT_CONVS_PATH` | — | Override path to conversations JSONL file |
