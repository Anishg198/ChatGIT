import os
import re
import shutil
import json
from pathlib import Path
from dotenv import load_dotenv
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# LlamaIndex
from llama_index.core import Settings, VectorStoreIndex, StorageContext
from llama_index.core import Document
from llama_index.embeddings.langchain import LangchainEmbedding

# ChromaDB
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore

# Groq
from groq import Groq

# GitPython
from git import Repo, GitCommandError

# Helper Modules
from chatgit.core.embeddings import load_embedding_model
from chatgit.core.ast_parser import generate_repo_ast
from chatgit.core.chunker import chunk_repository
from chatgit.core.reranker import rerank
from chatgit.core.graph.dependency import FunctionDependencyAnalyzer
from chatgit.core.snippets import ImprovedCodeSnippetExtractor
from chatgit.core.graph.pagerank import CodePageRankAnalyzer

load_dotenv()

# Token counting
try:
    import tiktoken
    TOKENIZER = tiktoken.get_encoding("cl100k_base")
    print("Tokenizer initialized (cl100k_base)")
except Exception as e:
    TOKENIZER = None
    print(f"Warning: tiktoken not available ({e})")

# Persistent ChromaDB directory
CHROMA_DIR = Path(os.getenv("CHROMA_DIR", Path.home() / ".chatgit_cache" / "chroma_db"))
CHROMA_DIR.mkdir(parents=True, exist_ok=True)

# --- Request/Response Models ---
class RepositoryLoadSchema(BaseModel):
    github_url: str

class MessagePayload(BaseModel):
    message: str
    enhance_code: bool = True

class RepoStatistics(BaseModel):
    total_files: int
    total_functions: int
    total_classes: int
    total_packages: int

# --- Global Context ---
class ServerContext:
    def __init__(self):
        self.repository_root: Optional[str] = None
        self.repo_key: Optional[str] = None          # sanitized "user_repo" key
        self.search_index: Optional[VectorStoreIndex] = None
        self.code_ast: Optional[Dict[str, Any]] = None
        self.graph_analyzer: Optional[CodePageRankAnalyzer] = None
        self.conversation_log: List[Dict[str, str]] = []
        self.llm_client: Optional[Groq] = None
        self.services_initialized: bool = False
        self.chroma_client: Optional[chromadb.PersistentClient] = None

    def clear_session(self):
        self.repository_root = None
        self.repo_key = None
        self.search_index = None
        self.code_ast = None
        self.graph_analyzer = None
        self.conversation_log = []

session = ServerContext()

# --- Utilities ---
def initialize_llm():
    key = os.getenv("GROQ_API_KEY")
    if not key:
        print("ERROR: GROQ_API_KEY not found in environment variables!")
        return None
    try:
        print(f"GROQ_API_KEY loaded ({len(key)} chars)")
        client = Groq(api_key=key)
        print("Groq client initialized")
        return client
    except Exception as e:
        print(f"ERROR: Failed to initialize Groq client: {e}")
        return None

def initialize_embedder():
    model = load_embedding_model(device="cpu")
    return LangchainEmbedding(model)

def ensure_services():
    """Lazy-load heavy models on first use."""
    if not session.services_initialized:
        print("Lazy loading AI models...")
        session.llm_client = initialize_llm()
        Settings.embed_model = initialize_embedder()
        session.chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        session.services_initialized = True
        print("AI models loaded.")

def sanitize_collection_name(name: str) -> str:
    """Sanitize a string for use as a ChromaDB collection name (3-63 chars, alphanumeric + underscores)."""
    sanitized = re.sub(r'[^a-zA-Z0-9_]', '_', name)
    sanitized = re.sub(r'_+', '_', sanitized).strip('_')
    if len(sanitized) < 3:
        sanitized = sanitized + "_repo"
    return sanitized[:63]

def determine_temperature(query: str) -> float:
    query_lower = query.lower()
    creative_kw = ['explain', 'how', 'why', 'what if', 'suggest', 'recommend',
                   'describe', 'compare', 'difference between', 'best way']
    factual_kw = ['find', 'show', 'where', 'which file', 'locate',
                  'what does', 'list', 'get']
    if any(k in query_lower for k in creative_kw):
        return 0.3
    elif any(k in query_lower for k in factual_kw):
        return 0.1
    return 0.2

def extract_github_segments(url: str):
    found = re.match(r"https://github\.com/([^/]+)/([^/]+?)(?:\.git)?/?$", url)
    return found.groups() if found else (None, None)

def build_file_tree(base_path):
    lines = ["# Project File Tree\n"]
    base = Path(base_path)
    ignored = {'venv', '__pycache__', '.git', 'node_modules', '.venv'}
    for root, folders, filenames in os.walk(base):
        folders[:] = [d for d in folders if d not in ignored]
        rel_path = Path(root).relative_to(base)
        depth = len(rel_path.parts)
        spacer = "  " * depth
        current_folder = rel_path.name if rel_path.parts else "root"
        lines.append(f"{spacer}[{current_folder}]")
        for fname in sorted(filenames)[:20]:
            lines.append(f"{spacer}  - {fname}")
    return "\n".join(lines)

def _count_tokens(text: str) -> int:
    if TOKENIZER:
        return int(len(TOKENIZER.encode(text)) * 1.2)
    return len(text) // 3

# --- App Lifecycle ---
@asynccontextmanager
async def app_lifespan(server):
    print("Application startup complete. Models are lazy-loaded on first use.")
    yield
    print("Application shutting down.")

app = FastAPI(lifespan=app_lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Endpoints ---

@app.get("/api/health")
async def health_check():
    return {"status": "active"}

@app.post("/api/load_repo")
async def ingest_repository(payload: RepositoryLoadSchema):
    ensure_services()
    url = payload.github_url
    user, project = extract_github_segments(url)

    if not user or not project:
        raise HTTPException(status_code=400, detail="Invalid GitHub URL format.")

    try:
        workspace = Path(os.getenv("WORKSPACE_DIR", Path.home() / "Documents" / "github_repos"))
        try:
            workspace.mkdir(parents=True, exist_ok=True)
            test_file = workspace / ".write_test"
            test_file.touch()
            test_file.unlink()
        except PermissionError:
            raise HTTPException(status_code=500, detail="Cannot write to workspace directory.")
        except OSError as e:
            raise HTTPException(status_code=500, detail=f"Filesystem error: {e}")

        stats_info = shutil.disk_usage(workspace)
        if stats_info.free < 100 * 1024 * 1024:
            raise HTTPException(status_code=507, detail="Insufficient disk space (need 100MB+)")

        target_path = workspace / project
        already_up_to_date = False

        if not target_path.exists():
            print(f"Cloning {url} via GitPython...")
            Repo.clone_from(url, str(target_path))
            print("Clone complete.")
        else:
            print(f"Repository exists. Pulling latest changes...")
            try:
                repo = Repo(str(target_path))
                pull_result = repo.remotes.origin.pull()
                fetch_info = pull_result[0] if pull_result else None
                # flags=4 means ALREADY_UP_TO_DATE in GitPython
                already_up_to_date = (fetch_info is not None and fetch_info.flags == 4)
                if already_up_to_date:
                    print("Repository already up to date.")
                else:
                    print("Repository updated with latest changes.")
            except GitCommandError as e:
                print(f"Warning: Git pull failed: {e}. Using existing local version.")
                already_up_to_date = True  # treat as up-to-date to reuse index

        # Build a stable collection key
        repo_key = sanitize_collection_name(f"{user}_{project}")

        # Decide whether to reuse existing ChromaDB collection
        chroma_collection = session.chroma_client.get_or_create_collection(
            name=repo_key,
            metadata={"hnsw:space": "cosine"}
        )
        reuse_index = already_up_to_date and chroma_collection.count() > 0
        ast_data = None  # will be set during indexing or below

        if reuse_index:
            print(f"Reusing existing vector index for '{repo_key}' ({chroma_collection.count()} chunks).")
            vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
            vector_db = VectorStoreIndex.from_vector_store(vector_store)
        else:
            if chroma_collection.count() > 0:
                print(f"Rebuilding index for '{repo_key}' (repo was updated)...")
                session.chroma_client.delete_collection(repo_key)
                chroma_collection = session.chroma_client.get_or_create_collection(
                    name=repo_key,
                    metadata={"hnsw:space": "cosine"}
                )

            print("Token-aware AST chunking...")
            documents = chunk_repository(str(target_path))

            # Add file-tree and overview docs
            tree_text = build_file_tree(target_path)
            documents.append(Document(
                text=tree_text,
                metadata={"file_name": "STRUCTURE.md", "node_type": "meta",
                          "node_name": "file_tree", "start_line": 1, "end_line": 0, "chunk_index": 0}
            ))

            print("Parsing AST...")
            ast_data = generate_repo_ast(str(target_path))
            stats = ast_data.get('stats', {})
            func_list = "\n".join([f"- {fn['name']} ({fn['file']})" for fn in ast_data.get('functions', [])[:50]])
            class_list = "\n".join([f"- {cl['name']} ({cl['file']})" for cl in ast_data.get('classes', [])[:50]])
            overview_text = f"""# Codebase Overview\n\n**Metrics:**\n- Files: {stats.get('total_files',0)}\n- Functions: {stats.get('total_functions',0)}\n- Classes: {stats.get('total_classes',0)}\n- Packages: {stats.get('total_packages',0)}\n\n**Key Functions:**\n{func_list}\n\n**Key Classes:**\n{class_list}\n"""
            documents.append(Document(
                text=overview_text,
                metadata={"file_name": "OVERVIEW.md", "node_type": "meta",
                          "node_name": "overview", "start_line": 1, "end_line": 0, "chunk_index": 0}
            ))

            print("Building vector index (ChromaDB)...")
            vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            vector_db = VectorStoreIndex.from_documents(
                documents,
                storage_context=storage_context,
                show_progress=True
            )
            print(f"Indexed {len(documents)} chunks into ChromaDB collection '{repo_key}'.")

        print("Running PageRank analysis...")
        if ast_data is None:
            ast_data = generate_repo_ast(str(target_path))
        pagerank = CodePageRankAnalyzer()
        pagerank.analyze_repository(str(target_path))

        session.repository_root = str(target_path)
        session.repo_key = repo_key
        session.code_ast = ast_data
        session.graph_analyzer = pagerank
        session.search_index = vector_db
        session.conversation_log = []

        return {"status": "success", "message": f"Loaded {project}", "repo_name": project}

    except HTTPException:
        raise
    except Exception as ex:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(ex))


@app.get("/api/current_repo")
async def get_active_repo():
    if not session.repository_root:
        return {"repo_name": None}
    return {"repo_name": Path(session.repository_root).name}

@app.post("/api/clear_repo")
async def reset_session():
    session.clear_session()
    return {"status": "cleared"}

@app.get("/api/stats")
async def fetch_statistics():
    if not session.code_ast:
        return {}
    return session.code_ast.get('stats', {})

@app.get("/api/structure")
async def fetch_structure():
    if not session.code_ast:
        return {"files": {}}
    return session.code_ast.get('files', {})

@app.get("/api/pagerank/files")
async def get_top_files():
    if not session.graph_analyzer:
        return []
    try:
        ranked = session.graph_analyzer.get_file_pagerank()[:10]
        return [{"name": f, "score": s} for f, s in ranked]
    except Exception as e:
        print(f"[API] Error in get_top_files: {e}")
        return []

@app.get("/api/pagerank/hubs_authorities")
async def get_network_metrics():
    if not session.graph_analyzer:
        return {"hubs": [], "authorities": []}
    try:
        hubs = session.graph_analyzer.get_hub_files(10)
        auths = session.graph_analyzer.get_authority_files(10)
        return {
            "hubs": [{"name": f, "count": c} for f, c in hubs if c > 0],
            "authorities": [{"name": f, "count": c} for f, c in auths if c > 0]
        }
    except Exception as e:
        print(f"[API] Error in get_network_metrics: {e}")
        return {"hubs": [], "authorities": []}

@app.get("/api/pagerank/functions")
async def get_top_functions():
    if not session.graph_analyzer:
        return []
    items = session.graph_analyzer.get_function_pagerank()[:10]
    return [{"name": f, "score": s} for f, s in items]

@app.get("/api/pagerank/central_functions")
async def get_centrality_metrics():
    if not session.graph_analyzer:
        return []
    items = session.graph_analyzer.get_central_functions(10)
    return [{"name": f, "score": s} for f, s in items if s > 0]

@app.get("/api/pagerank/modules")
async def get_module_importance():
    if not session.graph_analyzer:
        return []
    items = session.graph_analyzer.get_import_pagerank()[:10]
    return [{"name": m, "score": s, "is_local": m.endswith('.py')} for m, s in items]

@app.get("/api/call_graph")
async def retrieve_call_graph(target_function: Optional[str] = None):
    """Return list of all function nodes from the cached PageRank graph."""
    if not session.graph_analyzer:
        return {"error": "No repo loaded"}
    try:
        nodes = sorted(list(session.graph_analyzer.function_graph.nodes()))
        return {"functions": nodes}
    except Exception as e:
        return {"error": str(e)}

@app.post("/api/call_graph/visualize")
async def generate_graph_data(body: Dict[str, Any] = Body(...)):
    """Return call graph nodes/edges using the cached PageRank function graph."""
    if not session.graph_analyzer:
        return {"error": "No repo loaded"}

    focus = body.get("target")
    if focus == "Show All":
        focus = None

    try:
        fg = session.graph_analyzer.function_graph

        node_list = [{"id": n, "label": n.split("::")[-1]} for n in fg.nodes()]
        edge_list = [{"source": u, "target": v} for u, v in fg.edges()]

        meta = {}
        if focus:
            # Support both qualified ("file::func") and short names
            if "::" not in focus:
                # Try to find a matching qualified name
                candidates = [n for n in fg.nodes() if n.endswith(f"::{focus}")]
                focus_qualified = candidates[0] if candidates else focus
            else:
                focus_qualified = focus

            deps = list(fg.successors(focus_qualified)) if focus_qualified in fg else []
            callers = list(fg.predecessors(focus_qualified)) if focus_qualified in fg else []
            meta = {
                "target": focus_qualified,
                "dependencies": [d.split("::")[-1] for d in deps[:10]],
                "callers": [c.split("::")[-1] for c in callers[:10]]
            }

        return {"nodes": node_list, "edges": edge_list, "details": meta}

    except Exception as e:
        return {"error": str(e)}


@app.post("/api/chat")
async def process_chat(payload: MessagePayload):
    ensure_services()
    if not session.search_index:
        raise HTTPException(status_code=400, detail="Repository not loaded")

    query = payload.message
    session.conversation_log.append({"role": "user", "content": query})

    try:
        # -- Step 1: Retrieve top-k candidates --
        retriever = session.search_index.as_retriever(similarity_top_k=20)
        results = retriever.retrieve(query)

        # -- Step 2: Score candidates (vector score x PageRank boost) --
        analyzer = session.graph_analyzer
        ast_data = session.code_ast

        file_pr_map = dict(analyzer.get_file_pagerank()) if analyzer else {}
        func_pr_map = dict(analyzer.get_function_pagerank()) if analyzer else {}

        candidates = []
        context_metadata = {}

        for item in results:
            fname = item.metadata.get('file_name', 'unknown')
            content = item.text
            base_score = item.score if item.score else 1.0

            matched_funcs = []
            func_score = 0.0
            related_funcs = []

            if ast_data and fname.endswith(('.py', '.js', '.ts', '.java', '.cpp')):
                file_funcs = [f for f in ast_data.get('functions', []) if f['file'] == fname]
                for f in file_funcs:
                    func_name = f['name']
                    if re.search(r'\b' + re.escape(func_name) + r'\b', content):
                        matched_funcs.append(func_name)
                        pr_key = f"{fname}::{func_name}"
                        func_score = max(func_score, func_pr_map.get(pr_key, 0))
                        if analyzer and pr_key in analyzer.function_graph:
                            succs = list(analyzer.function_graph.successors(pr_key))[:3]
                            preds = list(analyzer.function_graph.predecessors(pr_key))[:3]
                            related_funcs.extend([s.split('::')[-1] for s in succs])
                            related_funcs.extend([p.split('::')[-1] for p in preds])

            pr_value = func_score if matched_funcs else file_pr_map.get(fname, 0)
            final_score = base_score * (1 + pr_value * 10)

            candidates.append({
                "snippet": item,
                "score": final_score,
                "pr_value": pr_value,
                "matched_funcs": matched_funcs,
                "related": list(set(related_funcs)),
            })

            if fname not in context_metadata:
                context_metadata[fname] = {
                    'functions': matched_funcs,
                    'pagerank': file_pr_map.get(fname, 0)
                }

        # Sort by heuristic score before feeding to cross-encoder
        candidates.sort(key=lambda x: x["score"], reverse=True)

        # -- Step 3: Cross-encoder reranking --
        top_results = rerank(query, candidates, top_n=8)

        # -- Step 4: File-diversity cap (max 3 chunks per file) --
        diverse_results = []
        file_count: Dict[str, int] = {}
        for res in top_results:
            fname = res['snippet'].metadata.get('file_name', 'unknown')
            if file_count.get(fname, 0) < 3:
                diverse_results.append(res)
                file_count[fname] = file_count.get(fname, 0) + 1

        # -- Step 5: Build context with token budget --
        by_file: Dict[str, list] = {}
        for res in diverse_results:
            fname = res['snippet'].metadata.get('file_name', 'unknown')
            by_file.setdefault(fname, []).append(res)

        context_blocks = []
        token_count = 0
        token_limit = 6000

        for fname, file_results in by_file.items():
            file_block = [f"### File: `{fname}`"]
            file_pr = file_pr_map.get(fname, 0)
            if file_pr > 0.01:
                file_block.append(f"**PageRank Score:** {file_pr:.4f}")

            for res in file_results:
                content = res['snippet'].text
                # Include line range from metadata if available
                meta = res['snippet'].metadata
                line_info = ""
                if meta.get('start_line') and meta.get('end_line'):
                    line_info = f" (Lines {meta['start_line']}-{meta['end_line']})"

                meta_parts = []
                if res['matched_funcs']:
                    meta_parts.append(f"Functions: {', '.join(res['matched_funcs'])}")
                if res['related']:
                    meta_parts.append(f"Calls: {', '.join(res['related'][:5])}")
                if meta.get('node_name') and meta.get('node_type') not in ('meta', 'module'):
                    meta_parts.append(f"{meta.get('node_type','').capitalize()}: {meta.get('node_name','')}")

                if meta_parts:
                    file_block.append(f"\n**{' | '.join(meta_parts)}{line_info}**")
                elif line_info:
                    file_block.append(f"\n**{line_info.strip()}**")

                file_block.append(f"```\n{content}\n```")

            block_text = "\n".join(file_block)
            est = _count_tokens(block_text)
            if token_count + est < token_limit:
                context_blocks.append(block_text)
                token_count += est
            else:
                break

        context_block = "\n\n---\n\n".join(context_blocks)

        # -- Step 6: Build prompt with conversation history --
        ast = session.code_ast
        stats = ast.get('stats', {}) if ast else {}

        file_index = "\n".join([
            f"- `{fname}`: {len(items)} snippet(s)"
            for fname, items in by_file.items()
        ])

        # Include last 3 turns (6 messages) of history for context
        history_turns = session.conversation_log[:-1]  # exclude current user message
        recent_history = history_turns[-6:] if len(history_turns) > 6 else history_turns
        history_block = ""
        if recent_history:
            history_lines = []
            for msg in recent_history:
                role = "User" if msg["role"] == "user" else "Assistant"
                # Truncate long history messages
                content_preview = msg["content"][:500] + "..." if len(msg["content"]) > 500 else msg["content"]
                history_lines.append(f"**{role}:** {content_preview}")
            history_block = "\n\n## Conversation History\n" + "\n\n".join(history_lines)

        final_prompt = f"""You are ChatGIT, an expert code analysis assistant.

# Repository Context

## Statistics
- Total Files: {stats.get('total_files', 0)}
- Total Functions: {stats.get('total_functions', 0)}
- Total Classes: {stats.get('total_classes', 0)}
- Total Packages: {stats.get('total_packages', 0)}

## Retrieved Files (Ranked by Relevance + PageRank + Cross-Encoder)
{file_index}

## Code Snippets
{context_block}
{history_block}

# Current Query
{query}

# Instructions
1. Answer using ONLY the code provided above.
2. Always specify the exact filename when referencing code.
3. Include line numbers when available.
4. If prior conversation is relevant, refer to it naturally.
5. If information is incomplete, say so - do not hallucinate.
6. Use code blocks with the correct language tag.

Answer:"""

        # -- Step 7: LLM inference --
        chat_messages = [
            {
                "role": "system",
                "content": "You are ChatGIT, an expert code assistant. Always cite exact filenames and line numbers. Be precise and grounded in the provided code."
            },
            {
                "role": "user",
                "content": final_prompt
            }
        ]

        temp = determine_temperature(query)
        completion = session.llm_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=chat_messages,
            temperature=temp,
            max_tokens=2048,
            stream=False
        )

        answer = completion.choices[0].message.content

        # -- Step 8: Enhance with precise line numbers --
        if payload.enhance_code and session.repository_root:
            try:
                enhancer = ImprovedCodeSnippetExtractor(session.repository_root)
                answer = enhancer.enhance_response(answer, session.repository_root,
                                                   context_metadata=context_metadata)
            except Exception as e:
                print(f"Enhancement failed: {e}")

        session.conversation_log.append({"role": "assistant", "content": answer})

        return {
            "response": answer,
            "history": session.conversation_log,
            "metadata": {
                "files_used": list(by_file.keys()),
                "total_snippets": len(diverse_results),
                "reranked": True,
            }
        }

    except Exception as err:
        import traceback
        error_detail = traceback.format_exc()
        print(f"Chat error:\n{error_detail}")
        error_resp = f"Error processing request: {str(err)}"
        if "413" in str(err) or "context" in str(err).lower():
            error_resp += "\n\nContext too large. Try a more specific question."
        session.conversation_log.append({"role": "assistant", "content": error_resp})
        return {"response": error_resp, "history": session.conversation_log}


@app.get("/api/chat/history")
async def fetch_history():
    return session.conversation_log


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
