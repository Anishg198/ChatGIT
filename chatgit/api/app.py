import os
import re
import subprocess
import shutil
import json
from pathlib import Path
from dotenv import load_dotenv
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# LlamaIndex
from llama_index.core import Settings, Document, SimpleDirectoryReader, VectorStoreIndex
from llama_index.embeddings.langchain import LangchainEmbedding
from llama_index.core.base.base_query_engine import BaseQueryEngine

# Groq
from groq import Groq

# Helper Modules
from chatgit.core.embeddings import load_embedding_model
from chatgit.core.ast_parser import generate_repo_ast
from chatgit.core.graph.dependency import FunctionDependencyAnalyzer
from chatgit.core.snippets import ImprovedCodeSnippetExtractor
from chatgit.core.graph.pagerank import CodePageRankAnalyzer

load_dotenv()

# Initialize tokenizer for accurate token counting
try:
    import tiktoken
    # Use cl100k_base encoding which is closer to Llama tokenizers
    # Note: We're using Groq's llama-3.1-8b-instant, not GPT-3.5
    TOKENIZER = tiktoken.get_encoding("cl100k_base")
    print("Tokenizer initialized successfully (cl100k_base)")
except Exception as e:
    TOKENIZER = None
    print(f"Warning: tiktoken not available ({e}), using fallback token estimation")

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
    total_packages: int  # Changed from total_imports - counts unique packages/libraries

# --- Global Context ---
class ServerContext:
    def __init__(self):
        self.repository_root: Optional[str] = None
        self.search_index: Optional[VectorStoreIndex] = None
        self.code_ast: Optional[Dict[str, Any]] = None
        self.graph_analyzer: Optional[CodePageRankAnalyzer] = None
        self.conversation_log: List[Dict[str, str]] = []
        self.llm_client: Optional[Groq] = None
        self.services_initialized: bool = False
        
    def clear_session(self):
        self.repository_root = None
        self.search_index = None
        self.code_ast = None
        self.graph_analyzer = None
        self.conversation_log = []

# Single global instance
session = ServerContext()

# --- Utilities ---
def initialize_llm():
    key = os.getenv("GROQ_API_KEY")
    if not key:
        print("❌ ERROR: GROQ_API_KEY not found in environment variables!")
        print("Make sure .env file exists with: GROQ_API_KEY=your_key_here")
        return None
    
    try:
        print(f"✓ GROQ_API_KEY loaded (length: {len(key)} chars)")
        client = Groq(api_key=key)
        print("✓ Groq client initialized successfully")
        return client
    except Exception as e:
        print(f"❌ ERROR: Failed to initialize Groq client: {e}")
        return None

def determine_temperature(query: str) -> float:
    """Determine appropriate temperature based on query type"""
    query_lower = query.lower()
    
    # Higher temp for creative/explanatory queries
    creative_keywords = ['explain', 'how', 'why', 'what if', 'suggest', 'recommend', 
                         'describe', 'compare', 'difference between', 'best way']
    
    # Lower temp for factual/lookup queries  
    factual_keywords = ['find', 'show', 'where', 'which file', 'locate', 
                        'what does', 'list', 'get']
    
    if any(kw in query_lower for kw in creative_keywords):
        return 0.3  # More creative
    elif any(kw in query_lower for kw in factual_keywords):
        return 0.1  # More deterministic
    
    return 0.2  # Default middle ground

def initialize_embedder():
    model = load_embedding_model(device="cpu")
    return LangchainEmbedding(model)

def extract_github_segments(url: str):
    regex = r"https://github\.com/([^/]+)/([^/]+)"
    found = re.match(regex, url)
    return found.groups() if found else (None, None)

def build_file_tree(base_path):
    """Creates a markdown representation of the file structure."""
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
        
        # Limit visible files per folder to avoid clutter
        for fname in sorted(filenames)[:20]:
            lines.append(f"{spacer}  - {fname}")
    
    return "\n".join(lines)

# --- App Lifecycle ---
@asynccontextmanager
async def app_lifespan(server: FastAPI):
    print("Application startup complete. Heavy services will be lazy-loaded.")
    yield
    print("Services shutting down.")
    
def ensure_services():
    """Lazy load heavy models securely"""
    if not session.services_initialized:
        print("Lazy loading AI models...")
        session.llm_client = initialize_llm()
        Settings.embed_model = initialize_embedder()
        session.services_initialized = True
        print("AI models loaded successfully.")

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

# clone and extract the code from the repository
@app.post("/api/load_repo")
async def ingest_repository(payload: RepositoryLoadSchema):
    ensure_services()
    url = payload.github_url
    user, project = extract_github_segments(url)
    
    if not user or not project:
        raise HTTPException(status_code=400, detail="Invalid GitHub URL format.")
        
    try:
        workspace = Path(os.getenv("WORKSPACE_DIR", Path.home() / "Documents" / "github_repos"))
        
        # Validate workspace - check write permissions
        try:
            workspace.mkdir(parents=True, exist_ok=True)
            # Test write permissions
            test_file = workspace / ".write_test"
            test_file.touch()
            test_file.unlink()
        except PermissionError:
            raise HTTPException(
                status_code=500, 
                detail="Cannot write to workspace directory. Check permissions."
            )
        except OSError as e:
            raise HTTPException(
                status_code=500, 
                detail=f"Filesystem error: {str(e)}"
            )
        
        # Check disk space (require at least 100MB free)
        try:
            stats = shutil.disk_usage(workspace)
            if stats.free < 100 * 1024 * 1024:  # Less than 100MB
                raise HTTPException(
                    status_code=507,
                    detail="Insufficient disk space (need at least 100MB free)"
                )
        except Exception as e:
            print(f"Warning: Could not check disk space: {e}")
        
        target_path = workspace / project
        
        if not target_path.exists():
            # Repository doesn't exist - clone it
            print(f"Cloning {url}...")
            proc = subprocess.run(
                ["git", "clone", url, str(target_path)],
                capture_output=True,
                text=True,
                timeout=120
            )
            if proc.returncode != 0:
                raise Exception(f"Git failed: {proc.stderr}")
        else:
            # Repository exists - pull latest changes
            print(f"Repository already exists. Pulling latest changes from {url}...")
            proc = subprocess.run(
                ["git", "-C", str(target_path), "pull"],
                capture_output=True,
                text=True,
                timeout=60
            )
            if proc.returncode != 0:
                print(f"Warning: Git pull failed: {proc.stderr}")
                print("Using existing local version (may be outdated)")
            else:
                print("✓ Repository updated with latest changes")
        
        
        print("Reading files...")
        reader = SimpleDirectoryReader(
            input_dir=str(target_path),
            required_exts=[".py", ".md", ".txt", ".js", ".ts", ".java", ".cpp", ".c", ".h"],
            recursive=True
        )
        documents = reader.load_data()
        
        if not documents:
            raise Exception("No valid source files found.")
            
        print("Structuring data...")
        tree_view = build_file_tree(target_path)
        documents.append(Document(
            text=tree_view,
            metadata={"file_name": "STRUCTURE.md", "type": "meta"}
        ))
        
        print("Parsing AST...")
        ast_data = generate_repo_ast(str(target_path))
        
        print("Running PageRank...")
        pagerank = CodePageRankAnalyzer()
        pagerank.analyze_repository(str(target_path))
        
        # Create summary document
        stats = ast_data.get('stats', {})
        func_list = "\n".join([f"- {fn['name']} ({fn['file']})" for fn in ast_data.get('functions', [])[:50]])
        class_list = "\n".join([f"- {cl['name']} ({cl['file']})" for cl in ast_data.get('classes', [])[:50]])
        
        summary_text = f"""# Codebase Overview

**Metrics:**
- Files: {stats.get('total_files', 0)}
- Functions: {stats.get('total_functions', 0)}
- Classes: {stats.get('total_classes', 0)}
- Packages: {stats.get('total_packages', 0)}

**Key Functions:**
{func_list}

**Key Classes:**
{class_list}
"""
        documents.append(Document(
            text=summary_text,
            metadata={"file_name": "OVERVIEW.md", "type": "meta"}
        ))
        
        print("Indexing...")
        vector_db = VectorStoreIndex.from_documents(documents, show_progress=True)
        
        # Save to session
        session.repository_root = str(target_path)
        session.code_ast = ast_data
        session.graph_analyzer = pagerank
        session.search_index = vector_db
        session.conversation_log = []
        
        return {"status": "success", "message": f"Loaded {project}", "repo_name": project}

    except Exception as ex:
        print(f"Ingestion error: {ex}")
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
        if not ranked:
            print("[API] Warning: No files returned from PageRank analysis")
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
        
        hub_results = [{"name": f, "count": c} for f, c in hubs if c > 0]
        auth_results = [{"name": f, "count": c} for f, c in auths if c > 0]
        
        if not hub_results:
            print("[API] Warning: No hub files found")
        if not auth_results:
            print("[API] Warning: No authority files found")
        
        return {
            "hubs": hub_results,
            "authorities": auth_results
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
    if not session.repository_root:
        return {"error": "No repo loaded"}
    
    try:
        analyzer = FunctionDependencyAnalyzer()
        analyzer.analyze_repository(session.repository_root)
        nodes = sorted(list(analyzer.graph.nodes()))
        return {"functions": nodes}
    except Exception as e:
         return {"error": str(e)}

@app.post("/api/call_graph/visualize")
async def generate_graph_data(body: Dict[str, Any] = Body(...)):
    if not session.repository_root:
        return {"error": "No repo loaded"}
    
    focus = body.get("target")
    if focus == "Show All":
        focus = None
        
    try:
        analyzer = FunctionDependencyAnalyzer()
        analyzer.analyze_repository(session.repository_root)
        
        main_graph = analyzer.graph
        
        node_list = []
        for n in main_graph.nodes():
            node_list.append({"id": n, "label": n})
            
        edge_list = []
        for u, v in main_graph.edges():
            edge_list.append({"source": u, "target": v})
            
        meta = {}
        if focus:
            deps = analyzer.find_dependencies(focus)
            invokers = analyzer.find_callers(focus)
            meta = {"target": focus, "dependencies": deps[:10], "callers": invokers[:10]}
            
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
        # Step 1: Retrieve relevant chunks
        search_engine = session.search_index.as_retriever(similarity_top_k=20)
        results = search_engine.retrieve(query)
        
        # Step 2: Score and re-rank results
        scored_results = []
        analyzer = session.graph_analyzer
        ast_data = session.code_ast
        
        file_pr_map = dict(analyzer.get_file_pagerank()) if analyzer else {}
        func_pr_map = dict(analyzer.get_function_pagerank()) if analyzer else {}
        
        # Track which files we've seen
        context_metadata = {}
        
        for item in results:
            fname = item.metadata.get('file_name', 'unknown')
            content = item.text
            base_score = item.score if item.score else 1.0
            
            # More precise function matching
            matched_funcs = []
            func_score = 0
            related_funcs = []
            
            if ast_data and fname.endswith(('.py', '.js', '.ts', '.java', '.cpp')):
                # Get functions from this file
                file_funcs = [f for f in ast_data.get('functions', []) if f['file'] == fname]
                
                for f in file_funcs:
                    func_name = f['name']
                    # Check if function is actually defined in this content chunk
                    # Use word boundaries to avoid partial matches
                    if re.search(r'\b' + re.escape(func_name) + r'\b', content):
                        matched_funcs.append(func_name)
                        pr_key = f"{fname}::{func_name}"
                        func_score = max(func_score, func_pr_map.get(pr_key, 0))
                        
                        # Get related functions
                        if analyzer and pr_key in analyzer.function_graph:
                            successors = list(analyzer.function_graph.successors(pr_key))[:3]
                            predecessors = list(analyzer.function_graph.predecessors(pr_key))[:3]
                            related_funcs.extend([s.split('::')[-1] for s in successors])
                            related_funcs.extend([p.split('::')[-1] for p in predecessors])
            
            # Calculate final score
            pr_value = func_score if matched_funcs else file_pr_map.get(fname, 0)
            
            # More balanced scoring: don't let PageRank dominate
            final_score = base_score * (1 + pr_value * 10)  # Reduced multiplier from 50 to 10
            
            scored_results.append({
                "snippet": item,
                "score": final_score,
                "pr_value": pr_value,
                "matched_funcs": matched_funcs,
                "related": list(set(related_funcs))
            })
            
            # Track file for context metadata
            if fname not in context_metadata:
                context_metadata[fname] = {
                    'functions': matched_funcs,
                    'pagerank': file_pr_map.get(fname, 0)
                }
        
        # Sort by score
        scored_results.sort(key=lambda x: x["score"], reverse=True)
        
        # Step 3: Select diverse top results (avoid too many from same file)
        top_results = []
        file_count = {}
        max_per_file = 3
        
        for res in scored_results:
            fname = res['snippet'].metadata.get('file_name', 'unknown')
            
            if file_count.get(fname, 0) < max_per_file:
                top_results.append(res)
                file_count[fname] = file_count.get(fname, 0) + 1
            
            if len(top_results) >= 8:
                break
        
        # Step 4: Build context with clear file boundaries
        context_blocks = []
        token_count = 0
        limit = 6000  # Increased limit
        
        # Group by file for better organization
        by_file = {}
        for res in top_results:
            fname = res['snippet'].metadata.get('file_name', 'unknown')
            if fname not in by_file:
                by_file[fname] = []
            by_file[fname].append(res)
        
        for fname, file_results in by_file.items():
            file_block = [f"### File: `{fname}`"]
            
            # Add file-level metadata
            file_pr = file_pr_map.get(fname, 0)
            if file_pr > 0.01:
                file_block.append(f"**PageRank Score:** {file_pr:.4f}")
            
            # Add snippets from this file
            for res in file_results:
                content = res['snippet'].text
                meta_parts = []
                
                if res['matched_funcs']:
                    meta_parts.append(f"Functions: {', '.join(res['matched_funcs'])}")
                if res['related']:
                    meta_parts.append(f"Calls: {', '.join(res['related'][:5])}")
                
                if meta_parts:
                    file_block.append(f"\n**{' | '.join(meta_parts)}**")
                
                file_block.append(f"```\n{content}\n```")
            
            # Check token budget with proper token counting
            block_text = "\n".join(file_block)
            
            # Use tiktoken if available, with 20% buffer for Llama tokenizer differences
            if TOKENIZER:
                base_tokens = len(TOKENIZER.encode(block_text))
                est_tokens = int(base_tokens * 1.2)  # 20% safety buffer
            else:
                est_tokens = len(block_text) // 3  # More conservative than //4
            
            if token_count + est_tokens < limit:
                context_blocks.append(block_text)
                token_count += est_tokens
            else:
                break
        
        context_block = "\n\n---\n\n".join(context_blocks)
        
        # Step 5: Build prompt with file index
        ast = session.code_ast
        stats = ast.get('stats', {}) if ast else {}
        
        # Create file reference index
        file_index = "\n".join([
            f"- `{fname}`: {len(items)} relevant snippets"
            for fname, items in by_file.items()
        ])

        final_prompt = f"""You are ChatGIT, an expert code analysis assistant.

# Repository Context

## Statistics
- Total Files: {stats.get('total_files', 0)}
- Total Functions: {stats.get('total_functions', 0)}
- Total Classes: {stats.get('total_classes', 0)}
- Total Packages: {stats.get('total_packages', 0)}

## Retrieved Files
{file_index}

## Code Snippets (Ranked by Relevance + PageRank)
{context_block}

# User Query
{query}

# Instructions
1. Answer using ONLY the code provided above
2. When referencing code, ALWAYS specify the exact filename
3. If you mention a function, indicate which file it's from
4. If information is incomplete, say so - don't make assumptions
5. Use code blocks with the correct language tag when showing code
6. Be precise about line numbers and file locations based on the context provided

Answer the query clearly and accurately:"""
        
        # Step 6: Get LLM response
        chat_messages = [
            {
                "role": "system",
                "content": "You are ChatGIT, an expert coding assistant. Always specify exact file names when referencing code. Be precise and factual."
            },
            {
                "role": "user",
                "content": final_prompt
            }
        ]
        
        # Determine appropriate temperature based on query type
        temp = determine_temperature(query)
        
        completion = session.llm_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=chat_messages,
            temperature=temp,
            max_tokens=2048,
            stream=False
        )
        
        answer = completion.choices[0].message.content
        
        # Step 7: Enhance with accurate locations
        if payload.enhance_code and session.repository_root:
            try:
                enhancer = ImprovedCodeSnippetExtractor(session.repository_root)
                answer = enhancer.enhance_response(
                    answer, 
                    session.repository_root,
                    context_metadata=context_metadata
                )
            except Exception as e:
                print(f"Enhancement failed: {e}")
                import traceback
                traceback.print_exc()
        
        session.conversation_log.append({"role": "assistant", "content": answer})
        
        return {
            "response": answer,
            "history": session.conversation_log,
            "metadata": {
                "files_used": list(by_file.keys()),
                "total_snippets": len(top_results)
            }
        }

    except Exception as err:
        import traceback
        error_detail = traceback.format_exc()
        print(f"Chat error: {error_detail}")
        
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
