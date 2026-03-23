# ChatGIT — Complete Project Overview

*A plain-English guide to what this project is, how it works, and why it matters.*

---

## What Is ChatGIT?

ChatGIT is a tool that lets you have a **conversation with any GitHub repository**.

Instead of manually reading hundreds of files, scrolling through code, or doing keyword searches that miss context, you can just ask questions in plain English:

- "How does authentication work in this project?"
- "Where are the database models defined?"
- "What does the `process_payment` function do?"
- "Show me all the API endpoints."

ChatGIT reads the entire repository, understands its structure, and gives you accurate, specific answers — pointing you to the exact file and line number where the relevant code lives.

---

## Why Does This Exist?

### The Problem With Reading Code Manually

Imagine you join a new team and are handed a codebase with 300 files, written over 5 years by 20 different developers. How do you understand it?

- **Keyword search** (Ctrl+F or `grep`) only finds exact word matches. If you search for "login", you miss functions called `authenticate`, `sign_in`, or `verify_user`.
- **Reading files one by one** takes days and you still might miss how things connect across files.
- **Documentation** is often outdated, incomplete, or missing entirely.

This is a real, everyday problem for developers, code reviewers, and anyone trying to understand unfamiliar software.

### What ChatGIT Does Instead

ChatGIT uses **AI** to understand code semantically — the way a human expert would. It can find relevant code even when the exact words you used don't appear in the source. And unlike a human expert, it has read every file in the repository before you ask your first question.

---

## The Big Picture: How It Works

At a high level, ChatGIT does four things:

```
1. READ     → Clone the repo and parse every source file
2. LEARN    → Convert all code into a searchable "understanding"
3. FIND     → When you ask a question, locate the most relevant code
4. ANSWER   → Use an AI to write a clear explanation using that code
```

Let's go through each step in detail.

---

## Step 1: Reading the Repository

When you paste a GitHub URL and click "Load Repository", ChatGIT:

1. **Clones the repository** to your local machine using Git. This downloads all the source code files.

2. **Walks through every file** in the project, skipping irrelevant folders like `node_modules`, `.git`, and `__pycache__`.

3. **Identifies the language** of each file by its extension (`.py` = Python, `.js` = JavaScript, `.java` = Java, etc.)

### What Counts as "Source Code"?

ChatGIT analyzes these file types:
- Python (`.py`)
- JavaScript / JSX (`.js`, `.jsx`)
- TypeScript / TSX (`.ts`, `.tsx`)
- Java (`.java`)
- Swift (`.swift`)
- C and C++ (`.c`, `.cpp`, `.h`, `.hpp`)
- Documentation (`.md`, `.txt`)

---

## Step 2: Parsing and Understanding the Code (AST)

This is where ChatGIT goes beyond keyword search.

### What Is an AST?

AST stands for **Abstract Syntax Tree**. It is a structured representation of code that a computer can understand as more than just text. Think of it like parsing an English sentence into subject, verb, and object — instead of just seeing a string of words.

For Python, ChatGIT uses Python's own built-in `ast` module (the same one the Python interpreter uses) to read each file. For other languages, it uses pattern-matching (regex) to identify code structures.

From each file, ChatGIT extracts:
- **Every function** — its name, line number, arguments, and docstring
- **Every class** — its name, line number, and methods
- **Every import** — what the file depends on

### Why Does This Matter?

Because the AI later works with **chunks** of code. If you cut code randomly (every 1000 characters), you might cut a function in half, losing context. By using the AST, ChatGIT cuts at function boundaries — each chunk is a complete, meaningful unit of code.

---

## Step 3: Chunking — Splitting Code Into Manageable Pieces

AI models can only read a limited amount of text at once (their "context window"). A repository might have 50,000 lines of code — far more than any AI can read in one go.

**Chunking** solves this by splitting the codebase into small, manageable pieces.

### How ChatGIT Chunks Code

For every function or class in the repository, ChatGIT creates one chunk. If the function is very long (more than ~512 tokens, which is roughly 400 words), it is split further — but with **overlap** so the boundary doesn't lose context.

For example, a long function of 1200 tokens might become:
- Chunk 1: lines 1–50 (512 tokens)
- Chunk 2: lines 45–95 (512 tokens, overlapping with Chunk 1 at lines 45–50)
- Chunk 3: lines 90–140 (512 tokens, overlapping with Chunk 2 at lines 90–95)

Each chunk is stored with its metadata:
- Which file it came from
- What line number it starts on
- What line number it ends on
- Whether it's a function, class, or module-level code
- The name of the function or class

---

## Step 4: Embeddings — Teaching AI to Understand Code Similarity

This is the core of the "semantic search" capability.

### What Is an Embedding?

An embedding is a list of numbers (a vector) that represents the meaning of a piece of text. The key property: **similar meanings produce similar vectors**.

For example:
- "authenticate user" and "login function" would produce very similar vectors
- "database connection" and "pizza recipe" would produce very different vectors

ChatGIT uses a model called **BGE-small** (by BAAI / HuggingFace) which is specifically trained to understand code and produce good embeddings for programming concepts.

### Building the Index

Every chunk of code is converted into its embedding vector and stored in a **vector database** (ChromaDB). This database is saved to disk so it only has to be built once per repository.

When you come back to the same repository later, ChatGIT skips all this work and loads the saved embeddings in seconds.

Think of it like building an index for a book — you do it once, and then lookups are instant.

---

## Step 5: PageRank — Finding the Most Important Code

Not all code is equally important. A utility function called 50 times by other functions is more "central" to the project than a helper called once.

ChatGIT uses the **PageRank algorithm** — the same algorithm Google originally used to rank web pages — to identify the most important parts of the codebase.

### How PageRank Works in ChatGIT

Instead of web pages linking to other web pages, ChatGIT looks at:
- **Function calls**: if function A calls function B, that's like A "linking to" B
- **File imports**: if `api.py` imports from `auth.py`, that's a link from `api.py` to `auth.py`

The PageRank score of a function or file reflects how many other things depend on it. Functions that are called everywhere score high. Files that are imported throughout the project score high.

### Three Types of Scores

ChatGIT calculates PageRank for three levels:

1. **File PageRank** — which files are most central to the project?
2. **Function PageRank** — which functions are called most by other functions?
3. **Module PageRank** — which imported packages are most depended upon?

These scores are shown in the Dashboard and also used to boost search results. If two code chunks score equally well on a query, the one with higher PageRank gets ranked higher — because it's more likely to be a core part of the system.

---

## Step 6: Answering Your Question (The RAG Pipeline)

When you type a question in the chat, here is exactly what happens:

### Stage 1: Embed the Query
Your question is converted into an embedding vector using the same BGE model that was used on the code chunks. This puts your question into the same "meaning space" as the code.

### Stage 2: Vector Search (Top-20)
The vector database finds the 20 code chunks whose embeddings are most similar to your question's embedding. This is done using cosine similarity — measuring the "angle" between vectors. Chunks that are semantically close to your question score highest.

### Stage 3: PageRank Boost
Each of the 20 chunks gets its score multiplied by a factor based on its PageRank. Important functions and files rank higher, breaking ties in favour of central code.

### Stage 4: Cross-Encoder Reranking
This is the most sophisticated step. A second AI model (a **cross-encoder**) reads your question and each chunk *together* — not separately. It produces a relevance score that reflects whether this chunk actually answers your question.

The difference matters:
- **Embedding search**: understands each thing separately, compares in a shared space — fast but imprecise
- **Cross-encoder**: reads both together and reasons about their relationship — slower but much more accurate

The cross-encoder reorders the 20 candidates and picks the best 8.

### Stage 5: Context Building
The top 8 chunks are assembled into a prompt, organized by file. The prompt includes:
- Repository statistics (number of files, functions, etc.)
- The code snippets, each labelled with file name and line numbers
- The last 3 turns of your conversation (so follow-up questions work)
- Your current question

### Stage 6: LLM Answer Generation
This assembled prompt is sent to **Groq's Llama 3.1-8B** — a large language model. The model is instructed to answer using only the provided code snippets, always citing exact filenames and line numbers.

Groq is used because it provides extremely fast inference (typically under 1 second for a response), making the chat feel responsive.

### Stage 7: Code Enhancement
After the LLM writes its answer, ChatGIT post-processes the response. Any code blocks in the answer are matched back to the original source files using similarity matching, and the response is annotated with:
- The exact file path
- The precise line numbers
- A confidence indicator

---

## The Call Graph Visualizer

Beyond text chat, ChatGIT has a visual call graph feature.

A **call graph** shows which functions call which other functions, displayed as a network diagram with nodes (functions) and arrows (function calls).

You can:
- See the entire function call network for the repository
- Select a specific function from a dropdown to focus on it
- See which functions it calls (dependencies) and which functions call it (callers)

This is useful for understanding the flow of execution in a large codebase — tracing how a user action triggers a chain of function calls all the way down to the database.

---

## The Dashboard

When a repository loads, the Dashboard shows:

| Metric | What It Means |
|--------|---------------|
| Total Files | Number of source code files analyzed |
| Functions | Total number of function definitions found |
| Classes | Total number of class definitions found |
| Packages | Number of unique external libraries imported |

Below the metrics, three ranked lists are shown:
- **Top Files by Importance** — the files with the highest PageRank scores
- **Top Functions by Importance** — the most-called functions
- **Top Modules by Importance** — the most-used imported packages

---

## The File Tree / Structure Explorer

This panel shows the entire repository structure as a collapsible file tree. Clicking a file expands it to show all the functions and classes inside it, as extracted by the AST parser. This gives you a quick structural overview without opening any files.

---

## What Makes ChatGIT Novel

Most existing tools for code understanding fall into one of two categories:

1. **Keyword search tools** (grep, GitHub search, IDE search) — fast but dumb. They miss semantic relationships.

2. **Single-file AI assistants** (GitHub Copilot, ChatGPT with a file pasted in) — understand code well but only see one file at a time.

ChatGIT combines the best of both:

### Multi-File Semantic Understanding
It understands the *entire repository* at once, not just individual files. When you ask how authentication works, it can pull together relevant code from `auth.py`, `middleware.py`, `routes.py`, and `models.py` — synthesizing an answer that spans the whole project.

### Code-Specific Chunking
By using the AST to chunk at function boundaries, ChatGIT ensures that each retrieved unit of code is a semantically complete thought — not an arbitrary text window.

### Graph-Augmented Retrieval
By layering PageRank on top of vector search, ChatGIT doesn't just find semantically similar code — it prioritizes code that is structurally central to the project. This helps surface the most important functions rather than obscure utilities that happen to match the query.

### Three-Stage Ranking
The combination of:
1. Vector similarity (semantic relevance)
2. PageRank boost (structural importance)
3. Cross-encoder reranking (precise relevance judgment)

is a complete, production-grade retrieval pipeline that goes beyond what most academic code-search systems implement.

### Persistent Indexing
By storing embeddings in ChromaDB, ChatGIT becomes more practical for daily use. You can close the app, restart your computer, and come back to the same repository without waiting for re-indexing.

---

## The Technology Stack

### Backend
| Component | Technology | Purpose |
|-----------|-----------|---------|
| Web framework | FastAPI | Handles API requests from the frontend |
| Embeddings | HuggingFace BGE-small | Converts code to semantic vectors |
| Vector database | ChromaDB | Stores and searches embeddings persistently |
| RAG framework | LlamaIndex | Manages retrieval and indexing pipeline |
| Cross-encoder | sentence-transformers | Reranks retrieved chunks precisely |
| LLM | Groq / Llama 3.1-8B | Generates natural language answers |
| Graph analysis | NetworkX | Builds and runs PageRank on call graphs |
| Git operations | GitPython | Clones repos and detects updates |
| AST parsing | Python `ast` module | Extracts functions/classes from Python files |
| Token counting | tiktoken | Measures chunk sizes in tokens |

### Frontend
| Component | Technology | Purpose |
|-----------|-----------|---------|
| Framework | React + Vite | Interactive user interface |
| HTTP client | Axios | Communicates with the backend API |
| Graph visualization | react-force-graph-2d | Renders the call graph as a force-directed network |
| Markdown rendering | react-markdown | Renders formatted AI responses |
| Code highlighting | react-syntax-highlighter | Syntax-highlighted code blocks in responses |

---

## System Architecture Flow

```
User
  │
  │  Types GitHub URL
  ▼
Frontend (React)
  │
  │  POST /api/load_repo
  ▼
Backend (FastAPI)
  │
  ├─ GitPython: clone / pull repository
  │
  ├─ chunker.py: AST-aware chunking with metadata
  │
  ├─ embeddings.py: BGE-small encodes each chunk → vector
  │
  ├─ ChromaDB: stores vectors persistently
  │
  ├─ pagerank.py: builds call graph → computes PageRank
  │
  └─ Session saved → "success" returned to frontend

User
  │
  │  Types a question in chat
  ▼
Frontend (React)
  │
  │  POST /api/chat
  ▼
Backend (FastAPI)
  │
  ├─ BGE-small: embed the query
  │
  ├─ ChromaDB: vector search → top 20 chunks
  │
  ├─ PageRank boost: re-score candidates
  │
  ├─ reranker.py: cross-encoder scores each candidate
  │
  ├─ Build prompt: top 8 chunks + conversation history + query
  │
  ├─ Groq API: Llama 3.1-8B generates answer
  │
  ├─ snippets.py: locate code in files, add line numbers
  │
  └─ Response returned to frontend → displayed in chat
```

---

## Limitations and Honest Assessment

### What It Does Well
- Understanding semantically-phrased questions about code
- Synthesizing answers from multiple files
- Identifying central/important code via PageRank
- Handling Python repositories with high accuracy (uses real AST)
- Caching repositories for fast subsequent access

### Current Limitations

**Language support quality varies**: Python gets the full AST treatment. JavaScript, Java, and others use regex-based parsing which is less reliable — it can miss some function definitions or misidentify others.

**Local-only for large repos**: Because the embedding model runs on CPU by default, very large repositories (500+ files) take a long time to index on the first load. A GPU would reduce this to under a minute.

**Single-user server**: The backend maintains one active repository session. Multiple users running it simultaneously would interfere with each other. It is designed as a personal developer tool, not a multi-user service.

**LLM can only see retrieved code**: Answers are grounded in the top 8 retrieved chunks. If the relevant code is not retrieved, the answer will be incomplete. The three-stage ranking pipeline significantly reduces this risk, but does not eliminate it.

**No code execution**: ChatGIT reads and explains code, but cannot run it, test it, or modify it.

---

## How to Run ChatGIT

### Requirements
- Python 3.8 or newer
- Node.js 18 or newer (but below 20.19 — use Node 20.10 or 18.x)
- A free Groq API key from [console.groq.com](https://console.groq.com)

### Backend Setup
```bash
# Clone the project
git clone https://github.com/organicall/chatgit.git
cd chatgit

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Create your environment file
echo "GROQ_API_KEY=your_key_here" > .env

# Start the backend
uvicorn api:app --port 8000
```

### Frontend Setup
```bash
# In a separate terminal
cd chatgit-react/frontend
npm install
npm run dev
```

Open `http://localhost:5173` in your browser.

### Using ChatGIT
1. Paste any public GitHub repository URL (e.g., `https://github.com/pallets/flask`)
2. Click **Load Repository** and wait for processing (2–5 minutes first time, ~10 seconds after)
3. Once loaded, explore the Dashboard for stats and PageRank insights
4. Type questions in the Chat panel
5. Optionally enable the Call Graph or File Tree views from the sidebar

### Environment Variables
```bash
GROQ_API_KEY=your_key          # Required — get from console.groq.com
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5   # Optional — bge-large for more accuracy
WORKSPACE_DIR=/path/to/repos   # Optional — where repos are cloned
CHROMA_DIR=/path/to/cache      # Optional — where ChromaDB stores embeddings
```
