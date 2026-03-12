# ChatGIT

**AI-Powered Code Analysis & Chat Interface for GitHub Repositories**

ChatGIT is an intelligent code analysis tool that lets you have natural language conversations with your codebase. Using advanced RAG (Retrieval Augmented Generation), PageRank algorithms, and multi-language AST parsing, LLM helps you understand, navigate, and analyze any GitHub repository.

Website – [ChatGIT](https://chatgit-x4l2.vercel.app)

---

## What is ChatGIT?

ChatGIT combines cutting-edge AI with code analysis to provide:

- **Natural Language Code Chat**: Ask questions about your codebase in plain English
- **Intelligent Code Search**: Powered by vector embeddings and semantic search
- **Code Importance Analysis**: PageRank algorithm identifies critical files and functions
- **Multi-Language Support**: Python, JavaScript, TypeScript, Java, Swift, C/C++
- **Dependency Graphs**: Visualize function calls and dependencies
- **Context-Aware Responses**: File locations, line numbers, and relevant code snippets
- **Fast & Accurate**: LlamaIndex vector database with BGE embeddings

---

## Architecture

ChatGIT uses a sophisticated pipeline:

1. **Repository Analysis**: Clones and parses the repository using AST (Abstract Syntax Tree)
2. **Vector Indexing**: Creates semantic embeddings of code using HuggingFace BGE models
3. **PageRank Analysis**: Calculates importance scores for files, functions, and modules
4. **RAG Pipeline**: Retrieves relevant context and generates responses using Groq's Llama 3.1
5. **Enhancement**: Adds precise file locations and line numbers to responses

**Tech Stack:**
- **Backend**: FastAPI, LlamaIndex, LangChain, Groq API
- **Frontend**: React, Vite, vis-network (for graph visualization)
- **Models**: Llama 3.1-8B (via Groq), BGE-large-en-v1.5 (embeddings)

---

## Quick Start

### Prerequisites

- **Python 3.12+**
- **Node.js 18+**
- **Git**
- **Groq API Key** 

### Step 1: Clone the Repository

```bash
git clone https://github.com/organicall/chatgit.git
cd chatgit
```

### Step 2: Backend Setup

```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install Python dependencies
pip install -r requirements.txt

# Create .env file with your Groq API key
echo "GROQ_API_KEY=your_groq_api_key_here" > .env
```

### Step 3: Frontend Setup

```bash
# Navigate to frontend directory
cd chatgit-react/frontend

# Install Node dependencies
npm install
```

### Step 4: Launch the Application

**Terminal 1 - Backend:**
```bash
# From project root
source venv/bin/activate
uvicorn api:app --reload --port 8000
```

**Terminal 2 - Frontend:**
```bash
# From chatgit-react/frontend
npm run dev
```

### Step 5: Access the Application

Open your browser and go to:
- **Frontend**: http://localhost:5173

---

## How to Use

### Step 1: Load a Repository

1. Enter any **public GitHub repository URL** (e.g., `https://github.com/pallets/flask`)
2. Click **"Load Repository"**
3. Wait for the analysis to complete (parsing, indexing, PageRank calculation)

### Step 2: Explore the Dashboard

View automatically generated insights:
- **Statistics**: Total files, functions, classes, packages
- **Top Files**: Most important files by PageRank score
- **Top Functions**: Most important functions
- **Module Dependencies**: Package importance rankings

### Step 3: Chat with Your Code

Ask questions like:
- "What does the main function do?"
- "How is authentication implemented?"
- "Show me the database models"
- "Explain the routing logic"
- "Find all API endpoints"

The LLM provides intelligent, context-rich answers to your questions by:
- Retrieving the most relevant code snippets from the repository
- Ranking results by importance using PageRank analysis
- Including exact file names and line numbers for easy navigation
- Delivering context-aware explanations tailored to your query

### Step 4: Visualize Dependencies

- Click **"Call Graph"** to see function dependencies
- Pick a specific function to explore its callers and callees
- Visualize the entire codebase structure

---

## Configuration

### Environment Variables

Create a `.env` file in the project root:

```bash
# Required
GROQ_API_KEY=your_groq_api_key_here

# Optional (defaults shown)
MODEL_NAME=llama-3.1-8b-instant
EMBEDDING_MODEL=BAAI/bge-large-en-v1.5
WORKSPACE_PATH=~/Documents/github_repos
```

### Supported Languages

ChatGIT can analyze:
- Python (`.py`)
- JavaScript/JSX (`.js`, `.jsx`)
- TypeScript/TSX (`.ts`, `.tsx`)
- Java (`.java`)
- Swift (`.swift`)
- C/C++ (`.c`, `.cpp`, `.h`, `.hpp`)

---

## Features in Detail

### 1. RAG (Retrieval Augmented Generation)

ChatGIT uses a multi-stage RAG pipeline:
- **Semantic Search**: Vector embeddings find relevant code
- **PageRank Scoring**: Boosts important files/functions
- **Context Building**: Organizes snippets by file with metadata
- **LLM Generation**: Groq's Llama 3.1 generates accurate responses

### 2. Abstract Syntax Tree (AST) Parsing

- **Python**: Uses Python's built-in `ast` module for precise parsing
- **Other Languages**: Regex-based parsing for functions, classes, imports
- **Multi-file Analysis**: Analyzes entire repository structure
- **Statistics**: Counts functions, classes, imports per file

### 3. PageRank Analysis

Identifies the most important parts of your codebase:
- **File PageRank**: Which files are most central to the project?
- **Function PageRank**: Which functions are most called?
- **Module PageRank**: Which packages are most critical?
- **Hub/Authority Detection**: Find architectural patterns

### 4. Code Enhancement

Responses include:
- **Exact file paths**: `src/api/routes.py`
- **Line numbers**: `Lines 45-67`
- **Function relationships**: "Calls `validate_user()` at line 123"
- **PageRank scores**: Importance indicators

---

## Project Structure

```
chatgit/
├── api.py                          # Backward compatibility wrapper
├── chatgit/                        # Main package
│   ├── api/
│   │   └── app.py                  # FastAPI application
│   ├── core/
│   │   ├── embeddings.py           # Embedding model
│   │   ├── ast_parser.py           # Multi-language AST parsing
│   │   ├── snippets.py             # Code snippet extraction
│   │   └── graph/
│   │       ├── dependency.py       # Dependency analysis
│   │       └── pagerank.py         # PageRank algorithm
│   └── utils/                      # Utilities
├── chatgit-react/                  # React frontend
│   └── frontend/
│       └── src/
│           ├── App.jsx
│           └── components/
└── requirements.txt
```

See [FINAL_STRUCTURE.md](FINAL_STRUCTURE.md) for complete details.


---

## Acknowledgments

- **Llama 3.1** by Meta AI
- **Groq** for ultra-fast LLM inference
- **LlamaIndex** for RAG framework
- **LangChain** for embeddings
- **HuggingFace** for BGE models

--- 





