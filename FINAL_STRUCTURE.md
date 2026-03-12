# ChatGIT - Codebase Structure

## Complete Folder Structure

```
ChatGit/
│
├── api.py                              # Backward compatibility wrapper (10 lines)
├── requirements.txt                    # Python dependencies
├── .env                                # Environment variables (gitignored)
├── .gitignore                          # Git ignore rules
│
├── Documentation/
│   ├── README.md                          # Main project README
│   ├── RUN_INSTRUCTIONS.md                # How to run the app
│   ├── CODE_EXPLANATION.md                # Code documentation
│   ├── SNIPPET_ENHANCEMENT.md             # Feature documentation
│   ├── PHASE2_RESTRUCTURING.md            # Refactoring documentation
│   └── architecture-diagram.png           # Architecture diagram
│
├── chatgit/                            # MAIN PYTHON PACKAGE
│   ├── __init__.py
│   │
│   ├── api/                            # API Layer
│   │   ├── __init__.py
│   │   ├── app.py                         # FastAPI application (663 lines)
│   │   │                                  # - All endpoints (/api/*)
│   │   │                                  # - Request/response models
│   │   │                                  # - Session management
│   │   │                                  # - CORS configuration
│   │   └── routes/                        # (Ready for future route splitting)
│   │       └── __init__.py
│   │
│   ├── core/                           # Core Business Logic
│   │   ├── __init__.py
│   │   │
│   │   ├── embeddings.py                  # Embedding Model (25 lines)
│   │   │                                  # - load_embedding_model()
│   │   │                                  # - HuggingFace BGE embeddings
│   │   │
│   │   ├── ast_parser.py                  # AST Parser (395 lines)
│   │   │                                  # - generate_repo_ast()
│   │   │                                  # - Multi-language support:
│   │   │                                  #   • Python (AST)
│   │   │                                  #   • JavaScript/TypeScript (regex)
│   │   │                                  #   • Java (regex)
│   │   │                                  #   • Swift (regex)
│   │   │                                  #   • C/C++ (regex)
│   │   │
│   │   ├── snippets.py                    # Code Snippet Extractor (269 lines)
│   │   │                                  # - ImprovedCodeSnippetExtractor
│   │   │                                  # - enhance_response()
│   │   │                                  # - Line number tracking
│   │   │
│   │   └── graph/                         # Graph Analysis
│   │       ├── __init__.py
│   │       │
│   │       ├── dependency.py              # Dependency Analyzer (171 lines)
│   │       │                              # - FunctionDependencyAnalyzer
│   │       │                              # - Call graph generation
│   │       │                              # - find_dependencies()
│   │       │                              # - find_callers()
│   │       │
│   │       └── pagerank.py                # PageRank Analyzer (655 lines)
│   │                                      # - CodePageRankAnalyzer
│   │                                      # - File/function importance
│   │                                      # - Hub/authority detection
│   │                                      # - Module importance
│   │
│   └── 🛠️  utils/                         # Utilities
│       └── __init__.py                    # (Ready for future utilities)
│
├──  chatgit-react/                     # REACT FRONTEND
│   ├── README.md
│   ├── start.sh                           # Script to start frontend
│   │
│   ├── backend/                           # (Empty - backend is in root)
│   │
│   └── frontend/                          # React Application
│       ├── package.json                   # NPM dependencies
│       ├── package-lock.json
│       ├── vite.config.js                 # Vite configuration
│       ├── eslint.config.js               # ESLint configuration
│       ├── index.html                     # HTML entry point
│       │
│       ├── public/
│       │   └── vite.svg                   # Vite logo
│       │
│       └── src/                           # React source code
│           ├── main.jsx                   # React entry point
│           ├── App.jsx                    # Main app component
│           ├── App.css                    # App styles
│           ├── index.css                  # Global styles
│           │
│           ├── assets/
│           │   └── react.svg              # React logo
│           │
│           └── components/                # React Components
│               ├── Sidebar.jsx            # Repository sidebar
│               ├── Dashboard.jsx          # Statistics dashboard
│               ├── Chat.jsx               # Chat interface
│               ├── CallGraph.jsx          # Call graph visualization
│               └── StructureExplorer.jsx  # File structure explorer
│
├── .agent/                             # Agent workflows
│   └── workflows/
│       └── deploy_to_github.md            # GitHub deployment workflow
│
├──  .vscode/                           # VS Code settings
│   └── settings.json
│
└── venv/                               # Python virtual environment
    └── (excluded from git)

```

---

## Key Files

| File | Location | Purpose | Lines |
|------|----------|---------|-------|
| **Main API** | `chatgit/api/app.py` | FastAPI application, all endpoints | 663 |
| **AST Parser** | `chatgit/core/ast_parser.py` | Multi-language code parsing | 395 |
| **PageRank** | `chatgit/core/graph/pagerank.py` | Code importance analysis | 655 |
| **Snippets** | `chatgit/core/snippets.py` | Code snippet extraction | 269 |
| **Dependency** | `chatgit/core/graph/dependency.py` | Call graph analysis | 171 |
| **Embeddings** | `chatgit/core/embeddings.py` | Vector embeddings | 25 |
| **React App** | `chatgit-react/frontend/src/App.jsx` | Frontend main component | ~200 |
| **Chat UI** | `chatgit-react/frontend/src/components/Chat.jsx` | Chat interface | ~150 |

---

## Package Organization

### `chatgit.api` - API Layer
- HTTP endpoints
- Request/response handling
- CORS middleware
- Session management

### `chatgit.core` - Business Logic
- AST parsing (multi-language)
- Embedding model loading
- Code snippet extraction

### `chatgit.core.graph` - Graph Analysis
- Dependency analysis
- PageRank algorithm
- Call graph generation
- Importance metrics

### `chatgit.utils` - Utilities
- Ready for future helper functions
- Token counting
- File utilities

---

## Entry Points

### Backend
```bash
# From project root
uvicorn api:app --reload --port 8000
```

### Frontend
```bash
# From chatgit-react/frontend
npm run dev
```

---



