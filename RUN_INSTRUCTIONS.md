# How to Run ChatGit

## Backend (Terminal 1)
```bash
source venv/bin/activate
uvicorn api:app --reload --port 8000
```

## Frontend (Terminal 2)
```bash
cd chatgit-react/frontend
npm run dev
```