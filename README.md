# RAG Chatbot

This project is a PDF-based RAG chatbot built with FastAPI and React.

## Included

- `backend/` FastAPI app, PDF processing, chunking, retrieval, and LLM service
- `frontend/` Vite + React UI shell
- test scripts for PDF, retrieval, vector store, and RAG pipeline flows
- `.env.example` files with placeholder configuration only

## Excluded from submission

- `.env` files with secrets
- virtual environments
- `node_modules`
- cache directories
- generated ChromaDB data
- temporary build output

## Setup

Backend:

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Frontend:

```bash
cd frontend
npm install
npm run dev
```

## Status

The project structure is ready for PDF RAG development, with frontend UI and backend pipeline components in place.
