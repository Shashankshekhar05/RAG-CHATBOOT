import json
from pathlib import Path
from typing import Any, AsyncIterator

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from llm_service import LLMGenerationError, LLMService, passes_relevance_gate
from pdf_processor import chunk_pages, extract_pages
from retrieval_service import RetrievalService
from vector_store import add_chunks


UPLOAD_DIRECTORY = Path(__file__).parent / "data" / "uploads"
retrieval_service = RetrievalService()


class ChatRequest(BaseModel):
    question: str

app = FastAPI(
    title="RAG Chatbot API",
    description="Backend scaffold for a PDF-based RAG chatbot.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)) -> dict[str, Any]:
    filename = Path(file.filename or "").name
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    UPLOAD_DIRECTORY.mkdir(parents=True, exist_ok=True)
    destination = UPLOAD_DIRECTORY / filename
    content = await file.read()
    destination.write_bytes(content)

    try:
        pages = extract_pages(destination)
        chunks = chunk_pages(pages, source=filename)
        indexed = add_chunks(chunks)
    except Exception as exc:
        destination.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=f"Could not process PDF: {exc}") from exc

    return {
        "message": "PDF processed successfully",
        "filename": filename,
        "pages": len(pages),
        "chunks": indexed,
    }


def _sse_event(event: str, payload: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


async def _stream_chat(question: str) -> AsyncIterator[str]:
    try:
        results = retrieval_service.retrieve(question)
        best_distance = results[0]["distance"] if results else None
        if not passes_relevance_gate(best_distance):
            yield _sse_event(
                "done",
                {
                    "answer": "The information is not available in the PDF.",
                    "source_pages": [],
                },
            )
            return

        context = retrieval_service.build_context(results)
        source_pages = retrieval_service.get_source_pages(results)
        llm = LLMService()
        answer_parts: list[str] = []
        for part in llm.stream_answer(question, context):
            answer_parts.append(part)
            yield _sse_event("token", {"content": part})

        yield _sse_event(
            "done",
            {"answer": "".join(answer_parts), "source_pages": source_pages},
        )
    except Exception as exc:
        message = str(exc) if isinstance(exc, LLMGenerationError) else "Chat request failed."
        yield _sse_event("error", {"message": message})


@app.post("/chat")
async def chat(request: ChatRequest) -> StreamingResponse:
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    return StreamingResponse(
        _stream_chat(question),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )
