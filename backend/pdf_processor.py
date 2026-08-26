from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pymupdf as fitz
from langchain_text_splitters import RecursiveCharacterTextSplitter


class PDFProcessorError(Exception):
    """Base exception for PDF extraction and chunking failures."""


class PDFReadError(PDFProcessorError):
    """Raised when a PDF cannot be opened or read."""


class PDFContentError(PDFProcessorError):
    """Raised when a PDF page does not contain usable text."""


def _coerce_path(pdf_path: str | Path) -> Path:
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {path}")
    if not path.is_file():
        raise PDFReadError(f"PDF path is not a file: {path}")
    return path


def _clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_pages(pdf_path: str | Path) -> list[dict[str, Any]]:
    """Extract cleaned text from a PDF, preserving page numbers."""

    path = _coerce_path(pdf_path)

    try:
        with fitz.open(path) as doc:
            if doc.page_count == 0:
                raise PDFContentError(f"PDF contains no pages: {path}")

            pages: list[dict[str, Any]] = []
            for index in range(doc.page_count):
                page_number = index + 1
                page = doc.load_page(index)
                raw_text = page.get_text("text")
                cleaned_text = _clean_text(raw_text)

                if not cleaned_text:
                    raise PDFContentError(
                        f"Page {page_number} contains no extractable text in {path}"
                    )

                pages.append(
                    {
                        "page_number": page_number,
                        "text": cleaned_text,
                    }
                )

            return pages
    except (fitz.FileDataError, fitz.EmptyFileError, RuntimeError) as exc:
        raise PDFReadError(f"Could not read PDF {path}: {exc}") from exc
    except PDFProcessorError:
        raise
    except Exception as exc:  # pragma: no cover - safety net for unexpected fitz errors
        raise PDFReadError(f"Unexpected PDF processing error for {path}: {exc}") from exc


def validate_chunks(chunks: list[dict[str, Any]]) -> None:
    """Validate chunk structure and uniqueness."""

    chunk_ids: set[str] = set()
    for chunk in chunks:
        chunk_id = chunk.get("chunk_id")
        page_number = chunk.get("page_number")
        text = chunk.get("text")

        if not chunk_id or not isinstance(chunk_id, str):
            raise PDFProcessorError("Every chunk must include a non-empty chunk_id.")
        if not isinstance(page_number, int):
            raise PDFProcessorError(f"Chunk {chunk_id} must include an integer page_number.")
        if not text or not isinstance(text, str) or not text.strip():
            raise PDFProcessorError(f"Chunk {chunk_id} must include non-empty text.")
        if chunk_id in chunk_ids:
            raise PDFProcessorError(f"Duplicate chunk_id detected: {chunk_id}")

        chunk_ids.add(chunk_id)


def chunk_pages(
    pages: list[dict[str, Any]],
    chunk_size: int = 800,
    chunk_overlap: int = 150,
    source: str = "document.pdf",
) -> list[dict[str, Any]]:
    """Chunk each PDF page independently while preserving page boundaries."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap cannot be negative.")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    chunks: list[dict[str, Any]] = []
    for page in pages:
        page_number = page.get("page_number")
        page_text = page.get("text", "")

        if not isinstance(page_number, int):
            raise PDFProcessorError("Each page must include an integer page_number.")
        if not isinstance(page_text, str) or not page_text.strip():
            raise PDFProcessorError(f"Page {page_number} must include non-empty text.")

        page_chunks = splitter.split_text(page_text)
        for index, chunk_text in enumerate(page_chunks, start=1):
            cleaned_chunk_text = _clean_text(chunk_text)
            if not cleaned_chunk_text:
                continue

            chunks.append(
                {
                    "chunk_id": f"page{page_number}_chunk{index}",
                    "page_number": page_number,
                    "source": source,
                    "text": cleaned_chunk_text,
                }
            )

    validate_chunks(chunks)
    return chunks
