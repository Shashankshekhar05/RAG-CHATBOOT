from __future__ import annotations

from dataclasses import dataclass
import os
from statistics import mean
from typing import Any

from dotenv import load_dotenv

from vector_store import initialize_vector_store


load_dotenv()


@dataclass
class RetrievalService:
    """Thin retrieval layer over the persistent Chroma vector store."""

    def __post_init__(self) -> None:
        self.vector_store = initialize_vector_store()

    def retrieve(self, question: str, top_k: int | None = None) -> list[dict[str, Any]]:
        if top_k is None:
            top_k = int(os.getenv("TOP_K", "5"))

        raw_results = self.vector_store.query_chunks(question, top_k=top_k)
        results: list[dict[str, Any]] = []

        for result in raw_results:
            metadata = result["metadata"]
            results.append(
                {
                    "text": result["document"],
                    "page_number": metadata["page_number"],
                    "chunk_id": metadata["chunk_id"],
                    "source": metadata["source"],
                    "distance": result["distance"],
                }
            )

        return results

    def evaluate_relevance(self, results: list[dict[str, Any]]) -> dict[str, Any]:
        if not results:
            return {
                "best_distance": None,
                "worst_distance": None,
                "average_distance": None,
                "result_count": 0,
                "pages": [],
            }

        distances = [float(result["distance"]) for result in results]
        pages = sorted({int(result["page_number"]) for result in results})

        return {
            "best_distance": min(distances),
            "worst_distance": max(distances),
            "average_distance": mean(distances),
            "result_count": len(results),
            "pages": pages,
        }

    def build_context(self, results: list[dict[str, Any]]) -> str:
        seen_texts: set[str] = set()
        blocks: list[str] = []

        for result in results:
            text = result["text"].strip()
            if text in seen_texts:
                continue
            seen_texts.add(text)
            blocks.append(
                f"[Source: Page {result['page_number']}]\n"
                f"[Chunk: {result['chunk_id']}]\n\n"
                f"{text}"
            )

        return "\n\n".join(blocks)

    def get_source_pages(self, results: list[dict[str, Any]]) -> list[int]:
        return sorted({int(result["page_number"]) for result in results})
