from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any
import hashlib
import math
import re

import chromadb
from sentence_transformers import SentenceTransformer


EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHROMA_COLLECTION_NAME = "pdf_chunks"
CHROMA_PERSIST_DIRECTORY = Path(__file__).parent / "data" / "chroma_db"
FALLBACK_EMBEDDING_DIMENSIONS = 384

_VECTOR_STORE_SERVICE: "VectorStoreService | None" = None


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    """Load the embedding model once and reuse it."""

    return SentenceTransformer(EMBEDDING_MODEL_NAME, local_files_only=True)


def _normalize_vector(vector: list[float]) -> list[float]:
    magnitude = math.sqrt(sum(value * value for value in vector))
    if magnitude == 0:
        return vector
    return [value / magnitude for value in vector]


def _fallback_embed_text(text: str) -> list[float]:
    """Create a deterministic local embedding when SentenceTransformer is unavailable."""

    vector = [0.0] * FALLBACK_EMBEDDING_DIMENSIONS
    tokens = re.findall(r"[A-Za-z0-9]+", text.lower())

    for token in tokens:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "big") % FALLBACK_EMBEDDING_DIMENSIONS
        weight = 1.0 + (len(token) % 7) * 0.1
        vector[index] += weight

    return _normalize_vector(vector)


def _embed_texts(texts: list[str]) -> list[list[float]]:
    try:
        model = get_embedding_model()
    except Exception:
        return [_fallback_embed_text(text) for text in texts]

    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return embeddings.tolist()


def _embed_query(query: str) -> list[float]:
    return _embed_texts([query])[0]


@dataclass
class VectorStoreService:
    collection_name: str = CHROMA_COLLECTION_NAME
    persist_directory: Path = CHROMA_PERSIST_DIRECTORY

    def __post_init__(self) -> None:
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(self.persist_directory))
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(self, chunks: list[dict[str, Any]]) -> int:
        if not chunks:
            return 0

        ids = [chunk["chunk_id"] for chunk in chunks]
        documents = [chunk["text"] for chunk in chunks]
        metadatas = [
            {
                "page_number": int(chunk["page_number"]),
                "chunk_id": chunk["chunk_id"],
                "source": chunk["source"],
            }
            for chunk in chunks
        ]
        embeddings = _embed_texts(documents)

        self.collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
        )
        return len(chunks)

    def query_chunks(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        results = self.collection.query(
            query_embeddings=[_embed_query(query)],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        items: list[dict[str, Any]] = []
        ids = results.get("ids", [[]])[0]
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        for item_id, document, metadata, distance in zip(
            ids, documents, metadatas, distances, strict=False
        ):
            items.append(
                {
                    "id": item_id,
                    "document": document,
                    "metadata": metadata,
                    "distance": distance,
                }
            )

        return items

    def clear_collection(self) -> None:
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )


def initialize_vector_store(force_new: bool = False) -> VectorStoreService:
    global _VECTOR_STORE_SERVICE

    if force_new or _VECTOR_STORE_SERVICE is None:
        _VECTOR_STORE_SERVICE = VectorStoreService()

    return _VECTOR_STORE_SERVICE


def add_chunks(chunks: list[dict[str, Any]]) -> int:
    return initialize_vector_store().add_chunks(chunks)


def query_chunks(query: str, top_k: int = 5) -> list[dict[str, Any]]:
    return initialize_vector_store().query_chunks(query, top_k=top_k)


def clear_collection() -> None:
    initialize_vector_store().clear_collection()
