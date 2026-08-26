from __future__ import annotations

from pathlib import Path

from pdf_processor import chunk_pages, extract_pages
from vector_store import (
    CHROMA_COLLECTION_NAME,
    CHROMA_PERSIST_DIRECTORY,
    add_chunks,
    get_embedding_model,
    initialize_vector_store,
    query_chunks,
)


DEFAULT_PDF = (
    Path(__file__).parent
    / "data"
    / "uploads"
    / "116456321_FY25-26_AnnualLoanStatement (1).pdf"
)


QUESTIONS = [
    "What is the EMI amount?",
    "What is the annual percentage rate?",
    "What is the interest rate?",
    "Which EMIs were unpaid?",
    "Which PDCs were cleared?",
]


def _preview(text: str, limit: int = 240) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[:limit] + "..."


def _print_results(question: str, results: list[dict]) -> None:
    print(f"QUESTION:\n{question}\n")
    for index, result in enumerate(results, start=1):
        metadata = result["metadata"]
        print(f"RESULT {index}:")
        print(f"Page: {metadata['page_number']}")
        print(f"Chunk ID: {metadata['chunk_id']}")
        print(f"Distance: {result['distance']}")
        print(f"Text: {result['document']}")
        print()


def main() -> int:
    pdf_path = DEFAULT_PDF

    model = get_embedding_model()
    print(f"Embedding model loaded successfully: {model.__class__.__name__}")
    print(f"Collection name: {CHROMA_COLLECTION_NAME}")
    print(f"Chroma persistence path: {CHROMA_PERSIST_DIRECTORY}")
    print()

    pages = extract_pages(pdf_path)
    chunks = chunk_pages(
        pages,
        chunk_size=800,
        chunk_overlap=150,
        source=pdf_path.name,
    )

    indexed_count = add_chunks(chunks)
    print(f"Number of chunks indexed: {indexed_count}")
    print(f"Total chunks (input): {len(chunks)}")
    print()

    store = initialize_vector_store(force_new=True)
    print(f"Collection count after index: {store.collection.count()}")
    print()

    for question in QUESTIONS:
        results = query_chunks(question, top_k=5)
        _print_results(question, results)

    fresh_store = initialize_vector_store(force_new=True)
    persisted_count = fresh_store.collection.count()
    persistence_verified = persisted_count >= len(chunks)

    print(f"Collection count from fresh initialize: {persisted_count}")
    print(f"Persistence verified: {persistence_verified}")
    print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
