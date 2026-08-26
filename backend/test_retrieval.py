from __future__ import annotations

from pathlib import Path

from pdf_processor import chunk_pages, extract_pages
from retrieval_service import RetrievalService
from vector_store import add_chunks


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
    "What is the customer's salary?",
    "What is the capital of France?",
]


def _print_result_block(index: int, result: dict[str, object]) -> None:
    print(f"RESULT #{index}")
    print(f"Chunk ID: {result['chunk_id']}")
    print(f"Page: {result['page_number']}")
    print(f"Distance: {result['distance']}")
    print(f"Text: {result['text']}")
    print()


def main() -> int:
    pdf_path = DEFAULT_PDF
    service = RetrievalService()

    pages = extract_pages(pdf_path)
    chunks = chunk_pages(
        pages,
        chunk_size=800,
        chunk_overlap=150,
        source=pdf_path.name,
    )
    indexed = add_chunks(chunks)

    print(f"Indexed chunks: {indexed}")
    print()

    for question in QUESTIONS:
        results = service.retrieve(question, top_k=5)
        diagnostics = service.evaluate_relevance(results)
        source_pages = service.get_source_pages(results)
        context = service.build_context(results)

        print(f"QUESTION:\n{question}\n")
        for index, result in enumerate(results, start=1):
            _print_result_block(index, result)

        print(f"Best distance: {diagnostics['best_distance']}")
        print(f"Worst distance: {diagnostics['worst_distance']}")
        print(f"Average distance: {diagnostics['average_distance']}")
        print(f"Result count: {diagnostics['result_count']}")
        print(f"Source pages: {source_pages}")
        print(f"Context:\n{context}")
        print("=" * 80)
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
