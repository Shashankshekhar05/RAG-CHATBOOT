from __future__ import annotations

from pathlib import Path

from llm_service import LLMConfigurationError, LLMGenerationError, LLMService, passes_relevance_gate
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


def _build_result_payload(question: str, results: list[dict], service: RetrievalService) -> None:
    best_distance = results[0]["distance"] if results else None
    accepted = passes_relevance_gate(best_distance)
    context = service.build_context(results)
    source_pages = service.get_source_pages(results)

    print(f"QUESTION:\n{question}\n")
    print(f"RETRIEVED PAGES: {source_pages}")
    print(f"BEST DISTANCE: {best_distance}")
    print(f"RELEVANCE: {'ACCEPTED' if accepted else 'REJECTED'}")
    print(f"CONTEXT:\n{context}\n")

    if not accepted:
        print("ANSWER:")
        print("The information is not available in the PDF.")
        print()
        print(f"SOURCE PAGES: {source_pages}")
        print("=" * 80)
        print()
        return

    try:
        llm = LLMService()
        answer = llm.generate_answer(question, context)
        print("ANSWER:")
        print(answer)
    except LLMConfigurationError as exc:
        print("ANSWER:")
        print(f"Configuration error: {exc}")
    except LLMGenerationError as exc:
        print("ANSWER:")
        print(f"Generation error: {exc}")

    print()
    print(f"SOURCE PAGES: {source_pages}")
    print("=" * 80)
    print()


def main() -> int:
    pdf_path = DEFAULT_PDF
    retrieval_service = RetrievalService()

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
        try:
            results = retrieval_service.retrieve(question, top_k=5)
        except Exception as exc:  # pragma: no cover - retrieval failure path
            print(f"QUESTION:\n{question}\n")
            print("ANSWER:")
            print(f"Retrieval error: {exc}")
            print()
            print("=" * 80)
            print()
            continue

        _build_result_payload(question, results, retrieval_service)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
