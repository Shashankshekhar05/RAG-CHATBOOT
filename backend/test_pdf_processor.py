from __future__ import annotations

import sys
from pathlib import Path

from pdf_processor import chunk_pages, extract_pages


DEFAULT_PDF = (
    Path(__file__).parent
    / "data"
    / "uploads"
    / "116456321_FY25-26_AnnualLoanStatement (1).pdf"
)


def _preview(text: str, limit: int = 120) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[:limit] + "..."


def main() -> int:
    pdf_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PDF

    pages = extract_pages(pdf_path)
    print(f"Pages extracted: {len(pages)}")
    print()

    page_text_paths = {
        1: Path(__file__).parent / "data" / "page1_extracted.txt",
        2: Path(__file__).parent / "data" / "page2_extracted.txt",
    }

    for page in pages:
        print(f"PAGE {page['page_number']}")
        print(f"Total characters: {len(page['text'])}")
        print(f"Preview: {_preview(page['text'])}")
        print()

        page_text_path = page_text_paths.get(page["page_number"])
        if page_text_path is not None:
            page_text_path.write_text(page["text"], encoding="utf-8")

    chunks = chunk_pages(
        pages,
        chunk_size=800,
        chunk_overlap=150,
        source=pdf_path.name,
    )

    print(f"Total chunks: {len(chunks)}")
    print()

    chunk_counts_by_page: dict[int, int] = {}
    for chunk in chunks:
        page_number = chunk["page_number"]
        chunk_counts_by_page[page_number] = chunk_counts_by_page.get(page_number, 0) + 1

    for page_number in sorted(chunk_counts_by_page):
        print(f"Page {page_number} chunk count: {chunk_counts_by_page[page_number]}")
    print()

    for chunk in chunks:
        print(chunk["chunk_id"])
        print(f"Page: {chunk['page_number']}")
        print(f"Preview: {_preview(chunk['text'])}")
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
