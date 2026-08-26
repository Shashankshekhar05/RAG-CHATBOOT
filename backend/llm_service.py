from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI


DOTENV_PATH = Path(__file__).resolve().with_name(".env")
load_dotenv(dotenv_path=DOTENV_PATH)

PDF_ONLY_SYSTEM_PROMPT = """You are a PDF question-answering assistant.

Answer the user's question ONLY using the provided PDF context.

Rules:

1. Use only the provided PDF context.
2. Do not use outside knowledge.
3. Do not guess.
4. Do not invent facts.
5. Do not infer unsupported numerical values.
6. Preserve numbers exactly as they appear in the context.
7. If the answer is not available in the provided PDF context, say exactly:

The information is not available in the PDF.

8. Keep the answer concise and directly answer the user's question."""


class LLMConfigurationError(RuntimeError):
    """Raised when Grok/OpenAI-compatible configuration is incomplete."""


class LLMGenerationError(RuntimeError):
    """Raised when the LLM call fails or returns an empty/invalid response."""


def get_retrieval_threshold(default: float = 0.736) -> float:
    raw_value = os.getenv("RETRIEVAL_THRESHOLD", "").strip()
    if not raw_value:
        return default

    try:
        return float(raw_value)
    except ValueError as exc:
        raise LLMConfigurationError(
            "RETRIEVAL_THRESHOLD must be a valid numeric distance value."
        ) from exc


def passes_relevance_gate(best_distance: float | None, threshold: float | None = None) -> bool:
    if best_distance is None:
        return False

    if threshold is None:
        threshold = get_retrieval_threshold()

    return best_distance <= threshold


@dataclass
class LLMService:
    client: OpenAI = field(init=False)
    api_key: str = field(init=False)
    base_url: str = field(init=False)
    model: str = field(init=False)

    def __post_init__(self) -> None:
        self.api_key = os.getenv("LLM_API_KEY", "").strip()
        self.base_url = os.getenv("LLM_BASE_URL", "").strip()
        self.model = os.getenv("LLM_MODEL", "").strip()

        missing = [
            name
            for name, value in (
                ("LLM_API_KEY", self.api_key),
                ("LLM_BASE_URL", self.base_url),
                ("LLM_MODEL", self.model),
            )
            if not value
        ]
        if missing:
            raise LLMConfigurationError(
                "Missing required LLM configuration: " + ", ".join(missing)
            )

        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def generate_answer(self, question: str, context: str) -> str:
        if not context.strip():
            return "The information is not available in the PDF."

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=0,
                messages=[
                    {"role": "system", "content": PDF_ONLY_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": f"Question:\n{question}\n\nPDF Context:\n{context}",
                    },
                ],
            )
        except Exception as exc:  # pragma: no cover - external API/network failures
            raise LLMGenerationError(f"Grok request failed: {exc}") from exc

        try:
            choice = response.choices[0]
            message = choice.message.content if choice.message else None
        except Exception as exc:  # pragma: no cover - defensive parsing
            raise LLMGenerationError("Grok returned an invalid response payload.") from exc

        if not message or not message.strip():
            raise LLMGenerationError("Grok returned an empty answer.")

        return message.strip()

    def stream_answer(self, question: str, context: str):
        if not context.strip():
            yield "The information is not available in the PDF."
            return

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=0,
                stream=True,
                messages=[
                    {"role": "system", "content": PDF_ONLY_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": f"Question:\n{question}\n\nPDF Context:\n{context}",
                    },
                ],
            )
            emitted = False
            for chunk in response:
                delta = chunk.choices[0].delta.content if chunk.choices else None
                if delta:
                    emitted = True
                    yield delta
        except Exception as exc:  # pragma: no cover - external API/network failures
            raise LLMGenerationError(f"Grok request failed: {exc}") from exc

        if not emitted:
            raise LLMGenerationError("Grok returned an empty answer.")
