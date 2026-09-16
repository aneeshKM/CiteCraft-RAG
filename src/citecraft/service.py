"""Application service orchestrating ingestion, retrieval, and generation."""

from __future__ import annotations

import time
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

from citecraft.config import Settings
from citecraft.hashing import sha256_file
from citecraft.models import Answer, IngestionResult, RetrievedElement
from citecraft.pdf_processor import extract_pdf_elements
from citecraft.prompts import ANSWER_SYSTEM_PROMPT, build_context
from citecraft.storage import DocumentRepository
from citecraft.summarizer import ElementSummarizer


class Repository(Protocol):
    def ping(self) -> None: ...

    def indexed_metadata(self, file_hash: str) -> dict[str, object] | None: ...

    def add(self, elements: Sequence, summaries: Sequence[str]) -> None: ...

    def mark_indexed(self, *, file_hash: str, filename: str, element_count: int) -> None: ...

    def search(
        self, query: str, *, top_k: int, allowed_hashes: Sequence[str]
    ) -> list[RetrievedElement]: ...


class RagService:
    def __init__(
        self,
        settings: Settings,
        *,
        repository: Repository | None = None,
        summarizer: ElementSummarizer | None = None,
    ) -> None:
        settings.validate()
        self.settings = settings
        self.repository = repository or DocumentRepository(settings)
        self.summarizer = summarizer or ElementSummarizer(
            model_name=settings.chat_model,
            api_key=settings.openai_api_key,
            max_concurrency=settings.summary_concurrency,
        )
        self._answer_model = None

    def check_connections(self) -> None:
        self.repository.ping()

    def ingest(self, path: str | Path, *, filename: str) -> IngestionResult:
        file_hash = sha256_file(path)
        cached = self.repository.indexed_metadata(file_hash)
        if cached:
            return IngestionResult(
                file_hash=file_hash,
                filename=str(cached.get("filename", filename)),
                element_count=int(cached.get("element_count", 0)),
                was_cached=True,
            )

        elements = extract_pdf_elements(
            path,
            filename=filename,
            file_hash=file_hash,
            strategy=self.settings.partition_strategy,
        )
        summaries = self.summarizer.summarize(elements)
        self.repository.add(elements, summaries)
        self.repository.mark_indexed(
            file_hash=file_hash,
            filename=filename,
            element_count=len(elements),
        )
        return IngestionResult(
            file_hash=file_hash,
            filename=filename,
            element_count=len(elements),
            was_cached=False,
        )

    def answer(
        self,
        question: str,
        *,
        allowed_hashes: Sequence[str],
        chat_history: Sequence[dict[str, str]] = (),
    ) -> Answer:
        start = time.perf_counter()
        retrieved = self.repository.search(
            question,
            top_k=self.settings.top_k,
            allowed_hashes=allowed_hashes,
        )
        context, sources = build_context(
            retrieved,
            max_characters=self.settings.max_context_characters,
        )
        if not context:
            return Answer(
                text="I could not find relevant content in the selected documents.",
                sources=(),
                latency_seconds=time.perf_counter() - start,
            )

        from langchain_core.messages import HumanMessage, SystemMessage
        from langchain_openai import ChatOpenAI

        history_text = _format_history(chat_history[-6:])
        user_prompt = (
            f"Document context:\n{context}\n\n"
            f"Recent conversation:\n{history_text or '(none)'}\n\n"
            f"Question: {question}"
        )
        if self._answer_model is None:
            self._answer_model = ChatOpenAI(
                model=self.settings.chat_model,
                api_key=self.settings.openai_api_key,
            )
        response = self._answer_model.invoke(
            [
                SystemMessage(content=ANSWER_SYSTEM_PROMPT),
                HumanMessage(content=user_prompt),
            ]
        )
        return Answer(
            text=str(response.content),
            sources=sources,
            latency_seconds=time.perf_counter() - start,
        )


def _format_history(messages: Sequence[dict[str, str]]) -> str:
    return "\n".join(
        f"{message.get('role', 'user').title()}: {message.get('content', '')}"
        for message in messages
        if message.get("content")
    )
