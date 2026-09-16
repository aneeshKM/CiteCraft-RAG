"""Domain models shared by ingestion, storage, and generation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ExtractedElement:
    parent_id: str
    text: str
    source: str
    file_hash: str
    page: int | None
    kind: str

    @property
    def citation(self) -> str:
        location = f"page {self.page}" if self.page else "page unknown"
        return f"{self.source}, {location}"


@dataclass(frozen=True, slots=True)
class RetrievedElement:
    element: ExtractedElement
    rank: int


@dataclass(frozen=True, slots=True)
class SourceCitation:
    index: int
    source: str
    page: int | None
    kind: str

    @property
    def label(self) -> str:
        page_label = f"p. {self.page}" if self.page else "page unavailable"
        return f"[{self.index}] {self.source} — {page_label} ({self.kind})"


@dataclass(frozen=True, slots=True)
class IngestionResult:
    file_hash: str
    filename: str
    element_count: int
    was_cached: bool


@dataclass(frozen=True, slots=True)
class Answer:
    text: str
    sources: tuple[SourceCitation, ...]
    latency_seconds: float

