"""PDF partitioning with text and table metadata preservation."""

from __future__ import annotations

import re
from pathlib import Path

from citecraft.hashing import stable_parent_id
from citecraft.models import ExtractedElement


def extract_pdf_elements(
    path: str | Path,
    *,
    filename: str,
    file_hash: str,
    strategy: str,
) -> list[ExtractedElement]:
    """Partition a PDF into citation-ready text and table elements."""
    try:
        from unstructured.partition.pdf import partition_pdf
    except ImportError as exc:  # pragma: no cover - installation-specific
        raise RuntimeError(
            "PDF support is not installed. Run `pip install -r requirements.txt`."
        ) from exc

    elements = partition_pdf(
        filename=str(path),
        strategy=strategy,
        infer_table_structure=strategy in {"auto", "hi_res"},
        chunking_strategy="by_title",
        max_characters=8_000,
        new_after_n_chars=4_000,
        combine_text_under_n_chars=1_000,
    )

    extracted: list[ExtractedElement] = []
    for position, element in enumerate(elements):
        category = str(getattr(element, "category", type(element).__name__))
        if category.lower() in {"header", "footer", "pagenumber"}:
            continue

        metadata = getattr(element, "metadata", None)
        table_html = getattr(metadata, "text_as_html", None)
        raw_text = table_html if category.lower() == "table" and table_html else str(element)
        text = normalize_text(raw_text)
        if not text:
            continue

        page = getattr(metadata, "page_number", None)
        extracted.append(
            ExtractedElement(
                parent_id=stable_parent_id(file_hash, position, text),
                text=text,
                source=filename,
                file_hash=file_hash,
                page=int(page) if page else None,
                kind="table" if category.lower() == "table" else "text",
            )
        )

    if not extracted:
        raise ValueError(
            "No readable text or tables were found. Try PARTITION_STRATEGY=ocr_only "
            "for a scanned PDF."
        )
    return extracted


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()

