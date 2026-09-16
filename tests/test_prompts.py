from citecraft.models import ExtractedElement, RetrievedElement
from citecraft.prompts import build_context


def _retrieved(text: str, rank: int = 1) -> RetrievedElement:
    return RetrievedElement(
        element=ExtractedElement(
            parent_id=f"id-{rank}",
            text=text,
            source="annual-report.pdf",
            file_hash="hash",
            page=7,
            kind="text",
        ),
        rank=rank,
    )


def test_build_context_keeps_citations_aligned():
    context, citations = build_context([_retrieved("Revenue was $10M.")], max_characters=500)

    assert "[1] Source: annual-report.pdf | Page: 7" in context
    assert citations[0].label == "[1] annual-report.pdf — p. 7 (text)"


def test_build_context_respects_character_budget():
    context, citations = build_context([_retrieved("x" * 500)], max_characters=100)

    assert len(context) <= 100
    assert len(citations) == 1

