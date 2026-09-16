from citecraft.models import ExtractedElement
from citecraft.storage import (
    _element_from_dict,
    _element_to_dict,
    _normalize_postgres_url,
    _vector_literal,
)


def test_element_serialization_round_trip():
    element = ExtractedElement(
        parent_id="parent-1",
        text="A source-grounded fact.",
        source="paper.pdf",
        file_hash="hash-1",
        page=3,
        kind="table",
    )

    assert _element_from_dict(_element_to_dict(element)) == element


def test_postgres_url_is_normalized_for_psycopg():
    value = "postgresql+psycopg://user:pass@localhost:6024/db"

    assert _normalize_postgres_url(value) == "postgresql://user:pass@localhost:6024/db"


def test_vector_literal_uses_pgvector_format():
    assert _vector_literal([0.5, -1, 0.00025]) == "[0.5,-1,0.00025]"
