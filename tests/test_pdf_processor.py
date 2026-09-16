from citecraft.pdf_processor import normalize_text


def test_normalize_text_collapses_pdf_whitespace():
    assert normalize_text("  Revenue\n\n  increased\t 12%  ") == "Revenue increased 12%"

