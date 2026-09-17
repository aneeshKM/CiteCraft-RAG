from benchmarks.run_financebench import relevant_pages


def test_relevant_pages_supports_the_pinned_financebench_schema():
    question = {
        "evidence": [
            {
                "doc_name": "REPORT_2024_10K",
                "evidence_page_num": 6,
            }
        ]
    }

    assert relevant_pages(question, page_offset=1) == {("REPORT_2024_10K.pdf", 7)}


def test_relevant_pages_supports_the_documented_financebench_schema():
    question = {
        "evidence": [
            {
                "evidence_doc_name": "REPORT_2024_10K",
                "evidence_page_num": 6,
            }
        ]
    }

    assert relevant_pages(question, page_offset=1) == {("REPORT_2024_10K.pdf", 7)}
