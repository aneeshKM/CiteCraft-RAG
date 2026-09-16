from citecraft.config import Settings
from citecraft.service import RagService


class EmptyRepository:
    def ping(self):
        return None

    def search(self, query, *, top_k, allowed_hashes):
        return []


class UnusedSummarizer:
    def summarize(self, elements):
        raise AssertionError("Summarization should not run while answering")


def test_answer_has_explicit_no_evidence_path():
    service = RagService(
        Settings(openai_api_key="test-key"),
        repository=EmptyRepository(),
        summarizer=UnusedSummarizer(),
    )

    answer = service.answer("What is the revenue?", allowed_hashes=["hash"])

    assert "could not find relevant content" in answer.text
    assert answer.sources == ()

