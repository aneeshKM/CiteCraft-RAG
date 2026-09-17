from langchain_core.messages import AIMessage

from citecraft.config import Settings
from citecraft.models import ExtractedElement, RetrievedElement
from citecraft.service import RagService


class EmptyRepository:
    def ping(self):
        return None

    def search(self, query, *, top_k, allowed_hashes):
        return []


class UnusedSummarizer:
    def summarize(self, elements):
        raise AssertionError("Summarization should not run while answering")


class EvidenceRepository(EmptyRepository):
    def search(self, query, *, top_k, allowed_hashes):
        return [
            RetrievedElement(
                element=ExtractedElement(
                    parent_id="parent-1",
                    text="Revenue was $10 million.",
                    source="report.pdf",
                    file_hash="hash",
                    page=3,
                    kind="text",
                ),
                rank=1,
            )
        ]


def test_answer_has_explicit_no_evidence_path():
    service = RagService(
        Settings(openai_api_key="test-key"),
        repository=EmptyRepository(),
        summarizer=UnusedSummarizer(),
    )

    answer = service.answer("What is the revenue?", allowed_hashes=["hash"])

    assert "could not find relevant content" in answer.text
    assert answer.sources == ()


def test_answer_requests_are_stored_with_observability_metadata(monkeypatch):
    created_with = {}
    invoked_with = {}

    class FakeChatOpenAI:
        def __init__(self, **kwargs):
            created_with.update(kwargs)

        def invoke(self, messages, **kwargs):
            invoked_with.update(kwargs)
            return AIMessage(content="Revenue was $10 million. [1]")

    monkeypatch.setattr("langchain_openai.ChatOpenAI", FakeChatOpenAI)
    service = RagService(
        Settings(openai_api_key="test-key"),
        repository=EvidenceRepository(),
        summarizer=UnusedSummarizer(),
    )

    service.answer("What was revenue?", allowed_hashes=["hash"])

    assert created_with["model_kwargs"] == {"store": True}
    assert invoked_with["metadata"] == {
        "application": "citecraft",
        "operation": "rag_answer",
    }
