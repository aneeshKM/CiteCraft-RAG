from langchain_core.messages import AIMessage

from citecraft.models import ExtractedElement
from citecraft.summarizer import ElementSummarizer


def test_summary_requests_are_stored_with_observability_metadata(monkeypatch):
    created_with = {}
    batched_with = {}

    class FakeChatOpenAI:
        def __init__(self, **kwargs):
            created_with.update(kwargs)

        def batch(self, messages, **kwargs):
            batched_with.update(kwargs)
            return [AIMessage(content="A retrieval summary.")]

    monkeypatch.setattr("langchain_openai.ChatOpenAI", FakeChatOpenAI)
    summarizer = ElementSummarizer(
        model_name="gpt-4o-mini",
        api_key="test-key",
        store_responses=True,
    )
    element = ExtractedElement(
        parent_id="parent-1",
        text="A" * 501,
        source="report.pdf",
        file_hash="hash",
        page=1,
        kind="text",
    )

    summaries = summarizer.summarize([element])

    assert summaries == ["A retrieval summary."]
    assert created_with["model_kwargs"] == {"store": True}
    assert batched_with["metadata"] == {
        "application": "citecraft",
        "operation": "retrieval_summary",
    }
