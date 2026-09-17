"""LLM-generated retrieval summaries."""

from __future__ import annotations

import logging
from collections.abc import Sequence

from citecraft.models import ExtractedElement
from citecraft.prompts import SUMMARY_SYSTEM_PROMPT

LOGGER = logging.getLogger(__name__)


class ElementSummarizer:
    def __init__(
        self,
        *,
        model_name: str,
        api_key: str,
        max_concurrency: int = 4,
        store_responses: bool = True,
    ) -> None:
        from langchain_openai import ChatOpenAI

        self._model = ChatOpenAI(
            model=model_name,
            api_key=api_key,
            model_kwargs={"store": store_responses},
        )
        self._max_concurrency = max_concurrency

    def summarize(self, elements: Sequence[ExtractedElement]) -> list[str]:
        from langchain_core.messages import HumanMessage, SystemMessage

        results: list[str | None] = [None] * len(elements)
        pending_indexes: list[int] = []
        pending_messages: list[list[SystemMessage | HumanMessage]] = []

        for index, element in enumerate(elements):
            if len(element.text) <= 500:
                results[index] = element.text
                continue
            pending_indexes.append(index)
            pending_messages.append(
                [
                    SystemMessage(content=SUMMARY_SYSTEM_PROMPT),
                    HumanMessage(content=f"Element type: {element.kind}\n\n{element.text}"),
                ]
            )

        if pending_messages:
            try:
                responses = self._model.batch(
                    pending_messages,
                    config={"max_concurrency": self._max_concurrency},
                    metadata={
                        "application": "citecraft",
                        "operation": "retrieval_summary",
                    },
                )
                for index, response in zip(pending_indexes, responses, strict=True):
                    results[index] = str(response.content)
            except Exception:
                LOGGER.exception("Summary generation failed; indexing raw element text instead.")
                for index in pending_indexes:
                    results[index] = elements[index].text

        return [result or elements[index].text for index, result in enumerate(results)]
