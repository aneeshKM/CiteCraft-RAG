"""Deterministic ranking and aggregation metrics for RAG evaluation."""

from __future__ import annotations

import math
import random
from collections.abc import Iterable, Sequence

SourcePage = tuple[str, int | None]


def normalize_source(value: str) -> str:
    """Normalize source names for case-insensitive page matching."""
    return value.rsplit("/", maxsplit=1)[-1].casefold()


def ranking_metrics(
    retrieved: Sequence[SourcePage],
    relevant: Iterable[SourcePage],
    *,
    k: int,
) -> dict[str, float]:
    """Calculate binary page-level retrieval metrics at a fixed cutoff."""
    if k < 1:
        raise ValueError("k must be greater than zero")

    normalized_relevant = {
        (normalize_source(source), page) for source, page in relevant
    }
    if not normalized_relevant:
        return {"hit": 0.0, "recall": 0.0, "mrr": 0.0, "ndcg": 0.0}

    normalized_retrieved = [
        (normalize_source(source), page) for source, page in retrieved[:k]
    ]
    seen_relevant: set[SourcePage] = set()
    relevance: list[float] = []
    for item in normalized_retrieved:
        is_new_relevant = item in normalized_relevant and item not in seen_relevant
        relevance.append(1.0 if is_new_relevant else 0.0)
        if is_new_relevant:
            seen_relevant.add(item)
    unique_relevant_retrieved = set(normalized_retrieved) & normalized_relevant

    first_relevant_rank = next(
        (rank for rank, score in enumerate(relevance, start=1) if score),
        None,
    )
    dcg = sum(score / math.log2(rank + 1) for rank, score in enumerate(relevance, start=1))
    ideal_count = min(len(normalized_relevant), k)
    ideal_dcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_count + 1))

    return {
        "hit": float(bool(unique_relevant_retrieved)),
        "recall": len(unique_relevant_retrieved) / len(normalized_relevant),
        "mrr": 1.0 / first_relevant_rank if first_relevant_rank else 0.0,
        "ndcg": dcg / ideal_dcg if ideal_dcg else 0.0,
    }


def percentile(values: Sequence[float], probability: float) -> float:
    """Return a linearly interpolated percentile without an external dependency."""
    if not values:
        return 0.0
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be between zero and one")

    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def bootstrap_mean_interval(
    values: Sequence[float],
    *,
    confidence: float = 0.95,
    samples: int = 2_000,
    seed: int = 42,
) -> tuple[float, float]:
    """Calculate a deterministic percentile-bootstrap interval for a sample mean."""
    if not values:
        return (0.0, 0.0)
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between zero and one")
    if samples < 1:
        raise ValueError("samples must be greater than zero")

    generator = random.Random(seed)
    numeric_values = [float(value) for value in values]
    sample_size = len(numeric_values)
    means = [
        sum(generator.choice(numeric_values) for _ in range(sample_size)) / sample_size
        for _ in range(samples)
    ]
    tail = (1.0 - confidence) / 2.0
    return percentile(means, tail), percentile(means, 1.0 - tail)


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0
