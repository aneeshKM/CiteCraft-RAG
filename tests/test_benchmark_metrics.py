import pytest

from benchmarks.metrics import bootstrap_mean_interval, percentile, ranking_metrics


def test_ranking_metrics_match_sources_and_pages_case_insensitively():
    retrieved = [("OTHER.pdf", 1), ("REPORT.PDF", 7), ("report.pdf", 9)]
    relevant = {("report.pdf", 7), ("report.pdf", 8)}

    metrics = ranking_metrics(retrieved, relevant, k=3)

    assert metrics["hit"] == 1.0
    assert metrics["recall"] == 0.5
    assert metrics["mrr"] == 0.5
    assert metrics["ndcg"] == pytest.approx(0.3868528072)


def test_ranking_metrics_return_zero_without_a_match():
    metrics = ranking_metrics([("wrong.pdf", 1)], {("report.pdf", 2)}, k=5)

    assert metrics == {"hit": 0.0, "recall": 0.0, "mrr": 0.0, "ndcg": 0.0}


def test_ranking_metrics_only_credit_a_relevant_page_once():
    retrieved = [("report.pdf", 7), ("REPORT.PDF", 7), ("report.pdf", 8)]
    relevant = {("report.pdf", 7), ("report.pdf", 8)}

    metrics = ranking_metrics(retrieved, relevant, k=3)

    assert metrics["hit"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["mrr"] == 1.0
    assert metrics["ndcg"] <= 1.0


def test_percentile_interpolates_and_bootstrap_is_reproducible():
    assert percentile([1, 2, 3, 4], 0.5) == 2.5
    assert bootstrap_mean_interval([0, 1, 1], samples=100) == bootstrap_mean_interval(
        [0, 1, 1], samples=100
    )
