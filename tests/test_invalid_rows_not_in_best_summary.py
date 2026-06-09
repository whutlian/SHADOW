from __future__ import annotations

from shadow_hgc.audit.reporting import best_rows_by_dataset


def test_invalid_rows_are_not_eligible_for_best_summary():
    rows = [
        {"dataset": "acm", "variant": "bad", "status": "invalid_config", "accuracy": 0.99},
        {"dataset": "acm", "variant": "good", "status": "completed", "accuracy": 0.75},
        {"dataset": "dblp", "variant": "timeout", "status": "timeout_dropped", "accuracy": ""},
    ]

    best = best_rows_by_dataset(rows)

    assert best["acm"]["variant"] == "good"
    assert "dblp" not in best
