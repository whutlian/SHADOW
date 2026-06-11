from __future__ import annotations

import torch

from scripts.run_t29_reddit_bonsai import build_bonsai_rows
from scripts.run_t29_reddit_pltc import build_pltc_rows
from shadow_hgc.sft.bonsai_sft_sketch import build_bonsai_sketch, lsh_bonsai_select
from shadow_hgc.sft.pseudo_label_transport import build_pltc_soft_labels, confidence_bins, select_pltc_indices
from shadow_hgc.sft.t29_contract import validate_t29_row


def test_t29_pltc_flags_teacher_logits_and_forbidden_in_safe_track():
    args = type("Args", (), {"ratios": [0.001], "prototype_inits": ["current_sft_signature_random"], "seed": 42, "smoke": True})()
    row = build_pltc_rows(args)[0]
    assert row["promotion_track"] == "sota_chase"
    assert row["uses_teacher_logits"] is True
    safe = dict(row)
    safe["promotion_track"] = "safe_mainline"
    safe["promotion_status"] = "promoted"
    safe["accuracy"] = 0.927
    safe["macro_f1"] = 0.889
    safe["predicted_classes"] = 41
    safe["status"] = "completed_long"
    assert "uses_teacher_logits" in validate_t29_row(safe)["forbidden_flags"]


def test_t29_pltc_no_valid_test_labels_as_input():
    probs = torch.softmax(torch.randn(10, 4), dim=1)
    labels = build_pltc_soft_labels(probs)
    assert labels.shape == (10, 4)
    selected = select_pltc_indices(probs, total_budget=6, seed=3)
    assert selected.selected_idx.numel() == 6
    assert selected.diagnostics["uses_valid_labels_as_input"] is False
    assert selected.diagnostics["uses_test_labels_as_input"] is False


def test_t29_pltc_confidence_bins_budget():
    probs = torch.tensor(
        [
            [0.90, 0.05, 0.05],
            [0.70, 0.20, 0.10],
            [0.45, 0.35, 0.20],
            [0.34, 0.33, 0.33],
        ]
    )
    bins = confidence_bins(probs)
    assert bins.tolist() == ["high", "medium", "low", "low"]


def test_t29_bonsai_sketch_shape_and_lsh_no_full_pairwise():
    blocks = {"X0": torch.randn(20, 8), "X1": torch.randn(20, 8), "X2": torch.randn(20, 8)}
    labels = torch.arange(20) % 4
    sketch = build_bonsai_sketch(blocks, labels=labels, degree=torch.arange(20), output_dim=16, seed=42)
    assert sketch.sketch.shape == (20, 16)
    selected = lsh_bonsai_select(sketch.sketch, labels, total_budget=8, seed=42)
    assert selected.selected_idx.numel() == 8
    assert selected.diagnostics["uses_full_pairwise"] is False
    assert selected.diagnostics["class_floor_respected"] is True


def test_t29_bonsai_runner_budget_and_class_floor():
    args = type("Args", (), {"ratios": [0.001], "sketch_dims": [64], "lsh_buckets": [256], "students": ["table_head"], "seed": 42, "smoke": True})()
    row = build_bonsai_rows(args)[0]
    assert row["method"] == "reddit_sft_bonsai_sketch"
    assert row["actual_condensed_nodes"] == 233
    assert row["uses_exact_pairwise"] is False
