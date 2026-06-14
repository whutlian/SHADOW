from __future__ import annotations

import numpy as np
import torch

from shadow_hgc.sft.stt_gated_mixer import STTGatedMixer
from shadow_hgc.sft.unified_objective import build_unified_ranked_prefixes
from shadow_hgc.sft.unified_schedule import compute_unified_schedule
from shadow_hgc.sft.unified_stt import make_t38_row, validate_t38_main_row


def test_t39_unified_schedule_exposes_image_required_aliases() -> None:
    row = make_t38_row(
        dataset="Reddit",
        requested_full_node_ratio=0.01,
        condensed_nodes=2329,
        num_classes=41,
        accuracy=0.93,
        macro_f1=0.91,
        promotion_status="promoted",
        shared_cache_time_sec=12.5,
        post_cache_time_sec=34.0,
        total_storage_bytes=5678,
    )

    assert row["budget_phase"] == row["budget_phase_tau"]
    assert row["teacher_cache_k"] == 4
    assert row["student_capacity"] == "stt_gated_mixer:gamlp_like:h256:e220"
    assert row["storage"] == 5678
    assert validate_t38_main_row(row)["valid"] is True


def test_t39_teacher_reliability_controls_soft_branch() -> None:
    weak = compute_unified_schedule(
        condensed_nodes=400,
        num_classes=40,
        teacher_valid_acc=0.14,
        majority_valid_acc=0.14,
        num_nodes=169_343,
    )
    strong = compute_unified_schedule(
        condensed_nodes=2329,
        num_classes=41,
        teacher_valid_acc=0.94,
        majority_valid_acc=0.31,
        num_nodes=232_965,
    )

    assert weak.teacher_reliability_q == 0.0
    assert weak.selection_weights["soft"] == 0.0
    assert weak.selection_weights["boundary"] == 0.0
    assert strong.teacher_reliability_q > 0.8
    assert strong.selection_weights["soft"] > weak.selection_weights["soft"]
    assert strong.loss_weights["alpha_soft"] > 0.0


def test_t39_unified_ranked_prefixes_are_nested() -> None:
    labels = torch.tensor([0, 0, 1, 1, 2, 2, 2, 1], dtype=torch.long)
    train_rows = torch.arange(labels.numel(), dtype=torch.long)
    features = np.asarray(
        [
            [0.1, 0.0],
            [0.9, 0.1],
            [0.0, 0.2],
            [0.0, 0.8],
            [0.2, 0.4],
            [0.4, 0.4],
            [0.8, 0.6],
            [0.3, 0.7],
        ],
        dtype=np.float32,
    )

    prefixes = build_unified_ranked_prefixes(
        labels=labels,
        train_rows=train_rows,
        feature_values=features,
        budgets=[3, 5, 7],
        num_classes=3,
        seed=7,
        selection_weights={"coverage": 0.6, "hard": 0.3, "soft": 0.0, "boundary": 0.0, "rare": 0.1, "diversity": 0.05},
    )

    assert prefixes[3].tolist() == prefixes[5][:3].tolist()
    assert prefixes[5].tolist() == prefixes[7][:5].tolist()
    assert len(set(prefixes[7].tolist())) == 7


def test_t39_stt_gated_mixer_forward_and_diagnostics() -> None:
    model = STTGatedMixer(
        {"self": 3, "x1": 3, "y1": 2, "structure": 1},
        num_classes=4,
        hidden_dim=8,
        dropout=0.0,
        internal_style="gamlp_like",
    )
    for normalizer in model.normalizers.values():
        normalizer.fitted = True
        normalizer.frozen = True
    logits = model(
        {
            "self": torch.randn(5, 3),
            "x1": torch.randn(5, 3),
            "y1": torch.randn(5, 2),
            "structure": torch.randn(5, 1),
        }
    )

    assert logits.shape == (5, 4)
    diag = model.diagnostics()
    assert diag["student_family"] == "stt_gated_mixer"
    assert diag["has_label_reuse_gates"] is True
    assert diag["uses_teacher_logits"] is False
