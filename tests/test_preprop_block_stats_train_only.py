from __future__ import annotations

import json

import torch

from shadow_hgc.preprop.true_preprop import compute_preprop_blocks


def test_preprop_block_stats_are_fit_on_train_rows_only(tmp_path):
    x0 = torch.tensor([[0.0, 0.0], [2.0, 2.0], [100.0, 100.0]], dtype=torch.float32)
    compute_preprop_blocks(
        dataset_name="tiny",
        target_type="paper",
        x_provider={"paper": x0, "train_rows": torch.tensor([0, 1], dtype=torch.long)},
        relations={},
        output_dir=str(tmp_path),
        blocks=["X0"],
        feature_dim=2,
        dtype="float32",
        seed=42,
    )

    stats = json.loads((tmp_path / "block_X0_stats.json").read_text(encoding="utf-8"))
    assert stats["fit_scope"] == "train_target_rows"
    assert stats["fit_rows"] == [0, 1]
    assert stats["mean"] == [1.0, 1.0]
