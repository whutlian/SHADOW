from __future__ import annotations

import json

import torch

from shadow_hgc.data.schemas import DirectedRelation
from shadow_hgc.preprop.true_preprop import compute_preprop_blocks


def test_t21_preprop_manifest_has_required_schema_fields(tmp_path):
    relation = DirectedRelation("paper", "cite_ref", "paper")
    manifest = compute_preprop_blocks(
        dataset_name="tiny",
        target_type="paper",
        x_provider={"paper": torch.eye(3), "train_rows": torch.tensor([0, 1], dtype=torch.long)},
        relations={relation: torch.tensor([[0, 1], [1, 2]], dtype=torch.long)},
        output_dir=str(tmp_path),
        blocks=["X0", "X1"],
        feature_dim=3,
        dtype="float32",
        seed=42,
    )

    payload = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    for key in [
        "dataset",
        "target_type",
        "seed",
        "feature_dim",
        "blocks",
        "total_cache_bytes",
        "full_edge_scans",
        "uses_logits_as_input",
        "uses_teacher_logits",
        "uses_kd",
        "uses_diffusion_legacy",
        "uses_e_by_d_materialization",
        "uses_dense_p2",
        "uses_bounded_edges",
    ]:
        assert key in payload
    assert payload["feature_dim"] == 3
    assert payload["total_cache_bytes"] == manifest.total_cache_bytes

    x1_meta = next(block for block in payload["blocks"] if block["name"] == "X1")
    for key in ["path", "shape", "dtype", "source_relations", "edge_scans", "disk_bytes", "normalization", "stats_fit_source"]:
        assert key in x1_meta
    assert x1_meta["source_relations"] == ["paper--cite_ref-->paper"]
    assert x1_meta["normalization"] == "destination_row"
    assert x1_meta["stats_fit_source"] == "train_target_rows"
