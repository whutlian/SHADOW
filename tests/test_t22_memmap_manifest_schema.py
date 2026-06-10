from __future__ import annotations

import json

import torch

from shadow_hgc.data.schemas import DirectedRelation
from shadow_hgc.preprop.filter_bank import compute_preprop_filter_bank


def test_t22_manifest_has_required_files_and_block_schema(tmp_path):
    rel = DirectedRelation("product", "co_purchase", "product")
    compute_preprop_filter_bank(
        dataset_name="tiny-products",
        graph_spec={"target_type": "product", "relations": {rel: torch.tensor([[0, 1], [1, 2]], dtype=torch.long)}, "num_nodes": {"product": 3}},
        feature_provider={"product": torch.randn(3, 4)},
        target_node_ids=torch.arange(3),
        train_target_ids=torch.tensor([0, 1]),
        labels=torch.tensor([0, 1, 0]),
        out_dir=tmp_path,
        blocks=("X0", "X1", "Y1", "structure"),
        feature_dim=4,
        dtype="float16",
        edge_chunk_size=1,
    )

    manifest = json.loads((tmp_path / "preprop_manifest.json").read_text(encoding="utf-8"))
    assert manifest["uses_logits_as_input"] is False
    assert manifest["uses_kd"] is False
    assert manifest["uses_dense_p2"] is False
    assert manifest["uses_e_by_d_materialization"] is False
    for block in manifest["blocks"]:
        for field in ["name", "shape", "dtype", "path", "cache_bytes", "edge_scans", "normalization", "source_relation"]:
            assert field in block
        assert block["dtype"] == "float16"
        assert block["uses_logits_as_input"] is False
        assert block["uses_kd"] is False
    assert "X0" in (tmp_path / "block_index.csv").read_text(encoding="utf-8")
