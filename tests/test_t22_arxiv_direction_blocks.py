from __future__ import annotations

import json

import torch

from shadow_hgc.data.schemas import DirectedRelation
from shadow_hgc.preprop.filter_bank import compute_preprop_filter_bank


def test_arxiv_forward_and_reverse_blocks_are_distinct_and_logged(tmp_path):
    cite_ref = DirectedRelation("paper", "cite_ref", "paper")
    cited_by = DirectedRelation("paper", "cited_by", "paper")
    x = torch.eye(3, dtype=torch.float32)
    forward = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    reverse = torch.stack([forward[1], forward[0]], dim=0)

    compute_preprop_filter_bank(
        dataset_name="ogbn-arxiv",
        graph_spec={"target_type": "paper", "relations": {cite_ref: forward, cited_by: reverse}, "num_nodes": {"paper": 3}},
        feature_provider={"paper": x},
        target_node_ids=torch.arange(3),
        train_target_ids=torch.tensor([0, 1]),
        labels=torch.tensor([0, 1, 0]),
        out_dir=tmp_path,
        blocks=("X1_cite_ref", "X1_cited_by", "Y1_cite_ref", "Y1_cited_by"),
        feature_dim=3,
        dtype="float32",
        edge_chunk_size=1,
    )

    manifest = json.loads((tmp_path / "preprop_manifest.json").read_text(encoding="utf-8"))
    by_name = {block["name"]: block for block in manifest["blocks"]}
    assert by_name["X1_cite_ref"]["source_relation"] == "paper--cite_ref-->paper"
    assert by_name["X1_cited_by"]["source_relation"] == "paper--cited_by-->paper"
    assert by_name["X1_cite_ref"]["spec_hash"] != by_name["X1_cited_by"]["spec_hash"]
    assert by_name["Y1_cite_ref"]["is_train_label_block"] is True
    assert by_name["Y1_cited_by"]["is_train_label_block"] is True
