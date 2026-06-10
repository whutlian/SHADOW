from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from shadow_hgc.data.schemas import DirectedRelation
from shadow_hgc.preprop.filter_bank import compute_preprop_filter_bank


def _read_block(root: Path, meta: dict) -> torch.Tensor:
    array = np.memmap(root / meta["path"], mode="r", dtype=np.dtype(meta["dtype"]), shape=tuple(meta["shape"]))
    return torch.from_numpy(np.asarray(array).copy()).to(torch.float32)


def _dense_row_spmm(edge_index: torch.Tensor, x: torch.Tensor, num_nodes: int) -> torch.Tensor:
    out = torch.zeros(num_nodes, x.shape[1], dtype=torch.float32)
    deg = torch.zeros(num_nodes, dtype=torch.float32)
    deg.index_add_(0, edge_index[1], torch.ones(edge_index.shape[1], dtype=torch.float32))
    for src, dst in edge_index.t().tolist():
        out[dst] += x[src] / deg[dst].clamp_min(1.0)
    return out


def test_t22_filter_bank_x3_residuals_and_label_reuse_match_dense(tmp_path):
    rel = DirectedRelation("paper", "cite_ref", "paper")
    x = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 2.0],
            [3.0, 1.0],
            [2.0, 4.0],
        ],
        dtype=torch.float32,
    )
    labels = torch.tensor([0, 1, 2, 1], dtype=torch.long)
    train = torch.tensor([0, 1], dtype=torch.long)
    edge_index = torch.tensor([[0, 2, 1, 2, 3], [1, 1, 2, 3, 3]], dtype=torch.long)

    manifest = compute_preprop_filter_bank(
        dataset_name="tiny-arxiv",
        graph_spec={"target_type": "paper", "relations": {rel: edge_index}, "num_nodes": {"paper": 4}},
        feature_provider={"paper": x},
        target_node_ids=torch.arange(4),
        train_target_ids=train,
        labels=labels,
        out_dir=tmp_path,
        blocks=("X0", "X1_cite_ref", "X2_cite_ref", "X3_mix", "Xres1_cite_ref", "Xres2_cite_ref", "Y1_cite_ref", "Y2_cite_ref", "Y3_mix", "structure"),
        feature_dim=2,
        dtype="float32",
        edge_chunk_size=2,
        dst_chunk_size=2,
        normalization="row",
    )

    raw_manifest = json.loads((tmp_path / "preprop_manifest.json").read_text(encoding="utf-8"))
    by_name = {block["name"]: block for block in raw_manifest["blocks"]}
    x1 = _read_block(tmp_path, by_name["X1_cite_ref"])
    x2 = _read_block(tmp_path, by_name["X2_cite_ref"])
    x3 = _read_block(tmp_path, by_name["X3_mix"])
    xres1 = _read_block(tmp_path, by_name["Xres1_cite_ref"])
    xres2 = _read_block(tmp_path, by_name["Xres2_cite_ref"])
    y1 = _read_block(tmp_path, by_name["Y1_cite_ref"])
    y2 = _read_block(tmp_path, by_name["Y2_cite_ref"])
    y3 = _read_block(tmp_path, by_name["Y3_mix"])

    expected_x1 = _dense_row_spmm(edge_index, x, 4)
    expected_x2 = _dense_row_spmm(edge_index, expected_x1, 4)
    expected_x3 = _dense_row_spmm(edge_index, expected_x2, 4)
    y0 = torch.zeros(4, 3)
    y0[train, labels[train]] = 1.0
    expected_y1 = _dense_row_spmm(edge_index, y0, 4)
    expected_y2 = _dense_row_spmm(edge_index, expected_y1, 4)
    expected_y3 = _dense_row_spmm(edge_index, expected_y2, 4)

    assert torch.allclose(x1, expected_x1, atol=1e-6)
    assert torch.allclose(x2, expected_x2, atol=1e-6)
    assert torch.allclose(x3, expected_x3, atol=1e-6)
    assert torch.allclose(xres1, x - expected_x1, atol=1e-6)
    assert torch.allclose(xres2, expected_x1 - expected_x2, atol=1e-6)
    assert torch.allclose(y1, expected_y1, atol=1e-6)
    assert torch.allclose(y2, expected_y2, atol=1e-6)
    assert torch.allclose(y3, expected_y3, atol=1e-6)
    assert manifest.uses_logits_as_input is False
    assert raw_manifest["uses_e_by_d_materialization"] is False
    assert (tmp_path / "block_index.csv").exists()
    assert (tmp_path / "block_stats.json").exists()
