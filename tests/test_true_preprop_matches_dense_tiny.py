from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from shadow_hgc.data.schemas import DirectedRelation
from shadow_hgc.preprop.true_preprop import compute_preprop_blocks


def _read_block(path: Path, shape: list[int], dtype: str) -> torch.Tensor:
    array = np.memmap(path, mode="r", dtype=np.dtype(dtype), shape=tuple(shape))
    return torch.from_numpy(np.asarray(array).copy()).to(torch.float32)


def _dense_destination_spmm(edge_index: torch.Tensor, x: torch.Tensor, num_nodes: int) -> torch.Tensor:
    out = torch.zeros(num_nodes, x.shape[1], dtype=torch.float32)
    deg = torch.zeros(num_nodes, dtype=torch.float32)
    deg.index_add_(0, edge_index[1], torch.ones(edge_index.shape[1], dtype=torch.float32))
    for src, dst in edge_index.t().tolist():
        out[dst] += x[src] / deg[dst].clamp_min(1.0)
    return out


def test_true_preprop_x1_x2_and_residual_match_dense_destination_normalization(tmp_path):
    relation = DirectedRelation("paper", "cite_ref", "paper")
    x0 = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 2.0],
            [3.0, 1.0],
            [2.0, 4.0],
        ],
        dtype=torch.float32,
    )
    edge_index = torch.tensor(
        [
            [0, 2, 1, 2, 3],
            [1, 1, 2, 3, 3],
        ],
        dtype=torch.long,
    )
    manifest = compute_preprop_blocks(
        dataset_name="tiny",
        target_type="paper",
        x_provider={"paper": x0, "train_rows": torch.tensor([0, 1], dtype=torch.long)},
        relations={relation: edge_index},
        output_dir=str(tmp_path),
        blocks=["X0", "X1", "X2", "Xres"],
        feature_dim=2,
        dtype="float32",
        edge_chunk_size=2,
        dst_chunk_size=2,
        max_ram_gb=1.0,
        force_memmap=True,
        seed=42,
    )

    meta_by_name = {block.name: block for block in manifest.blocks}
    x1 = _read_block(tmp_path / meta_by_name["X1"].path, meta_by_name["X1"].shape, meta_by_name["X1"].dtype)
    x2 = _read_block(tmp_path / meta_by_name["X2"].path, meta_by_name["X2"].shape, meta_by_name["X2"].dtype)
    xres = _read_block(tmp_path / meta_by_name["Xres"].path, meta_by_name["Xres"].shape, meta_by_name["Xres"].dtype)

    expected_x1 = _dense_destination_spmm(edge_index, x0, num_nodes=4)
    expected_x2 = _dense_destination_spmm(edge_index, expected_x1, num_nodes=4)
    assert torch.allclose(x1, expected_x1, atol=1e-6)
    assert torch.allclose(x2, expected_x2, atol=1e-6)
    assert torch.allclose(xres, x0 - expected_x1, atol=1e-6)

    manifest_json = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest_json["uses_logits_as_input"] is False
    assert manifest_json["uses_teacher_logits"] is False
    assert manifest_json["uses_kd"] is False
    assert manifest_json["uses_e_by_d_materialization"] is False
