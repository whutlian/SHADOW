from __future__ import annotations

import numpy as np
import torch

from shadow_hgc.demand.normalize import destination_row_normalize
from shadow_hgc.features.streaming_diffusion import compute_streaming_diffusion_blocks


def _dense_step(x: torch.Tensor, edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    alpha = destination_row_normalize(edge_index, num_nodes).to(dtype=x.dtype)
    out = torch.zeros(num_nodes, x.shape[1], dtype=x.dtype)
    out.index_add_(0, edge_index[1], x[edge_index[0]] * alpha[:, None])
    return out


def _read_block(path, shape, dtype) -> torch.Tensor:
    arr = np.memmap(path, mode="r", dtype=np.dtype(dtype), shape=shape)
    return torch.from_numpy(np.asarray(arr).copy())


def test_streaming_diffusion_matches_dense_destination_row_normalized_blocks(tmp_path):
    x = torch.tensor(
        [
            [1.0, 2.0, 0.0],
            [3.0, 0.0, 1.0],
            [5.0, 4.0, 2.0],
            [7.0, 6.0, 3.0],
        ],
        dtype=torch.float32,
    )
    edge_index = torch.tensor(
        [
            [0, 1, 2, 0, 3, 1],
            [1, 1, 1, 2, 2, 3],
        ],
        dtype=torch.long,
    )

    result = compute_streaming_diffusion_blocks(
        x_provider=x,
        edge_index=edge_index,
        num_nodes=x.shape[0],
        steps=(1, 2),
        include_highpass=True,
        out_dir=tmp_path,
        dtype="float32",
        edge_chunk_size=2,
        overwrite=True,
    )

    dense_x1 = _dense_step(x, edge_index, x.shape[0])
    dense_x2 = _dense_step(dense_x1, edge_index, x.shape[0])
    expected = {"X1": dense_x1, "X2": dense_x2, "Xhp": x - dense_x1}

    assert result.block_names == ["X1", "X2", "Xhp"]
    assert result.stats["normalize"] == "destination_row"
    for name, expected_tensor in expected.items():
        actual = _read_block(
            result.block_paths[name],
            result.block_shapes[name],
            result.block_dtypes[name],
        )
        torch.testing.assert_close(actual, expected_tensor, atol=1e-6, rtol=1e-6)
