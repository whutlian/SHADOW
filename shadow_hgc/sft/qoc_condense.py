from __future__ import annotations

from typing import Any

import torch

from shadow_hgc.sft.operator_match import apply_sparse_operator


def build_qoc_table(
    *,
    z0: torch.Tensor,
    y0: torch.Tensor,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    structure: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    z0f = z0.to(torch.float32)
    y0f = y0.to(torch.float32)
    pz0 = apply_sparse_operator(z0f, edge_index, edge_weight)
    p2z0 = apply_sparse_operator(pz0, edge_index, edge_weight)
    py0 = apply_sparse_operator(y0f, edge_index, edge_weight)
    p2y0 = apply_sparse_operator(py0, edge_index, edge_weight)
    blocks = [z0f, pz0, p2z0, y0f, py0, p2y0]
    names = ["Z0", "PZ0", "P2Z0", "Y0", "PY0", "P2Y0"]
    if structure is not None:
        blocks.append(structure.to(torch.float32))
        names.append("structure")
    table = torch.cat(blocks, dim=1)
    return table, {"table_blocks": ",".join(names), "uses_dense_adjacency": False, "uses_e_by_d_materialization": False}
