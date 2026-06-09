from __future__ import annotations

import torch

from shadow_hgc.demand.normalize import destination_row_normalize


def destination_alpha(edge_index: torch.Tensor, *, num_dst_nodes: int) -> torch.Tensor:
    return destination_row_normalize(edge_index.to(torch.long), int(num_dst_nodes)).to(torch.float32)
