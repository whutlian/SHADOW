from __future__ import annotations

import time
from dataclasses import dataclass

import torch

from shadow_hgc.demand.normalize import destination_row_normalize


@dataclass(frozen=True)
class LogitPropagationResult:
    block: torch.Tensor
    diagnostics: dict


def _one_step(edge_index: torch.Tensor, z: torch.Tensor, num_nodes: int, chunk_size: int) -> torch.Tensor:
    out = torch.zeros(int(num_nodes), int(z.shape[1]), dtype=torch.float32, device=z.device)
    if edge_index.numel() == 0:
        return out
    edge_index = edge_index.to(device=z.device, dtype=torch.long)
    alpha = destination_row_normalize(edge_index, int(num_nodes)).to(device=z.device, dtype=torch.float32)
    for start in range(0, int(edge_index.shape[1]), int(chunk_size)):
        end = min(int(edge_index.shape[1]), start + int(chunk_size))
        src = edge_index[0, start:end]
        dst = edge_index[1, start:end]
        out.index_add_(0, dst, z[src] * alpha[start:end].unsqueeze(1))
    return out


def propagate_logits(
    *,
    edge_index: torch.Tensor,
    logits: torch.Tensor,
    num_nodes: int,
    target_rows: torch.Tensor,
    steps: int = 1,
    lam: float = 0.5,
    input_mode: str = "logits",
    chunk_size: int = 65536,
) -> LogitPropagationResult:
    if input_mode not in {"logits", "probabilities"}:
        raise ValueError("input_mode must be logits or probabilities")
    started = time.perf_counter()
    z0 = logits.to(torch.float32)
    if input_mode == "probabilities":
        z0 = torch.softmax(z0, dim=1)
    z = z0
    for _ in range(int(steps)):
        z = _one_step(edge_index, z, int(num_nodes), int(chunk_size))
    combined = (1.0 - float(lam)) * z0 + float(lam) * z if int(steps) > 0 else z0
    rows = target_rows.to(device=combined.device, dtype=torch.long)
    block = combined[rows]
    diagnostics = {
        "logit_prop_steps": int(steps),
        "logit_prop_lambda": float(lam),
        "logit_prop_input": input_mode,
        "logit_prop_cache_bytes": int(block.numel() * block.element_size()),
        "logit_prop_precompute_time_s": float(time.perf_counter() - started),
        "uses_labels": False,
        "uses_validation_labels": False,
        "uses_test_labels": False,
        "propagates_features": False,
        "normalization": "destination_row",
    }
    return LogitPropagationResult(block=block, diagnostics=diagnostics)


def validate_logit_propagation_config(*, num_classes: int, input_dim: int, propagates_features: bool) -> dict:
    invalid: list[str] = []
    if propagates_features:
        invalid.append("feature_diffusion_forbidden")
    if int(input_dim) != int(num_classes) and propagates_features:
        invalid.append("input_dim_exceeds_num_classes")
    return {"valid": len(invalid) == 0, "invalid_reasons": invalid}
