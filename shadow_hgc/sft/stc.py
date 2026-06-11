from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class BlockSpec:
    name: str
    start: int
    end: int

    def slice(self) -> slice:
        if self.end <= self.start:
            raise ValueError(f"invalid block {self.name}: end must be greater than start")
        return slice(int(self.start), int(self.end))


def actual_full_node_ratio(syn_rows: int, original_num_nodes: int) -> float:
    if int(original_num_nodes) <= 0:
        raise ValueError("original_num_nodes must be positive")
    return int(syn_rows) / float(original_num_nodes)


def apply_tanh_bounded_delta(
    z_init: torch.Tensor,
    raw_delta: torch.Tensor,
    blocks: list[BlockSpec] | tuple[BlockSpec, ...],
    *,
    rho: float,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor]:
    if z_init.shape != raw_delta.shape:
        raise ValueError("z_init and raw_delta must have the same shape")
    if float(rho) < 0:
        raise ValueError("rho must be non-negative")
    parts: list[torch.Tensor] = []
    cursor = 0
    for block in blocks:
        block_slice = block.slice()
        if block.start != cursor:
            raise ValueError("blocks must be contiguous and sorted")
        init_block = z_init[:, block_slice]
        raw_block = raw_delta[:, block_slice]
        dim = max(1, init_block.shape[1])
        init_norm = init_block.norm(dim=1, keepdim=True)
        bounded_direction = torch.tanh(raw_block) / (dim**0.5)
        delta_block = float(rho) * init_norm.clamp_min(0.0) * bounded_direction
        zero_mask = init_norm <= eps
        if zero_mask.any():
            delta_block = torch.where(zero_mask, torch.zeros_like(delta_block), delta_block)
        parts.append(delta_block)
        cursor = block.end
    if cursor != z_init.shape[1]:
        raise ValueError("blocks must cover every feature dimension")
    delta = torch.cat(parts, dim=1)
    return z_init + delta, delta


def delta_bound_ratios(
    z_init: torch.Tensor,
    delta: torch.Tensor,
    blocks: list[BlockSpec] | tuple[BlockSpec, ...],
    *,
    eps: float = 1e-12,
) -> dict[str, float]:
    ratios: dict[str, float] = {}
    for block in blocks:
        block_slice = block.slice()
        numerator = delta[:, block_slice].norm(dim=1)
        denominator = z_init[:, block_slice].norm(dim=1)
        ratio = torch.where(denominator > eps, numerator / denominator.clamp_min(eps), torch.zeros_like(numerator))
        ratios[block.name] = float(ratio.max().item()) if ratio.numel() else 0.0
    return ratios


def class_histogram_json(labels: np.ndarray | torch.Tensor | list[int], *, num_classes: int) -> str:
    if isinstance(labels, torch.Tensor):
        array = labels.detach().cpu().numpy()
    else:
        array = np.asarray(labels)
    histogram = {str(cls): 0 for cls in range(int(num_classes))}
    for value in array.astype(np.int64).tolist():
        if 0 <= int(value) < int(num_classes):
            histogram[str(int(value))] += 1
    return json.dumps(histogram, sort_keys=True)


def count_nonzero_histogram(histogram_json: str | dict[str, int]) -> int:
    histogram = json.loads(histogram_json) if isinstance(histogram_json, str) else histogram_json
    return int(sum(1 for value in histogram.values() if int(value) > 0))


def products_balanced_gate(
    *,
    predicted_class_histogram_json: str | dict[str, int],
    macro_f1: float,
    accuracy: float,
    min_predicted_classes: int = 38,
    min_macro_f1: float = 0.400,
    min_accuracy: float = 0.735,
) -> bool:
    return (
        count_nonzero_histogram(predicted_class_histogram_json) >= int(min_predicted_classes)
        and float(macro_f1) >= float(min_macro_f1)
        and float(accuracy) >= float(min_accuracy)
    )


def synthetic_class_count_stats(labels: np.ndarray | torch.Tensor | list[int], *, num_classes: int) -> dict[str, float]:
    histogram = json.loads(class_histogram_json(labels, num_classes=num_classes))
    values = np.asarray(list(histogram.values()), dtype=np.float64)
    return {
        "selected_or_syn_class_count_min": float(values.min()) if values.size else 0.0,
        "selected_or_syn_class_count_median": float(np.median(values)) if values.size else 0.0,
        "selected_or_syn_class_count_max": float(values.max()) if values.size else 0.0,
    }
