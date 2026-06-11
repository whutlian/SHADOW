from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class TimeAwareFeatureResult:
    features: torch.Tensor
    diagnostics: dict[str, Any]


@dataclass(frozen=True)
class TemporalLabelReuseResult:
    features: torch.Tensor
    raw_weights: torch.Tensor
    diagnostics: dict[str, Any]


def _as_long(values: torch.Tensor) -> torch.Tensor:
    return values.detach().to(torch.long).cpu()


def _train_mask(num_nodes: int, train_idx: torch.Tensor) -> torch.Tensor:
    mask = torch.zeros(int(num_nodes), dtype=torch.bool)
    train_idx = _as_long(train_idx)
    if train_idx.numel():
        mask[train_idx] = True
    return mask


def normalized_year_scalar(years: torch.Tensor, train_idx: torch.Tensor) -> torch.Tensor:
    years_f = years.detach().to(torch.float32).cpu()
    train = years_f[_train_mask(years_f.numel(), train_idx)]
    source = train if train.numel() else years_f
    mean = source.mean()
    std = source.std(unbiased=False).clamp_min(1e-6)
    return ((years_f - mean) / std).unsqueeze(1)


def year_bucket_onehot_v2(years: torch.Tensor, train_idx: torch.Tensor) -> torch.Tensor:
    years_l = _as_long(years)
    train = years_l[_train_mask(years_l.numel(), train_idx)]
    source = train if train.numel() else years_l
    min_year = int(source.min().item())
    max_year = int(source.max().item())
    clipped = years_l.clamp(min_year, max_year) - min_year
    out = torch.zeros((years_l.numel(), max_year - min_year + 1), dtype=torch.float32)
    out[torch.arange(years_l.numel()), clipped] = 1.0
    return out


def relative_year_features(years: torch.Tensor, *, train_boundary: int = 2017, valid_boundary: int = 2018) -> torch.Tensor:
    years_f = years.detach().to(torch.float32).cpu()
    return torch.stack(
        [
            years_f - float(train_boundary),
            years_f - float(valid_boundary),
        ],
        dim=1,
    )


def train_year_class_prior_features(years: torch.Tensor, labels: torch.Tensor, train_idx: torch.Tensor, *, num_classes: int) -> torch.Tensor:
    years_l = _as_long(years)
    labels_l = _as_long(labels)
    train_idx = _as_long(train_idx)
    priors: dict[int, torch.Tensor] = {}
    if train_idx.numel():
        for year in torch.unique(years_l[train_idx], sorted=True):
            mask = train_idx[years_l[train_idx] == int(year.item())]
            hist = torch.bincount(labels_l[mask].clamp(0, int(num_classes) - 1), minlength=int(num_classes)).to(torch.float32)
            priors[int(year.item())] = hist / hist.sum().clamp_min(1.0)
    global_hist = torch.bincount(labels_l[train_idx].clamp(0, int(num_classes) - 1), minlength=int(num_classes)).to(torch.float32) if train_idx.numel() else torch.ones(int(num_classes))
    global_prior = global_hist / global_hist.sum().clamp_min(1.0)
    out = torch.zeros((years_l.numel(), int(num_classes)), dtype=torch.float32)
    for i, year in enumerate(years_l.tolist()):
        out[i] = priors.get(int(year), global_prior)
    return out


def temporal_labelreuse_decay_v2(
    edge_index: torch.Tensor,
    years: torch.Tensor,
    labels: torch.Tensor,
    train_idx: torch.Tensor,
    *,
    num_classes: int,
    gamma: float,
) -> TemporalLabelReuseResult:
    edge_index = edge_index.detach().to(torch.long).cpu()
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, E]")
    years_l = _as_long(years)
    labels_l = _as_long(labels)
    train = _train_mask(years_l.numel(), train_idx)
    src = edge_index[0]
    dst = edge_index[1]
    keep = train[src]
    out = torch.zeros((years_l.numel(), int(num_classes)), dtype=torch.float32)
    raw = torch.zeros(edge_index.shape[1], dtype=torch.float32)
    for edge_pos, (s, d) in enumerate(zip(src[keep].tolist(), dst[keep].tolist())):
        lag = max(0, int(years_l[d].item()) - int(years_l[s].item()))
        weight = float(torch.exp(torch.tensor(-float(gamma) * float(lag))).item())
        cls = int(labels_l[s].item())
        if 0 <= cls < int(num_classes):
            out[d, cls] += weight
        kept_positions = torch.nonzero(keep, as_tuple=False).view(-1)
        raw[int(kept_positions[edge_pos].item())] = weight
    row_sum = out.sum(dim=1, keepdim=True)
    out = torch.where(row_sum > 0, out / row_sum.clamp_min(1e-12), out)
    return TemporalLabelReuseResult(
        features=out,
        raw_weights=raw,
        diagnostics={
            "uses_train_labels_only": True,
            "uses_valid_labels_as_input": False,
            "uses_test_labels_as_input": False,
            "temporal_decay_gamma": float(gamma),
        },
    )


def build_timeaware_arxiv_features(
    years: torch.Tensor,
    labels: torch.Tensor,
    train_idx: torch.Tensor,
    valid_idx: torch.Tensor,
    test_idx: torch.Tensor,
    *,
    num_classes: int,
    include_prior: bool = True,
) -> TimeAwareFeatureResult:
    _ = valid_idx.numel()
    _ = test_idx.numel()
    pieces = [
        normalized_year_scalar(years, train_idx),
        year_bucket_onehot_v2(years, train_idx),
        relative_year_features(years),
    ]
    if include_prior:
        pieces.append(train_year_class_prior_features(years, labels, train_idx, num_classes=int(num_classes)))
    features = torch.cat([piece.to(torch.float32) for piece in pieces], dim=1)
    return TimeAwareFeatureResult(
        features=features,
        diagnostics={
            "uses_year_metadata": True,
            "uses_temporal_features": True,
            "uses_train_labels_only": True,
            "uses_valid_labels_as_input": False,
            "uses_test_labels_as_input": False,
            "year_feature_dim": int(features.shape[1]),
        },
    )
