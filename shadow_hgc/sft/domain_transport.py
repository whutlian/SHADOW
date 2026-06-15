from __future__ import annotations

from dataclasses import dataclass
import json
import math
from typing import Any

import numpy as np
import torch

from shadow_hgc.sft.domain_coverage import build_domain_bucket_ids


EPS = 1e-12


@dataclass(frozen=True)
class DomainTransportConfig:
    gap_threshold: float = 0.10
    gap_tau: float = 0.03
    capacity_threshold: float = 24.0
    capacity_tau: float = 8.0
    max_domain_frac_medium: float = 0.30
    max_domain_frac_large: float = 0.35
    max_domain_frac_ultra: float = 0.10
    lambda_mix_min: float = 0.20
    lambda_mix_max: float = 0.40
    bucket_candidate_cap: int = 2048
    projection_dim: int = 64
    num_quantile_bins: int = 16
    seed: int = 42
    activation_threshold: float = 0.25
    deficit_alpha: float = 1.0
    deficit_beta: float = 0.5


@dataclass(frozen=True)
class DomainBucketStats:
    bucket_ids: np.ndarray
    target_idx: np.ndarray
    train_idx: np.ndarray
    train_labels: np.ndarray
    all_bucket_mass: dict[int, float]
    train_bucket_mass: dict[int, float]
    domain_gap_train_all: float
    metadata: dict[str, Any]


@dataclass(frozen=True)
class DomainTransportRows:
    selected_rows: np.ndarray
    anchor_rows: np.ndarray
    source_bucket_ids: np.ndarray
    lambda_mix: float
    domain_transport_rows: int
    domain_row_frac: float
    domain_transport_active: bool
    domain_transport_strength: float
    domain_gap_before: float
    domain_gap_after: float
    domain_transport_gain: float
    domain_overfit_proxy: float
    row_type_counts: dict[str, int]
    metadata: dict[str, Any]

    def row_type_counts_json(self) -> str:
        return json.dumps(self.row_type_counts, sort_keys=True)


def _as_numpy(values: np.ndarray | torch.Tensor | list[int] | tuple[int, ...], *, dtype: np.dtype | type | None = None) -> np.ndarray:
    if isinstance(values, torch.Tensor):
        out = values.detach().cpu().numpy()
    else:
        out = np.asarray(values)
    return out.astype(dtype, copy=False) if dtype is not None else out


def _sigmoid(value: float) -> float:
    value = float(value)
    if value >= 0.0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def _mass_from_buckets(bucket_ids: np.ndarray) -> dict[int, float]:
    buckets = np.asarray(bucket_ids, dtype=np.uint64)
    if buckets.size == 0:
        return {}
    unique, counts = np.unique(buckets, return_counts=True)
    denom = float(counts.sum())
    return {int(bucket): float(count) / max(denom, EPS) for bucket, count in zip(unique.tolist(), counts.tolist())}


def _bucket_subset(bucket_ids: np.ndarray, rows: np.ndarray | list[int] | tuple[int, ...]) -> np.ndarray:
    buckets = np.asarray(bucket_ids, dtype=np.uint64)
    idx = np.asarray(rows, dtype=np.int64)
    if idx.size == 0:
        return np.zeros(0, dtype=np.uint64)
    if int(idx.max(initial=0)) >= buckets.shape[0]:
        raise ValueError("row ids must align with bucket_ids")
    return buckets[idx]


def _js_from_mass(left: dict[int, float], right: dict[int, float]) -> float:
    keys = sorted(set(left) | set(right))
    if not keys:
        return 0.0
    p = np.asarray([max(0.0, float(left.get(key, 0.0))) for key in keys], dtype=np.float64)
    q = np.asarray([max(0.0, float(right.get(key, 0.0))) for key in keys], dtype=np.float64)
    p = p / max(float(p.sum()), EPS)
    q = q / max(float(q.sum()), EPS)
    m = 0.5 * (p + q)
    p_mask = p > 0
    q_mask = q > 0
    kl_pm = float((p[p_mask] * (np.log(p[p_mask]) - np.log(m[p_mask]))).sum())
    kl_qm = float((q[q_mask] * (np.log(q[q_mask]) - np.log(m[q_mask]))).sum())
    return float(0.5 * (kl_pm + kl_qm))


def compute_domain_transport_strength(
    domain_gap_train_all: float,
    class_capacity_b: float,
    cfg: DomainTransportConfig | None = None,
) -> float:
    cfg = cfg or DomainTransportConfig()
    gap_term = _sigmoid((float(domain_gap_train_all) - float(cfg.gap_threshold)) / max(float(cfg.gap_tau), EPS))
    capacity_term = _sigmoid((float(class_capacity_b) - float(cfg.capacity_threshold)) / max(float(cfg.capacity_tau), EPS))
    return float(gap_term * capacity_term)


def scale_bucket_for_num_nodes(num_nodes: int) -> str:
    n = int(num_nodes)
    if n > 10_000_000:
        return "ultra"
    if n > 1_000_000:
        return "large"
    return "medium"


def _max_domain_frac(scale_bucket: str, cfg: DomainTransportConfig) -> float:
    scale = str(scale_bucket).lower()
    if scale == "ultra":
        return float(cfg.max_domain_frac_ultra)
    if scale == "large":
        return float(cfg.max_domain_frac_large)
    return float(cfg.max_domain_frac_medium)


def allocate_domain_row_budget(
    total_budget: int,
    domain_transport_strength: float,
    scale_bucket: str,
    cfg: DomainTransportConfig | None = None,
    *,
    min_core_rows: int = 0,
) -> tuple[int, float]:
    cfg = cfg or DomainTransportConfig()
    total = max(0, int(total_budget))
    strength = max(0.0, min(1.0, float(domain_transport_strength)))
    frac = _max_domain_frac(scale_bucket, cfg) * strength
    if strength < float(cfg.activation_threshold) or total <= 0:
        return 0, float(frac)
    rows = int(round(float(total) * frac))
    rows = max(0, rows)
    rows = min(rows, max(0, total - max(0, int(min_core_rows))))
    return int(rows), float(frac)


def build_domain_bucket_stats(
    signature_reader: np.ndarray | np.memmap,
    *,
    target_idx: np.ndarray | torch.Tensor | list[int] | tuple[int, ...],
    train_idx: np.ndarray | torch.Tensor | list[int] | tuple[int, ...],
    train_labels: np.ndarray | torch.Tensor | list[int] | tuple[int, ...],
    teacher_cache: Any | None = None,
    cfg: DomainTransportConfig | None = None,
    degree: np.ndarray | None = None,
    teacher_pred_class: np.ndarray | None = None,
    teacher_confidence: np.ndarray | None = None,
    valid_labels: Any | None = None,
    test_labels: Any | None = None,
) -> DomainBucketStats:
    del teacher_cache, valid_labels, test_labels
    cfg = cfg or DomainTransportConfig()
    x = np.asarray(signature_reader, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError("signature_reader must expose a 2D signature matrix")
    target_rows = _as_numpy(target_idx, dtype=np.int64)
    train_rows = _as_numpy(train_idx, dtype=np.int64)
    labels = _as_numpy(train_labels, dtype=np.int64)
    if target_rows.size == 0:
        target_rows = np.arange(x.shape[0], dtype=np.int64)
    if int(target_rows.max(initial=0)) < x.shape[0] and target_rows.size != x.shape[0]:
        signatures = np.asarray(x[target_rows], dtype=np.float32)
        local_pos = {int(row): pos for pos, row in enumerate(target_rows.tolist())}
        train_positions = np.asarray([local_pos[int(row)] for row in train_rows if int(row) in local_pos], dtype=np.int64)
    else:
        signatures = x
        train_positions = train_rows
    train_mask = np.zeros(signatures.shape[0], dtype=bool)
    train_mask[train_positions] = True
    bucket_ids = build_domain_bucket_ids(
        signatures,
        train_mask=train_mask,
        labels=None,
        seed=int(cfg.seed),
        projection_dim=int(cfg.projection_dim),
        num_quantiles=int(cfg.num_quantile_bins),
        degree=degree,
        teacher_pred_class=teacher_pred_class,
        teacher_confidence=teacher_confidence,
    )
    all_mass = _mass_from_buckets(bucket_ids)
    train_mass = _mass_from_buckets(bucket_ids[train_positions])
    return DomainBucketStats(
        bucket_ids=np.asarray(bucket_ids, dtype=np.uint64),
        target_idx=target_rows,
        train_idx=train_rows,
        train_labels=labels,
        all_bucket_mass=all_mass,
        train_bucket_mass=train_mass,
        domain_gap_train_all=_js_from_mass(train_mass, all_mass),
        metadata={
            "uses_valid_labels_as_input": False,
            "uses_test_labels_as_input": False,
            "uses_all_pair_distance": False,
            "bucket_count": int(len(all_mass)),
        },
    )


def compute_bucket_deficits(selected_bucket_mass: dict[int, float], all_bucket_mass: dict[int, float]) -> dict[int, float]:
    return {
        int(bucket): max(0.0, float(all_bucket_mass.get(bucket, 0.0)) - float(selected_bucket_mass.get(bucket, 0.0)))
        for bucket in sorted(set(selected_bucket_mass) | set(all_bucket_mass))
    }


def _virtual_mass(bucket_ids: np.ndarray, rows: np.ndarray, transport_buckets: np.ndarray | None = None) -> dict[int, float]:
    parts = [_bucket_subset(bucket_ids, rows)]
    if transport_buckets is not None and transport_buckets.size:
        parts.append(np.asarray(transport_buckets, dtype=np.uint64))
    combined = np.concatenate(parts) if parts else np.zeros(0, dtype=np.uint64)
    return _mass_from_buckets(combined)


def _rank_deficit_buckets(deficits: dict[int, float], all_mass: dict[int, float], cfg: DomainTransportConfig) -> list[int]:
    scored = []
    for bucket, deficit in deficits.items():
        score = (max(0.0, float(deficit)) ** float(cfg.deficit_alpha)) * (max(EPS, float(all_mass.get(bucket, 0.0))) ** float(cfg.deficit_beta))
        if score > 0.0:
            scored.append((score, int(bucket)))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [bucket for _, bucket in scored]


def _anchor_for_bucket(
    *,
    bucket: int,
    bucket_ids: np.ndarray,
    train_rows: np.ndarray,
    train_labels: np.ndarray,
    used_count: int,
    cfg: DomainTransportConfig,
) -> int:
    del train_labels
    if train_rows.size == 0:
        raise ValueError("domain transport requires at least one train anchor")
    train_buckets = _bucket_subset(bucket_ids, train_rows)
    local = np.flatnonzero(train_buckets == np.uint64(bucket))
    if local.size:
        capped = local[: max(1, int(cfg.bucket_candidate_cap))]
        return int(train_rows[int(capped[used_count % capped.size])])
    return int(train_rows[used_count % train_rows.size])


def build_domain_transport_rows(
    *,
    bucket_ids: np.ndarray,
    train_rows: np.ndarray | torch.Tensor | list[int] | tuple[int, ...],
    train_labels: np.ndarray | torch.Tensor | list[int] | tuple[int, ...],
    selected_rows: np.ndarray | torch.Tensor | list[int] | tuple[int, ...],
    total_budget: int,
    num_classes: int,
    domain_transport_strength: float,
    cfg: DomainTransportConfig | None = None,
    scale_bucket: str = "medium",
    domain_gap_train_all: float | None = None,
) -> DomainTransportRows:
    cfg = cfg or DomainTransportConfig()
    rows = _as_numpy(selected_rows, dtype=np.int64)
    train = _as_numpy(train_rows, dtype=np.int64)
    labels = _as_numpy(train_labels, dtype=np.int64)
    all_mass = _mass_from_buckets(np.asarray(bucket_ids, dtype=np.uint64))
    selected_mass = _virtual_mass(bucket_ids, rows)
    gap_before = _js_from_mass(selected_mass, all_mass)
    min_core = min(max(0, int(num_classes)), max(0, int(total_budget)))
    domain_rows, frac = allocate_domain_row_budget(
        int(total_budget),
        float(domain_transport_strength),
        scale_bucket,
        cfg,
        min_core_rows=min_core,
    )
    active = domain_rows > 0 and float(domain_transport_strength) >= float(cfg.activation_threshold)
    if not active:
        after = gap_before
        return DomainTransportRows(
            selected_rows=rows[: int(total_budget)].copy(),
            anchor_rows=np.zeros(0, dtype=np.int64),
            source_bucket_ids=np.zeros(0, dtype=np.uint64),
            lambda_mix=float(cfg.lambda_mix_min),
            domain_transport_rows=0,
            domain_row_frac=float(frac),
            domain_transport_active=False,
            domain_transport_strength=float(domain_transport_strength),
            domain_gap_before=float(gap_before),
            domain_gap_after=float(after),
            domain_transport_gain=0.0,
            domain_overfit_proxy=abs(float(after) - float(domain_gap_train_all if domain_gap_train_all is not None else 0.0)),
            row_type_counts={"hard_anchor": int(min(int(total_budget), rows.size)), "domain_transport": 0},
            metadata={"uses_all_pair_distance": False, "bounded_bucket_reservoir": True},
        )
    deficits = compute_bucket_deficits(selected_mass, all_mass)
    ranked = _rank_deficit_buckets(deficits, all_mass, cfg)
    if not ranked:
        ranked = [int(bucket) for bucket in sorted(all_mass)]
    source_buckets = np.asarray([ranked[idx % len(ranked)] for idx in range(domain_rows)], dtype=np.uint64)
    anchors = np.asarray(
        [
            _anchor_for_bucket(
                bucket=int(bucket),
                bucket_ids=bucket_ids,
                train_rows=train,
                train_labels=labels,
                used_count=idx,
                cfg=cfg,
            )
            for idx, bucket in enumerate(source_buckets.tolist())
        ],
        dtype=np.int64,
    )
    lambda_mix = min(float(cfg.lambda_mix_max), max(float(cfg.lambda_mix_min), float(cfg.lambda_mix_min) + 0.20 * float(domain_transport_strength)))
    virtual_after = _virtual_mass(bucket_ids, rows, source_buckets)
    gap_after = _js_from_mass(virtual_after, all_mass)
    gain = max(0.0, float(gap_before) - float(gap_after))
    train_gap = float(domain_gap_train_all if domain_gap_train_all is not None else 0.0)
    return DomainTransportRows(
        selected_rows=np.concatenate([rows, anchors]).astype(np.int64, copy=False),
        anchor_rows=anchors,
        source_bucket_ids=source_buckets,
        lambda_mix=float(lambda_mix),
        domain_transport_rows=int(domain_rows),
        domain_row_frac=float(frac),
        domain_transport_active=True,
        domain_transport_strength=float(domain_transport_strength),
        domain_gap_before=float(gap_before),
        domain_gap_after=float(gap_after),
        domain_transport_gain=float(gain),
        domain_overfit_proxy=abs(float(gap_after) - train_gap),
        row_type_counts={"hard_anchor": int(rows.size), "domain_transport": int(domain_rows)},
        metadata={
            "uses_all_pair_distance": False,
            "bounded_bucket_reservoir": True,
            "lambda_mix": float(lambda_mix),
            "domain_deficit_buckets": int(len(ranked)),
        },
    )


def apply_domain_transport_to_selection(
    *,
    base_selected_rows: np.ndarray | torch.Tensor | list[int] | tuple[int, ...],
    bucket_ids: np.ndarray,
    train_rows: np.ndarray | torch.Tensor | list[int] | tuple[int, ...],
    train_labels: np.ndarray | torch.Tensor | list[int] | tuple[int, ...],
    total_budget: int,
    num_classes: int,
    domain_gap_train_all: float,
    cfg: DomainTransportConfig | None = None,
    scale_bucket: str | None = None,
) -> DomainTransportRows:
    cfg = cfg or DomainTransportConfig()
    total = max(1, int(total_budget))
    class_capacity_b = float(total) / float(max(1, int(num_classes)))
    strength = compute_domain_transport_strength(float(domain_gap_train_all), class_capacity_b, cfg)
    scale = scale_bucket or scale_bucket_for_num_nodes(int(np.asarray(bucket_ids).shape[0]))
    domain_rows, _ = allocate_domain_row_budget(total, strength, scale, cfg, min_core_rows=min(max(0, int(num_classes)), total))
    base = _as_numpy(base_selected_rows, dtype=np.int64)
    train_np = _as_numpy(train_rows, dtype=np.int64)
    if base.size < total and train_np.size:
        need = total - int(base.size)
        filler = np.resize(train_np, need).astype(np.int64, copy=False)
        base = np.concatenate([base, filler]).astype(np.int64, copy=False)
    before_base = base[:total]
    core_budget = max(0, total - int(domain_rows))
    core = base[:core_budget]
    if int(domain_rows) <= 0:
        all_mass = _mass_from_buckets(np.asarray(bucket_ids, dtype=np.uint64))
        before_gap = _js_from_mass(_virtual_mass(bucket_ids, before_base), all_mass)
        return DomainTransportRows(
            selected_rows=before_base.copy(),
            anchor_rows=np.zeros(0, dtype=np.int64),
            source_bucket_ids=np.zeros(0, dtype=np.uint64),
            lambda_mix=float(cfg.lambda_mix_min),
            domain_transport_rows=0,
            domain_row_frac=0.0,
            domain_transport_active=False,
            domain_transport_strength=float(strength),
            domain_gap_before=float(before_gap),
            domain_gap_after=float(before_gap),
            domain_transport_gain=0.0,
            domain_overfit_proxy=abs(float(before_gap) - float(domain_gap_train_all)),
            row_type_counts={"hard_anchor": int(before_base.size), "domain_transport": 0},
            metadata={"uses_all_pair_distance": False, "bounded_bucket_reservoir": True},
        )
    rows = build_domain_transport_rows(
        bucket_ids=bucket_ids,
        train_rows=train_rows,
        train_labels=train_labels,
        selected_rows=core,
        total_budget=total,
        num_classes=int(num_classes),
        domain_transport_strength=float(strength),
        cfg=cfg,
        scale_bucket=scale,
        domain_gap_train_all=float(domain_gap_train_all),
    )
    all_mass = _mass_from_buckets(np.asarray(bucket_ids, dtype=np.uint64))
    before_gap = _js_from_mass(_virtual_mass(bucket_ids, before_base), all_mass)
    after_gap = rows.domain_gap_after
    selected = rows.selected_rows[:total]
    if selected.size < total and base.size > selected.size:
        selected = np.concatenate([selected, base[selected.size : total]]).astype(np.int64, copy=False)
    return DomainTransportRows(
        selected_rows=selected,
        anchor_rows=rows.anchor_rows,
        source_bucket_ids=rows.source_bucket_ids,
        lambda_mix=rows.lambda_mix,
        domain_transport_rows=int(rows.domain_transport_rows),
        domain_row_frac=float(rows.domain_row_frac),
        domain_transport_active=rows.domain_transport_active,
        domain_transport_strength=float(strength),
        domain_gap_before=float(before_gap),
        domain_gap_after=float(after_gap),
        domain_transport_gain=max(0.0, float(before_gap) - float(after_gap)),
        domain_overfit_proxy=abs(float(after_gap) - float(domain_gap_train_all)),
        row_type_counts={"hard_anchor": int(total - rows.domain_transport_rows), "domain_transport": int(rows.domain_transport_rows)},
        metadata=rows.metadata,
    )
