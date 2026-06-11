from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TTCCondensedTable:
    z_syn: torch.Tensor
    y_syn_soft: torch.Tensor
    y_syn_hard: torch.Tensor
    hard_anchor_mask: torch.Tensor
    source_node_ids: torch.Tensor
    bucket_types: list[str]
    sample_weight: torch.Tensor
    diagnostics: dict[str, Any]


def _as_probabilities(probs: torch.Tensor) -> torch.Tensor:
    values = probs.detach().float()
    values = values.clamp_min(0.0)
    denom = values.sum(dim=1, keepdim=True).clamp_min(1e-12)
    return values / denom


def teacher_probability_diagnostics(probs: torch.Tensor, disagreement: torch.Tensor | None = None) -> dict[str, Any]:
    p = _as_probabilities(probs)
    entropy = -(p.clamp_min(1e-12) * p.clamp_min(1e-12).log()).sum(dim=1)
    top2 = torch.topk(p, k=min(2, p.shape[1]), dim=1).values
    margin = top2[:, 0] - (top2[:, 1] if top2.shape[1] > 1 else 0.0)
    predicted = p.argmax(dim=1)
    if disagreement is None:
        disagreement_mean = 0.0
    else:
        disagreement_mean = float(disagreement.detach().float().mean().item())
    return {
        "teacher_entropy_mean": float(entropy.mean().item()) if entropy.numel() else 0.0,
        "teacher_margin_mean": float(margin.mean().item()) if margin.numel() else 0.0,
        "teacher_disagreement_mean": disagreement_mean,
        "predicted_classes": int(predicted.unique().numel()) if predicted.numel() else 0,
        "teacher_confidence_mean": float(p.max(dim=1).values.mean().item()) if p.numel() else 0.0,
    }


def _budget_allocation(num_rows: int, *, mixup: bool) -> dict[str, int]:
    weights = {
        "confidence_core": 0.40,
        "boundary": 0.25,
        "disagreement": 0.15,
        "rare_structure": 0.10,
        "prior_repair": 0.10,
    }
    if mixup:
        weights = {
            "confidence_core": 0.32,
            "boundary": 0.22,
            "disagreement": 0.13,
            "rare_structure": 0.08,
            "prior_repair": 0.10,
            "mixup": 0.15,
        }
    raw = {key: num_rows * value for key, value in weights.items()}
    alloc = {key: int(math.floor(value)) for key, value in raw.items()}
    remaining = num_rows - sum(alloc.values())
    order = sorted(raw, key=lambda key: (raw[key] - alloc[key], weights[key]), reverse=True)
    for key in order[:remaining]:
        alloc[key] += 1
    return {key: value for key, value in alloc.items() if value > 0}


def _take_unique(scores: torch.Tensor, count: int, used: set[int], *, largest: bool = True) -> list[int]:
    if count <= 0:
        return []
    order = torch.argsort(scores, descending=largest).tolist()
    selected: list[int] = []
    for idx in order:
        if idx in used:
            continue
        selected.append(int(idx))
        used.add(int(idx))
        if len(selected) == count:
            break
    if len(selected) < count:
        for idx in order:
            selected.append(int(idx))
            if len(selected) == count:
                break
    return selected


def _round_robin_by_class(scores: torch.Tensor, pred: torch.Tensor, count: int, used: set[int], *, largest: bool = True) -> list[int]:
    if count <= 0:
        return []
    classes = torch.unique(pred).tolist()
    per_class: dict[int, list[int]] = {}
    for cls in classes:
        mask_idx = torch.nonzero(pred == int(cls), as_tuple=False).flatten()
        class_scores = scores[mask_idx]
        order = mask_idx[torch.argsort(class_scores, descending=largest)].tolist()
        per_class[int(cls)] = [int(v) for v in order]
    selected: list[int] = []
    while len(selected) < count and any(per_class.values()):
        for cls in sorted(per_class):
            bucket = per_class[cls]
            while bucket:
                idx = bucket.pop(0)
                if idx not in used:
                    selected.append(idx)
                    used.add(idx)
                    break
            if len(selected) == count:
                break
    if len(selected) < count:
        selected.extend(_take_unique(scores, count - len(selected), used, largest=largest))
    return selected


def _rare_structure_scores(features: torch.Tensor) -> torch.Tensor:
    norms = features.detach().float().norm(dim=1)
    if norms.numel() == 0:
        return norms
    q = torch.quantile(norms, torch.tensor([0.2, 0.4, 0.6, 0.8], device=norms.device))
    bucket = torch.bucketize(norms, q)
    counts = torch.bincount(bucket.cpu(), minlength=5).float().clamp_min(1.0)
    return (1.0 / counts[bucket.cpu()]).to(norms.device)


def _coverage(scores: torch.Tensor, selected: torch.Tensor) -> float:
    if selected.numel() == 0 or scores.numel() == 0:
        return 0.0
    threshold = torch.quantile(scores.detach().float(), 0.5)
    return float((scores[selected] >= threshold).float().mean().item())


def build_ttc_condensed_table(
    *,
    features: torch.Tensor,
    teacher_probs: torch.Tensor,
    labels: torch.Tensor | None,
    train_idx: torch.Tensor,
    valid_idx: torch.Tensor | None = None,
    test_idx: torch.Tensor | None = None,
    num_rows: int,
    mode: str = "ttc_coverage_plus_boundary",
    seed: int = 42,
    mixup_alpha: float = 0.4,
    disagreement: torch.Tensor | None = None,
) -> TTCCondensedTable:
    del valid_idx, test_idx
    if num_rows <= 0:
        raise ValueError("num_rows must be positive")
    x = features.detach().float().cpu()
    probs = _as_probabilities(teacher_probs).cpu()
    n, d = x.shape
    if probs.shape[0] != n:
        raise ValueError("features and teacher_probs must have the same first dimension")
    confidence = probs.max(dim=1).values
    entropy = -(probs.clamp_min(1e-12) * probs.clamp_min(1e-12).log()).sum(dim=1)
    top2 = torch.topk(probs, k=min(2, probs.shape[1]), dim=1).values
    margin = top2[:, 0] - (top2[:, 1] if top2.shape[1] > 1 else 0.0)
    pred = probs.argmax(dim=1)
    dis = disagreement.detach().float().cpu() if disagreement is not None else entropy - entropy.mean()
    rare = _rare_structure_scores(x)
    mixup = "mixup" in mode
    allocation = _budget_allocation(num_rows, mixup=mixup)
    used: set[int] = set()
    selected_ids: list[int] = []
    bucket_types: list[str] = []

    def add(bucket: str, ids: list[int]) -> None:
        selected_ids.extend(ids)
        bucket_types.extend([bucket] * len(ids))

    add("confidence_core", _round_robin_by_class(confidence, pred, allocation.get("confidence_core", 0), used, largest=True))
    add("boundary", _round_robin_by_class(margin, pred, allocation.get("boundary", 0), used, largest=False))
    add("disagreement", _take_unique(dis, allocation.get("disagreement", 0), used, largest=True))
    add("rare_structure", _take_unique(rare, allocation.get("rare_structure", 0), used, largest=True))
    add("prior_repair", _round_robin_by_class(-torch.bincount(pred, minlength=probs.shape[1]).float()[pred], pred, allocation.get("prior_repair", 0), used, largest=True))

    real_budget = num_rows - allocation.get("mixup", 0)
    if len(selected_ids) < real_budget:
        add("confidence_core", _take_unique(confidence, real_budget - len(selected_ids), used, largest=True))
    selected_ids = selected_ids[:real_budget]
    bucket_types = bucket_types[:real_budget]

    z_rows = [x[idx] for idx in selected_ids]
    y_rows = [probs[idx] for idx in selected_ids]
    src_rows = selected_ids[:]
    rng = torch.Generator(device="cpu")
    rng.manual_seed(int(seed))
    mix_count = allocation.get("mixup", 0)
    if mix_count > 0:
        base = selected_ids if selected_ids else list(range(n))
        boundary_order = torch.argsort(margin, descending=False).tolist()
        for i in range(mix_count):
            a = int(base[i % len(base)])
            b = int(boundary_order[(i + seed) % len(boundary_order)])
            lam = float(torch.distributions.Beta(mixup_alpha, mixup_alpha).sample((1,)).item())
            z_rows.append(lam * x[a] + (1.0 - lam) * x[b])
            y_rows.append(lam * probs[a] + (1.0 - lam) * probs[b])
            src_rows.append(-1)
            bucket_types.append("mixup")

    while len(z_rows) < num_rows:
        idx = int(torch.argsort(confidence, descending=True)[len(z_rows) % n])
        z_rows.append(x[idx])
        y_rows.append(probs[idx])
        src_rows.append(idx)
        bucket_types.append("confidence_core")

    z_syn = torch.stack(z_rows[:num_rows], dim=0).view(num_rows, d)
    y_syn_soft = _as_probabilities(torch.stack(y_rows[:num_rows], dim=0))
    source_node_ids = torch.tensor(src_rows[:num_rows], dtype=torch.long)
    train_set = {int(v) for v in train_idx.detach().cpu().tolist()}
    hard_anchor_mask = torch.tensor([int(v) in train_set and int(v) >= 0 for v in source_node_ids.tolist()], dtype=torch.bool)
    y_syn_hard = torch.full((num_rows,), -1, dtype=torch.long)
    if labels is not None and hard_anchor_mask.any():
        labels_cpu = labels.detach().cpu().long()
        y_syn_hard[hard_anchor_mask] = labels_cpu[source_node_ids[hard_anchor_mask]]
    selected_bucket_counts: dict[str, int] = {}
    for bucket in bucket_types[:num_rows]:
        selected_bucket_counts[bucket] = selected_bucket_counts.get(bucket, 0) + 1
    diagnostics = {
        **teacher_probability_diagnostics(probs, disagreement=disagreement),
        "candidate_nodes": "all",
        "candidate_node_count": int(n),
        "candidate_bucket_counts": {
            "confidence_core": int(n),
            "boundary": int(n),
            "disagreement": int(n),
            "rare_structure": int(n),
            "prior_repair": int(n),
        },
        "selected_bucket_counts": selected_bucket_counts,
        "soft_class_mass_coverage": float(y_syn_soft.sum(dim=0).clamp_max(1.0).sum().item() / max(1, probs.shape[1])),
        "entropy_bucket_coverage": _coverage(entropy, source_node_ids[source_node_ids >= 0]),
        "margin_bucket_coverage": _coverage(-margin, source_node_ids[source_node_ids >= 0]),
        "degree_bucket_coverage": _coverage(rare, source_node_ids[source_node_ids >= 0]),
        "hard_anchor_count": int(hard_anchor_mask.sum().item()),
        "soft_only_count": int((~hard_anchor_mask).sum().item()),
        "mixup_row_count": int(sum(1 for bucket in bucket_types[:num_rows] if bucket == "mixup")),
        "uses_valid_labels_as_input": False,
        "uses_test_labels_as_input": False,
    }
    return TTCCondensedTable(
        z_syn=z_syn,
        y_syn_soft=y_syn_soft,
        y_syn_hard=y_syn_hard,
        hard_anchor_mask=hard_anchor_mask,
        source_node_ids=source_node_ids,
        bucket_types=bucket_types[:num_rows],
        sample_weight=torch.ones(num_rows, dtype=torch.float32),
        diagnostics=diagnostics,
    )


class _SoftStudent(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def macro_f1_from_predictions(pred: torch.Tensor, labels: torch.Tensor, num_classes: int) -> float:
    pred_cpu = pred.detach().cpu()
    labels_cpu = labels.detach().cpu()
    values: list[float] = []
    for cls in range(num_classes):
        p = pred_cpu == cls
        y = labels_cpu == cls
        tp = int((p & y).sum().item())
        fp = int((p & ~y).sum().item())
        fn = int((~p & y).sum().item())
        denom = 2 * tp + fp + fn
        values.append((2 * tp / denom) if denom else 0.0)
    return float(sum(values) / len(values)) if values else 0.0


def train_soft_label_condensed_student(
    *,
    z_syn: torch.Tensor,
    y_syn_soft: torch.Tensor,
    eval_features: torch.Tensor,
    eval_labels: torch.Tensor,
    train_anchor_hard: torch.Tensor | None = None,
    hard_anchor_mask: torch.Tensor | None = None,
    valid_features: torch.Tensor | None = None,
    valid_labels: torch.Tensor | None = None,
    hidden_dim: int = 128,
    epochs: int = 120,
    lr: float = 0.01,
    weight_decay: float = 5e-4,
    dropout: float = 0.1,
    temperature: float = 2.0,
    lambda_hard: float = 0.5,
    lambda_prior: float = 0.05,
    target_prior: torch.Tensor | None = None,
    device: str = "cpu",
    seed: int = 42,
) -> dict[str, Any]:
    torch.manual_seed(int(seed))
    dev = torch.device(device if device == "cpu" or torch.cuda.is_available() else "cpu")
    z = z_syn.float().to(dev)
    y = _as_probabilities(y_syn_soft).to(dev)
    model = _SoftStudent(z.shape[1], int(hidden_dim), y.shape[1], float(dropout)).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    prior = _as_probabilities(target_prior.view(1, -1)).flatten().to(dev) if target_prior is not None else y.mean(dim=0)
    hard_labels = train_anchor_hard.to(dev) if train_anchor_hard is not None else None
    hard_mask = hard_anchor_mask.to(dev) if hard_anchor_mask is not None else None
    for _ in range(int(epochs)):
        model.train()
        logits = model(z)
        logp_t = F.log_softmax(logits / temperature, dim=1)
        soft_loss = F.kl_div(logp_t, y, reduction="batchmean") * (temperature * temperature)
        loss = soft_loss
        if hard_labels is not None and hard_mask is not None and hard_mask.any():
            loss = loss + float(lambda_hard) * F.cross_entropy(logits[hard_mask], hard_labels[hard_mask])
        mean_pred = F.softmax(logits, dim=1).mean(dim=0).clamp_min(1e-12)
        loss = loss + float(lambda_prior) * F.kl_div(mean_pred.log(), prior, reduction="sum")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        test_logits = model(eval_features.float().to(dev))
        pred = test_logits.argmax(dim=1).cpu()
        labels_cpu = eval_labels.long().cpu()
        acc = float((pred == labels_cpu).float().mean().item())
        macro = macro_f1_from_predictions(pred, labels_cpu, y.shape[1])
        valid_acc = ""
        if valid_features is not None and valid_labels is not None:
            valid_pred = model(valid_features.float().to(dev)).argmax(dim=1).cpu()
            valid_acc = float((valid_pred == valid_labels.long().cpu()).float().mean().item())
    return {
        "accuracy": acc,
        "macro_f1": macro,
        "valid_acc": valid_acc,
        "predicted_classes": int(pred.unique().numel()),
        "model": model.cpu(),
    }
