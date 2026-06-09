from __future__ import annotations

import time
from dataclasses import dataclass

import torch

from shadow_hgc.eval.metrics import macro_f1_score
from shadow_hgc.features.metapath_blocks import compute_metapath_feature_blocks
from shadow_hgc.features.metapath_schema import schema_default_metapath_blocks
from shadow_hgc.features.path_lad_v2 import compute_path_lad_v2_blocks
from shadow_hgc.models.losses import prototype_cross_entropy
from shadow_hgc.models.sehgnn_lite import SeHGNNLite
from shadow_hgc.prototype.cluster import class_wise_prototypes


@dataclass
class SeHGNNTargetRun:
    summary: dict
    blocks: dict[str, torch.Tensor]


def _num_classes(labels: torch.Tensor) -> int:
    valid = labels[labels >= 0]
    return 0 if valid.numel() == 0 else int(valid.max().item()) + 1


def _weighted_f1(pred: torch.Tensor, labels: torch.Tensor, idx: torch.Tensor, num_classes: int) -> float:
    selected_pred = pred[idx]
    selected_labels = labels[idx]
    total = 0.0
    score = 0.0
    for class_id in range(num_classes):
        true_mask = selected_labels == class_id
        pred_mask = selected_pred == class_id
        support = float(true_mask.sum().item())
        if support == 0:
            continue
        tp = float((true_mask & pred_mask).sum().item())
        fp = float((~true_mask & pred_mask).sum().item())
        fn = float((true_mask & ~pred_mask).sum().item())
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        score += support * f1
        total += support
    return score / total if total > 0 else 0.0


def _prediction_summary(logits: torch.Tensor, labels: torch.Tensor, idx: torch.Tensor, num_classes: int) -> dict:
    pred = logits.argmax(dim=1).to(torch.long)
    selected = pred[idx]
    acc = float((selected == labels[idx]).to(torch.float32).mean().item()) if idx.numel() else 0.0
    hist = torch.bincount(selected.clamp_min(0), minlength=num_classes).to(torch.float64)
    probs = hist / hist.sum().clamp_min(1.0)
    entropy = float(-(probs[probs > 0] * probs[probs > 0].log()).sum().item()) if hist.numel() else 0.0
    return {
        "accuracy": acc,
        "macro_f1": macro_f1_score(selected, labels[idx], num_classes=num_classes),
        "weighted_f1": _weighted_f1(pred, labels, idx, num_classes),
        "predicted_class_count": int((hist > 0).sum().item()),
        "prediction_entropy": entropy,
        "predicted_class_histogram": {str(i): int(value.item()) for i, value in enumerate(hist) if int(value.item()) > 0},
    }


def build_schema_default_blocks(
    graph,
    *,
    include_self: bool = True,
    include_metapath: bool = True,
    include_path_lad_v2: bool = False,
    requested_blocks: list[str] | None = None,
) -> tuple[dict[str, torch.Tensor], dict]:
    target_type = graph.target_type
    target_features = graph.node_features[target_type].to(torch.float32)
    blocks: dict[str, torch.Tensor] = {}
    if include_self:
        blocks["self"] = target_features
    metadata: dict = {
        "metapath_blocks": [],
        "metapath_skipped_blocks": [],
        "path_lad_blocks": [],
    }
    if include_metapath:
        defaults = schema_default_metapath_blocks(
            dataset_name=graph.dataset_name,
            target_type=target_type,
            relations=list(graph.relations),
            requested_blocks=requested_blocks,
        )
        result = compute_metapath_feature_blocks(
            edge_index=graph.edge_index,
            relations=list(graph.relations),
            target_type=target_type,
            target_features=target_features,
            num_nodes=graph.num_nodes,
            requested_blocks=defaults.available_blocks,
        )
        blocks.update(result.blocks)
        metadata["metapath_blocks"] = list(result.blocks)
        metadata["metapath_skipped_blocks"] = defaults.skipped_blocks
        metadata["metapath_required_blocks"] = defaults.required_blocks
    if include_path_lad_v2:
        train_mask = torch.zeros(graph.num_nodes[target_type], dtype=torch.bool)
        train_mask[graph.train_idx] = True
        path_result = compute_path_lad_v2_blocks(
            graph,
            requested_paths=requested_blocks,
            train_target_mask=train_mask,
            train_labels=graph.labels,
            num_classes=_num_classes(graph.labels),
        )
        blocks.update({f"path_lad:{name}": block for name, block in path_result.blocks.items()})
        metadata.update(path_result.diagnostics)
    return blocks, metadata


def _slice_blocks(blocks: dict[str, torch.Tensor], rows: torch.Tensor) -> dict[str, torch.Tensor]:
    return {name: tensor[rows] for name, tensor in blocks.items()}


def train_fullgraph_sehgnn_lite(
    graph,
    *,
    blocks: dict[str, torch.Tensor],
    metadata: dict,
    seed: int,
    epochs: int,
    hidden_dim: int = 128,
    dropout: float = 0.3,
    lr: float = 0.01,
    weight_decay: float = 1e-4,
    loss_type: str = "weighted",
    label_smoothing: float = 0.0,
) -> SeHGNNTargetRun:
    torch.manual_seed(seed)
    start = time.perf_counter()
    num_classes = _num_classes(graph.labels)
    block_dims = {name: int(tensor.shape[1]) for name, tensor in blocks.items()}
    model = SeHGNNLite(block_dims, num_classes=num_classes, hidden_dim=hidden_dim, dropout=dropout, block_norm="standardize", block_gate=True)
    model.fit_block_stats(_slice_blocks(blocks, graph.train_idx), source="train_full_target_rows")
    model.freeze_block_stats()
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    train_labels = graph.labels[graph.train_idx].to(torch.long)
    train_weights = torch.ones(train_labels.numel(), dtype=torch.float32)
    for _ in range(int(epochs)):
        model.train()
        opt.zero_grad()
        logits = model(_slice_blocks(blocks, graph.train_idx))
        loss = prototype_cross_entropy(
            logits,
            train_labels,
            train_weights,
            loss_type=loss_type,
            label_smoothing=label_smoothing,
        )
        loss.backward()
        opt.step()
    train_time = time.perf_counter() - start
    infer_start = time.perf_counter()
    model.eval()
    with torch.no_grad():
        all_logits = model(blocks)
    infer_time = time.perf_counter() - infer_start
    summary = {
        "model_type": "sehgnn_lite",
        "block_dims": block_dims,
        "feature_blocks": list(blocks),
        "block_norm_stats_source": "train_full_target_rows",
        "final_logits_activation": "none",
        "block_gates": model.block_gate_values(),
        "num_classes": num_classes,
        "train_time": train_time,
        "infer_time": infer_time,
        "loss_type": loss_type,
        "label_smoothing": float(label_smoothing),
        **metadata,
        **_prediction_summary(all_logits, graph.labels, graph.test_idx, num_classes),
    }
    train_summary = _prediction_summary(all_logits, graph.labels, graph.train_idx, num_classes)
    summary["train_accuracy"] = train_summary["accuracy"]
    return SeHGNNTargetRun(summary=summary, blocks=blocks)


def train_prototype_sehgnn_lite(
    graph,
    *,
    blocks: dict[str, torch.Tensor],
    metadata: dict,
    requested_ratio: float,
    seed: int,
    epochs: int,
    hidden_dim: int = 128,
    dropout: float = 0.3,
    lr: float = 0.01,
    weight_decay: float = 1e-4,
    loss_type: str = "weighted",
    min_proto_per_class: int = 1,
) -> SeHGNNTargetRun:
    torch.manual_seed(seed)
    num_train = int(graph.train_idx.numel())
    requested_budget = max(1, int(round(float(requested_ratio) * num_train)))
    signature = torch.cat([tensor[graph.train_idx] for tensor in blocks.values()], dim=1)
    phi_for_proto = blocks.get("self", next(iter(blocks.values())))
    proto = class_wise_prototypes(
        phi_target=phi_for_proto,
        signatures=signature,
        labels=graph.labels,
        train_idx=graph.train_idx,
        M_tau=requested_budget,
        signature_idx=graph.train_idx,
        min_proto_per_class=min_proto_per_class,
        seed=seed,
    )
    proto_blocks = {
        name: torch.stack([tensor[members].mean(dim=0) for members in proto.cell_members], dim=0)
        for name, tensor in blocks.items()
    }
    start = time.perf_counter()
    num_classes = _num_classes(graph.labels)
    block_dims = {name: int(tensor.shape[1]) for name, tensor in proto_blocks.items()}
    model = SeHGNNLite(block_dims, num_classes=num_classes, hidden_dim=hidden_dim, dropout=dropout, block_norm="standardize", block_gate=True)
    model.fit_block_stats(_slice_blocks(blocks, graph.train_idx), source="train_full_target_rows")
    model.freeze_block_stats()
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    for _ in range(int(epochs)):
        model.train()
        opt.zero_grad()
        logits = model(proto_blocks)
        loss = prototype_cross_entropy(logits, proto.prototype_labels, proto.prototype_weights, loss_type=loss_type)
        loss.backward()
        opt.step()
    train_time = time.perf_counter() - start
    infer_start = time.perf_counter()
    model.eval()
    with torch.no_grad():
        all_logits = model(blocks)
    infer_time = time.perf_counter() - infer_start
    summary = {
        "model_type": "sehgnn_lite",
        "block_dims": block_dims,
        "feature_blocks": list(blocks),
        "block_norm_stats_source": "train_full_target_rows",
        "final_logits_activation": "none",
        "block_gates": model.block_gate_values(),
        "num_classes": num_classes,
        "requested_ratio": float(requested_ratio),
        "ratio_mode": "train_target",
        "effective_M_tau": int(proto.effective_M_tau),
        "condensed_nodes_total": int(proto.effective_M_tau),
        "total_condensed_node_ratio": float(proto.effective_M_tau / max(1, sum(graph.num_nodes.values()))),
        "effective_target_ratio": float(proto.effective_M_tau / max(1, num_train)),
        "prototype_mode": "class_wise_kmeans_mean",
        "train_time": train_time,
        "infer_time": infer_time,
        "loss_type": loss_type,
        **metadata,
        **_prediction_summary(all_logits, graph.labels, graph.test_idx, num_classes),
    }
    return SeHGNNTargetRun(summary=summary, blocks=blocks)
