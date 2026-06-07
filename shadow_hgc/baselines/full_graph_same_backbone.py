from __future__ import annotations

import time
from pathlib import Path

import torch
import torch.nn.functional as F

from shadow_hgc.data.loaders import HeteroGraphData
from shadow_hgc.models.losses import prototype_cross_entropy
from shadow_hgc.models.weighted_rel_linear import RelationMessageEncoderMLP, WeightedRelationLinearConv
from shadow_hgc.pipeline.core import (
    infer_class_metadata,
    prepare_model_features,
    prediction_diagnostics,
    _relation_demand,
)
from shadow_hgc.eval.metrics import macro_f1_score
from shadow_hgc.eval.logging import attach_run_metadata, write_json_summary


def run_full_graph_same_backbone(
    graph: HeteroGraphData,
    *,
    output_path: str | Path,
    seed: int,
    epochs: int,
    feature_dim: int,
    projection_type: str = "random",
    degree_scale: float = 0.1,
    model_type: str = "relation_linear",
    hidden_dim: int = 128,
    dropout: float = 0.3,
    lr: float = 0.03,
    weight_decay: float = 1e-4,
    loss_type: str = "weighted",
    inference_edge_chunk_size: int | None = 500_000,
) -> dict:
    torch.manual_seed(seed)
    _, phi, _, target_relations = prepare_model_features(
        graph,
        feature_dim=feature_dim,
        seed=seed,
        projection_type=projection_type,
        degree_scale=degree_scale,
    )
    _, alpha = _relation_demand(graph, phi, target_relations, edge_chunk_size=inference_edge_chunk_size)
    class_metadata = infer_class_metadata(graph.labels, graph.train_idx, graph.test_idx)
    num_classes = class_metadata["num_classes_global"]
    in_channels = {node_type: features.shape[1] for node_type, features in phi.items()}
    if model_type == "relation_mlp":
        model = RelationMessageEncoderMLP(
            in_channels=in_channels,
            out_channels=num_classes,
            node_types=list(phi),
            relations=target_relations,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
    else:
        model = WeightedRelationLinearConv(
            in_channels=in_channels,
            out_channels=num_classes,
            node_types=list(phi),
            relations=target_relations,
            activation=None,
        )
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    edge_weight = {relation: alpha[relation] for relation in target_relations}
    train_loss_start = None
    train_loss_end = None
    start = time.perf_counter()
    for _ in range(epochs):
        opt.zero_grad()
        out = model(phi, graph.edge_index, edge_weight, edge_chunk_size=inference_edge_chunk_size)
        logits = out[graph.target_type][graph.train_idx]
        weights = torch.ones(logits.shape[0], dtype=torch.float32)
        loss = prototype_cross_entropy(logits, graph.labels[graph.train_idx], weights, loss_type=loss_type)
        if train_loss_start is None:
            train_loss_start = float(loss.detach().item())
        loss.backward()
        opt.step()
        train_loss_end = float(loss.detach().item())
    training_time = time.perf_counter() - start
    with torch.no_grad():
        out = model(phi, graph.edge_index, edge_weight, edge_chunk_size=inference_edge_chunk_size)
        pred = out[graph.target_type].argmax(dim=1)
    accuracy = None
    macro_f1 = None
    if graph.test_idx.numel() > 0:
        accuracy = float((pred[graph.test_idx] == graph.labels[graph.test_idx]).to(torch.float32).mean().item())
        macro_f1 = macro_f1_score(pred[graph.test_idx], graph.labels[graph.test_idx], num_classes=num_classes)
    summary = {
        "method": "Full-WRL-GNN",
        "baseline": "Full-WRL-GNN",
        "dataset": graph.dataset_name,
        "target_type": graph.target_type,
        "directed_relations": [str(relation) for relation in target_relations],
        "feature_dim": feature_dim,
        "projection_type": projection_type,
        "degree_scale": degree_scale,
        "model": model_type,
        "loss_type": loss_type,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "training_time": training_time,
        "train_loss_start": train_loss_start,
        "train_loss_end": train_loss_end,
        "num_optimizer_steps": int(epochs),
        "num_epochs": int(epochs),
        **class_metadata,
        **prediction_diagnostics(pred, graph.labels, graph.test_idx, num_classes=num_classes),
    }
    config_for_hash = {
        "baseline": "Full-WRL-GNN",
        "dataset": graph.dataset_name,
        "seed": seed,
        "epochs": epochs,
        "feature_dim": feature_dim,
        "projection_type": projection_type,
        "degree_scale": degree_scale,
        "model": model_type,
        "hidden_dim": hidden_dim,
        "dropout": dropout,
        "lr": lr,
        "weight_decay": weight_decay,
        "loss_type": loss_type,
    }
    summary = attach_run_metadata(summary, config=config_for_hash)
    write_json_summary(output_path, summary, config=config_for_hash)
    return summary
