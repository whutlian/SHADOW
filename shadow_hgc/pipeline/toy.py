from __future__ import annotations

from pathlib import Path

from shadow_hgc.data.loaders import build_toy_graph
from shadow_hgc.pipeline.core import run_shadow_hgc_experiment


def run_toy_experiment(
    *,
    output_path: str | Path = "experiments/logs/toy/summary.json",
    seed: int = 0,
    epochs: int = 40,
    M_tau: int = 4,
    M_r: int | dict | None = 3,
    k_s: int = 2,
    feature_dim: int = 4,
    projection_type: str = "random",
    degree_scale: float = 0.1,
    loss_type: str = "weighted",
    model_type: str = "relation_linear",
    hidden_dim: int = 128,
    dropout: float = 0.3,
    lr: float = 0.03,
    weight_decay: float = 1e-4,
    min_proto_per_class: int = 1,
    shadow_mode: str = "virtual_demand_shadow",
    self_only: bool = False,
) -> dict:
    graph = build_toy_graph(seed=seed)
    return run_shadow_hgc_experiment(
        graph,
        output_path=output_path,
        seed=seed,
        epochs=epochs,
        M_tau=M_tau,
        M_r=M_r,
        k_s=k_s,
        feature_dim=feature_dim,
        projection_type=projection_type,
        degree_scale=degree_scale,
        loss_type=loss_type,
        model_type=model_type,
        hidden_dim=hidden_dim,
        dropout=dropout,
        lr=lr,
        weight_decay=weight_decay,
        min_proto_per_class=min_proto_per_class,
        shadow_mode=shadow_mode,
        self_only=self_only,
    )
