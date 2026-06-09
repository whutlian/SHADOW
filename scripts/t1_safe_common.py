from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

from scripts.run_t1_available_logit_verification import _build_blocks, _combined_target_edge_index, _metrics, _split_train_valid
from scripts.run_t0s_sfb_v2_fullgraph import _load_graph, _num_classes
from shadow_hgc.fullgraph.sfb_v2_train import train_sfb_v2_table_model
from shadow_hgc.logits import LogitCacheMeta, load_logits_cache, save_logits_cache
from shadow_hgc.logits.metadata import now_iso


SAFE_BASES: list[dict[str, Any]] = [
    {"dataset": "acm", "base_variant": "SFB-v2 B3_scap_v2 retained", "cache_variant": "B3_scap_v2", "expected_acc": 0.915486, "macro_f1": 0.916580},
    {"dataset": "dblp", "base_variant": "R+ relation-linear current-best", "cache_variant": "rplus_current_best", "expected_acc": 0.836972, "macro_f1": 0.829937},
    {"dataset": "imdb", "base_variant": "clean S1 MAM/MDM/MKM", "cache_variant": "s1_clean_mam_mdm_mkm", "expected_acc": 0.424110, "macro_f1": 0.353932},
    {"dataset": "ogbn-arxiv", "base_variant": "LAD_reference", "cache_variant": "lad_reference", "expected_acc": 0.596774, "macro_f1": 0.415452},
    {"dataset": "ogbn-products", "base_variant": "R++ base shadow-fusion", "cache_variant": "rpp_base_shadow_fusion", "expected_acc": 0.668908, "macro_f1": 0.307981},
    {"dataset": "ogbn-products", "base_variant": "LAD_reference", "cache_variant": "lad_reference", "expected_acc": 0.658674, "macro_f1": 0.338064},
]


def stable_hash(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def read_csv(path: str | Path) -> list[dict[str, str]]:
    if not Path(path).exists():
        return []
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def small_train_args(seed: int) -> SimpleNamespace:
    return SimpleNamespace(
        seed=int(seed),
        medium_edge_limit=0,
        medium_feature_dim=64,
        edge_chunk_size=65536,
        scap_topk=8,
    )


def train_sfb_v2_cache(
    *,
    dataset: str,
    cache_variant: str,
    seed: int,
    gate_mode: bool,
    epochs: int,
    val_fraction: float = 0.2,
) -> tuple[Any, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    graph = _load_graph(dataset)
    args = small_train_args(seed)
    blocks = _build_blocks(graph, dataset, cache_variant, args)
    if gate_mode:
        train_rows, valid_rows = _split_train_valid(graph.labels, graph.train_idx, seed=seed, val_fraction=val_fraction)
    else:
        train_rows = graph.train_idx
        valid_rows = torch.empty(0, dtype=torch.long)
    model_valid_rows = valid_rows if valid_rows.numel() else graph.train_idx
    result = train_sfb_v2_table_model(
        blocks,
        graph.labels,
        train_rows,
        model_valid_rows,
        graph.test_idx,
        num_classes=_num_classes(graph.labels),
        seed=seed,
        epochs=epochs,
        patience=20,
        hidden_dim=256,
        branch_dropout=0.3,
        lr=0.003,
        weight_decay=5e-4,
        batch_size=None,
    )
    metrics = {
        "train": _metrics(result.logits, graph.labels, train_rows, _num_classes(graph.labels)),
        "valid": _metrics(result.logits, graph.labels, model_valid_rows, _num_classes(graph.labels)),
        "test": _metrics(result.logits, graph.labels, graph.test_idx, _num_classes(graph.labels)),
    }
    return graph, result.logits, train_rows, valid_rows, metrics


def save_cache_for_graph(
    *,
    root: Path,
    graph,
    logits: torch.Tensor,
    train_rows: torch.Tensor,
    valid_rows: torch.Tensor,
    base: dict[str, Any],
    seed: int,
    role: str,
    metrics: dict[str, Any],
    dtype: str,
) -> Path:
    cache_dir = root / f"{base['dataset']}_{base['cache_variant']}_{role}_seed{seed}"
    num_classes = _num_classes(graph.labels)
    meta = LogitCacheMeta(
        dataset=base["dataset"],
        variant=base["base_variant"],
        seed=int(seed),
        num_target_nodes=int(graph.num_nodes[graph.target_type]),
        num_classes=int(num_classes),
        target_type=graph.target_type,
        split_hash=stable_hash({"dataset": base["dataset"], "role": role, "seed": seed, "train": train_rows.tolist(), "valid": valid_rows.tolist()}),
        feature_hash=stable_hash({"dataset": base["dataset"], "variant": base["cache_variant"], "role": role}),
        uses_diffusion=False,
        uses_dense_p2=False,
        uses_bounded_edges=False,
        uses_source_anchors=False,
        uses_coverage_medoid=False,
        uses_old_kd=False,
        accuracy=float(metrics["test"]["accuracy"]),
        macro_f1=float(metrics["test"]["macro_f1"]),
        predicted_class_count=int(metrics["test"]["predicted_class_count"]),
        created_at=now_iso(),
    )
    valid_logits = logits[valid_rows] if valid_rows.numel() else torch.empty(0, num_classes)
    y_valid = graph.labels[valid_rows] if valid_rows.numel() else None
    valid_idx = valid_rows if valid_rows.numel() else None
    path = save_logits_cache(
        cache_dir,
        train_logits=logits[train_rows],
        valid_logits=valid_logits,
        test_logits=logits[graph.test_idx],
        all_target_logits=logits,
        y_train=graph.labels[train_rows],
        y_valid=y_valid,
        y_test=graph.labels[graph.test_idx],
        train_idx=train_rows,
        valid_idx=valid_idx,
        test_idx=graph.test_idx,
        meta=meta,
        dtype=dtype,
    )
    metadata_path = path / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.update(
        {
            "model_config_hash": stable_hash({"model": "sfb_v2", "cache_variant": base["cache_variant"], "role": role, "seed": seed}),
            "cache_role": role,
            "base_variant": base["base_variant"],
            "historical_expected_acc": float(base["expected_acc"]),
            "cache_status": "available_unreplayed",
        }
    )
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def target_edge_for_cache(cache_dir: str | Path) -> torch.Tensor | None:
    loaded = load_logits_cache(cache_dir)
    graph = _load_graph(loaded.meta.dataset)
    return _combined_target_edge_index(graph)
