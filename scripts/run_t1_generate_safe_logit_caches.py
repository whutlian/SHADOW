from __future__ import annotations

import argparse
from dataclasses import replace
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_lad_common import DATASET_LOSS, write_csv
from scripts.run_t1_available_logit_verification import _metrics, _split_train_valid
from scripts.t1_safe_common import SAFE_BASES, save_cache_for_graph, stable_hash, train_sfb_v2_cache
from shadow_hgc.data.ogb import load_ogb_node_property_dataset
from shadow_hgc.data.small import load_processed_small_dataset
from shadow_hgc.eval.budgeting import ratio_slug
from shadow_hgc.eval.logging import write_json_summary
from shadow_hgc.eval.status import exception_status
from shadow_hgc.logits import load_logits_cache
from shadow_hgc.pipeline.core import run_shadow_hgc_experiment
from shadow_hgc.train.sehgnn_lite_target import build_schema_default_blocks, train_prototype_sehgnn_lite


FIELDS = [
    "dataset",
    "base_variant",
    "cache_status",
    "historical_test_acc",
    "replay_test_acc",
    "delta_replay",
    "macro_f1",
    "predicted_class_count",
    "train_nodes",
    "valid_nodes",
    "test_nodes",
    "all_target_nodes",
    "split_hash",
    "feature_hash",
    "model_config_hash",
    "cache_path",
    "gate_cache_path",
    "forbidden_component_flags",
    "blocked_reason",
]


def _num_classes(labels: torch.Tensor) -> int:
    valid = labels[labels >= 0]
    return 0 if valid.numel() == 0 else int(valid.max().item()) + 1


def _metadata(path: Path) -> dict:
    return json.loads((path / "metadata.json").read_text(encoding="utf-8"))


def _cache_dir(cache_root: Path, base: dict, role: str, seed: int) -> Path:
    return cache_root / f"{base['dataset']}_{base['cache_variant']}_{role}_seed{seed}"


def _model_config_payload(base: dict, args, role: str) -> dict:
    common = {
        "dataset": base["dataset"],
        "base_variant": base["base_variant"],
        "cache_variant": base["cache_variant"],
        "role": role,
        "seed": int(args.seed),
    }
    if base["dataset"] == "acm":
        return {
            **common,
            "model": "sfb_v2_table",
            "epochs": int(args.epochs),
            "source": "train_sfb_v2_cache",
        }
    if base["dataset"] == "dblp":
        return {
            **common,
            "model": "pipeline_rplus_relation_linear",
            "epochs": int(args.dblp_epochs),
            "ratio": 0.065,
            "feature_mode": "metapath",
            "loss_type": "clipped",
            "projection_type": "raw",
            "shadow_policy": "rank_adaptive",
            "adaptive_b": True,
            "relation_gate": True,
        }
    if base["dataset"] == "imdb":
        return {
            **common,
            "model": "sehgnn_lite_prototype",
            "epochs": int(args.imdb_epochs),
            "ratio": 0.05,
            "feature_blocks": "schema_default_metapath",
            "loss_type": DATASET_LOSS["imdb"],
            "hidden_dim": 128,
        }
    if base["dataset"] in {"ogbn-arxiv", "ogbn-products"} and base["cache_variant"] == "lad_reference":
        return {
            **common,
            "model": "pipeline_lad_reference_compiled_demand_mlp",
            "epochs": int(args.medium_lad_epochs),
            "ratio": 0.12,
            "feature_mode": "label_affinity",
            "label_affinity_mode": "target_target",
            "compiled_head": True,
            "compiled_block_stats_source": "train_full_demand_table",
            "medium_edge_chunk_size": int(args.medium_edge_chunk_size),
            "medium_inference_batch_size": int(args.medium_inference_batch_size),
        }
    if base["dataset"] == "ogbn-products" and base["cache_variant"] == "rpp_base_shadow_fusion":
        return {
            **common,
            "model": "pipeline_rpp_base_shadow_fusion",
            "epochs": int(args.products_rpp_epochs),
            "ratio": 0.12,
            "feature_mode": "base",
            "loss_type": "sqrt_weighted",
            "relation_gate": True,
            "products_inference_dst_chunk_size": int(args.products_inference_dst_chunk_size),
        }
    return common


def _validate_reusable_cache(cache_dir: Path, base: dict, args, *, role: str) -> tuple[bool, str]:
    if not cache_dir.exists():
        return False, "cache_dir_missing"
    try:
        loaded = load_logits_cache(cache_dir)
        metadata = _metadata(cache_dir)
    except Exception as exc:
        return False, f"cache_load_failed:{exc}"
    if loaded.all_target_logits is None:
        return False, "missing_all_target_logits"
    if loaded.meta.dataset != base["dataset"]:
        return False, "dataset_mismatch"
    if loaded.meta.variant != base["base_variant"]:
        return False, "base_variant_mismatch"
    if int(loaded.meta.seed) != int(args.seed):
        return False, "seed_mismatch"
    if metadata.get("cache_role") != role:
        return False, "cache_role_mismatch"
    if metadata.get("base_variant") != base["base_variant"]:
        return False, "metadata_base_variant_mismatch"
    if abs(float(metadata.get("historical_expected_acc", -1.0)) - float(base["expected_acc"])) > 1e-12:
        return False, "historical_expected_acc_mismatch"
    expected_hash = stable_hash(_model_config_payload(base, args, role))
    if metadata.get("model_config_hash") != expected_hash:
        return False, "model_config_hash_mismatch"
    if metadata.get("uses_diffusion") or metadata.get("uses_dense_p2") or metadata.get("uses_bounded_edges"):
        return False, "forbidden_component_flag"
    if metadata.get("uses_source_anchors") or metadata.get("uses_coverage_medoid") or metadata.get("uses_old_kd"):
        return False, "forbidden_component_flag"
    if loaded.storage.get("forbidden_reasons"):
        return False, "storage_forbidden_reasons"
    return True, ""


def _existing_cache_row(base: dict, args, *, gate_same_as_replay: bool = False) -> dict | None:
    cache_root = Path(args.cache_root)
    replay_cache = _cache_dir(cache_root, base, "historical_replay", args.seed)
    gate_cache = replay_cache if gate_same_as_replay else _cache_dir(cache_root, base, "gate_selection", args.seed)
    replay_ok, _ = _validate_reusable_cache(replay_cache, base, args, role="historical_replay")
    gate_ok = replay_ok if gate_same_as_replay else _validate_reusable_cache(gate_cache, base, args, role="gate_selection")[0]
    if not replay_ok or not gate_ok:
        return None
    loaded = load_logits_cache(replay_cache)
    meta = _metadata(replay_cache)
    valid_nodes = 0 if loaded.valid_idx is None else int(loaded.valid_idx.shape[0])
    return {
        "dataset": base["dataset"],
        "base_variant": base["base_variant"],
        "cache_status": "available_unreplayed",
        "historical_test_acc": base["expected_acc"],
        "replay_test_acc": "",
        "delta_replay": "",
        "macro_f1": "" if loaded.meta.macro_f1 is None else float(loaded.meta.macro_f1),
        "predicted_class_count": "" if loaded.meta.predicted_class_count is None else int(loaded.meta.predicted_class_count),
        "train_nodes": int(loaded.train_idx.shape[0]),
        "valid_nodes": valid_nodes,
        "test_nodes": 0 if loaded.test_idx is None else int(loaded.test_idx.shape[0]),
        "all_target_nodes": int(loaded.meta.num_target_nodes),
        "split_hash": meta["split_hash"],
        "feature_hash": meta["feature_hash"],
        "model_config_hash": meta["model_config_hash"],
        "cache_path": str(replay_cache),
        "gate_cache_path": str(gate_cache),
        "forbidden_component_flags": "[]",
        "blocked_reason": "",
    }


def _cache_row(
    *,
    base: dict,
    graph,
    replay_cache: Path,
    gate_cache: Path,
    metrics: dict,
    train_rows: torch.Tensor,
    valid_rows: torch.Tensor,
) -> dict:
    meta = _metadata(replay_cache)
    return {
        "dataset": base["dataset"],
        "base_variant": base["base_variant"],
        "cache_status": "available_unreplayed",
        "historical_test_acc": base["expected_acc"],
        "replay_test_acc": "",
        "delta_replay": "",
        "macro_f1": metrics["test"]["macro_f1"],
        "predicted_class_count": metrics["test"]["predicted_class_count"],
        "train_nodes": int(train_rows.numel()),
        "valid_nodes": int(valid_rows.numel()),
        "test_nodes": int(graph.test_idx.numel()),
        "all_target_nodes": int(graph.num_nodes[graph.target_type]),
        "split_hash": meta["split_hash"],
        "feature_hash": meta["feature_hash"],
        "model_config_hash": meta["model_config_hash"],
        "cache_path": str(replay_cache),
        "gate_cache_path": str(gate_cache),
        "forbidden_component_flags": "[]",
        "blocked_reason": "",
    }


def _blocked_row(base: dict, reason: str) -> dict:
    return {
        "dataset": base["dataset"],
        "base_variant": base["base_variant"],
        "cache_status": "blocked_cache_generation_failed",
        "historical_test_acc": base["expected_acc"],
        "replay_test_acc": "",
        "delta_replay": "",
        "macro_f1": base["macro_f1"],
        "predicted_class_count": "",
        "train_nodes": "",
        "valid_nodes": "",
        "test_nodes": "",
        "all_target_nodes": "",
        "split_hash": "",
        "feature_hash": "",
        "model_config_hash": "",
        "cache_path": "",
        "gate_cache_path": "",
        "forbidden_component_flags": "[]",
        "blocked_reason": reason,
    }


def _split_graph_for_gate(graph, *, seed: int, val_fraction: float):
    train_rows, valid_rows = _split_train_valid(graph.labels, graph.train_idx, seed=seed, val_fraction=val_fraction)
    return replace(graph, train_idx=train_rows, val_idx=valid_rows), train_rows, valid_rows


def _save_pipeline_cache(
    *,
    base: dict,
    graph,
    summary: dict,
    cache_root: Path,
    seed: int,
    role: str,
    train_rows: torch.Tensor,
    valid_rows: torch.Tensor,
    dtype: str,
    model_config_payload: dict,
) -> tuple[Path, dict]:
    logits = summary.get("_target_logits")
    if logits is None:
        raise RuntimeError("pipeline run did not return _target_logits")
    num_classes = _num_classes(graph.labels)
    metrics = {
        "train": _metrics(logits, graph.labels, train_rows, num_classes),
        "valid": _metrics(logits, graph.labels, valid_rows, num_classes) if valid_rows.numel() else {"accuracy": 0.0, "macro_f1": 0.0, "predicted_class_count": 0},
        "test": _metrics(logits, graph.labels, graph.test_idx, num_classes),
    }
    cache = save_cache_for_graph(
        root=cache_root,
        graph=graph,
        logits=logits,
        train_rows=train_rows,
        valid_rows=valid_rows,
        base=base,
        seed=seed,
        role=role,
        metrics=metrics,
        dtype=dtype,
        model_config_payload=model_config_payload,
    )
    return cache, metrics


def _run_dblp_rplus(base: dict, args) -> dict:
    if args.reuse_existing_caches:
        existing = _existing_cache_row(base, args)
        if existing is not None:
            return existing
    cache_root = Path(args.cache_root)
    graph = load_processed_small_dataset("dblp")
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    common = dict(
        seed=args.seed,
        epochs=args.dblp_epochs,
        budget_mode="ratio",
        ratio=0.065,
        ratio_base="train_target",
        feature_dim=64,
        projection_type="raw",
        loss_type="clipped",
        k_s=2,
        min_proto_per_class=4,
        budget_alpha=0.5,
        method_name="Shadow-HGC-R+",
        feature_mode="metapath",
        metapath_signature=True,
        metapath_model_input=True,
        shadow_policy="rank_adaptive",
        adaptive_b=True,
        relation_gate=True,
        model_type="relation_linear",
        return_logits=True,
    )
    replay_summary = run_shadow_hgc_experiment(
        graph,
        output_path=log_dir / f"dblp_rplus_current_best_{ratio_slug(0.065)}_historical_seed{args.seed}.json",
        **common,
    )
    replay_cache, replay_metrics = _save_pipeline_cache(
        base=base,
        graph=graph,
        summary=replay_summary,
        cache_root=cache_root,
        seed=args.seed,
        role="historical_replay",
        train_rows=graph.train_idx,
        valid_rows=torch.empty(0, dtype=torch.long),
        dtype=args.dtype,
        model_config_payload=_model_config_payload(base, args, "historical_replay"),
    )
    gate_graph, gate_train, gate_valid = _split_graph_for_gate(graph, seed=args.seed, val_fraction=args.val_fraction)
    gate_summary = run_shadow_hgc_experiment(
        gate_graph,
        output_path=log_dir / f"dblp_rplus_current_best_{ratio_slug(0.065)}_gate_seed{args.seed}.json",
        **common,
    )
    gate_cache, _ = _save_pipeline_cache(
        base=base,
        graph=gate_graph,
        summary=gate_summary,
        cache_root=cache_root,
        seed=args.seed,
        role="gate_selection",
        train_rows=gate_train,
        valid_rows=gate_valid,
        dtype=args.dtype,
        model_config_payload=_model_config_payload(base, args, "gate_selection"),
    )
    return _cache_row(base=base, graph=graph, replay_cache=replay_cache, gate_cache=gate_cache, metrics=replay_metrics, train_rows=graph.train_idx, valid_rows=torch.empty(0, dtype=torch.long))


def _run_imdb_s1(base: dict, args) -> dict:
    if args.reuse_existing_caches:
        existing = _existing_cache_row(base, args)
        if existing is not None:
            return existing
    cache_root = Path(args.cache_root)
    graph = load_processed_small_dataset("imdb")
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    def train(graph_for_run, *, role: str, train_rows: torch.Tensor, valid_rows: torch.Tensor) -> tuple[Path, dict]:
        blocks, metadata = build_schema_default_blocks(graph_for_run, include_self=True, include_metapath=True)
        run = train_prototype_sehgnn_lite(
            graph_for_run,
            blocks=blocks,
            metadata=metadata,
            requested_ratio=0.05,
            seed=args.seed,
            epochs=args.imdb_epochs,
            hidden_dim=128,
            dropout=0.3,
            lr=0.01,
            weight_decay=1e-4,
            loss_type=DATASET_LOSS["imdb"],
            min_proto_per_class=4,
        )
        summary = {
            "dataset": "imdb",
            "variant": "S1_clean_MAM_MDM_MKM",
            "seed": args.seed,
            "status": "completed",
            "target_type": graph_for_run.target_type,
            "teacher_type": "none",
            "use_kd": False,
            "use_diffusion": False,
            "use_source_anchors": False,
            "use_coverage_medoid": False,
            **run.summary,
        }
        write_json_summary(log_dir / f"imdb_s1_clean_mam_mdm_mkm_r{ratio_slug(0.05)}_{role}_seed{args.seed}.json", summary)
        if run.logits is None:
            raise RuntimeError("SeHGNNLite run did not return logits")
        num_classes = _num_classes(graph_for_run.labels)
        metrics = {
            "train": _metrics(run.logits, graph_for_run.labels, train_rows, num_classes),
            "valid": _metrics(run.logits, graph_for_run.labels, valid_rows, num_classes) if valid_rows.numel() else {"accuracy": 0.0, "macro_f1": 0.0, "predicted_class_count": 0},
            "test": _metrics(run.logits, graph_for_run.labels, graph_for_run.test_idx, num_classes),
        }
        cache = save_cache_for_graph(
            root=cache_root,
            graph=graph_for_run,
            logits=run.logits,
            train_rows=train_rows,
            valid_rows=valid_rows,
            base=base,
            seed=args.seed,
            role=role,
            metrics=metrics,
            dtype=args.dtype,
            model_config_payload=_model_config_payload(base, args, role),
        )
        return cache, metrics

    replay_cache, replay_metrics = train(
        graph,
        role="historical_replay",
        train_rows=graph.train_idx,
        valid_rows=torch.empty(0, dtype=torch.long),
    )
    gate_graph, gate_train, gate_valid = _split_graph_for_gate(graph, seed=args.seed, val_fraction=args.val_fraction)
    gate_cache, _ = train(gate_graph, role="gate_selection", train_rows=gate_train, valid_rows=gate_valid)
    return _cache_row(base=base, graph=graph, replay_cache=replay_cache, gate_cache=gate_cache, metrics=replay_metrics, train_rows=graph.train_idx, valid_rows=torch.empty(0, dtype=torch.long))


def _run_medium_lad_reference(base: dict, args, dataset: str) -> dict:
    if args.reuse_existing_caches:
        existing = _existing_cache_row(base, args, gate_same_as_replay=True)
        if existing is not None:
            return existing
    graph = load_ogb_node_property_dataset(dataset, download=args.download)
    cache_root = Path(args.cache_root)
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    summary = run_shadow_hgc_experiment(
        graph,
        output_path=log_dir / f"{dataset}_lad_reference_r{ratio_slug(0.12)}_historical_seed{args.seed}.json",
        method_name="Shadow-HGC-LAD-clean",
        stage="medium_no_diffusion_refine",
        seed=args.seed,
        epochs=args.medium_lad_epochs,
        budget_mode="ratio",
        ratio=0.12,
        ratio_base="train_target",
        budget_rounding="nearest",
        feature_dim=128,
        projection_type="random",
        model_type="shadow_fusion",
        hidden_dim=128,
        dropout=0.3,
        lr=0.03,
        weight_decay=1e-4,
        loss_type=DATASET_LOSS[dataset],
        logit_adjustment_tau=1.0,
        feature_mode="label_affinity",
        diffusion_enabled=False,
        diffusion_status="diagnostic_only",
        label_affinity=True,
        label_affinity_mode="target_target",
        label_affinity_self_exclude=True,
        label_affinity_block_norm="row_l1",
        path_label_affinity=False,
        compiled_head=True,
        compiled_head_fusion="concat_mlp",
        compiled_hidden_dim=256,
        compiled_dropout=0.3,
        compiled_block_gate=True,
        compiled_demand_source="shadow_reconstructed",
        compiled_block_stats_source="train_full_demand_table",
        min_proto_per_class=4,
        budget_alpha=0.5,
        shadow_policy="rank_adaptive",
        rank_adaptive_global_cap=True,
        max_total_condensed_ratio=0.12,
        assignment_chunk_size=8192,
        inference_dst_chunk_size=args.medium_inference_batch_size,
        demand_edge_chunk_size=args.medium_edge_chunk_size,
        inference_edge_chunk_size=args.medium_edge_chunk_size,
        return_logits=True,
    )
    replay_cache, replay_metrics = _save_pipeline_cache(
        base=base,
        graph=graph,
        summary=summary,
        cache_root=cache_root,
        seed=args.seed,
        role="historical_replay",
        train_rows=graph.train_idx,
        valid_rows=graph.val_idx,
        dtype=args.dtype,
        model_config_payload=_model_config_payload(base, args, "historical_replay"),
    )
    return _cache_row(base=base, graph=graph, replay_cache=replay_cache, gate_cache=replay_cache, metrics=replay_metrics, train_rows=graph.train_idx, valid_rows=graph.val_idx)


def _run_products_rpp_base(base: dict, args) -> dict:
    if args.reuse_existing_caches:
        existing = _existing_cache_row(base, args, gate_same_as_replay=True)
        if existing is not None:
            return existing
    graph = load_ogb_node_property_dataset("ogbn-products", download=args.download)
    cache_root = Path(args.cache_root)
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    summary = run_shadow_hgc_experiment(
        graph,
        output_path=log_dir / f"ogbn-products_rpp_base_shadow_fusion_r{ratio_slug(0.12)}_historical_seed{args.seed}.json",
        method_name="Shadow-HGC-R++",
        seed=args.seed,
        epochs=args.products_rpp_epochs,
        budget_mode="ratio",
        ratio=0.12,
        ratio_base="train_target",
        feature_dim=128,
        projection_type="random",
        feature_mode="base",
        model_type="shadow_fusion",
        relation_gate=True,
        loss_type="sqrt_weighted",
        min_proto_per_class=4,
        budget_alpha=0.5,
        assignment_chunk_size=4096,
        inference_edge_chunk_size=args.medium_edge_chunk_size,
        inference_dst_chunk_size=args.products_inference_dst_chunk_size,
        return_logits=True,
    )
    replay_cache, replay_metrics = _save_pipeline_cache(
        base=base,
        graph=graph,
        summary=summary,
        cache_root=cache_root,
        seed=args.seed,
        role="historical_replay",
        train_rows=graph.train_idx,
        valid_rows=graph.val_idx,
        dtype=args.dtype,
        model_config_payload=_model_config_payload(base, args, "historical_replay"),
    )
    return _cache_row(base=base, graph=graph, replay_cache=replay_cache, gate_cache=replay_cache, metrics=replay_metrics, train_rows=graph.train_idx, valid_rows=graph.val_idx)


def run(args) -> list[dict]:
    rows = []
    cache_root = Path(args.cache_root)
    for base in SAFE_BASES:
        try:
            if base["dataset"] == "acm" and base["cache_variant"] == "B3_scap_v2":
                if args.reuse_existing_caches:
                    existing = _existing_cache_row(base, args)
                    if existing is not None:
                        rows.append(existing)
                        continue
                graph, logits, train_rows, valid_rows, metrics = train_sfb_v2_cache(
                    dataset="acm",
                    cache_variant="B3_scap_v2",
                    seed=args.seed,
                    gate_mode=False,
                    epochs=args.epochs,
                )
                replay_cache = save_cache_for_graph(
                    root=cache_root,
                    graph=graph,
                    logits=logits,
                    train_rows=train_rows,
                    valid_rows=valid_rows,
                    base=base,
                    seed=args.seed,
                    role="historical_replay",
                    metrics=metrics,
                    dtype=args.dtype,
                    model_config_payload=_model_config_payload(base, args, "historical_replay"),
                )
                gate_graph, gate_logits, gate_train, gate_valid, gate_metrics = train_sfb_v2_cache(
                    dataset="acm",
                    cache_variant="B3_scap_v2",
                    seed=args.seed,
                    gate_mode=True,
                    epochs=args.epochs,
                    val_fraction=args.val_fraction,
                )
                gate_cache = save_cache_for_graph(
                    root=cache_root,
                    graph=gate_graph,
                    logits=gate_logits,
                    train_rows=gate_train,
                    valid_rows=gate_valid,
                    base=base,
                    seed=args.seed,
                    role="gate_selection",
                    metrics=gate_metrics,
                    dtype=args.dtype,
                    model_config_payload=_model_config_payload(base, args, "gate_selection"),
                )
                rows.append(_cache_row(base=base, graph=graph, replay_cache=replay_cache, gate_cache=gate_cache, metrics=metrics, train_rows=train_rows, valid_rows=valid_rows))
            elif base["dataset"] == "dblp":
                rows.append(_run_dblp_rplus(base, args))
            elif base["dataset"] == "imdb":
                rows.append(_run_imdb_s1(base, args))
            elif base["dataset"] == "ogbn-arxiv":
                rows.append(_run_medium_lad_reference(base, args, "ogbn-arxiv"))
            elif base["dataset"] == "ogbn-products" and base["cache_variant"] == "rpp_base_shadow_fusion":
                if args.run_products_rpp_base:
                    rows.append(_run_products_rpp_base(base, args))
                else:
                    rows.append(_blocked_row(base, "products R++ base 500-epoch cache generation was dropped as too slow on local machine; rerun with --run-products-rpp-base to force it"))
            elif base["dataset"] == "ogbn-products" and base["cache_variant"] == "lad_reference":
                rows.append(_run_medium_lad_reference(base, args, "ogbn-products"))
            else:
                rows.append(_blocked_row(base, "no cache generation branch matched this safe base"))
        except Exception as exc:
            rows.append(_blocked_row(base, f"{exception_status(exc)}: {exc}"))
            continue
    write_csv(args.output, rows, FIELDS)
    report = Path(args.report)
    report.parent.mkdir(parents=True, exist_ok=True)
    available = [row for row in rows if row["cache_status"] == "available_unreplayed"]
    blocked = [row for row in rows if row["cache_status"] != "available_unreplayed"]
    lines = [
        "# T1.1 Safe Logit Cache Summary",
        "",
        "Safe-row cache generation now attempts ACM, DBLP, IMDB, ogbn-arxiv, and ogbn-products historical entries with replayable all-target logits.",
        "",
        f"- Available cache rows: `{len(available)}`",
        f"- Blocked cache rows: `{len(blocked)}`",
        f"- CSV: `{args.output}`",
    ]
    if blocked:
        lines.extend(["", "## Blocked Rows", ""])
        for row in blocked:
            lines.append(f"- {row['dataset']} / {row['base_variant']}: {row['blocked_reason']}")
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate T1.1 safe-row logit caches.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--dblp-epochs", type=int, default=500)
    parser.add_argument("--imdb-epochs", type=int, default=200)
    parser.add_argument("--medium-lad-epochs", type=int, default=200)
    parser.add_argument("--products-rpp-epochs", type=int, default=500)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--reuse-existing-caches", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--run-products-rpp-base", action="store_true")
    parser.add_argument("--medium-edge-chunk-size", type=int, default=250000)
    parser.add_argument("--medium-inference-batch-size", type=int, default=250000)
    parser.add_argument("--products-inference-dst-chunk-size", type=int, default=8192)
    parser.add_argument("--cache-root", default="experiments/logit_caches/t1_safe_seed42")
    parser.add_argument("--log-dir", default="experiments/logs/t1_safe_cache_generation_seed42")
    parser.add_argument("--output", default="experiments/tables/t1_safe_logit_cache_index_seed42.csv")
    parser.add_argument("--report", default="experiments/reports/t1_safe_logit_cache_summary.md")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
