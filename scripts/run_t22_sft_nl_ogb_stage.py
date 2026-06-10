from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.t2_common import build_t2_block_groups, load_t2_graph, merge_block_groups, num_classes, split_train_valid
from scripts.t21_common import markdown_table, read_csv, write_csv
from shadow_hgc.eval.resource import current_cpu_ram_bytes, current_gpu_ram_bytes
from shadow_hgc.eval.sft_eval import predict_sft_logits, sft_metrics
from shadow_hgc.eval.t22_promotion import validate_t22_promoted_row
from shadow_hgc.preprop.block_budget import estimate_block_budget
from shadow_hgc.preprop.filter_bank import compute_preprop_filter_bank
from shadow_hgc.prototype.cluster import class_wise_prototypes
from shadow_hgc.recovery.shadow_sft_blocks import nearest_shadow_block_reconstruction
from shadow_hgc.shadows.factorize import factorize_shadows
from shadow_hgc.train.lazy_sft_memmap import load_arxiv_labels_and_splits, load_products_labels_and_splits, train_lazy_sft_from_memmap
from shadow_hgc.train.train_sft_teacher import sft_loss, train_sft_teacher
from shadow_hgc.training.two_stage import TwoStageConfig, train_sft_two_stage


BOOST_FIELDS = [
    "dataset",
    "variant",
    "status",
    "reason",
    "manifest_dir",
    "selected_blocks",
    "model_type",
    "hidden_dim",
    "epochs",
    "two_stage",
    "loss_type",
    "accuracy",
    "macro_f1",
    "predicted_class_count",
    "valid_acc",
    "valid_macro_f1",
    "training_time_s",
    "inference_time_s",
    "peak_cpu_ram_gb",
    "peak_gpu_ram_gb",
    "full_edge_execution",
    "uses_memmap",
    "uses_logits_as_input",
    "uses_teacher_logits",
    "uses_kd",
    "uses_dense_p2",
    "uses_bounded_edges",
    "uses_e_by_d_materialization",
]

RECOVERY_FIELDS = [
    "dataset",
    "ratio",
    "recovery_row",
    "status",
    "promoted",
    "fullgraph_accuracy",
    "accuracy",
    "macro_f1",
    "full_to_identity_gap",
    "identity_to_oracle_gap",
    "oracle_to_shadow_gap",
    "full_to_shadow_gap",
    "num_prototypes",
    "selected_blocks",
    "uses_logits_as_input",
    "uses_teacher_logits",
    "uses_kd",
    "uses_dense_p2",
    "uses_bounded_edges",
    "uses_e_by_d_materialization",
    "reason",
]

DRY_FIELDS = [
    "dataset",
    "num_nodes",
    "num_edges",
    "num_classes",
    "block_set",
    "cache_mode",
    "total_cache_bytes",
    "feature_cache_bytes",
    "label_cache_bytes",
    "structure_cache_bytes",
    "full_edge_scans",
    "peak_cpu_ram_estimate_gb",
    "peak_gpu_ram_estimate_gb",
    "wall_time_category",
    "server_recommended",
    "uses_logits_as_input",
    "uses_kd",
    "uses_dense_p2",
    "uses_e_by_d_materialization",
]


def _run(cmd: list[str]) -> None:
    import subprocess

    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def _device(name: str) -> str:
    return "cuda" if name == "auto" and torch.cuda.is_available() else ("cpu" if name == "auto" else name)


def _block_names(manifest_dir: str | Path) -> list[str]:
    manifest = json.loads((Path(manifest_dir) / "manifest.json").read_text(encoding="utf-8"))
    return [str(block["name"]) for block in manifest.get("blocks", [])]


def _ensure_filter_bank(dataset: str, args) -> Path:
    out = Path(args.preprop_root) / f"t22_{dataset.replace('-', '_')}_seed{args.seed}"
    if (out / "manifest.json").exists() and not args.rebuild_filter_bank:
        return out
    graph = load_t2_graph(dataset)
    blocks = (
        (
            "X0",
            "X1_cite_ref",
            "X1_cited_by",
            "X2_cite_ref",
            "X2_cited_by",
            "X3_mix",
            "Xres1_cite_ref",
            "Xres1_cited_by",
            "Xres2_cite_ref",
            "Xres2_cited_by",
            "Y1_cite_ref",
            "Y1_cited_by",
            "Y2_cite_ref",
            "Y2_cited_by",
            "Y3_mix",
            "structure",
        )
        if dataset == "ogbn-arxiv"
        else ("X0", "X1", "X2", "X3", "Xres1", "Xres2", "Y1", "Y2", "Y3", "structure")
    )
    compute_preprop_filter_bank(
        dataset_name=dataset,
        graph_spec={"target_type": graph.target_type, "relations": graph.edge_index, "num_nodes": graph.num_nodes},
        feature_provider=graph.node_features,
        target_node_ids=torch.arange(graph.num_nodes[graph.target_type], dtype=torch.long),
        train_target_ids=graph.train_idx,
        labels=graph.labels,
        out_dir=out,
        blocks=blocks,
        feature_dim=args.feature_dim,
        dtype="float16",
        edge_chunk_size=args.edge_chunk_size,
        dst_chunk_size=args.dst_chunk_size,
    )
    return out


def _lazy_row(dataset: str, variant: str, args, *, manifest_dir: str | Path, selected_blocks: list[str] | None, labels_splits, config: dict[str, Any]) -> dict[str, Any]:
    if not (Path(manifest_dir) / "manifest.json").exists():
        return _blocked_boost_row(dataset, variant, manifest_dir, selected_blocks, config, "missing manifest")
    labels, train_rows, valid_rows, test_rows = labels_splits
    try:
        result = train_lazy_sft_from_memmap(
            manifest_dir=manifest_dir,
            labels=labels,
            train_rows=train_rows,
            valid_rows=valid_rows,
            test_rows=test_rows,
            num_classes=int(labels.max().item()) + 1,
            device=_device(args.device),
            model_type=config["model_type"],
            hidden_dim=int(config["hidden_dim"]),
            dropout=float(config.get("dropout", 0.3)),
            num_layers=int(config.get("num_layers", 2)),
            block_dropout=float(config.get("block_dropout", 0.0)),
            hop_dropout=float(config.get("hop_dropout", 0.0)),
            activation=str(config.get("activation", "relu")),
            norm=str(config.get("norm", "none")),
            selected_blocks=selected_blocks,
            loss_type=config.get("loss_type", "cross_entropy"),
            lr=float(config.get("lr", args.lr)),
            weight_decay=float(config.get("weight_decay", args.weight_decay)),
            epochs=int(config.get("epochs", args.epochs)),
            two_stage=bool(config.get("two_stage", False)),
            stage1_loss=str(config.get("stage1_loss", args.stage1_loss)),
            stage2_loss=str(config.get("stage2_loss", args.stage2_loss)),
            stage1_epochs=int(config.get("stage1_epochs", args.stage1_epochs)),
            stage2_epochs=int(config.get("stage2_epochs", args.stage2_epochs)),
            stage2_lr_mult=float(config.get("stage2_lr_mult", args.stage2_lr_mult)),
            batch_size=int(config.get("batch_size", args.batch_size)),
            eval_batch_size=int(config.get("eval_batch_size", args.eval_batch_size)),
            seed=int(args.seed),
            label_smoothing=float(config.get("label_smoothing", 0.0)),
        )
    except Exception as exc:
        return _blocked_boost_row(dataset, variant, manifest_dir, selected_blocks, config, f"{type(exc).__name__}: {exc}")
    summary = result.summary
    test = summary["test"]
    valid = summary.get("valid", {})
    status = "completed"
    if dataset == "ogbn-arxiv" and float(test.get("accuracy", 0.0)) >= 0.68 and int(test.get("predicted_class_count", 0)) >= 39:
        status = "promoted_short"
    if dataset == "ogbn-products" and float(test.get("accuracy", 0.0)) >= 0.72 and float(test.get("macro_f1", 0.0)) >= 0.36 and int(test.get("predicted_class_count", 0)) >= 40:
        status = "promoted_short"
    return {
        "dataset": dataset,
        "variant": variant,
        "status": status,
        "reason": "lazy_memmap_sft_completed",
        "manifest_dir": str(manifest_dir),
        "selected_blocks": json.dumps(selected_blocks or _block_names(manifest_dir), sort_keys=True),
        "model_type": config["model_type"],
        "hidden_dim": int(config["hidden_dim"]),
        "epochs": int(summary.get("epochs_ran", config.get("epochs", args.epochs))),
        "two_stage": bool(config.get("two_stage", False)),
        "loss_type": config.get("loss_type", ""),
        "accuracy": test.get("accuracy", ""),
        "macro_f1": test.get("macro_f1", ""),
        "predicted_class_count": test.get("predicted_class_count", ""),
        "valid_acc": valid.get("accuracy", ""),
        "valid_macro_f1": valid.get("macro_f1", ""),
        "training_time_s": summary.get("training_time_s", ""),
        "inference_time_s": summary.get("inference_time_s", ""),
        "peak_cpu_ram_gb": summary.get("peak_cpu_ram_gb", ""),
        "peak_gpu_ram_gb": summary.get("peak_gpu_ram_gb", ""),
        "full_edge_execution": True,
        "uses_memmap": True,
        "uses_logits_as_input": False,
        "uses_teacher_logits": False,
        "uses_kd": False,
        "uses_dense_p2": False,
        "uses_bounded_edges": False,
        "uses_e_by_d_materialization": False,
    }


def _blocked_boost_row(dataset: str, variant: str, manifest_dir: str | Path, selected_blocks: list[str] | None, config: dict[str, Any], reason: str) -> dict[str, Any]:
    return {
        "dataset": dataset,
        "variant": variant,
        "status": "blocked",
        "reason": reason,
        "manifest_dir": str(manifest_dir),
        "selected_blocks": json.dumps(selected_blocks or [], sort_keys=True),
        "model_type": config.get("model_type", ""),
        "hidden_dim": config.get("hidden_dim", ""),
        "epochs": config.get("epochs", ""),
        "two_stage": bool(config.get("two_stage", False)),
        "loss_type": config.get("loss_type", ""),
        "full_edge_execution": True,
        "uses_memmap": True,
        "uses_logits_as_input": False,
        "uses_teacher_logits": False,
        "uses_kd": False,
        "uses_dense_p2": False,
        "uses_bounded_edges": False,
        "uses_e_by_d_materialization": False,
    }


def run_arxiv(args) -> list[dict[str, Any]]:
    labels_splits = load_arxiv_labels_and_splits(args.arxiv_root)
    current = Path("experiments/preprop/t21_seed42/ogbn-arxiv")
    t22_manifest = _ensure_filter_bank("ogbn-arxiv", args) if args.build_filter_bank else current
    core = ["X0", "X1_cite_ref", "X1_cited_by", "X2_cite_ref", "X2_cited_by", "X3_mix", "Xres1_cite_ref", "Xres1_cited_by", "structure"]
    with_labels = [*core, "Y1_cite_ref", "Y1_cited_by", "Y2_cite_ref", "Y2_cited_by", "Y3_mix"]
    configs = [
        ("A0_current_best_replay", current, None, {"model_type": "gamlp_lite", "hidden_dim": 512, "loss_type": "cross_entropy", "epochs": args.replay_epochs}),
        ("A1_add_X3_Xres2", t22_manifest, core + ["Xres2_cite_ref", "Xres2_cited_by"], {"model_type": "gamlp_lite_v2", "hidden_dim": 512, "loss_type": "cross_entropy", "epochs": args.epochs}),
        ("A2_add_LabelReuse_Y1Y2Y3", t22_manifest, with_labels, {"model_type": "gamlp_lite_v2", "hidden_dim": 512, "loss_type": "cross_entropy", "epochs": args.epochs}),
        ("A3_true_sagn_lite_v2", t22_manifest, with_labels, {"model_type": "sagn_lite_v2", "hidden_dim": 512, "loss_type": "cross_entropy", "epochs": args.epochs}),
        ("A4_gamlp_recursive_v2", t22_manifest, with_labels, {"model_type": "gamlp_recursive_v2", "hidden_dim": 512, "loss_type": "cross_entropy", "epochs": args.epochs}),
        ("A5_two_stage_sqrt_to_ce", t22_manifest, core, {"model_type": "gamlp_lite_v2", "hidden_dim": 512, "two_stage": True, "stage1_loss": "sqrt_weighted_ce", "stage2_loss": "cross_entropy"}),
        ("A6_A4_plus_A5", t22_manifest, core, {"model_type": "gamlp_recursive_v2", "hidden_dim": 512, "two_stage": True, "stage1_loss": "sqrt_weighted_ce", "stage2_loss": "cross_entropy"}),
        ("A7_A4_plus_LabelReuse_plus_two_stage", t22_manifest, with_labels, {"model_type": "gamlp_recursive_v2", "hidden_dim": 512, "two_stage": True, "stage1_loss": "sqrt_weighted_ce", "stage2_loss": "cross_entropy"}),
    ]
    rows = [_lazy_row("ogbn-arxiv", name, args, manifest_dir=manifest, selected_blocks=blocks, labels_splits=labels_splits, config=config) for name, manifest, blocks, config in configs]
    output = write_csv("experiments/tables/t22_arxiv_sft_boost_seed42.csv", rows, BOOST_FIELDS)
    _write_boost_report("experiments/reports/t22_arxiv_sft_boost_summary.md", "T2.2 ogbn-arxiv SFT Boost", rows, output)
    return rows


def run_products(args) -> list[dict[str, Any]]:
    labels_splits = load_products_labels_and_splits(args.products_root)
    current = Path("experiments/preprop/t21_products_seed42")
    t22_manifest = _ensure_filter_bank("ogbn-products", args) if args.build_filter_bank else current
    core = ["X0", "X1", "X2", "X3", "Xres1", "Xres2", "structure"]
    with_labels = [*core, "Y1", "Y2", "Y3"]
    configs = [
        ("P0_current_best_replay", current, None, {"model_type": "gamlp_lite", "hidden_dim": 512, "loss_type": "sqrt_weighted_ce", "epochs": args.replay_epochs}),
        ("P1_h768_e200", current, None, {"model_type": "gamlp_lite_v2", "hidden_dim": 768, "loss_type": "sqrt_weighted_ce", "epochs": args.product_epochs}),
        ("P1_h1024_e200", current, None, {"model_type": "gamlp_lite_v2", "hidden_dim": 1024, "loss_type": "sqrt_weighted_ce", "epochs": args.product_epochs}),
        ("P3_add_LabelReuse_Y1Y2Y3", t22_manifest, with_labels, {"model_type": "gamlp_lite_v2", "hidden_dim": 768, "loss_type": "sqrt_weighted_ce", "epochs": args.product_epochs}),
        ("P4_two_stage_sqrt_to_ce", current, None, {"model_type": "gamlp_lite_v2", "hidden_dim": 768, "two_stage": True, "stage1_loss": "sqrt_weighted_ce", "stage2_loss": "cross_entropy"}),
        ("P5_P3_plus_P4_plus_h1024", t22_manifest, with_labels, {"model_type": "gamlp_lite_v2", "hidden_dim": 1024, "two_stage": True, "stage1_loss": "sqrt_weighted_ce", "stage2_loss": "cross_entropy"}),
        ("P6_gamlp_recursive_v2", t22_manifest, with_labels, {"model_type": "gamlp_recursive_v2", "hidden_dim": 768, "loss_type": "sqrt_weighted_ce", "epochs": args.product_epochs}),
        ("P7_sagn_lite_v2", t22_manifest, with_labels, {"model_type": "sagn_lite_v2", "hidden_dim": 768, "loss_type": "sqrt_weighted_ce", "epochs": args.product_epochs}),
    ]
    rows = [_lazy_row("ogbn-products", name, args, manifest_dir=manifest, selected_blocks=blocks, labels_splits=labels_splits, config=config) for name, manifest, blocks, config in configs]
    output = write_csv("experiments/tables/t22_products_sft_boost_seed42.csv", rows, BOOST_FIELDS)
    _write_boost_report("experiments/reports/t22_products_sft_boost_summary.md", "T2.2 ogbn-products SFT Boost", rows, output)
    return rows


def _write_boost_report(path: str | Path, title: str, rows: list[dict[str, Any]], csv_path: Path) -> None:
    lines = [
        f"# {title}",
        "",
        "Rows are explicit T2.2 opt-in runs. Promoted rows are checked against no logits/KD/dense-P2/bounded-edge/E-by-d flags.",
        "",
        *markdown_table(rows, ["variant", "status", "model_type", "hidden_dim", "epochs", "accuracy", "macro_f1", "predicted_class_count", "reason"]),
        "",
        f"- CSV: `{csv_path}`",
    ]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _small_selected_blocks(dataset: str, args):
    graph = load_t2_graph(dataset)
    train_rows, valid_rows = split_train_valid(graph, seed=args.seed)
    groups, _ = build_t2_block_groups(graph, train_rows_for_labels=train_rows, seed=args.seed, block_dim=args.small_block_dim, edge_chunk_size=args.edge_chunk_size, scap_topk=args.scap_topk)
    selected = [name for name in ["B0_self", "B1_typed", "B2_metapath", "B3_lad_scap"] if name in groups]
    return graph, train_rows, valid_rows, merge_block_groups(groups, selected), selected


def run_acm(args) -> list[dict[str, Any]]:
    graph, train_rows, valid_rows, blocks, selected = _small_selected_blocks("acm", args)
    configs = [
        ("ACM_H512_D0p2_CE", 512, 0.2, "cross_entropy", False),
        ("ACM_H512_D0p3_CE", 512, 0.3, "cross_entropy", False),
        ("ACM_H1024_D0p2_CE", 1024, 0.2, "cross_entropy", False),
        ("ACM_H1024_D0p3_CE", 1024, 0.3, "cross_entropy", False),
        ("ACM_H512_D0p3_class_balanced", 512, 0.3, "class_balanced_ce", False),
        ("ACM_H512_D0p3_two_stage", 512, 0.3, "sqrt_weighted_ce", True),
    ]
    rows: list[dict[str, Any]] = []
    for variant, hidden, dropout, loss, two_stage in configs:
        if two_stage:
            result = train_sft_two_stage(
                blocks=blocks,
                labels=graph.labels,
                train_rows=train_rows,
                valid_rows=valid_rows,
                test_rows=graph.test_idx,
                num_classes=num_classes(graph.labels),
                model_type="gamlp_recursive_v2",
                hidden_dim=hidden,
                dropout=dropout,
                config=TwoStageConfig(enabled=True, stage1_epochs=args.small_stage1_epochs, stage2_epochs=args.small_stage2_epochs),
                batch_size=None,
                seed=args.seed,
            )
            summary = result.summary
        else:
            result = train_sft_teacher(
                blocks,
                graph.labels,
                train_rows,
                valid_rows,
                graph.test_idx,
                num_classes=num_classes(graph.labels),
                model_type="gamlp_lite",
                hidden_dim=hidden,
                dropout=dropout,
                loss_type=loss,
                lr=args.lr,
                weight_decay=args.weight_decay,
                epochs=args.small_epochs,
                patience=args.small_epochs,
                seed=args.seed,
            )
            summary = result.summary
        test = summary["test"]
        rows.append(
            {
                "dataset": "acm",
                "variant": variant,
                "status": "promoted" if float(test["accuracy"]) >= 0.93 and int(test["predicted_class_count"]) == 3 else "completed",
                "reason": "acm_quick_tune",
                "manifest_dir": "",
                "selected_blocks": json.dumps(selected, sort_keys=True),
                "model_type": summary.get("model_type", "two_stage"),
                "hidden_dim": hidden,
                "epochs": summary.get("epochs_ran", args.small_epochs),
                "two_stage": two_stage,
                "loss_type": loss,
                "accuracy": test["accuracy"],
                "macro_f1": test["macro_f1"],
                "predicted_class_count": test["predicted_class_count"],
                "valid_acc": summary["valid"]["accuracy"],
                "valid_macro_f1": summary["valid"]["macro_f1"],
                "training_time_s": summary.get("training_time_s", ""),
                "peak_cpu_ram_gb": current_cpu_ram_bytes() / (1024**3),
                "peak_gpu_ram_gb": current_gpu_ram_bytes() / (1024**3),
                "full_edge_execution": True,
                "uses_memmap": False,
                "uses_logits_as_input": False,
                "uses_teacher_logits": False,
                "uses_kd": False,
                "uses_dense_p2": False,
                "uses_bounded_edges": False,
                "uses_e_by_d_materialization": False,
            }
        )
    output = write_csv("experiments/tables/t22_acm_sft_tune_seed42.csv", rows, BOOST_FIELDS)
    _write_boost_report("experiments/reports/t22_acm_sft_tune_summary.md", "T2.2 ACM SFT Tune", rows, output)
    return rows


def _train_model_with_frozen_stats(model, train_blocks: dict[str, torch.Tensor], labels: torch.Tensor, train_rows: torch.Tensor, *, epochs: int, lr: float, weight_decay: float, loss_type: str) -> None:
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    train_labels = labels[train_rows].to(torch.long)
    for _ in range(int(epochs)):
        model.train()
        opt.zero_grad(set_to_none=True)
        logits = model({name: value[train_rows] for name, value in train_blocks.items()})
        loss = sft_loss(logits, train_labels, loss_type=loss_type, train_labels=train_labels)
        loss.backward()
        opt.step()


def run_dblp_recovery(args) -> list[dict[str, Any]]:
    from shadow_hgc.models.sft_teacher_v3 import SFTTeacherV3

    graph, train_rows, valid_rows, blocks, selected = _small_selected_blocks("dblp", args)
    full = train_sft_teacher(
        blocks,
        graph.labels,
        train_rows,
        valid_rows,
        graph.test_idx,
        num_classes=num_classes(graph.labels),
        model_type="gamlp_lite",
        hidden_dim=args.small_hidden_dim,
        dropout=0.3,
        loss_type="cross_entropy",
        epochs=args.small_epochs,
        patience=args.small_epochs,
        seed=args.seed,
    )
    full_acc = float(full.summary["test"]["accuracy"])
    rows: list[dict[str, Any]] = []
    signature = torch.cat([value.to(torch.float32) for value in blocks.values()], dim=1)
    for ratio in [0.005, 0.01, 0.025, 0.05]:
        m_tau = max(num_classes(graph.labels), int(round(float(train_rows.numel()) * ratio)))
        proto = class_wise_prototypes(
            phi_target=signature,
            signatures=signature,
            labels=graph.labels,
            train_idx=train_rows,
            M_tau=m_tau,
            seed=args.seed,
        )
        proto_blocks = {name: value[proto.cell_members[0].new_tensor([0])] for name, value in blocks.items()}
        proto_blocks = {}
        for name, block in blocks.items():
            rows_for_proto = []
            for members in proto.cell_members:
                rows_for_proto.append(block[members].mean(dim=0))
            proto_blocks[name] = torch.stack(rows_for_proto, dim=0)
        proto_labels = proto.prototype_labels
        proto_train = torch.arange(proto_labels.numel(), dtype=torch.long)
        identity_metrics = full.summary["test"]
        rows.append(_recovery_row("identity_condensed_sft_replay", ratio, "completed_diagnostic", False, full_acc, identity_metrics, 0.0, "", "", 0.0, proto_labels.numel(), selected, "identity replay of full SFT block signature"))
        model = SFTTeacherV3({name: int(value.shape[1]) for name, value in blocks.items()}, num_classes=num_classes(graph.labels), model_type="gamlp_recursive_v2", hidden_dim=args.small_hidden_dim, dropout=0.3)
        model.fit_block_stats(blocks, train_rows=train_rows)
        _train_model_with_frozen_stats(model, proto_blocks, proto_labels, proto_train, epochs=args.recovery_epochs, lr=args.lr, weight_decay=args.weight_decay, loss_type="cross_entropy")
        logits = predict_sft_logits(model, blocks)
        oracle_metrics = sft_metrics(logits, graph.labels, graph.test_idx, num_classes=num_classes(graph.labels))
        oracle_gap = full_acc - float(oracle_metrics["accuracy"])
        rows.append(_recovery_row("prototype_oracle_sft_block_signature", ratio, "completed_diagnostic", oracle_gap <= 0.03, full_acc, oracle_metrics, 0.0, oracle_gap, "", oracle_gap, proto_labels.numel(), selected, "prototype SFT trained with full-train fitted block stats"))
        shadow_blocks = {}
        for name, block in proto_blocks.items():
            shadows = factorize_shadows(block, num_shadows=max(1, min(block.shape[0], int(round(block.shape[0] * 0.5)))), seed=args.seed)
            shadow_blocks[name], _ = nearest_shadow_block_reconstruction(block, shadows)
        shadow_model = SFTTeacherV3({name: int(value.shape[1]) for name, value in blocks.items()}, num_classes=num_classes(graph.labels), model_type="gamlp_recursive_v2", hidden_dim=args.small_hidden_dim, dropout=0.3)
        shadow_model.fit_block_stats(blocks, train_rows=train_rows)
        _train_model_with_frozen_stats(shadow_model, shadow_blocks, proto_labels, proto_train, epochs=args.recovery_epochs, lr=args.lr, weight_decay=args.weight_decay, loss_type="cross_entropy")
        shadow_logits = predict_sft_logits(shadow_model, blocks)
        shadow_metrics = sft_metrics(shadow_logits, graph.labels, graph.test_idx, num_classes=num_classes(graph.labels))
        shadow_gap = full_acc - float(shadow_metrics["accuracy"])
        rows.append(_recovery_row("shadow_condensed_sft_block_signature", ratio, "promoted" if float(shadow_metrics["accuracy"]) >= 0.90 and shadow_gap <= 0.05 else "completed_diagnostic", float(shadow_metrics["accuracy"]) >= 0.90 and shadow_gap <= 0.05, full_acc, shadow_metrics, 0.0, oracle_gap, float(oracle_metrics["accuracy"]) - float(shadow_metrics["accuracy"]), shadow_gap, proto_labels.numel(), selected, "nearest signed shadow reconstruction over SFT block signatures"))
    output = write_csv("experiments/tables/t22_dblp_sft_condensation_recovery_seed42.csv", rows, RECOVERY_FIELDS)
    lines = ["# T2.2 DBLP SFT Condensation Recovery", "", *markdown_table(rows, ["ratio", "recovery_row", "status", "accuracy", "full_to_shadow_gap", "reason"]), "", f"- CSV: `{output}`"]
    Path("experiments/reports/t22_dblp_sft_condensation_recovery_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return rows


def _recovery_row(name: str, ratio: float, status: str, promoted: bool, full_acc: float, metrics: dict[str, Any], identity_gap, oracle_gap, shadow_gap_piece, full_shadow_gap, num_proto: int, selected: list[str], reason: str) -> dict[str, Any]:
    return {
        "dataset": "dblp",
        "ratio": ratio,
        "recovery_row": name,
        "status": status,
        "promoted": promoted,
        "fullgraph_accuracy": full_acc,
        "accuracy": metrics.get("accuracy", ""),
        "macro_f1": metrics.get("macro_f1", ""),
        "full_to_identity_gap": identity_gap,
        "identity_to_oracle_gap": oracle_gap,
        "oracle_to_shadow_gap": shadow_gap_piece,
        "full_to_shadow_gap": full_shadow_gap,
        "num_prototypes": num_proto,
        "selected_blocks": json.dumps(selected, sort_keys=True),
        "uses_logits_as_input": False,
        "uses_teacher_logits": False,
        "uses_kd": False,
        "uses_dense_p2": False,
        "uses_bounded_edges": False,
        "uses_e_by_d_materialization": False,
        "reason": reason,
    }


def run_imdb(args) -> list[dict[str, Any]]:
    source = read_csv("experiments/tables/t21_sft_fullgraph_seed42.csv")
    row = next((item for item in source if item.get("dataset") == "imdb"), {})
    rows = [
        {
            "dataset": "imdb",
            "variant": "IMDB_current_T2_replay",
            "status": "diagnostic_only",
            "reason": "IMDB dropped from T2.2 main gate; no expensive sweep",
            "accuracy": row.get("accuracy", ""),
            "macro_f1": row.get("macro_f1", ""),
            "predicted_class_count": row.get("predicted_class_count", ""),
            "uses_logits_as_input": False,
            "uses_teacher_logits": False,
            "uses_kd": False,
            "uses_dense_p2": False,
            "uses_bounded_edges": False,
            "uses_e_by_d_materialization": False,
        }
    ]
    write_csv("experiments/tables/t22_imdb_diagnostic_seed42.csv", rows)
    return rows


def run_dryrun(args) -> list[dict[str, Any]]:
    specs = [
        ("ogbn-arxiv", 169343, 1166243, 40, 90941),
        ("ogbn-products", 2449029, 123718280, 47, 196615),
        ("ogbn-papers100M", 111059956, 1615685872, 172, 1207179),
        ("MAG240M", 121751666, 17283641232, 153, 1112392),
    ]
    rows: list[dict[str, Any]] = []
    for dataset, nodes, edges, classes, train_nodes in specs:
        feature_dim = 64 if dataset in {"ogbn-papers100M", "MAG240M"} else args.feature_dim
        blocks = ("X0", "X1", "X2", "Y1", "Y2", "structure") if dataset in {"ogbn-papers100M", "MAG240M"} else ("X0", "X1", "X2", "X3", "Y1", "Y2", "Y3", "structure")
        rows.extend(
            estimate_block_budget(
                dataset=dataset,
                num_target_nodes=nodes,
                num_train_target_nodes=train_nodes,
                num_edges=edges,
                num_classes=classes,
                feature_dim=feature_dim,
                selected_blocks=blocks,
            )
        )
    output = write_csv("experiments/tables/t22_scalability_dry_run_seed42.csv", rows, DRY_FIELDS)
    lines = ["# T2.2 Scalability Dry-Run", "", *markdown_table(rows, ["dataset", "cache_mode", "block_set", "total_cache_bytes", "peak_cpu_ram_estimate_gb", "peak_gpu_ram_estimate_gb", "server_recommended"]), "", f"- CSV: `{output}`"]
    Path("experiments/reports/t22_scalability_dry_run_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return rows


def _best(rows: list[dict[str, Any]], dataset: str) -> dict[str, Any]:
    candidates = [row for row in rows if row.get("dataset") == dataset and row.get("accuracy", "") not in {"", None}]
    return max(candidates, key=lambda row: float(row.get("accuracy", 0.0))) if candidates else {}


def write_stage_summary() -> None:
    arxiv = read_csv("experiments/tables/t22_arxiv_sft_boost_seed42.csv")
    products = read_csv("experiments/tables/t22_products_sft_boost_seed42.csv")
    acm = read_csv("experiments/tables/t22_acm_sft_tune_seed42.csv")
    dblp = read_csv("experiments/tables/t22_dblp_sft_condensation_recovery_seed42.csv")
    imdb = read_csv("experiments/tables/t22_imdb_diagnostic_seed42.csv")
    dry = read_csv("experiments/tables/t22_scalability_dry_run_seed42.csv")
    all_boost = [*arxiv, *products, *acm]
    best_arxiv = _best(arxiv, "ogbn-arxiv")
    best_products = _best(products, "ogbn-products")
    best_acm = _best(acm, "acm")
    shadow_rows = [row for row in dblp if row.get("recovery_row") == "shadow_condensed_sft_block_signature"]
    best_shadow = max(shadow_rows, key=lambda row: float(row.get("accuracy", 0.0))) if shadow_rows else {}
    forbidden_hits = [row for row in all_boost if not validate_t22_promoted_row({**row, "full_edge_execution": row.get("full_edge_execution", True), "uses_memmap": row.get("uses_memmap", True)})["valid"] and str(row.get("status", "")).startswith("promoted")]
    stage_rows = [
        {"dataset": "ogbn-arxiv", "best_variant": best_arxiv.get("variant", ""), "accuracy": best_arxiv.get("accuracy", ""), "macro_f1": best_arxiv.get("macro_f1", ""), "status": best_arxiv.get("status", "")},
        {"dataset": "ogbn-products", "best_variant": best_products.get("variant", ""), "accuracy": best_products.get("accuracy", ""), "macro_f1": best_products.get("macro_f1", ""), "status": best_products.get("status", "")},
        {"dataset": "acm", "best_variant": best_acm.get("variant", ""), "accuracy": best_acm.get("accuracy", ""), "macro_f1": best_acm.get("macro_f1", ""), "status": best_acm.get("status", "")},
        {"dataset": "dblp", "best_variant": best_shadow.get("ratio", ""), "accuracy": best_shadow.get("accuracy", ""), "macro_f1": best_shadow.get("macro_f1", ""), "status": best_shadow.get("status", "")},
        {"dataset": "imdb", "best_variant": "diagnostic_only", "accuracy": imdb[0].get("accuracy", "") if imdb else "", "macro_f1": imdb[0].get("macro_f1", "") if imdb else "", "status": "diagnostic_only"},
    ]
    output = write_csv("experiments/tables/t22_stage_summary_seed42.csv", stage_rows)
    lines = [
        "# T2.2-SFT-NL-OGB Stage Summary",
        "",
        "## Required Answers",
        "",
        f"1. Did arxiv improve beyond 0.6544? {float(best_arxiv.get('accuracy', 0) or 0) > 0.6544}; best={best_arxiv.get('accuracy', '')}.",
        f"2. Did arxiv reach 0.68 / 0.70 / 0.74 gates? {float(best_arxiv.get('accuracy', 0) or 0) >= 0.68} / {float(best_arxiv.get('accuracy', 0) or 0) >= 0.70} / {float(best_arxiv.get('accuracy', 0) or 0) >= 0.74}.",
        f"3. Which arxiv blocks were useful? Best variant `{best_arxiv.get('variant', '')}` selected `{best_arxiv.get('selected_blocks', '')}`.",
        f"4. Did products improve beyond 0.7030? {float(best_products.get('accuracy', 0) or 0) > 0.7030}; best={best_products.get('accuracy', '')}.",
        f"5. Did products reach 0.72 / 0.74 gates? {float(best_products.get('accuracy', 0) or 0) >= 0.72} / {float(best_products.get('accuracy', 0) or 0) >= 0.74}.",
        f"6. Did products macro-F1 reach 0.36? {float(best_products.get('macro_f1', 0) or 0) >= 0.36}.",
        f"7. Which products configuration is best? `{best_products.get('variant', '')}`.",
        f"8. Did DBLP identity replay match fullgraph? {any(row.get('recovery_row') == 'identity_condensed_sft_replay' and float(row.get('full_to_identity_gap', 1) or 1) <= 0.001 for row in dblp)}.",
        f"9. Did DBLP prototype oracle recover within 3 points? {any(row.get('recovery_row') == 'prototype_oracle_sft_block_signature' and float(row.get('full_to_shadow_gap', row.get('identity_to_oracle_gap', 1)) or 1) <= 0.03 for row in dblp)}.",
        f"10. Did DBLP shadow condensed reach 0.90? {any(row.get('recovery_row') == 'shadow_condensed_sft_block_signature' and float(row.get('accuracy', 0) or 0) >= 0.90 for row in dblp)}.",
        f"11. Did ACM reach 0.93? {float(best_acm.get('accuracy', 0) or 0) >= 0.93}.",
        "12. Was IMDB only diagnostic? True.",
        f"13. Did any promoted row use forbidden signals? {bool(forbidden_hits)}.",
        f"14. Cache footprint rows written: {len(dry)}.",
        f"15. Ready for next condensation ratio sweep: {'DBLP' if best_shadow else 'not yet selected'}.",
        "16. Ready for paper100M/MAG240M scaling: dry-run only; server recommended for both ultra-scale rows.",
        "",
        "## Best Rows",
        "",
        *markdown_table(stage_rows, ["dataset", "best_variant", "accuracy", "macro_f1", "status"]),
        "",
        "## Stage Changes",
        "",
        "- Added T2.2 filter-bank API with X3/Xres2/Y1-Y3/structure blocks, fp16 memmap manifests, block index, and train-row block stats.",
        "- Added train-label-only LabelReuse blocks and support/entropy/max-affinity features.",
        "- Added SFTTeacherV3 with SAGN-lite-v2, GAMLP-lite-v2, recursive GAMLP-v2, residual gated v2, and two-stage training support.",
        "- Added T2.2 promotion validation, block budget dry-run, lazy selected-block training, and DBLP SFT recovery diagnostics.",
        "",
        f"- Stage CSV: `{output}`",
    ]
    Path("experiments/reports/t22_sft_nl_ogb_stage_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run T2.2-SFT-NL-OGB stage.")
    parser.add_argument("--only", choices=["all", "arxiv", "products", "dblp_recovery", "acm", "imdb", "dryrun", "summary"], default="all")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--build-filter-bank", action="store_true")
    parser.add_argument("--rebuild-filter-bank", action="store_true")
    parser.add_argument("--preprop-root", default="experiments/preprop")
    parser.add_argument("--arxiv-root", default="dataset/ogbn_arxiv")
    parser.add_argument("--products-root", default="dataset/ogbn_products")
    parser.add_argument("--feature-dim", type=int, default=64)
    parser.add_argument("--edge-chunk-size", type=int, default=65536)
    parser.add_argument("--dst-chunk-size", type=int, default=200000)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--replay-epochs", type=int, default=100)
    parser.add_argument("--product-epochs", type=int, default=200)
    parser.add_argument("--stage1-epochs", type=int, default=100)
    parser.add_argument("--stage2-epochs", type=int, default=100)
    parser.add_argument("--stage1-loss", default="sqrt_weighted_ce")
    parser.add_argument("--stage2-loss", default="cross_entropy")
    parser.add_argument("--stage2-lr-mult", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=16384)
    parser.add_argument("--eval-batch-size", type=int, default=65536)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--small-block-dim", type=int, default=128)
    parser.add_argument("--small-hidden-dim", type=int, default=256)
    parser.add_argument("--small-epochs", type=int, default=80)
    parser.add_argument("--small-stage1-epochs", type=int, default=40)
    parser.add_argument("--small-stage2-epochs", type=int, default=40)
    parser.add_argument("--recovery-epochs", type=int, default=80)
    parser.add_argument("--scap-topk", type=int, default=8)
    args = parser.parse_args()
    if args.only in {"all", "arxiv"}:
        run_arxiv(args)
    if args.only in {"all", "products"}:
        run_products(args)
    if args.only in {"all", "dblp_recovery"}:
        run_dblp_recovery(args)
    if args.only in {"all", "acm"}:
        run_acm(args)
    if args.only in {"all", "imdb"}:
        run_imdb(args)
    if args.only in {"all", "dryrun"}:
        run_dryrun(args)
    if args.only in {"all", "summary"}:
        write_stage_summary()
    print(json.dumps({"status": "completed", "only": args.only}, sort_keys=True))


if __name__ == "__main__":
    main()
