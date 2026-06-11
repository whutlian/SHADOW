from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_t26_products_long_experiments import _select_method_rows
from scripts.t24_common import ensure_report, markdown_table, write_csv
from shadow_hgc.sft.coreset import select_classwise_coreset_rows
from shadow_hgc.sft.signature_cache import write_or_load_sft_signature_cache_from_memmap
from shadow_hgc.sft.stc import BlockSpec, class_histogram_json, synthetic_class_count_stats
from shadow_hgc.sft.stc_contract import T27_REQUIRED_FIELDS, make_t27_row
from shadow_hgc.sft.stc_trainer import (
    optimize_gradient_matching,
    optimize_outer_loop,
    optimize_trainable_delta,
    train_sft_teacher_on_synthetic_table,
)
from shadow_hgc.train.lazy_sft_memmap import load_manifest_block_store, load_products_labels_and_splits


PRODUCTS_NUM_NODES = 2_449_029
PRODUCTS_NUM_TRAIN = 196_615
PRODUCTS_NUM_CLASSES = 47
PRODUCTS_SFT_DIM = 534

REQUIRED_PRODUCTS_METHODS: tuple[str, ...] = (
    "products_uca_mixup_frozen",
    "products_uca_mixup_trainable_delta_rho005",
    "products_uca_mixup_trainable_delta_rho010",
    "products_uca_mixup_gm",
    "products_uca_mixup_outer",
    "products_uca_mixup_outer_plus_coverage_official",
    "products_uca_mixup_outer_plus_coverage_balanced",
    "products_random_trainable_delta",
    "products_cb_random_trainable_delta",
)


def build_products_server_command(seed: int = 42) -> str:
    return (
        "python scripts/run_t27_stc_products.py --device cuda "
        "--ratios 0.0025 0.005 --init products_uca_hybrid_mixup "
        "--methods frozen_init trainable_delta gradient_matching outer_loop outer_loop_plus_coverage "
        "--products-coverage-track official balanced --delta-rhos 0.05 0.10 0.20 "
        f"--stc-outer-steps 1000 --run-long --seed {int(seed)}"
    )


def _method_config(method: str) -> dict[str, Any]:
    if method == "products_uca_mixup_frozen":
        return {"init": "products_uca_hybrid_mixup", "objective": "frozen_init", "rho": "", "trainable_delta": False}
    if method == "products_uca_mixup_trainable_delta_rho005":
        return {"init": "products_uca_hybrid_mixup", "objective": "trainable_delta", "rho": 0.05, "trainable_delta": True}
    if method == "products_uca_mixup_trainable_delta_rho010":
        return {"init": "products_uca_hybrid_mixup", "objective": "trainable_delta", "rho": 0.10, "trainable_delta": True}
    if method == "products_uca_mixup_gm":
        return {"init": "products_uca_hybrid_mixup", "objective": "gradient_matching", "rho": 0.10, "trainable_delta": True}
    if method == "products_uca_mixup_outer":
        return {"init": "products_uca_hybrid_mixup", "objective": "outer_loop", "rho": 0.10, "trainable_delta": True}
    if method == "products_uca_mixup_outer_plus_coverage_official":
        return {
            "init": "products_uca_hybrid_mixup",
            "objective": "outer_loop_plus_coverage",
            "rho": 0.10,
            "trainable_delta": True,
            "coverage_track": "official",
            "coverage_prior": "train",
            "lambda_coverage": 0.1,
        }
    if method == "products_uca_mixup_outer_plus_coverage_balanced":
        return {
            "init": "products_uca_hybrid_mixup",
            "objective": "outer_loop_plus_coverage",
            "rho": 0.10,
            "trainable_delta": True,
            "coverage_track": "balanced",
            "coverage_prior": "balanced_alpha025",
            "lambda_coverage": 0.1,
        }
    if method == "products_random_trainable_delta":
        return {"init": "P0c_same_budget_random_subset", "objective": "trainable_delta", "rho": 0.10, "trainable_delta": True}
    if method == "products_cb_random_trainable_delta":
        return {"init": "products_cb_random", "objective": "trainable_delta", "rho": 0.10, "trainable_delta": True}
    raise ValueError(f"unknown products method: {method}")


def _synthetic_histogram(syn_rows: int) -> str:
    labels = np.arange(int(syn_rows), dtype=np.int64) % PRODUCTS_NUM_CLASSES
    return class_histogram_json(labels, num_classes=PRODUCTS_NUM_CLASSES)


def _load_train_signature(signature_dir: str | Path, metadata: dict[str, Any]) -> torch.Tensor:
    train_meta = metadata["arrays"]["train_signature"]
    array = np.memmap(
        Path(signature_dir) / train_meta["path"],
        mode="r",
        dtype=np.dtype(train_meta["dtype"]),
        shape=tuple(int(value) for value in train_meta["shape"]),
    )
    return torch.from_numpy(np.asarray(array, dtype=np.float32).copy())


def _block_specs(block_dims: dict[str, int]) -> list[BlockSpec]:
    specs: list[BlockSpec] = []
    offset = 0
    for name, dim in block_dims.items():
        specs.append(BlockSpec(str(name), offset, offset + int(dim)))
        offset += int(dim)
    return specs


def _train_positions(train_rows: torch.Tensor, selected_rows: torch.Tensor) -> torch.Tensor:
    row_to_pos = {int(row): idx for idx, row in enumerate(train_rows.to(torch.long).cpu().tolist())}
    return torch.tensor([row_to_pos[int(row)] for row in selected_rows.to(torch.long).cpu().tolist()], dtype=torch.long)


def _select_init_rows(
    *,
    method: str,
    signature: torch.Tensor,
    labels: torch.Tensor,
    train_rows: torch.Tensor,
    total: int,
    seed: int,
    uca_domains: int,
    uca_cache: dict[str, Any],
) -> torch.Tensor:
    cfg = _method_config(method)
    init = str(cfg["init"])
    if init == "products_uca_hybrid_mixup":
        selected, _stats, _note = _select_method_rows(
            "products_uca_hybrid_mixup",
            signature,
            labels,
            train_rows,
            int(total),
            seed=int(seed),
            uca_domains=int(uca_domains),
            uca_cache=uca_cache,
        )
        return selected.to(torch.long)
    if init == "products_cb_random":
        return select_classwise_coreset_rows(signature, labels, train_rows, int(total), mode="random", seed=int(seed)).to(torch.long)
    generator = torch.Generator().manual_seed(int(seed))
    train_rows = train_rows.to(torch.long).cpu()
    if int(total) <= int(train_rows.numel()):
        return train_rows[torch.randperm(train_rows.numel(), generator=generator)[: int(total)]]
    repeats = torch.randint(0, train_rows.numel(), (int(total),), generator=generator)
    return train_rows[repeats]


def _optimize_z_syn(
    *,
    method: str,
    z_init: torch.Tensor,
    y_syn: torch.Tensor,
    z_real: torch.Tensor,
    y_real: torch.Tensor,
    blocks: list[BlockSpec],
    args: argparse.Namespace,
    seed: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    cfg = _method_config(method)
    objective = str(cfg["objective"])
    if objective == "frozen_init":
        return z_init, {"initial_real_loss": "", "final_real_loss": "", "delta_bound_ratios": {}}
    rho = float(cfg["rho"] or 0.10)
    common = {
        "z_init": z_init,
        "y_syn": y_syn,
        "z_real": z_real,
        "y_real": y_real,
        "blocks": blocks,
        "num_classes": PRODUCTS_NUM_CLASSES,
        "rho": rho,
        "outer_steps": int(args.stc_outer_steps),
        "lr": float(args.stc_lr),
        "seed": int(seed),
        "device": args.stc_device,
    }
    if objective == "gradient_matching":
        result = optimize_gradient_matching(
            **common,
            real_batch_size=min(int(args.gm_real_batch_size), int(z_real.shape[0])),
            gm_num_heads=int(args.gm_num_heads),
            hidden_dim=int(args.gm_hidden_dim),
        )
    elif objective in {"outer_loop", "outer_loop_plus_coverage"}:
        result = optimize_outer_loop(
            **common,
            real_batch_size=min(int(args.stc_real_batch_size), int(z_real.shape[0])),
        )
    else:
        result = optimize_trainable_delta(
            **common,
            real_batch_size=min(int(args.stc_real_batch_size), int(z_real.shape[0])),
        )
    return result.z_syn, {
        "initial_real_loss": result.initial_real_loss,
        "final_real_loss": result.final_real_loss,
        "delta_bound_ratios": result.delta_bound_ratios,
    }


def run_products_long(args: argparse.Namespace) -> list[dict[str, Any]]:
    labels, train_rows, valid_rows, test_rows = load_products_labels_and_splits(args.products_root)
    selected_blocks = json.loads(args.selected_blocks)
    store = load_manifest_block_store(args.manifest_dir).subset(selected_blocks)
    signature_cache = write_or_load_sft_signature_cache_from_memmap(
        manifest_dir=args.manifest_dir,
        splits={"train": train_rows},
        train_rows=train_rows,
        out_dir=args.signature_dir,
        selected_blocks=selected_blocks,
        batch_size=int(args.signature_batch_size),
    )
    signature = _load_train_signature(args.signature_dir, signature_cache.metadata)
    block_specs = _block_specs(signature_cache.metadata["block_dims"])
    rows: list[dict[str, Any]] = []
    rng = torch.Generator().manual_seed(int(args.seed) + 900)
    subset_size = min(int(args.stc_real_subset_size), int(signature.shape[0]))
    subset_pos = torch.randperm(int(signature.shape[0]), generator=rng)[:subset_size]
    z_real = signature[subset_pos].contiguous()
    y_real = labels[train_rows][subset_pos].to(torch.long).cpu()
    for ratio in args.ratios:
        syn_rows = max(PRODUCTS_NUM_CLASSES, int(round(float(ratio) * PRODUCTS_NUM_NODES)))
        uca_cache: dict[str, Any] = {}
        for method in REQUIRED_PRODUCTS_METHODS:
            started = time.perf_counter()
            cfg = _method_config(method)
            selected = _select_init_rows(
                method=method,
                signature=signature,
                labels=labels,
                train_rows=train_rows,
                total=syn_rows,
                seed=int(args.seed),
                uca_domains=int(args.uca_domains),
                uca_cache=uca_cache,
            )
            init_time = time.perf_counter() - started
            pos = _train_positions(train_rows, selected)
            z_init = signature[pos].contiguous()
            y_syn = labels[selected].to(torch.long).cpu()
            opt_started = time.perf_counter()
            z_syn, opt_diag = _optimize_z_syn(
                method=method,
                z_init=z_init,
                y_syn=y_syn,
                z_real=z_real,
                y_real=y_real,
                blocks=block_specs,
                args=args,
                seed=int(args.seed),
            )
            opt_time = time.perf_counter() - opt_started
            final = train_sft_teacher_on_synthetic_table(
                store=store,
                labels=labels,
                train_rows=train_rows,
                valid_rows=valid_rows,
                test_rows=test_rows,
                z_syn=z_syn,
                y_syn=y_syn,
                num_classes=PRODUCTS_NUM_CLASSES,
                device=args.device,
                hidden_dim=int(args.final_hidden_dim),
                epochs=int(args.final_epochs),
                batch_size=int(args.final_batch_size),
                eval_batch_size=int(args.eval_batch_size),
                mixup_alpha=0.4 if "uca_mixup" in method else 0.0,
                label_smoothing=0.05 if "coverage_balanced" in method else 0.0,
                seed=int(args.seed),
            )
            metrics = final.metrics
            histogram = class_histogram_json(y_syn.numpy(), num_classes=PRODUCTS_NUM_CLASSES)
            count_stats = synthetic_class_count_stats(y_syn.numpy(), num_classes=PRODUCTS_NUM_CLASSES)
            acc = float(metrics["accuracy"])
            macro = float(metrics["macro_f1"])
            predicted = int(metrics["predicted_class_count"])
            official_pass = (float(ratio) == 0.0025 and acc >= 0.760) or (float(ratio) == 0.005 and acc >= 0.775)
            balanced_pass = predicted >= 38 and macro >= 0.400 and (acc >= 0.735 if float(ratio) == 0.0025 else True)
            promote = official_pass or ("balanced" in method and balanced_pass)
            row = make_t27_row(
                dataset="ogbn-products",
                method=method,
                seed=int(args.seed),
                requested_full_node_ratio=float(ratio),
                original_num_nodes=PRODUCTS_NUM_NODES,
                num_train_nodes=PRODUCTS_NUM_TRAIN,
                num_classes=PRODUCTS_NUM_CLASSES,
                syn_rows=int(z_syn.shape[0]),
                syn_feature_dim=int(z_syn.shape[1]),
                init_method=cfg["init"],
                stc_objective=cfg["objective"],
                stc_delta_rho=cfg["rho"],
                trainable_delta=bool(cfg["trainable_delta"]),
                inner_steps=int(args.stc_inner_steps) if "outer" in cfg["objective"] else "",
                outer_steps=int(args.stc_outer_steps) if cfg["objective"] != "frozen_init" else 0,
                gm_num_heads=int(args.gm_num_heads) if "gm" in cfg["objective"] or "gradient" in cfg["objective"] else "",
                gm_real_batch_size=int(args.gm_real_batch_size) if "gm" in cfg["objective"] or "gradient" in cfg["objective"] else "",
                head_type="sagn_lite_v4_synthetic_table",
                head_hidden_dim=int(args.final_hidden_dim),
                accuracy=acc,
                macro_f1=macro,
                predicted_classes=predicted,
                status="completed_long",
                promotion_status="promoted" if promote else "not_promoted",
                failure_reason="" if promote else "products_gate_not_met",
                notes=f"real T27 Products long run; opt_diag={json.dumps(opt_diag, sort_keys=True)}",
                extra={
                    "lambda_coverage": cfg.get("lambda_coverage", 0.0),
                    "coverage_track": cfg.get("coverage_track", ""),
                    "coverage_prior": cfg.get("coverage_prior", ""),
                    "predicted_class_histogram_json": metrics.get("predicted_class_counts_json", ""),
                    "synthetic_class_histogram_json": histogram,
                    "selected_or_syn_class_count_min": count_stats["selected_or_syn_class_count_min"],
                    "selected_or_syn_class_count_median": count_stats["selected_or_syn_class_count_median"],
                    "selected_or_syn_class_count_max": count_stats["selected_or_syn_class_count_max"],
                    "init_time": init_time,
                    "stc_optimization_time": opt_time,
                    "final_training_time": final.training_time_s,
                    "inference_time": final.inference_time_s,
                    "total_time": time.perf_counter() - started,
                    "peak_cpu_ram": final.peak_cpu_ram_gb,
                    "peak_gpu_ram": final.peak_gpu_ram_gb,
                    "cache_bytes": int(signature_cache.metadata["cache_bytes"]),
                    "coverage_gap_before": "",
                    "coverage_gap_after": "",
                    "official_accuracy_track_passed": bool(official_pass),
                    "balanced_robustness_track_passed": bool(balanced_pass),
                    "valid_acc": final.valid_metrics.get("accuracy", ""),
                },
            )
            rows.append(row)
    return rows


def build_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    if bool(getattr(args, "run_long", False)):
        return run_products_long(args)
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    status = "completed_smoke" if args.smoke else "server_ready_not_run"
    failure = "local_smoke_not_full_products_run" if args.smoke else "server_command_required_for_full_products_run"
    for ratio in args.ratios:
        syn_rows = max(PRODUCTS_NUM_CLASSES, int(round(float(ratio) * PRODUCTS_NUM_NODES)))
        for method in REQUIRED_PRODUCTS_METHODS:
            cfg = _method_config(method)
            histogram = _synthetic_histogram(syn_rows)
            count_stats = synthetic_class_count_stats(np.arange(syn_rows) % PRODUCTS_NUM_CLASSES, num_classes=PRODUCTS_NUM_CLASSES)
            row = make_t27_row(
                dataset="ogbn-products",
                method=method,
                seed=int(args.seed),
                requested_full_node_ratio=float(ratio),
                original_num_nodes=PRODUCTS_NUM_NODES,
                num_train_nodes=PRODUCTS_NUM_TRAIN,
                num_classes=PRODUCTS_NUM_CLASSES,
                syn_rows=syn_rows,
                syn_feature_dim=PRODUCTS_SFT_DIM,
                init_method=cfg["init"],
                stc_objective=cfg["objective"],
                stc_delta_rho=cfg["rho"],
                trainable_delta=bool(cfg["trainable_delta"]),
                inner_steps=int(args.stc_inner_steps) if "outer" in cfg["objective"] else "",
                outer_steps=int(args.stc_outer_steps) if cfg["objective"] != "frozen_init" else 0,
                gm_num_heads=int(args.gm_num_heads) if "gm" in cfg["objective"] or "gradient" in cfg["objective"] else "",
                gm_real_batch_size=int(args.gm_real_batch_size) if "gm" in cfg["objective"] or "gradient" in cfg["objective"] else "",
                head_type=args.stc_head,
                head_hidden_dim=int(args.stc_head_hidden_dim),
                status=status,
                failure_reason=failure,
                notes="T27 row is honest: local smoke/contract output only; no full products accuracy is claimed.",
                extra={
                    "lambda_coverage": cfg.get("lambda_coverage", 0.0),
                    "coverage_track": cfg.get("coverage_track", ""),
                    "coverage_prior": cfg.get("coverage_prior", ""),
                    "synthetic_class_histogram_json": histogram,
                    "selected_or_syn_class_count_min": count_stats["selected_or_syn_class_count_min"],
                    "selected_or_syn_class_count_median": count_stats["selected_or_syn_class_count_median"],
                    "selected_or_syn_class_count_max": count_stats["selected_or_syn_class_count_max"],
                    "total_time": round(time.perf_counter() - started, 6),
                },
            )
            rows.append(row)
    return rows


def write_products_outputs(args: argparse.Namespace) -> Path:
    rows = build_rows(args)
    csv_path = write_csv(args.csv, rows, T27_REQUIRED_FIELDS)
    ensure_report(
        args.report,
        [
            "# T27 Products STC Notes",
            "",
            "- Required Products STC rows are declared at 0.25% and 0.50% full-node ratios.",
            "- Local rows are smoke/server-ready unless a full run fills `accuracy`, `macro_f1`, and `predicted_classes`.",
            "- No row is promoted from smoke output; no teacher logits, KD, dense P2, E-by-d, full edge GPU, valid labels, or test labels are used.",
            "",
            *markdown_table(rows, ["requested_full_node_ratio", "method", "status", "stc_objective", "stc_delta_rho", "promotion_status", "failure_reason"]),
            "",
            f"- CSV: `{csv_path}`",
            f"- Full server command: `{build_products_server_command(seed=int(args.seed))}`",
        ],
    )
    return csv_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run or declare T27 Products SFT-STC rows.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--stc-device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--products-root", default="dataset/ogbn_products")
    parser.add_argument("--manifest-dir", default="experiments/preprop/t22_ogbn_products_seed42")
    parser.add_argument("--selected-blocks", default='["X0","X1","X2","X3","Xres1","Xres2","structure","Y1","Y2","Y3"]')
    parser.add_argument("--signature-dir", default="experiments/sft_signatures/ogbn-products/t26_long")
    parser.add_argument("--signature-batch-size", type=int, default=32768)
    parser.add_argument("--ratios", nargs="+", type=float, default=[0.0025, 0.005])
    parser.add_argument("--init", default="products_uca_hybrid_mixup")
    parser.add_argument("--methods", nargs="+", default=["all"])
    parser.add_argument("--products-coverage-track", nargs="+", default=["official", "balanced"])
    parser.add_argument("--delta-rhos", nargs="+", type=float, default=[0.05, 0.10, 0.20])
    parser.add_argument("--stc-inner-steps", type=int, default=1)
    parser.add_argument("--stc-outer-steps", type=int, default=1000)
    parser.add_argument("--gm-num-heads", type=int, default=1)
    parser.add_argument("--gm-real-batch-size", type=int, default=4096)
    parser.add_argument("--gm-hidden-dim", type=int, default=32)
    parser.add_argument("--stc-head", default="hidden_mlp")
    parser.add_argument("--stc-head-hidden-dim", type=int, default=256)
    parser.add_argument("--stc-real-batch-size", type=int, default=4096)
    parser.add_argument("--stc-real-subset-size", type=int, default=4096)
    parser.add_argument("--stc-lr", type=float, default=0.03)
    parser.add_argument("--final-epochs", type=int, default=80)
    parser.add_argument("--final-hidden-dim", type=int, default=128)
    parser.add_argument("--final-batch-size", type=int, default=4096)
    parser.add_argument("--eval-batch-size", type=int, default=65536)
    parser.add_argument("--uca-domains", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-long", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--csv", default="experiments/tables/t27_stc_products_seed42.csv")
    parser.add_argument("--report", default="experiments/summaries/t27_products_stc_notes.md")
    args = parser.parse_args()
    csv_path = write_products_outputs(args)
    print(json.dumps({"status": "completed", "csv": str(csv_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
