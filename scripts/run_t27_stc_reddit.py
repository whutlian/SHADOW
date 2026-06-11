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

from scripts.t24_common import ensure_report, markdown_table, write_csv
from shadow_hgc.data.reddit_stream import load_reddit_raw_memmap_labels_and_splits
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
from shadow_hgc.train.lazy_sft_memmap import load_manifest_block_store


REDDIT_NUM_NODES = 232_965
REDDIT_NUM_TRAIN = 153_431
REDDIT_NUM_CLASSES = 41
REDDIT_SFT_DIM = 602

REQUIRED_REDDIT_METHODS: tuple[str, ...] = (
    "reddit_random_frozen_init",
    "reddit_random_trainable_delta_rho005",
    "reddit_random_trainable_delta_rho010",
    "reddit_random_gm",
    "reddit_random_outer",
    "reddit_random_gm_plus_moment",
    "reddit_kcenter_trainable_delta",
    "reddit_medoid_trainable_delta",
)


def build_reddit_server_command(seeds: list[int] | tuple[int, ...] = (1, 2, 3, 4, 5)) -> str:
    seed_text = " ".join(str(int(seed)) for seed in seeds)
    return (
        "python scripts/run_t27_stc_reddit.py --device cuda --ratios 0.005 0.01 "
        "--init current_sft_signature_random --methods frozen_init trainable_delta gradient_matching outer_loop "
        "--delta-rhos 0.05 0.10 "
        f"--run-long --seeds {seed_text}"
    )


def _method_config(method: str) -> dict[str, Any]:
    if method == "reddit_random_frozen_init":
        return {"init": "current_sft_signature_random", "objective": "frozen_init", "rho": "", "trainable_delta": False}
    if method == "reddit_random_trainable_delta_rho005":
        return {"init": "current_sft_signature_random", "objective": "trainable_delta", "rho": 0.05, "trainable_delta": True}
    if method == "reddit_random_trainable_delta_rho010":
        return {"init": "current_sft_signature_random", "objective": "trainable_delta", "rho": 0.10, "trainable_delta": True}
    if method == "reddit_random_gm":
        return {"init": "current_sft_signature_random", "objective": "gradient_matching", "rho": 0.10, "trainable_delta": True}
    if method == "reddit_random_outer":
        return {"init": "current_sft_signature_random", "objective": "outer_loop", "rho": 0.10, "trainable_delta": True}
    if method == "reddit_random_gm_plus_moment":
        return {"init": "current_sft_signature_random", "objective": "gradient_matching", "rho": 0.10, "trainable_delta": True, "lambda_moment": 0.1}
    if method == "reddit_kcenter_trainable_delta":
        return {"init": "current_sft_signature_kcenter", "objective": "trainable_delta", "rho": 0.10, "trainable_delta": True}
    if method == "reddit_medoid_trainable_delta":
        return {"init": "current_sft_signature_medoid", "objective": "trainable_delta", "rho": 0.10, "trainable_delta": True}
    raise ValueError(f"unknown Reddit method: {method}")


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
) -> torch.Tensor:
    init = str(_method_config(method)["init"])
    mode = "random"
    if "kcenter" in init:
        mode = "kcenter"
    elif "medoid" in init:
        mode = "medoid"
    return select_classwise_coreset_rows(signature, labels, train_rows, int(total), mode=mode, seed=int(seed)).to(torch.long)


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
        "num_classes": REDDIT_NUM_CLASSES,
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
    elif objective == "outer_loop":
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


def run_reddit_long(args: argparse.Namespace) -> list[dict[str, Any]]:
    labels, train_rows, valid_rows, test_rows = load_reddit_raw_memmap_labels_and_splits(args.memmap_root)
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
    seeds = args.seeds if getattr(args, "seeds", None) else [args.seed]
    for seed in seeds:
        rng = torch.Generator().manual_seed(int(seed) + 900)
        subset_size = min(int(args.stc_real_subset_size), int(signature.shape[0]))
        subset_pos = torch.randperm(int(signature.shape[0]), generator=rng)[:subset_size]
        z_real = signature[subset_pos].contiguous()
        y_real = labels[train_rows][subset_pos].to(torch.long).cpu()
        for ratio in args.ratios:
            syn_rows = max(REDDIT_NUM_CLASSES, int(round(float(ratio) * REDDIT_NUM_NODES)))
            for method in REQUIRED_REDDIT_METHODS:
                started = time.perf_counter()
                cfg = _method_config(method)
                selected = _select_init_rows(
                    method=method,
                    signature=signature,
                    labels=labels,
                    train_rows=train_rows,
                    total=syn_rows,
                    seed=int(seed),
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
                    seed=int(seed),
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
                    num_classes=REDDIT_NUM_CLASSES,
                    device=args.device,
                    hidden_dim=int(args.final_hidden_dim),
                    epochs=int(args.final_epochs),
                    batch_size=int(args.final_batch_size),
                    eval_batch_size=int(args.eval_batch_size),
                    seed=int(seed),
                )
                metrics = final.metrics
                labels_np = y_syn.numpy()
                histogram = class_histogram_json(labels_np, num_classes=REDDIT_NUM_CLASSES)
                count_stats = synthetic_class_count_stats(labels_np, num_classes=REDDIT_NUM_CLASSES)
                acc = float(metrics["accuracy"])
                macro = float(metrics["macro_f1"])
                predicted = int(metrics["predicted_class_count"])
                passed = (float(ratio) == 0.005 and acc >= 0.928 and macro >= 0.890) or (
                    float(ratio) == 0.01 and acc >= 0.932 and macro >= 0.895
                )
                rows.append(
                    make_t27_row(
                        dataset="Reddit",
                        method=method,
                        seed=int(seed),
                        requested_full_node_ratio=float(ratio),
                        original_num_nodes=REDDIT_NUM_NODES,
                        num_train_nodes=REDDIT_NUM_TRAIN,
                        num_classes=REDDIT_NUM_CLASSES,
                        syn_rows=int(z_syn.shape[0]),
                        syn_feature_dim=int(z_syn.shape[1]),
                        init_method=cfg["init"],
                        stc_objective=cfg["objective"],
                        stc_delta_rho=cfg["rho"],
                        trainable_delta=bool(cfg["trainable_delta"]),
                        outer_steps=int(args.stc_outer_steps) if cfg["objective"] != "frozen_init" else 0,
                        gm_num_heads=int(args.gm_num_heads) if "gm" in cfg["objective"] or "gradient" in cfg["objective"] else "",
                        gm_real_batch_size=int(args.gm_real_batch_size) if "gm" in cfg["objective"] or "gradient" in cfg["objective"] else "",
                        head_type="sagn_lite_v4_synthetic_table",
                        head_hidden_dim=int(args.final_hidden_dim),
                        accuracy=acc,
                        macro_f1=macro,
                        predicted_classes=predicted,
                        status="completed_long",
                        promotion_status="promoted" if passed else "not_promoted",
                        failure_reason="" if passed else "reddit_gate_not_met",
                        notes=f"real T27 Reddit long run; opt_diag={json.dumps(opt_diag, sort_keys=True)}",
                        extra={
                            "lambda_moment": cfg.get("lambda_moment", 0.1),
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
                            "valid_acc": final.valid_metrics.get("accuracy", ""),
                        },
                    )
                )
    return rows


def build_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    if bool(getattr(args, "run_long", False)):
        return run_reddit_long(args)
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    status = "completed_smoke" if args.smoke else "server_ready_not_run"
    failure = "local_smoke_not_full_reddit_run" if args.smoke else "server_command_required_for_reddit_seed_sweep"
    seeds = args.seeds if getattr(args, "seeds", None) else [args.seed]
    for seed in seeds:
        for ratio in args.ratios:
            syn_rows = max(REDDIT_NUM_CLASSES, int(round(float(ratio) * REDDIT_NUM_NODES)))
            labels = np.arange(syn_rows, dtype=np.int64) % REDDIT_NUM_CLASSES
            histogram = class_histogram_json(labels, num_classes=REDDIT_NUM_CLASSES)
            count_stats = synthetic_class_count_stats(labels, num_classes=REDDIT_NUM_CLASSES)
            for method in REQUIRED_REDDIT_METHODS:
                cfg = _method_config(method)
                rows.append(
                    make_t27_row(
                        dataset="Reddit",
                        method=method,
                        seed=int(seed),
                        requested_full_node_ratio=float(ratio),
                        original_num_nodes=REDDIT_NUM_NODES,
                        num_train_nodes=REDDIT_NUM_TRAIN,
                        num_classes=REDDIT_NUM_CLASSES,
                        syn_rows=syn_rows,
                        syn_feature_dim=REDDIT_SFT_DIM,
                        init_method=cfg["init"],
                        stc_objective=cfg["objective"],
                        stc_delta_rho=cfg["rho"],
                        trainable_delta=bool(cfg["trainable_delta"]),
                        outer_steps=int(args.stc_outer_steps) if cfg["objective"] != "frozen_init" else 0,
                        gm_num_heads=int(args.gm_num_heads) if "gm" in cfg["objective"] or "gradient" in cfg["objective"] else "",
                        gm_real_batch_size=int(args.gm_real_batch_size) if "gm" in cfg["objective"] or "gradient" in cfg["objective"] else "",
                        head_type=args.stc_head,
                        head_hidden_dim=int(args.stc_head_hidden_dim),
                        status=status,
                        failure_reason=failure,
                        notes="T27 Reddit row is honest smoke/server-ready output; full seed metrics require the listed server command.",
                        extra={
                            "lambda_moment": cfg.get("lambda_moment", 0.1),
                            "synthetic_class_histogram_json": histogram,
                            "selected_or_syn_class_count_min": count_stats["selected_or_syn_class_count_min"],
                            "selected_or_syn_class_count_median": count_stats["selected_or_syn_class_count_median"],
                            "selected_or_syn_class_count_max": count_stats["selected_or_syn_class_count_max"],
                            "total_time": round(time.perf_counter() - started, 6),
                        },
                    )
                )
    return rows


def write_reddit_outputs(args: argparse.Namespace) -> Path:
    rows = build_rows(args)
    csv_path = write_csv(args.csv, rows, T27_REQUIRED_FIELDS)
    ensure_report(
        args.report,
        [
            "# T27 Reddit STC Notes",
            "",
            "- Required Reddit STC rows are declared at 0.50% and 1.00% full-node ratios.",
            "- Seed 42 smoke is local; seeds 1..5 are emitted by the server command and must not be inferred from seed 42.",
            "- No smoke row is promoted.",
            "",
            *markdown_table(rows, ["requested_full_node_ratio", "seed", "method", "status", "stc_objective", "promotion_status", "failure_reason"]),
            "",
            f"- CSV: `{csv_path}`",
            f"- Full server command: `{build_reddit_server_command(seeds=[1, 2, 3, 4, 5])}`",
        ],
    )
    return csv_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run or declare T27 Reddit SFT-STC rows.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--stc-device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--manifest-dir", default="experiments/preprop/t24_reddit_streaming_seed42")
    parser.add_argument("--memmap-root", default="dataset/Reddit/processed/raw_memmap")
    parser.add_argument("--selected-blocks", default='["X0","X1","X2","X3","Xres1","Y1","Y2","Y3","structure"]')
    parser.add_argument("--signature-dir", default="experiments/sft_signatures/Reddit/t24_streaming")
    parser.add_argument("--signature-batch-size", type=int, default=32768)
    parser.add_argument("--ratios", nargs="+", type=float, default=[0.005, 0.01])
    parser.add_argument("--init", default="current_sft_signature_random")
    parser.add_argument("--methods", nargs="+", default=["all"])
    parser.add_argument("--delta-rhos", nargs="+", type=float, default=[0.05, 0.10])
    parser.add_argument("--stc-outer-steps", type=int, default=1000)
    parser.add_argument("--gm-num-heads", type=int, default=1)
    parser.add_argument("--gm-real-batch-size", type=int, default=4096)
    parser.add_argument("--gm-hidden-dim", type=int, default=32)
    parser.add_argument("--stc-head", default="hidden_mlp")
    parser.add_argument("--stc-head-hidden-dim", type=int, default=256)
    parser.add_argument("--stc-real-batch-size", type=int, default=4096)
    parser.add_argument("--stc-real-subset-size", type=int, default=4096)
    parser.add_argument("--stc-lr", type=float, default=0.03)
    parser.add_argument("--final-epochs", type=int, default=30)
    parser.add_argument("--final-hidden-dim", type=int, default=128)
    parser.add_argument("--final-batch-size", type=int, default=4096)
    parser.add_argument("--eval-batch-size", type=int, default=65536)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--run-long", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--csv", default="experiments/tables/t27_stc_reddit_seed42.csv")
    parser.add_argument("--report", default="experiments/summaries/t27_reddit_stc_notes.md")
    args = parser.parse_args()
    csv_path = write_reddit_outputs(args)
    print(json.dumps({"status": "completed", "csv": str(csv_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
