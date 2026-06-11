from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_t33_reddit_ratio_curve import build_ratio_curve_rows
from scripts.t24_common import ensure_report, markdown_table, write_csv
from shadow_hgc.sft.t33_contract import T33_REQUIRED_FIELDS
from shadow_hgc.sft.ttcpp_topk_cache import dense_probs_to_topk_cache


def _load_dense_probs(path: Path) -> torch.Tensor:
    return torch.from_numpy(np.asarray(np.load(path, mmap_mode="r"), dtype=np.float32))


def _prepare_cache(args: argparse.Namespace, mode: str) -> Path:
    source_dir = Path(args.source_teacher_cache_dir)
    source_probs = source_dir / "teacher_probs.npy"
    target_dir = Path(args.cache_work_dir) / str(mode)
    target_dir.mkdir(parents=True, exist_ok=True)
    target_probs = target_dir / "teacher_probs.npy"
    target_meta = target_dir / "metadata.json"
    if mode == "dense_fp16":
        probs = _load_dense_probs(source_probs).to(torch.float16)
        np.save(target_probs, probs.cpu().numpy())
        metadata = {"cache_mode": mode, "teacher_cache_bytes": int(target_probs.stat().st_size), "source": str(source_probs)}
    else:
        dense = _load_dense_probs(source_probs)
        k = 4 if mode == "topk4_fp16" else 8
        cache = dense_probs_to_topk_cache(dense, k=k, include_entropy_margin=mode == "topk8_plus_entropy_margin")
        reconstructed = cache.to_dense(num_classes=dense.shape[1]).to(torch.float16)
        np.save(target_probs, reconstructed.cpu().numpy())
        aux_bytes = int(cache.topk_class_ids.numel() * cache.topk_class_ids.element_size() + cache.topk_probs.numel() * cache.topk_probs.element_size() + cache.residual_mass.numel() * cache.residual_mass.element_size())
        if cache.entropy is not None:
            aux_bytes += int(cache.entropy.numel() * cache.entropy.element_size())
        if cache.margin is not None:
            aux_bytes += int(cache.margin.numel() * cache.margin.element_size())
        metadata = {
            "cache_mode": mode,
            "teacher_cache_bytes": aux_bytes,
            "source": str(source_probs),
            "reconstructed_probs_path": str(target_probs),
            "topk_residual_policy": "drop_and_renormalize_in_selector",
        }
    if (source_dir / "metadata.json").exists():
        try:
            src_meta = json.loads((source_dir / "metadata.json").read_text(encoding="utf-8"))
            if "diagnostics" in src_meta:
                metadata.update(src_meta["diagnostics"])
            metadata.setdefault("teacher_accuracy", src_meta.get("teacher_accuracy", ""))
            metadata.setdefault("teacher_valid_acc", src_meta.get("teacher_valid_acc", ""))
        except json.JSONDecodeError:
            pass
    target_meta.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return target_dir


def build_cache_ablation_rows(args: argparse.Namespace) -> list[dict]:
    rows: list[dict] = []
    for mode in args.teacher_cache_modes:
        cache_dir = _prepare_cache(args, mode)
        local = argparse.Namespace(**vars(args))
        local.teacher_ensemble_cache_dir = str(cache_dir)
        local.teacher_cache_mode = str(mode)
        mode_rows = build_ratio_curve_rows(local)
        for row in mode_rows:
            row["method"] = f"{row['method']}_{mode}"
            row["teacher_cache_mode"] = str(mode)
            meta = json.loads((cache_dir / "metadata.json").read_text(encoding="utf-8"))
            row["teacher_cache_bytes"] = meta.get("teacher_cache_bytes", row.get("teacher_cache_bytes", ""))
            row["notes"] = (str(row.get("notes", "")) + f"; cache_ablation_source={cache_dir}").strip("; ")
        rows.extend(mode_rows)
    return rows


def write_outputs(args: argparse.Namespace) -> Path:
    if bool(args.clean_cache_work_dir) and Path(args.cache_work_dir).exists():
        shutil.rmtree(args.cache_work_dir)
    rows = build_cache_ablation_rows(args)
    csv_path = write_csv(args.csv, rows, T33_REQUIRED_FIELDS)
    ensure_report(
        args.report,
        [
            "# T33 Reddit Teacher Cache Ablation",
            "",
            *markdown_table(rows, ["method", "requested_full_node_ratio", "accuracy", "macro_f1", "teacher_cache_mode", "teacher_cache_bytes", "promotion_status", "failure_reason"]),
            "",
            f"- CSV: `{csv_path}`",
        ],
    )
    return csv_path


def main() -> None:
    parser = argparse.ArgumentParser(description="T33 Reddit dense vs top-k teacher cache ablation.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--source-teacher-cache-dir", default="experiments/cache/t31_reddit_ttc_teacher_seed42")
    parser.add_argument("--cache-work-dir", default="experiments/cache/t33_reddit_teacher_cache_ablation")
    parser.add_argument("--clean-cache-work-dir", action="store_true")
    parser.add_argument("--manifest-dir", default="experiments/preprop/t24_reddit_streaming_seed42")
    parser.add_argument("--memmap-root", default="dataset/Reddit/processed/raw_memmap")
    parser.add_argument("--selected-blocks", default=json.dumps(["X0", "X1", "X2", "X3", "Xres1", "Y1", "Y2", "Y3", "structure"]))
    parser.add_argument("--teacher-cache-dir", default="experiments/cache/t31_reddit_ttc_teacher_seed42")
    parser.add_argument("--teacher-ensemble-cache-dir", default="experiments/cache/t32_reddit_teacher_ensemble_seed42")
    parser.add_argument("--teacher-model-type", default="sagn_lite_v4")
    parser.add_argument("--teacher-hidden-dim", type=int, default=128)
    parser.add_argument("--teacher-dropout", type=float, default=0.3)
    parser.add_argument("--teacher-num-layers", type=int, default=2)
    parser.add_argument("--teacher-epochs", type=int, default=30)
    parser.add_argument("--ratios", nargs="+", type=float, default=[0.001, 0.005])
    parser.add_argument("--teacher-cache-modes", nargs="+", default=["dense_fp16", "topk4_fp16", "topk8_fp16", "topk8_plus_entropy_margin"])
    parser.add_argument("--methods", nargs="+", default=["reddit_ttcpp_gamlp_table_student", "reddit_ttcpp_sagn_table_student"])
    parser.add_argument("--budget-policy", default="ratio_adaptive_v2")
    parser.add_argument("--temperatures", nargs="+", type=float, default=[2.0])
    parser.add_argument("--lambda-hard", nargs="+", type=float, default=[0.25])
    parser.add_argument("--lambda-prior", nargs="+", type=float, default=[0.02])
    parser.add_argument("--lambda-conf", type=float, default=0.0)
    parser.add_argument("--lambda-mix", type=float, default=0.0)
    parser.add_argument("--student-model-type", default="sagn_lite_v4")
    parser.add_argument("--student-lr", type=float, default=0.003)
    parser.add_argument("--student-batch-size", type=int, default=2048)
    parser.add_argument("--teacher-eval-batch-size", type=int, default=65536)
    parser.add_argument("--hidden-dims", nargs="+", type=int, default=[256])
    parser.add_argument("--epochs", nargs="+", type=int, default=[220])
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--mixup-alpha", type=float, default=0.4)
    parser.add_argument("--checkpoint-selection", default="best_valid_acc")
    parser.add_argument("--run-long", action="store_true")
    parser.add_argument("--csv", default="experiments/tables/t33_reddit_teacher_cache_ablation.csv")
    parser.add_argument("--report", default="experiments/summaries/t33_reddit_teacher_cache_ablation.md")
    args = parser.parse_args()
    path = write_outputs(args)
    print(json.dumps({"status": "completed", "csv": str(path)}, sort_keys=True))


if __name__ == "__main__":
    main()
