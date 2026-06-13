from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_t34_reddit_stt_ratio_curve import aggregate_t34_ratio_rows, build_ratio_curve_rows
from scripts.t24_common import ensure_report, markdown_table, write_csv
from shadow_hgc.sft.stt_cache import dense_to_stt_cache, estimate_stt_cache_bytes
from shadow_hgc.sft.t34_contract import T34_REQUIRED_FIELDS


def _load_dense_probs(path: Path) -> torch.Tensor:
    return torch.from_numpy(np.asarray(np.load(path, mmap_mode="r"), dtype=np.float32))


def _method_for_cache(base: str, mode: str) -> str:
    if mode == "topk8_tail":
        return "reddit_stt_topk8_tail"
    if mode == "topk16_tail":
        return "reddit_stt_topk16_tail"
    return f"{base}_{mode}"


def _prepare_cache(args: argparse.Namespace, mode: str) -> tuple[Path, dict[str, Any]]:
    source_dir = Path(args.source_teacher_cache_dir)
    source_probs = source_dir / "teacher_probs.npy"
    target_dir = Path(args.cache_work_dir) / str(mode)
    target_dir.mkdir(parents=True, exist_ok=True)
    target_probs = target_dir / "teacher_probs.npy"
    dense = _load_dense_probs(source_probs)
    if mode == "dense_fp16":
        np.save(target_probs, dense.to(torch.float16).cpu().numpy())
    else:
        cache = dense_to_stt_cache(dense, mode=mode, tail_prior=dense.mean(dim=0))
        rows = torch.arange(dense.shape[0], dtype=torch.long)
        reconstructed = cache.reconstruct_rows(rows, num_classes=dense.shape[1]).to(torch.float16)
        np.save(target_probs, reconstructed.cpu().numpy())
    estimate = estimate_stt_cache_bytes(num_nodes=int(dense.shape[0]), num_classes=int(dense.shape[1]), mode=mode)
    metadata = {
        "cache_mode": mode,
        "teacher_cache_bytes": estimate["teacher_cache_bytes"],
        "teacher_dense_cache_bytes_diagnostic": estimate["teacher_dense_cache_bytes_diagnostic"],
        "cache_compression_ratio": estimate["cache_compression_ratio"],
        "source": str(source_probs),
        "reconstructed_probs_path": str(target_probs),
        "medium_dataset_reconstruction_diagnostic": mode != "dense_fp16",
    }
    source_meta = source_dir / "metadata.json"
    if source_meta.exists():
        try:
            src = json.loads(source_meta.read_text(encoding="utf-8"))
            metadata.update(src.get("diagnostics", {}))
            metadata.setdefault("teacher_accuracy", src.get("teacher_accuracy", ""))
            metadata.setdefault("teacher_valid_acc", src.get("teacher_valid_acc", ""))
        except json.JSONDecodeError:
            pass
    (target_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return target_dir, metadata


def build_cache_ablation_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for mode in args.teacher_cache_modes:
        cache_dir, meta = _prepare_cache(args, str(mode))
        local = argparse.Namespace(**vars(args))
        local.teacher_ensemble_cache_dir = str(cache_dir)
        local.teacher_cache_mode = str(mode)
        mode_rows = build_ratio_curve_rows(local)
        for row in mode_rows:
            row["method"] = _method_for_cache(str(row["method"]).split("_dense_fp16")[0], str(mode))
            row["teacher_cache_mode"] = str(mode)
            row["teacher_cache_bytes"] = meta["teacher_cache_bytes"]
            row["teacher_dense_cache_bytes_diagnostic"] = meta["teacher_dense_cache_bytes_diagnostic"]
            row["cache_compression_ratio"] = meta["cache_compression_ratio"]
            row["notes"] = (str(row.get("notes", "")) + f"; cache_ablation_source={cache_dir}").strip("; ")
        rows.extend(mode_rows)
    return rows


def write_outputs(args: argparse.Namespace) -> Path:
    if bool(args.clean_cache_work_dir) and Path(args.cache_work_dir).exists():
        shutil.rmtree(args.cache_work_dir)
    rows = build_cache_ablation_rows(args)
    csv_path = write_csv(args.csv, rows, T34_REQUIRED_FIELDS)
    agg = aggregate_t34_ratio_rows(rows)
    ensure_report(
        args.report,
        [
            "# T34 Reddit STT Cache Ablation",
            "",
            *markdown_table(rows, ["method", "requested_full_node_ratio", "accuracy", "macro_f1", "teacher_cache_mode", "teacher_cache_bytes", "cache_compression_ratio", "promotion_status", "failure_reason"]),
            "",
            "## Aggregate",
            "",
            *markdown_table(agg, ["method", "requested_full_node_ratio", "seed_count", "accuracy_mean", "accuracy_std", "macro_f1_mean", "macro_f1_std"]),
            "",
            f"- CSV: `{csv_path}`",
        ],
    )
    return csv_path


def main() -> None:
    parser = argparse.ArgumentParser(description="T34 Reddit STT teacher cache ablation.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--source-teacher-cache-dir", default="experiments/cache/t31_reddit_ttc_teacher_seed42")
    parser.add_argument("--cache-work-dir", default="experiments/cache/t34_reddit_teacher_cache_ablation")
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
    parser.add_argument("--teacher-cache-modes", nargs="+", default=["dense_fp16", "topk4_fp16", "topk8_fp16", "topk8_tail", "topk16_tail"])
    parser.add_argument("--methods", nargs="+", default=["reddit_stt_gamlp_ratio_v2", "reddit_stt_sagn_ratio_v2"])
    parser.add_argument("--budget-policy", "--stt-budget-policy", dest="budget_policy", default="ratio_adaptive_v2")
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
    parser.add_argument("--csv", default="experiments/tables/t34_reddit_stt_cache_ablation.csv")
    parser.add_argument("--report", default="experiments/summaries/t34_reddit_stt_cache_ablation.md")
    args = parser.parse_args()
    path = write_outputs(args)
    print(json.dumps({"status": "completed", "csv": str(path)}, sort_keys=True))


if __name__ == "__main__":
    main()
