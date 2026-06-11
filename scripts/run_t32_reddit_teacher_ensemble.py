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
from scripts.run_t31_reddit_ttc import _train_teacher_and_cache
from shadow_hgc.data.reddit_stream import load_reddit_raw_memmap_labels_and_splits
from shadow_hgc.eval.resource import current_cpu_ram_bytes, current_gpu_ram_bytes
from shadow_hgc.sft.t32_contract import T32_REQUIRED_FIELDS, apply_t32_promotion_guard, make_t32_row
from shadow_hgc.sft.ttcpp_teacher_ensemble import (
    build_ensemble_probabilities,
    calibrate_teacher_temperature,
    compute_teacher_diagnostics,
    write_teacher_cache_manifest,
)


def _arg(args: argparse.Namespace, name: str, default: Any) -> Any:
    return getattr(args, name, default)


def build_teacher_ensemble_command() -> str:
    return (
        "python scripts/run_t32_reddit_teacher_ensemble.py --device cuda "
        "--teacher-cache-dirs experiments/cache/t31_reddit_ttc_teacher_seed42 "
        "--temperature-grid 0.75 1.0 1.5 2.0 3.0 --seed 42 --run-long"
    )


def _load_member(args: argparse.Namespace, cache_dir: Path) -> tuple[torch.Tensor, float, dict[str, Any]]:
    meta_path = cache_dir / "metadata.json"
    logits_path = cache_dir / "teacher_logits.npy"
    probs_path = cache_dir / "teacher_probs.npy"
    if not meta_path.exists() or not probs_path.exists():
        if bool(_arg(args, "train_missing_teachers", False)):
            metadata = _train_teacher_and_cache(args, cache_dir)
        else:
            raise FileNotFoundError(f"missing teacher cache: {cache_dir}")
    else:
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    labels, _train, valid, _test = load_reddit_raw_memmap_labels_and_splits(_arg(args, "memmap_root", "dataset/Reddit/processed/raw_memmap"))
    if logits_path.exists():
        logits = torch.from_numpy(np.asarray(np.load(logits_path, mmap_mode="r"), dtype=np.float32))
        calibration = calibrate_teacher_temperature(
            logits,
            labels,
            valid_idx=valid,
            temperatures=[float(v) for v in _arg(args, "temperature_grid", [1.0])],
        )
        return logits, float(calibration.temperature), metadata
    probs = torch.from_numpy(np.asarray(np.load(probs_path, mmap_mode="r"), dtype=np.float32))
    return probs, 1.0, metadata


def build_teacher_ensemble_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    started = time.perf_counter()
    cache_dirs = [Path(v) for v in _arg(args, "teacher_cache_dirs", ["experiments/cache/t31_reddit_ttc_teacher_seed42"])]
    members: list[torch.Tensor] = []
    temperatures: list[float] = []
    metadata_rows: list[dict[str, Any]] = []
    for cache_dir in cache_dirs:
        try:
            tensor, temp, meta = _load_member(args, cache_dir)
        except FileNotFoundError:
            continue
        members.append(tensor)
        temperatures.append(temp)
        metadata_rows.append({**meta, "cache_dir": str(cache_dir), "temperature": temp})
    if not members:
        return [
            make_t32_row(
                dataset="Reddit",
                method="reddit_ttcpp_teacher_ensemble_confidence",
                seed=int(_arg(args, "seed", 42)),
                status="blocked",
                failure_reason="missing_reddit_teacher_cache",
                promotion_track="sota_chase",
                promotion_status="not_promoted",
                uses_teacher_logits=True,
                uses_logits_as_input=False,
                next_action=build_teacher_ensemble_command(),
            )
        ]
    ensemble = build_ensemble_probabilities(members, temperatures=temperatures)
    diagnostics = compute_teacher_diagnostics(ensemble.probs, ensemble.disagreement)
    out_dir = Path(_arg(args, "output_cache_dir", "experiments/cache/t32_reddit_teacher_ensemble_seed42"))
    out_dir.mkdir(parents=True, exist_ok=True)
    probs_path = out_dir / "teacher_probs.npy"
    disagreement_path = out_dir / "teacher_disagreement.npy"
    np.save(probs_path, ensemble.probs.numpy().astype(np.float32, copy=False))
    np.save(disagreement_path, ensemble.disagreement.numpy().astype(np.float32, copy=False))
    write_teacher_cache_manifest(out_dir / "metadata.json", rows=metadata_rows, diagnostics=diagnostics)
    cache_bytes = int(probs_path.stat().st_size + disagreement_path.stat().st_size + (out_dir / "metadata.json").stat().st_size)
    rows: list[dict[str, Any]] = []
    for method in [
        "reddit_ttcpp_teacher_ensemble_confidence",
        "reddit_ttcpp_teacher_ensemble_disagreement",
        "reddit_ttcpp_teacher_ensemble_coverage_boundary",
    ]:
        row = make_t32_row(
            dataset="Reddit",
            method=method,
            seed=int(_arg(args, "seed", 42)),
            status="completed_long",
            failure_reason="",
            promotion_track="sota_chase",
            promotion_status="not_promoted",
            uses_teacher_logits=True,
            uses_logits_as_input=False,
            candidate_nodes="all",
            teacher_accuracy=metadata_rows[0].get("teacher_accuracy", metadata_rows[0].get("accuracy", "")),
            teacher_valid_acc=metadata_rows[0].get("teacher_valid_acc", ""),
            teacher_temperature=";".join(str(v) for v in temperatures),
            teacher_entropy_mean=diagnostics.get("teacher_entropy_mean", ""),
            teacher_disagreement_mean=diagnostics.get("teacher_disagreement_mean", ""),
            predicted_classes=diagnostics.get("predicted_classes", ""),
            cache_bytes=cache_bytes,
            precompute_time=float(time.perf_counter() - started),
            peak_cpu_ram=current_cpu_ram_bytes(),
            peak_gpu_ram=current_gpu_ram_bytes(),
            notes=f"ensemble_members={len(members)}; probs={probs_path}; disagreement={disagreement_path}",
            next_action=build_teacher_ensemble_command(),
        )
        rows.append(apply_t32_promotion_guard(row))
    return rows


def write_outputs(args: argparse.Namespace) -> Path:
    rows = build_teacher_ensemble_rows(args)
    csv_path = write_csv(_arg(args, "csv", "experiments/tables/t32_reddit_ttcpp_teacher_ensemble.csv"), rows, T32_REQUIRED_FIELDS)
    ensure_report(
        _arg(args, "report", "experiments/summaries/t32_reddit_teacher_ensemble.md"),
        [
            "# T32 Reddit Teacher Ensemble",
            "",
            *markdown_table(rows, ["method", "status", "teacher_accuracy", "teacher_temperature", "teacher_entropy_mean", "teacher_disagreement_mean", "cache_bytes", "failure_reason"]),
            "",
            f"- CSV: `{csv_path}`",
            f"- Next command: `{build_teacher_ensemble_command()}`",
        ],
    )
    return csv_path


def main() -> None:
    parser = argparse.ArgumentParser(description="T32 Reddit TTC++ teacher ensemble cache builder.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--memmap-root", default="dataset/Reddit/processed/raw_memmap")
    parser.add_argument("--manifest-dir", default="experiments/preprop/t24_reddit_streaming_seed42")
    parser.add_argument("--teacher-cache-dirs", nargs="+", default=["experiments/cache/t31_reddit_ttc_teacher_seed42"])
    parser.add_argument("--output-cache-dir", default="experiments/cache/t32_reddit_teacher_ensemble_seed42")
    parser.add_argument("--temperature-grid", nargs="+", type=float, default=[1.0])
    parser.add_argument("--train-missing-teachers", action="store_true")
    parser.add_argument("--teacher-model-type", default="sagn_lite_v4")
    parser.add_argument("--teacher-hidden-dim", type=int, default=128)
    parser.add_argument("--teacher-dropout", type=float, default=0.3)
    parser.add_argument("--teacher-num-layers", type=int, default=2)
    parser.add_argument("--teacher-epochs", type=int, default=30)
    parser.add_argument("--run-long", action="store_true")
    parser.add_argument("--csv", default="experiments/tables/t32_reddit_ttcpp_teacher_ensemble.csv")
    parser.add_argument("--report", default="experiments/summaries/t32_reddit_teacher_ensemble.md")
    args = parser.parse_args()
    path = write_outputs(args)
    print(json.dumps({"status": "completed", "csv": str(path)}, sort_keys=True))


if __name__ == "__main__":
    main()
