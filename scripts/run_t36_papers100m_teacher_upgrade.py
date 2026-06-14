from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.t24_common import write_csv
from shadow_hgc.eval.resource import current_cpu_ram_bytes
from shadow_hgc.ultra.papers100m_memmap import read_json, write_json
from shadow_hgc.ultra.papers100m_teacher_upgrade import install_teacher_upgrade, train_teacher_upgrade


FIELDS = [
    "method",
    "teacher_id",
    "cache_build_id",
    "edge_cache_id",
    "sft_cache_id",
    "feature_block_mode",
    "uses_streaming_logits",
    "teacher_cache_mode",
    "teacher_cache_id",
    "valid_acc",
    "test_acc",
    "macro_f1",
    "predicted_classes",
    "topk_cache_bytes",
    "train_time",
    "infer_time",
    "peak_cpu_ram",
    "peak_gpu_ram",
    "uses_dense_teacher_cache_in_ram",
    "uses_dense_all_node_teacher_cache",
    "uses_valid_labels_as_input",
    "uses_test_labels_as_input",
    "promotion_status",
    "failure_reason",
    "notes",
]


def _epochs_for(method: str, default_epochs: int) -> int:
    if default_epochs > 0:
        return int(default_epochs)
    if method == "sgc":
        return 20
    if method == "sign":
        return 30
    if method == "sagn_lite":
        return 40
    if method == "gamlp_lite":
        return 50
    return 1


def main() -> None:
    parser = argparse.ArgumentParser(description="T36 papers100M teacher upgrade runner.")
    parser.add_argument("--cache-root", default="caches/papers100m/stt_v1")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--teachers", nargs="+", default=["sgc", "sign", "sagn_lite", "gamlp_lite"])
    parser.add_argument("--feature-block-modes", nargs="+", default=["minimal"])
    parser.add_argument("--teacher-cache-modes", nargs="+", default=["topk8_tail"])
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--epochs", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--eval-batch-size", type=int, default=65536)
    parser.add_argument("--infer-batch-size", type=int, default=65536)
    parser.add_argument("--preload-train", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--install-best", action="store_true")
    parser.add_argument("--run-long", action="store_true")
    parser.add_argument("--tables-dir", default="experiments/tables")
    args = parser.parse_args()

    cache_root = Path(args.cache_root)
    rows: list[dict[str, Any]] = []
    for teacher in args.teachers:
        for feature_mode in args.feature_block_modes:
            for cache_mode in args.teacher_cache_modes:
                row = train_teacher_upgrade(
                    cache_root,
                    method=str(teacher),
                    feature_block_mode=str(feature_mode),
                    teacher_cache_mode=str(cache_mode),
                    seed=int(args.seed),
                    epochs=_epochs_for(str(teacher), int(args.epochs)),
                    batch_size=int(args.batch_size),
                    eval_batch_size=int(args.eval_batch_size),
                    infer_batch_size=int(args.infer_batch_size),
                    device=str(args.device),
                    force=bool(args.force),
                    preload_train=bool(args.preload_train),
                )
                row["peak_cpu_ram"] = current_cpu_ram_bytes()
                rows.append(row)
                write_csv(Path(args.tables_dir) / "t36_papers100m_teacher_upgrade.csv", rows, FIELDS)

    completed = [row for row in rows if row.get("test_acc", "") != ""]
    if completed:
        best = max(completed, key=lambda item: (float(item.get("test_acc", 0.0) or 0.0), float(item.get("valid_acc", 0.0) or 0.0)))
        write_json(cache_root / "teacher_upgrade" / "best_teacher.json", best)
        if bool(args.install_best):
            installed = install_teacher_upgrade(cache_root, str(best["teacher_id"]))
            print(f"installed_teacher_cache_id={installed.get('teacher_cache_id', '')}")
        print(f"best_teacher_id={best['teacher_id']}")
        print(f"best_teacher_test_acc={best.get('test_acc', '')}")
    print("status=completed")


if __name__ == "__main__":
    main()
