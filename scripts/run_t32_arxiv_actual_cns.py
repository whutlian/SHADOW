from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.t24_common import ensure_report, markdown_table, write_csv
from shadow_hgc.eval.resource import current_cpu_ram_bytes, current_gpu_ram_bytes
from shadow_hgc.sft.arxiv_actual_cns import run_actual_cns_grid
from shadow_hgc.sft.arxiv_base_predictors_v3 import load_validated_base_logits, validate_arxiv_split_and_feature_alignment
from shadow_hgc.sft.t32_arxiv_cns import cns_failure_reason, cns_grid_plan_v2, transform_arxiv_edge_index
from shadow_hgc.sft.t32_contract import ARXIV_NUM_CLASSES, T32_REQUIRED_FIELDS, apply_t32_promotion_guard, make_t32_row

from scripts.run_t31_arxiv_actual_cns import _load_arxiv_arrays, train_raw_x_mlp_logits


def _arg(args: argparse.Namespace, name: str, default: Any) -> Any:
    return getattr(args, name, default)


def build_arxiv_cns_server_command() -> str:
    return (
        "python scripts/run_t32_arxiv_actual_cns.py --device cuda --base-predictors raw_x_mlp "
        "--base-logits-dir experiments/logits/t32_arxiv --train-base-logits-if-missing "
        "--graph-directions cite_ref cited_by undirected_sym --normalization-modes dst_row "
        "--self-loop-modes none --correction-alphas 0.2 0.4 0.6 0.8 0.95 "
        "--smoothing-alphas 0.2 0.4 0.6 0.8 0.95 --correction-steps 10 20 50 100 "
        "--smoothing-steps 10 20 50 100 --autoscale on off --hidden-dims 512 768 1024 --epochs 300 --run-long"
    )


def _find_base_logits(base_dir: Path, predictor: str) -> Path | None:
    for suffix in (".pt", ".npy"):
        path = base_dir / f"{predictor}_logits{suffix}"
        if path.exists():
            return path
    path = base_dir / predictor
    return path if path.exists() else None


def _blocked_row(args: argparse.Namespace, predictor: str, reason: str, *, best_plan: dict[str, Any] | None = None, notes: str = "") -> dict[str, Any]:
    best_plan = best_plan or {}
    return make_t32_row(
        dataset="ogbn-arxiv",
        method=f"arxiv_{predictor}_actual_cns",
        seed=int(_arg(args, "seed", 42)),
        status="blocked",
        failure_reason=reason,
        promotion_track="safe_main",
        promotion_status="not_promoted",
        base_predictor=predictor,
        base_logit_cache_path="",
        graph_direction=best_plan.get("graph_direction", ""),
        correction_alpha=best_plan.get("correction_alpha", ""),
        smoothing_alpha=best_plan.get("smoothing_alpha", ""),
        correction_steps=best_plan.get("correction_steps", ""),
        smoothing_steps=best_plan.get("smoothing_steps", ""),
        autoscale=best_plan.get("autoscale", ""),
        normalization_mode=best_plan.get("normalization_mode", ""),
        self_loop_mode=best_plan.get("self_loop_mode", ""),
        notes=notes,
        next_action=build_arxiv_cns_server_command(),
    )


def build_arxiv_cns_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    predictors = [str(v) for v in _arg(args, "base_predictors", ["raw_x_mlp"])]
    plan = cns_grid_plan_v2(
        graph_directions=[str(v) for v in _arg(args, "graph_directions", ["cite_ref"])],
        correction_alphas=[float(v) for v in _arg(args, "correction_alphas", [0.2])],
        smoothing_alphas=[float(v) for v in _arg(args, "smoothing_alphas", [0.4])],
        correction_steps=[int(v) for v in _arg(args, "correction_steps", [10])],
        smoothing_steps=[int(v) for v in _arg(args, "smoothing_steps", [20])],
        autoscale=[str(v) for v in _arg(args, "autoscale", ["off"])],
        normalization_modes=[str(v) for v in _arg(args, "normalization_modes", ["dst_row"])],
        self_loop_modes=[str(v) for v in _arg(args, "self_loop_modes", ["none"])],
    )
    best_plan = plan[0] if plan else {}
    base_dir = Path(_arg(args, "base_logits_dir", "experiments/logits/t32_arxiv"))
    rows: list[dict[str, Any]] = []
    for predictor in predictors:
        cache_path = _find_base_logits(base_dir, predictor)
        if cache_path is None and bool(_arg(args, "train_base_logits_if_missing", False)) and predictor == "raw_x_mlp":
            cache_path = train_raw_x_mlp_logits(args, base_dir / "raw_x_mlp_logits.pt")
        if cache_path is None:
            rows.append(_blocked_row(args, predictor, "missing_base_logits", best_plan=best_plan))
            continue
        align = validate_arxiv_split_and_feature_alignment(_arg(args, "dataset_root", "dataset/ogbn_arxiv"))
        if align.get("blocked"):
            row = _blocked_row(args, predictor, str(align.get("failure_reason", "missing_arxiv_dataset")), best_plan=best_plan)
            row["base_logit_cache_path"] = str(cache_path)
            rows.append(row)
            continue
        started = time.perf_counter()
        loaded = load_validated_base_logits(cache_path)
        try:
            _features, labels, train_idx, valid_idx, test_idx, raw_edge_index = _load_arxiv_arrays(_arg(args, "dataset_root", "dataset/ogbn_arxiv"))
        except FileNotFoundError:
            row = _blocked_row(args, predictor, "missing_arxiv_dataset", best_plan=best_plan)
            row["base_logit_cache_path"] = str(cache_path)
            rows.append(row)
            continue
        candidates: list[tuple[float, dict[str, Any]]] = []
        failed_candidates: list[str] = []
        for item in plan:
            if item["normalization_mode"] != "dst_row":
                continue
            edge_index = transform_arxiv_edge_index(
                raw_edge_index,
                graph_direction=str(item["graph_direction"]),
                self_loop_mode=str(item["self_loop_mode"]),
            )
            try:
                result = run_actual_cns_grid(
                    logits=loaded.logits,
                    labels=labels,
                    train_idx=train_idx,
                    valid_idx=valid_idx,
                    test_idx=test_idx,
                    edge_index=edge_index,
                    num_classes=ARXIV_NUM_CLASSES,
                    correction_alphas=[float(item["correction_alpha"])],
                    smoothing_alphas=[float(item["smoothing_alpha"])],
                    correction_steps=[int(item["correction_steps"])],
                    smoothing_steps=[int(item["smoothing_steps"])],
                    autoscale=str(item["autoscale"]) == "on",
                )
            except (RuntimeError, IndexError, ValueError) as exc:
                failed_candidates.append(f"{item}:{type(exc).__name__}:{exc}")
                continue
            best = result.best_row
            payload = {**item, **best}
            candidates.append((float(best.get("valid_acc", 0.0)), payload))
        if not candidates:
            rows.append(_blocked_row(args, predictor, "empty_cns_grid", best_plan=best_plan))
            continue
        candidates.sort(key=lambda pair: (pair[0], float(pair[1].get("macro_f1", 0.0))), reverse=True)
        best = candidates[0][1]
        failure = cns_failure_reason(cns_accuracy=best.get("accuracy", ""), base_predictor=predictor)
        row = make_t32_row(
            dataset="ogbn-arxiv",
            method=f"arxiv_{predictor}_actual_cns",
            seed=int(_arg(args, "seed", 42)),
            accuracy=best.get("accuracy", ""),
            macro_f1=best.get("macro_f1", ""),
            valid_acc=best.get("valid_acc", ""),
            predicted_classes=best.get("predicted_classes", ""),
            status="completed_long",
            failure_reason=failure,
            promotion_track="safe_main",
            promotion_status="not_promoted" if failure else "promoted",
            base_predictor=predictor,
            base_logit_cache_path=str(cache_path),
            base_accuracy=loaded.metadata.get("test_acc", loaded.metadata.get("accuracy", "")),
            base_valid_acc=loaded.metadata.get("valid_acc", ""),
            cns_accuracy=best.get("accuracy", ""),
            cns_valid_acc=best.get("valid_acc", ""),
            graph_direction=best.get("graph_direction", ""),
            correction_alpha=best.get("correction_alpha", best.get("cns_correction_alpha", "")),
            smoothing_alpha=best.get("smoothing_alpha", best.get("cns_smoothing_alpha", "")),
            correction_steps=best.get("correction_steps", best.get("cns_correction_steps", "")),
            smoothing_steps=best.get("smoothing_steps", best.get("cns_smoothing_steps", "")),
            autoscale=best.get("autoscale", best.get("cns_autoscale", "")),
            normalization_mode=best.get("normalization_mode", "dst_row"),
            self_loop_mode=best.get("self_loop_mode", "none"),
            split_hash=align.get("split_hash", ""),
            feature_manifest_hash=align.get("feature_manifest_hash", ""),
            precompute_time=float(time.perf_counter() - started),
            peak_cpu_ram=current_cpu_ram_bytes(),
            peak_gpu_ram=current_gpu_ram_bytes(),
            notes=(
                f"actual C&S grid candidates={len(candidates)}; failed_candidates={len(failed_candidates)}; "
                "valid-selected only; no historical LAD fallback"
            ),
            next_action=build_arxiv_cns_server_command(),
        )
        rows.append(apply_t32_promotion_guard(row))
    return rows


def write_outputs(args: argparse.Namespace) -> Path:
    rows = build_arxiv_cns_rows(args)
    csv_path = write_csv(_arg(args, "csv", "experiments/tables/t32_arxiv_actual_cns_seed42.csv"), rows, T32_REQUIRED_FIELDS)
    ensure_report(
        _arg(args, "report", "experiments/summaries/t32_arxiv_cns_forensic_notes.md"),
        [
            "# T32 Arxiv Actual C&S",
            "",
            *markdown_table(rows, ["method", "base_predictor", "base_accuracy", "cns_accuracy", "cns_valid_acc", "graph_direction", "normalization_mode", "status", "failure_reason"]),
            "",
            f"- CSV: `{csv_path}`",
            f"- Next command: `{build_arxiv_cns_server_command()}`",
        ],
    )
    return csv_path


def main() -> None:
    parser = argparse.ArgumentParser(description="T32 ogbn-arxiv actual base-logit C&S forensic grid.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dataset-root", default="dataset/ogbn_arxiv")
    parser.add_argument("--base-logits-dir", default="experiments/logits/t32_arxiv")
    parser.add_argument("--base-predictors", nargs="+", default=["raw_x_mlp"])
    parser.add_argument("--train-base-logits-if-missing", action="store_true")
    parser.add_argument("--correction-alphas", nargs="+", type=float, default=[0.2])
    parser.add_argument("--smoothing-alphas", nargs="+", type=float, default=[0.4])
    parser.add_argument("--correction-steps", nargs="+", type=int, default=[10])
    parser.add_argument("--smoothing-steps", nargs="+", type=int, default=[20])
    parser.add_argument("--autoscale", nargs="+", default=["off"])
    parser.add_argument("--graph-directions", nargs="+", default=["cite_ref"])
    parser.add_argument("--normalization-modes", nargs="+", default=["dst_row"])
    parser.add_argument("--self-loop-modes", nargs="+", default=["none"])
    parser.add_argument("--hidden-dims", nargs="+", type=int, default=[512])
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--run-long", action="store_true")
    parser.add_argument("--csv", default="experiments/tables/t32_arxiv_actual_cns_seed42.csv")
    parser.add_argument("--report", default="experiments/summaries/t32_arxiv_cns_forensic_notes.md")
    args = parser.parse_args()
    path = write_outputs(args)
    print(json.dumps({"status": "completed", "csv": str(path)}, sort_keys=True))


if __name__ == "__main__":
    main()
