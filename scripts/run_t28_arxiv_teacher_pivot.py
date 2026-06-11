from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.t24_common import ensure_report, fvalue, markdown_table, read_csv, write_csv
from shadow_hgc.sft.t28_contract import (
    ARXIV_A1,
    ARXIV_TEACHER_FIELDS,
    make_arxiv_teacher_row,
)


DEFAULT_BASE_PREDICTORS: tuple[str, ...] = ("sft_v5", "mlp_on_sft", "sagn_lite_v5", "gamlp_lite_v5")
DEFAULT_TEMPORAL_GAMMAS: tuple[float, ...] = (0.01, 0.03, 0.05, 0.10, 0.20)


def _arg(args: argparse.Namespace, name: str, default: Any) -> Any:
    return getattr(args, name, default)


def _as_list(value: Any, default: Iterable[Any]) -> list[Any]:
    if value is None or value == "":
        return list(default)
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _best_reference(path: str | Path) -> dict[str, Any] | None:
    rows = read_csv(path)
    scored = [row for row in rows if row.get("accuracy") not in {"", None}]
    if not scored:
        return None
    return max(scored, key=lambda row: fvalue(row.get("accuracy")))


def _predictor_method(base: str) -> str:
    mapping = {
        "sft_v5": "arxiv_sft_v5_reference",
        "current_sft_v5_reference": "arxiv_sft_v5_reference",
        "mlp_on_sft": "arxiv_mlp_sft_cns",
        "sagn_lite_v5": "arxiv_sagn_lite_v5_cns",
        "gamlp_lite_v5": "arxiv_gamlp_lite_v5_cns",
    }
    return mapping.get(str(base), f"arxiv_{base}_cns")


def build_arxiv_server_command(seed: int = 42) -> str:
    return (
        "python scripts/run_t28_arxiv_teacher_pivot.py --device cuda "
        "--base-predictors sft_v5 mlp_on_sft sagn_lite_v5 gamlp_lite_v5 "
        "--enable-cns --correction-alphas 0.2 0.4 0.6 0.8 0.95 "
        "--smoothing-alphas 0.2 0.4 0.6 0.8 0.95 "
        "--correction-steps 10 20 50 --smoothing-steps 10 20 50 "
        "--variants year_features temporal_decay year_conditioned_hop_attention time_cns "
        "--temporal-decay-gammas 0.01 0.03 0.05 0.10 0.20 "
        "--hidden-dims 512 768 1024 "
        f"--seed {int(seed)} --run-long"
    )


def _reference_row(args: argparse.Namespace) -> dict[str, Any]:
    seed = int(_arg(args, "seed", 42))
    ref_path = _arg(args, "reference_csv", "experiments/tables/t26_arxiv_teacher_actual_seed42.csv")
    ref = _best_reference(ref_path)
    if not ref:
        return make_arxiv_teacher_row(
            method="arxiv_sft_v5_reference",
            seed=seed,
            status="blocked_missing_reference",
            promotion_status="not_promoted",
            failure_reason="missing_arxiv_reference_csv",
            notes=f"No local reference table found at {ref_path}; run the server command.",
        )
    acc = fvalue(ref.get("accuracy"))
    return make_arxiv_teacher_row(
        method="arxiv_sft_v5_reference",
        seed=seed,
        accuracy=ref.get("accuracy", ""),
        macro_f1=ref.get("macro_f1", ""),
        predicted_classes=ref.get("predicted_classes", ref.get("predicted_class_count", "")),
        valid_acc=ref.get("valid_acc", ref.get("valid_accuracy", "")),
        status="completed_reference",
        promotion_status="not_promoted",
        failure_reason="arxiv_teacher_below_A1" if acc < ARXIV_A1 else "",
        notes=f"Imported best local teacher reference from {ref_path}; no arxiv condensation is run below A1.",
    )


def _cns_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    seed = int(_arg(args, "seed", 42))
    smoke = bool(_arg(args, "smoke", False))
    run_long = bool(_arg(args, "run_long", False))
    base_predictors = _as_list(_arg(args, "base_predictors", DEFAULT_BASE_PREDICTORS), DEFAULT_BASE_PREDICTORS)
    correction_alphas = _as_list(_arg(args, "correction_alphas", [0.4]), [0.4])
    smoothing_alphas = _as_list(_arg(args, "smoothing_alphas", [0.4]), [0.4])
    correction_steps = _as_list(_arg(args, "correction_steps", [20]), [20])
    smoothing_steps = _as_list(_arg(args, "smoothing_steps", [20]), [20])
    autoscale_values = _as_list(_arg(args, "autoscale", [True]), [True])
    status = "completed_smoke" if smoke else "server_ready_not_run"
    if run_long:
        status = "blocked_missing_base_logits"
    rows: list[dict[str, Any]] = []
    for base in base_predictors:
        if str(base) in {"sft_v5", "current_sft_v5_reference"}:
            continue
        rows.append(
            make_arxiv_teacher_row(
                method=_predictor_method(str(base)),
                seed=seed,
                status=status,
                promotion_status="not_promoted",
                failure_reason="smoke_only_no_base_logits" if smoke else "server_command_required_for_cns_logits",
                notes=(
                    "C&S contract row. Uses train labels only in correction residuals; "
                    "valid/test labels are evaluation or selection only."
                ),
                uses_cns_postprocess=True,
                cns_correction_alpha=correction_alphas[0],
                cns_smoothing_alpha=smoothing_alphas[0],
                cns_correction_steps=correction_steps[0],
                cns_smoothing_steps=smoothing_steps[0],
                cns_autoscale=autoscale_values[0],
            )
        )
    return rows


def _temporal_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    seed = int(_arg(args, "seed", 42))
    smoke = bool(_arg(args, "smoke", False))
    status = "completed_smoke" if smoke else "server_ready_not_run"
    variants = set(str(v) for v in _as_list(_arg(args, "variants", ["year_features", "temporal_decay", "time_cns"]), []))
    rows: list[dict[str, Any]] = []
    if "year_features" in variants:
        rows.append(
            make_arxiv_teacher_row(
                method="arxiv_sft_v5_year_features",
                seed=seed,
                status=status,
                promotion_status="not_promoted",
                failure_reason="smoke_only_no_training" if smoke else "server_command_required_for_temporal_teacher",
                notes="Temporal year scalar/bucket/relative-boundary features; no valid/test labels as inputs.",
                uses_temporal_features=True,
                year_feature_dim=12,
            )
        )
    if "temporal_decay" in variants:
        for gamma in _as_list(_arg(args, "temporal_decay_gammas", DEFAULT_TEMPORAL_GAMMAS), DEFAULT_TEMPORAL_GAMMAS):
            gamma_text = f"{float(gamma):.2f}".replace(".", "")
            rows.append(
                make_arxiv_teacher_row(
                    method=f"arxiv_sft_v5_temporal_decay_gamma{gamma_text}",
                    seed=seed,
                    status=status,
                    promotion_status="not_promoted",
                    failure_reason="smoke_only_no_training" if smoke else "server_command_required_for_temporal_decay",
                    notes="Temporal label-reuse decay uses train labels only.",
                    uses_temporal_features=True,
                    uses_temporal_label_decay=True,
                    temporal_decay_gamma=float(gamma),
                    year_feature_dim=12,
                )
            )
    if "year_conditioned_hop_attention" in variants:
        rows.append(
            make_arxiv_teacher_row(
                method="arxiv_sft_v5_year_conditioned_hop_attention",
                seed=seed,
                status=status,
                promotion_status="not_promoted",
                failure_reason="smoke_only_no_training" if smoke else "server_command_required_for_temporal_head",
                notes="Declared temporal head option; scalable SFT path only, no fullgraph GNN teacher.",
                uses_temporal_features=True,
                year_feature_dim=12,
            )
        )
    if "time_cns" in variants:
        rows.append(
            make_arxiv_teacher_row(
                method="arxiv_sft_time_cns",
                seed=seed,
                status=status,
                promotion_status="not_promoted",
                failure_reason="smoke_only_no_base_logits" if smoke else "server_command_required_for_time_cns",
                notes="Temporal feature provider plus C&S postprocess; C&S outputs are not condensed features.",
                uses_cns_postprocess=True,
                uses_temporal_features=True,
                cns_correction_alpha=0.4,
                cns_smoothing_alpha=0.4,
                cns_correction_steps=20,
                cns_smoothing_steps=20,
                cns_autoscale=True,
                year_feature_dim=12,
            )
        )
    return rows


def build_arxiv_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows = [_reference_row(args)]
    if bool(_arg(args, "enable_cns", True)):
        rows.extend(_cns_rows(args))
    rows.extend(_temporal_rows(args))
    return rows


def write_arxiv_outputs(args: argparse.Namespace) -> Path:
    rows = build_arxiv_rows(args)
    csv_path = write_csv(_arg(args, "csv", "experiments/tables/t28_arxiv_teacher_pivot_seed42.csv"), rows, ARXIV_TEACHER_FIELDS)
    ensure_report(
        _arg(args, "report", "experiments/summaries/t28_arxiv_teacher_pivot_summary.md"),
        [
            "# T28 Arxiv Teacher Pivot",
            "",
            "- Arxiv condensation remains blocked until a teacher reaches A1 accuracy >= 0.715.",
            "- C&S rows are teacher-evaluation rows; corrected logits are not written into condensed SFT caches.",
            "- Valid/test labels are forbidden as inputs in every row.",
            "",
            *markdown_table(
                rows,
                [
                    "method",
                    "status",
                    "accuracy",
                    "macro_f1",
                    "valid_acc",
                    "teacher_gate_A1_passed",
                    "uses_cns_postprocess",
                    "uses_temporal_features",
                    "failure_reason",
                ],
            ),
            "",
            f"- CSV: `{csv_path}`",
            f"- Next command: `{build_arxiv_server_command(seed=int(_arg(args, 'seed', 42)))}`",
        ],
    )
    return csv_path


def main() -> None:
    parser = argparse.ArgumentParser(description="T28 arxiv teacher pivot: C&S and temporal-teacher rows.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--base-predictors", nargs="+", default=list(DEFAULT_BASE_PREDICTORS))
    parser.add_argument("--enable-cns", action="store_true")
    parser.add_argument("--correction-alphas", nargs="+", type=float, default=[0.4])
    parser.add_argument("--smoothing-alphas", nargs="+", type=float, default=[0.4])
    parser.add_argument("--correction-steps", nargs="+", type=int, default=[20])
    parser.add_argument("--smoothing-steps", nargs="+", type=int, default=[20])
    parser.add_argument("--autoscale", nargs="+", default=[True])
    parser.add_argument("--variants", nargs="+", default=["year_features", "temporal_decay", "time_cns"])
    parser.add_argument("--temporal-decay-gammas", nargs="+", type=float, default=list(DEFAULT_TEMPORAL_GAMMAS))
    parser.add_argument("--hidden-dims", nargs="+", type=int, default=[512, 768, 1024])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--run-long", action="store_true")
    parser.add_argument("--reference-csv", default="experiments/tables/t26_arxiv_teacher_actual_seed42.csv")
    parser.add_argument("--csv", default="experiments/tables/t28_arxiv_teacher_pivot_seed42.csv")
    parser.add_argument("--report", default="experiments/summaries/t28_arxiv_teacher_pivot_summary.md")
    args = parser.parse_args()
    csv_path = write_arxiv_outputs(args)
    print(json.dumps({"status": "completed", "csv": str(csv_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
