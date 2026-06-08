from __future__ import annotations

import argparse
import csv
import json
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shadow_hgc.data.small import load_processed_small_dataset
from shadow_hgc.eval.budgeting import ratio_slug
from shadow_hgc.eval.logging import write_json_summary
from shadow_hgc.eval.status import exception_status
from shadow_hgc.pipeline.core import run_shadow_hgc_experiment


def _variant_config(name: str) -> dict:
    base = {
        "method_name": "Shadow-HGC-R-1" if name == "base" else "Shadow-HGC-R+",
        "feature_mode": "base",
        "metapath_signature": False,
        "metapath_model_input": False,
        "shadow_policy": "fixed",
        "adaptive_b": False,
        "relation_gate": False,
    }
    if name in {"metapath", "full_rplus"}:
        base.update({"feature_mode": "metapath", "metapath_signature": True, "metapath_model_input": True})
    if name in {"rank_adaptive", "full_rplus"}:
        base.update({"shadow_policy": "rank_adaptive"})
    if name in {"adaptive_b", "full_rplus"}:
        base.update({"adaptive_b": True, "b_max": 4})
    if name in {"relation_gate", "full_rplus"}:
        base.update({"relation_gate": True, "relation_gate_init": 1.0})
    return base


def _row(path: Path, variant: str, loss: str, summary: dict) -> dict:
    diagnostics = summary.get("diagnostics", {})
    rank = diagnostics.get("rank", {})
    recon = {
        relation: values.get("ShadowReconErr")
        for relation, values in diagnostics.items()
        if isinstance(values, dict) and "ShadowReconErr" in values
    }
    return {
        "dataset": summary.get("dataset", "imdb"),
        "variant": variant,
        "loss_type": loss,
        "seed": summary.get("seed", ""),
        "ratio": summary.get("ratio", ""),
        "accuracy": summary.get("accuracy", ""),
        "macro_f1": summary.get("macro_f1", ""),
        "predicted_class_count": summary.get("predicted_class_count", summary.get("num_predicted_classes", "")),
        "prediction_entropy": summary.get("prediction_entropy", ""),
        "shadow_recon_err_by_relation": json.dumps(recon, sort_keys=True),
        "effective_rank_by_relation": json.dumps(
            {
                relation: max(float(values.get("stable_rank", 0.0)), float(values.get("entropy_effective_rank", 0.0)))
                for relation, values in rank.items()
            },
            sort_keys=True,
        ),
        "M_r_by_relation": json.dumps(summary.get("M_r", {}), sort_keys=True),
        "b_by_relation": json.dumps(summary.get("b_by_relation", {}), sort_keys=True),
        "relation_gate_values": json.dumps(summary.get("relation_gate_values", {}), sort_keys=True),
        "condensed_nodes_total": summary.get("condensed_nodes_total", ""),
        "condensed_edges_total": summary.get("condensed_edges_total", ""),
        "status": summary.get("status", "completed"),
        "source_log": str(path),
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _fmt(value) -> str:
    if value in ("", None):
        return ""
    return f"{float(value):.4f}"


def _write_report(path: Path, rows: list[dict], csv_path: Path) -> None:
    completed = [row for row in rows if row.get("status") == "completed" and row.get("accuracy") not in ("", None)]
    best = max(completed, key=lambda row: float(row["accuracy"])) if completed else None
    lines = [
        "# IMDB R+ Rescue Summary",
        "",
        "## Scope",
        "",
        "- Dataset: IMDB.",
        "- Seed: 42 only.",
        "- Ratios: 0.5%, 2.5%, 5.0%.",
        "- Variants: base and full R+ at all ratios; component variants at 2.5%.",
        "",
        "## Results",
        "",
        "| Variant | Loss | Ratio | Acc | Macro-F1 | Pred classes | Entropy |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in completed:
        lines.append(
            f"| {row['variant']} | {row['loss_type']} | {_fmt(float(row['ratio']) * 100.0)}% | "
            f"{_fmt(row['accuracy'])} | {_fmt(row['macro_f1'])} | {row['predicted_class_count']} | {_fmt(row['prediction_entropy'])} |"
        )
    lines.extend(["", "## Best Point", ""])
    if best:
        lines.append(
            f"- Best accuracy: `{_fmt(best['accuracy'])}` from `{best['variant']}` with `{best['loss_type']}` at `{_fmt(float(best['ratio']) * 100.0)}%`."
        )
    lines.extend(
        [
            "- Rescue is successful only if accuracy and macro-F1 improve without predicted-class collapse.",
            "",
            "## Files",
            "",
            f"- CSV: `{csv_path}`",
            f"- Report: `{path}`",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run IMDB Shadow-HGC-R+ rescue grid with seed 42.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--output", default="experiments/tables/imdb_rescue_rplus_seed42.csv")
    parser.add_argument("--report-output", default="experiments/reports/imdb_rescue_rplus_summary.md")
    parser.add_argument("--log-dir", default="experiments/logs/imdb_rescue_rplus_seed42")
    args = parser.parse_args()

    graph = load_processed_small_dataset("imdb")
    ratios = [0.005, 0.025, 0.05]
    losses = ["clipped", "sqrt_weighted", "class_balanced"]
    full_variants = ["base", "full_rplus"]
    component_variants = ["metapath", "rank_adaptive", "adaptive_b", "relation_gate"]
    specs = [(variant, ratio, loss) for variant in full_variants for ratio in ratios for loss in losses]
    specs.extend((variant, 0.025, loss) for variant in component_variants for loss in losses)
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for variant, ratio, loss in specs:
        path = log_dir / f"imdb_{variant}_{loss}_{ratio_slug(ratio)}_seed{args.seed}.json"
        try:
            if args.skip_existing and path.exists():
                summary = json.loads(path.read_text(encoding="utf-8"))
            else:
                summary = run_shadow_hgc_experiment(
                    graph,
                    output_path=path,
                    seed=args.seed,
                    epochs=args.epochs,
                    budget_mode="ratio",
                    ratio=ratio,
                    ratio_base="train_target",
                    feature_dim=64,
                    projection_type="raw",
                    loss_type=loss,
                    model_type="relation_linear",
                    k_s=2,
                    min_proto_per_class=4,
                    budget_alpha=0.5,
                    shadow_min_per_relation=8,
                    shadow_max_multiplier=2.0,
                    **_variant_config(variant),
                )
            rows.append(_row(path, variant, loss, summary))
        except Exception as exc:
            payload = {
                "dataset": "imdb",
                "variant": variant,
                "loss_type": loss,
                "seed": args.seed,
                "ratio": ratio,
                "status": exception_status(exc),
                "reason": str(exc),
                "traceback": traceback.format_exc(),
            }
            write_json_summary(path, payload)
            rows.append(_row(path, variant, loss, payload))
    output = Path(args.output)
    report = Path(args.report_output)
    _write_csv(output, rows)
    _write_report(report, rows, output)
    print(f"wrote {output}")
    print(f"wrote {report}")


if __name__ == "__main__":
    main()
