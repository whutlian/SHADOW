from __future__ import annotations

from pathlib import Path
import csv

from shadow_hgc.data.loaders import HeteroGraphData
from shadow_hgc.pipeline.core import run_shadow_hgc_experiment


def _mean_diag(summary: dict, key: str) -> float:
    values = [diag[key] for diag in summary["diagnostics"].values() if key in diag]
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _row(dataset: str, ablation: str, setting: str, seed: int, summary: dict, status: str = "completed") -> dict:
    return {
        "dataset": dataset,
        "ablation": ablation,
        "setting": setting,
        "seed": seed,
        "accuracy": "" if summary.get("accuracy") is None else f"{summary['accuracy']:.6f}",
        "skeleton_coverage_mean": f"{_mean_diag(summary, 'SkeletonMassCoverage'):.6f}",
        "residual_energy_mean": f"{_mean_diag(summary, 'ResidualEnergy'):.6f}",
        "shadow_recon_err_mean": f"{_mean_diag(summary, 'ShadowReconErr'):.6f}",
        "condensation_time": f"{summary.get('condensation_time', 0.0):.6f}",
        "training_time": f"{summary.get('training_time', 0.0):.6f}",
        "status": status,
    }


def run_ablation_suite(
    graph: HeteroGraphData,
    *,
    log_dir: str | Path,
    seed: int,
    epochs: int,
    M_tau: int,
    M_r: int,
    k_s: int,
    feature_dim: int,
    k_s_values: list[int] | None = None,
) -> list[dict]:
    log_dir = Path(log_dir)
    rows: list[dict] = []
    base_kwargs = {
        "seed": seed,
        "epochs": epochs,
        "M_tau": M_tau,
        "M_r": M_r,
        "k_s": k_s,
        "feature_dim": feature_dim,
    }

    variants = [
        ("mean_only_demand", "include_degree_features=false", {"include_degree_features": False}),
        ("residual_shadow_off", "residual_shadow=false", {"residual_shadow": False}),
        ("real_source_centroid", "shadow_mode=real_source_centroid", {"shadow_mode": "real_source_centroid"}),
        ("prototype_loss", "loss_type=unweighted", {"loss_type": "unweighted"}),
        ("prototype_loss", "loss_type=clipped", {"loss_type": "clipped"}),
        ("prototype_loss", "loss_type=class_balanced", {"loss_type": "class_balanced"}),
        ("relation_norm_calibration", "calibration_enabled=false", {"calibration_enabled": False}),
    ]
    for ablation, setting, overrides in variants:
        output_path = log_dir / f"{graph.dataset_name}_{ablation}_{setting.replace('=', '-').replace(',', '_')}_seed{seed}.json"
        summary = run_shadow_hgc_experiment(graph, output_path=output_path, **base_kwargs, **overrides)
        rows.append(_row(graph.dataset_name, ablation, setting, seed, summary))

    if any(relation.is_target_target(graph.target_type) for relation in graph.relations):
        for value in (k_s_values or [0, 1, 2, 4, 8]):
            output_path = log_dir / f"{graph.dataset_name}_target_target_skeleton_ks{value}_seed{seed}.json"
            summary = run_shadow_hgc_experiment(graph, output_path=output_path, **{**base_kwargs, "k_s": value})
            rows.append(_row(graph.dataset_name, "target_target_skeleton", f"k_s={value}", seed, summary))
    else:
        rows.append(
            {
                "dataset": graph.dataset_name,
                "ablation": "target_target_skeleton",
                "setting": "no_target_target_relation",
                "seed": seed,
                "accuracy": "",
                "skeleton_coverage_mean": "0.000000",
                "residual_energy_mean": "0.000000",
                "shadow_recon_err_mean": "0.000000",
                "condensation_time": "0.000000",
                "training_time": "0.000000",
                "status": "not_applicable",
            }
        )
    return rows


def write_skeleton_coverage_figure(rows: list[dict], *, csv_path: str | Path, svg_path: str | Path) -> None:
    points = []
    for row in rows:
        if row.get("ablation") != "target_target_skeleton" or row.get("status") != "completed":
            continue
        setting = row.get("setting", "")
        if not setting.startswith("k_s="):
            continue
        points.append(
            {
                "dataset": row["dataset"],
                "k_s": int(setting.split("=", 1)[1]),
                "skeleton_coverage": float(row["skeleton_coverage_mean"]),
                "accuracy": float(row["accuracy"]) if row["accuracy"] != "" else 0.0,
            }
        )
    csv_path = Path(csv_path)
    svg_path = Path(svg_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    svg_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["dataset", "k_s", "skeleton_coverage", "accuracy"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(points)

    width, height = 640, 360
    margin = 48
    if points:
        max_k = max(point["k_s"] for point in points) or 1
        max_acc = max(point["accuracy"] for point in points) or 1.0
        circles = []
        labels = []
        for point in points:
            x = margin + (width - 2 * margin) * point["k_s"] / max_k
            y = height - margin - (height - 2 * margin) * point["accuracy"] / max_acc
            radius = 4 + 10 * point["skeleton_coverage"]
            circles.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius:.1f}" fill="#2563eb" opacity="0.72" />')
            labels.append(
                f'<text x="{x + 7:.1f}" y="{y - 7:.1f}" font-size="11" fill="#111827">'
                f'{point["dataset"]} k={point["k_s"]}</text>'
            )
        body = "\n  ".join(circles + labels)
    else:
        body = '<text x="80" y="180" font-size="14" fill="#6b7280">No target-target skeleton rows available</text>'
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#ffffff" />
  <text x="{margin}" y="30" font-size="18" font-weight="700" fill="#111827">Skeleton Coverage vs Accuracy</text>
  <line x1="{margin}" y1="{height - margin}" x2="{width - margin}" y2="{height - margin}" stroke="#374151" />
  <line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height - margin}" stroke="#374151" />
  <text x="{width / 2 - 30:.1f}" y="{height - 12}" font-size="12" fill="#374151">k_s</text>
  <text x="8" y="{height / 2:.1f}" font-size="12" fill="#374151">accuracy</text>
  {body}
</svg>
"""
    svg_path.write_text(svg, encoding="utf-8")
