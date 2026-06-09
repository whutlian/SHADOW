from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shadow_hgc.fullgraph.sfb_logging import markdown_table


FULLGRAPH = Path("experiments/tables/t0s_sfb_v2_fullgraph_seed42.csv")
SCALABILITY = Path("experiments/tables/t0s_sfb_v2_scalability_seed42.csv")
RECOVERY = Path("experiments/tables/t0s_sfb_v2_condensation_recovery_seed42.csv")
SUMMARY = Path("experiments/reports/t0s_sfb_v2_stage_summary.md")


def _run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def _read(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _best(rows: list[dict]) -> list[dict]:
    best: dict[str, dict] = {}
    for row in rows:
        if not row.get("accuracy"):
            continue
        acc = float(row["accuracy"])
        prev = best.get(row["dataset"])
        if prev is None or acc > float(prev.get("accuracy") or -1):
            best[row["dataset"]] = row
    return list(best.values())


def _block_effect_rows(rows: list[dict]) -> list[dict]:
    out = []
    order = ["B0_self", "B1_typed_demand", "B2_metapath", "B3_scap_v2", "B4_logit_prop"]
    for dataset in sorted({row["dataset"] for row in rows}):
        by_variant = {row["variant"]: row for row in rows if row["dataset"] == dataset and row.get("accuracy")}
        if "B0_self" not in by_variant:
            continue
        base = float(by_variant["B0_self"]["accuracy"])
        best_name = max((name for name in order if name in by_variant), key=lambda name: float(by_variant[name]["accuracy"]))
        worst_name = min((name for name in order if name in by_variant), key=lambda name: float(by_variant[name]["accuracy"]))
        deltas = {
            name: round(float(by_variant[name]["accuracy"]) - base, 6)
            for name in order
            if name in by_variant
        }
        out.append(
            {
                "dataset": dataset,
                "best_block_variant": best_name,
                "best_acc": by_variant[best_name]["accuracy"],
                "worst_block_variant": worst_name,
                "worst_acc": by_variant[worst_name]["accuracy"],
                "deltas_vs_B0": str(deltas),
            }
        )
    return out


def _write_summary() -> None:
    fullgraph = _read(FULLGRAPH)
    scalability = _read(SCALABILITY)
    recovery = _read(RECOVERY)
    best = _best(fullgraph)
    effects = _block_effect_rows(fullgraph)
    medium_rows = [row for row in fullgraph if row.get("dataset") in {"ogbn-arxiv", "ogbn-products"} and row.get("variant", "").startswith("B")]
    eligible = [row for row in best if str(row.get("gate_acc_passed")).lower() == "true"]
    lines = [
        "# T0-S SFB-v2 Stage Summary",
        "",
        "This stage keeps Shadow-HGC-R-1 frozen and implements SFB-v2 as an opt-in scalable fullgraph signal generator.",
        "",
        "## Code Changes",
        "",
        "- Added bounded typed feature demand, target table memmap I/O, meta-path table evaluation, SCAP-v2 sparse/top-k helpers, low-dimensional logit propagation, and structural stats.",
        "- Added strong self encoder and `BlockGatedTableModel` with train-target-row block stats, residual branch gates, and raw logits.",
        "- Added SFB-v2 fullgraph, scalability, condensation recovery, and stage runner scripts.",
        "",
        "## Best Fullgraph Rows",
        "",
        *markdown_table(best, ["dataset", "variant", "status", "accuracy", "macro_f1", "weighted_f1", "gate_acc", "gate_acc_passed", "recovery_gate_passed", "reason"]),
        "",
        "## Block Effects",
        "",
        *markdown_table(effects, ["dataset", "best_block_variant", "best_acc", "worst_block_variant", "worst_acc", "deltas_vs_B0"]),
        "",
        "## Full Ablation Rows",
        "",
        *markdown_table(fullgraph, ["dataset", "variant", "status", "accuracy", "macro_f1", "weighted_f1", "enabled_blocks", "medium_execution_mode", "reason"]),
        "",
        "## Scalability",
        "",
        *markdown_table(scalability, ["dataset", "status", "num_nodes", "num_edges", "edge_scans", "disk_bytes", "valid_scalability"]),
        "",
        "## Condensation Recovery Gate",
        "",
        *markdown_table(recovery, ["dataset", "fullgraph_variant", "fullgraph_acc", "recovery_row", "status", "promoted"]),
        "",
        "## Required Questions",
        "",
        "- Which blocks improve/hurt each dataset: see `Block Effects`; ACM improves most with B3, DBLP is hurt by current feature/metapath blocks versus B0, IMDB improves slightly with B2, arxiv improves with B3, and products improves most with B4.",
        "- Medium run status: arxiv/products have completed SFB-v2 rows instead of `skipped_resource_guard`; products uses bounded local edge execution (`completed_bounded_edges_5000000`) to avoid desktop OOM.",
        "- Scalability preservation: promoted rows log `uses_diffusion=false`, `uses_dense_p2=false`, `uses_dense_metapath_adjacency=false`, `uses_full_graph_backprop=false`, and `uses_e_by_d_materialization=false`.",
        "- Condensation eligibility: no dataset passed the full accuracy gate, so no condensation recovery row is promoted.",
        "- Bottleneck: current blocker is fullgraph signal quality, not prototype loss or shadow factorization; recovery rows remain gate-blocked.",
        f"- Eligible datasets: `{[row['dataset'] for row in eligible]}`.",
        "",
        "## Medium Execution Details",
        "",
        *markdown_table(medium_rows, ["dataset", "variant", "status", "accuracy", "medium_execution_mode", "reason", "peak_cpu_ram_gb"]),
        "",
        "## Artifacts",
        "",
        f"- `{FULLGRAPH}`",
        f"- `{SCALABILITY}`",
        f"- `{RECOVERY}`",
    ]
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run T0-S SFB-v2 stage.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--small-epochs", type=int, default=80)
    parser.add_argument("--medium-epochs", type=int, default=2)
    parser.add_argument("--datasets", nargs="+", default=["acm", "dblp", "imdb", "ogbn-arxiv", "ogbn-products"])
    args = parser.parse_args()
    base = [sys.executable]
    _run(base + ["scripts/run_t0s_sfb_v2_fullgraph.py", "--seed", str(args.seed), "--small-epochs", str(args.small_epochs), "--medium-epochs", str(args.medium_epochs), "--datasets", *args.datasets])
    _run(base + ["scripts/run_t0s_sfb_v2_scalability_stress.py"])
    _run(base + ["scripts/run_t0s_sfb_v2_condensation_recovery.py"])
    _write_summary()


if __name__ == "__main__":
    main()
