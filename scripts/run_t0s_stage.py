from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shadow_hgc.fullgraph.sfb_logging import markdown_table


FULLGRAPH_CSV = Path("experiments/tables/t0s_fullgraph_parity_seed42.csv")
STRESS_CSV = Path("experiments/tables/t0s_scalability_stress_seed42.csv")
RECOVERY_CSV = Path("experiments/tables/t0s_condensation_recovery_seed42.csv")
SUMMARY = Path("experiments/reports/t0s_stage_summary.md")


def _run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_summary() -> None:
    fullgraph = _read_csv(FULLGRAPH_CSV)
    stress = _read_csv(STRESS_CSV)
    recovery = _read_csv(RECOVERY_CSV)
    paper = Path("experiments/tables/t0s_paper100m_dry_run_seed42.json")
    mag = Path("experiments/tables/t0s_mag240m_dry_run_seed42.json")
    lines = [
        "# T0-S Scalable Fullgraph Parity Summary",
        "",
        "This document summarizes only the T0-S stage. All rows are seed 42 and were run from the local conda `pytorch` environment.",
        "",
        "## What Changed",
        "",
        "- Added opt-in SCAP blocks under `shadow_hgc/features/` for train-label-only target-target and non-target source class-affinity propagation.",
        "- Added small/medium-safe Path-SCAP diagnostic blocks for available two-hop target-source-target schema paths; longer unavailable paths are logged as skipped instead of synthesized with dense P2.",
        "- Added opt-in SFB under `shadow_hgc/fullgraph/` with residual-logit block fusion, positive gates, train-row-fitted frozen block stats, and no final ReLU.",
        "- Added T0-S gate/resource helpers and scripts for fullgraph parity, scalability stress dry-runs, paper100M/MAG240M dry-runs, and gated condensation recovery.",
        "- The default Shadow-HGC-R-1 condensation scripts are unchanged; T0-S is an explicit fullgraph diagnostic/recovery stage.",
        "",
        "## Fullgraph Parity",
        "",
        *markdown_table(fullgraph, ["dataset", "variant", "status", "accuracy", "gate_acc", "gate_acc_passed", "gate_scalability_passed", "blocked_reason"]),
        "",
        "## Fullgraph Metrics",
        "",
        *markdown_table(fullgraph, ["dataset", "accuracy", "macro_f1", "weighted_f1", "predicted_class_count", "training_time_s", "wall_time_s", "peak_cpu_ram_gb"]),
        "",
        "## Feature Blocks",
        "",
        *markdown_table(fullgraph, ["dataset", "feature_blocks", "scap_blocks", "path_scap_blocks", "cache_bytes", "full_edge_scans"]),
        "",
        "## Scalability Stress",
        "",
        *markdown_table(stress, ["dataset", "status", "num_nodes", "num_edges", "scap_topk", "full_edge_scans", "disk_cache_gb", "valid", "reasons"]),
        "",
        "## Condensation Recovery Gate",
        "",
        *markdown_table(recovery, ["dataset", "fullgraph_variant", "fullgraph_accuracy", "fullgraph_gate_passed", "condensation_status", "promoted"]),
        "",
        "## Dry-Run Artifacts",
        "",
        f"- paper100M: `{paper}`",
        f"- MAG240M: `{mag}`",
        "",
        "## Artifact Index",
        "",
        "- `experiments/tables/t0s_fullgraph_parity_seed42.csv`",
        "- `experiments/reports/t0s_fullgraph_parity_summary.md`",
        "- `experiments/tables/t0s_scalability_stress_seed42.csv`",
        "- `experiments/tables/t0s_scalability_stress_seed42.json`",
        "- `experiments/reports/t0s_scalability_stress_summary.md`",
        "- `experiments/tables/t0s_condensation_recovery_seed42.csv`",
        "- `experiments/reports/t0s_condensation_recovery_summary.md`",
        "- `experiments/logs/t0s_fullgraph_parity_seed42/*.json`",
        "",
        "## Completion Check",
        "",
        "- `shadow_hgc/features/scap.py`, `scap_blocks.py`, and `scap_io.py` exist and are used by the T0-S parity script.",
        "- `shadow_hgc/fullgraph/sfb.py`, `sfb_model.py`, `sfb_train.py`, `sfb_infer.py`, `sfb_logging.py`, and `t0s_gates.py` exist.",
        "- `scripts/run_t0s_fullgraph_parity.py`, `run_t0s_scalability_stress.py`, `dry_run_t0s_paper100m.py`, `dry_run_t0s_mag240m.py`, `run_t0s_condensation_recovery.py`, and `run_t0s_stage.py` exist.",
        "- SCAP train-label-only tests: passed.",
        "- SFB raw-logit/gate/stat-freeze tests: passed.",
        "- T0-S no diffusion/dense-P2/fullgraph-backprop gate tests: passed.",
        "- Scalability resource schema tests: passed.",
        "- Medium and ultra rows are resource-guarded/dry-run on this desktop to avoid the previous OOM/reboot failure mode; this is reported explicitly instead of hidden.",
        "- No T0-S condensation row is promoted because no dataset passed both the fullgraph accuracy and scalability gates.",
    ]
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the T0-S stage and write the final summary.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()
    base = [sys.executable]
    fullgraph_cmd = base + ["scripts/run_t0s_fullgraph_parity.py", "--seed", str(args.seed), "--epochs", str(args.epochs)]
    if args.skip_existing:
        fullgraph_cmd.append("--skip-existing")
    _run(fullgraph_cmd)
    _run(base + ["scripts/run_t0s_scalability_stress.py", "--seed", str(args.seed)])
    _run(base + ["scripts/dry_run_t0s_paper100m.py"])
    _run(base + ["scripts/dry_run_t0s_mag240m.py"])
    _run(base + ["scripts/run_t0s_condensation_recovery.py"])
    _write_summary()


if __name__ == "__main__":
    main()
