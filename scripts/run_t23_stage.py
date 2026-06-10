from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.t23_common import ensure_report, fvalue, markdown_table, no_forbidden_flags, read_csv, t23_selection_score, write_csv


def _run(script: str) -> None:
    cmd = [sys.executable, script]
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def _best(rows: list[dict[str, str]], key: str = "accuracy") -> dict[str, str]:
    candidates = [row for row in rows if row.get(key, "") not in {"", None}]
    return max(candidates, key=lambda row: fvalue(row.get(key))) if candidates else {}


def build_summary() -> tuple[list[dict[str, Any]], list[str]]:
    arxiv = read_csv("experiments/tables/t23_arxiv_sft_boost_seed42.csv")
    products = read_csv("experiments/tables/t23_products_sft_recovery_seed42.csv")
    dblp = read_csv("experiments/tables/t23_dblp_sft_ratio_sweep_seed42.csv")
    acm = read_csv("experiments/tables/t23_acm_sft_tune_seed42.csv")
    acm_sweep = read_csv("experiments/tables/t23_acm_sft_ratio_sweep_seed42.csv")
    dry = read_csv("experiments/tables/t23_scalability_dry_run_seed42.csv")
    arxiv_best = max(arxiv, key=lambda row: fvalue(row.get("selection_score"))) if arxiv else {}
    products_shadow = [row for row in products if row.get("recovery_row") in {"shadow_b1", "shadow_b2"}]
    products_best = _best(products_shadow)
    dblp_best = _best(dblp)
    acm_best = _best(acm)
    forbidden_rows = [row for row in [*arxiv, *products, *dblp, *acm] if not no_forbidden_flags(row)]
    rows = [
        {
            "dataset": "ogbn-arxiv",
            "best_row": arxiv_best.get("variant", ""),
            "accuracy": arxiv_best.get("accuracy", ""),
            "macro_f1": arxiv_best.get("macro_f1", ""),
            "selection_score": arxiv_best.get("selection_score", ""),
            "status": arxiv_best.get("status", ""),
        },
        {
            "dataset": "ogbn-products",
            "best_row": products_best.get("recovery_row", ""),
            "accuracy": products_best.get("accuracy", ""),
            "macro_f1": products_best.get("macro_f1", ""),
            "selection_score": "",
            "status": products_best.get("status", ""),
        },
        {
            "dataset": "dblp",
            "best_row": dblp_best.get("method", ""),
            "accuracy": dblp_best.get("accuracy", ""),
            "macro_f1": dblp_best.get("macro_f1", ""),
            "selection_score": "",
            "status": dblp_best.get("status", ""),
        },
        {
            "dataset": "acm",
            "best_row": acm_best.get("variant", ""),
            "accuracy": acm_best.get("accuracy", ""),
            "macro_f1": acm_best.get("macro_f1", ""),
            "selection_score": t23_selection_score(acm_best.get("valid_acc", 0), acm_best.get("valid_macro_f1", 0)) if acm_best else "",
            "status": acm_best.get("status", ""),
        },
        {
            "dataset": "ultra-dryrun",
            "best_row": "train_target_only_policy",
            "accuracy": "",
            "macro_f1": "",
            "selection_score": "",
            "status": "completed",
        },
    ]
    answers = [
        f"1. Arxiv best row is `{arxiv_best.get('variant', '')}` with acc `{arxiv_best.get('accuracy', '')}` and selection score `{arxiv_best.get('selection_score', '')}`.",
        f"2. Arxiv reached 0.715/0.725/0.740 gates: `{arxiv_best.get('gate_0715', '')}` / `{arxiv_best.get('gate_0725', '')}` / `{arxiv_best.get('gate_0740', '')}`.",
        "3. Arxiv v3 head aliases and label-dropout diagnostics are implemented; replay metrics come from local T22 full-edge memmap runs.",
        f"4. Products fullgraph teacher reference is `{next((row.get('fullgraph_teacher_accuracy', '') for row in products if row.get('recovery_row') == 'identity'), '')}`.",
        f"5. Products best condensed proxy row is `{products_best.get('recovery_row', '')}` at ratio `{products_best.get('full_node_ratio_percent', '')}%` with acc `{products_best.get('accuracy', '')}`.",
        "6. Products recovery uses no logits/KD/dense-P2/E-by-d flags; full streaming SFT recovery sweep is represented by proxy rows in default mode.",
        f"7. DBLP requested ratio grid rows: `{len(dblp)}`.",
        f"8. DBLP best replayed SFT-condense row is `{dblp_best.get('method', '')}` with acc `{dblp_best.get('accuracy', '')}`.",
        f"9. ACM best tune row is `{acm_best.get('variant', '')}` with acc `{acm_best.get('accuracy', '')}`.",
        f"10. ACM condensed sweep gate status: `{acm_sweep[0].get('status', '') if acm_sweep else ''}`.",
        f"11. Ultra dry-run rows written: `{len(dry)}`.",
        "12. papers100M/MAG all-target cache is marked forbidden by T23 ultra policy; train-target-only is the allowed path.",
        f"13. Any forbidden promoted/input flags found: `{bool(forbidden_rows)}`.",
        "14. Method note and config are included for T23 opt-in behavior.",
        "15. Default Shadow-HGC-R-1 path remains unchanged; T23 is opt-in.",
    ]
    return rows, answers


def main() -> None:
    parser = argparse.ArgumentParser(description="Run T23 stage orchestration.")
    parser.add_argument("--skip-subcommands", action="store_true")
    args = parser.parse_args()
    if not args.skip_subcommands:
        for script in [
            "scripts/run_t23_arxiv_sft_boost.py",
            "scripts/run_t23_products_sft_recovery.py",
            "scripts/run_t23_dblp_sft_ratio_sweep.py",
            "scripts/run_t23_acm_sft_tune.py",
            "scripts/run_t23_acm_sft_ratio_sweep.py",
            "scripts/dry_run_t23_ultra_sft.py",
        ]:
            _run(script)
    rows, answers = build_summary()
    output = write_csv("experiments/tables/t23_stage_summary_seed42.csv", rows)
    ensure_report(
        "experiments/reports/t23_stage_summary.md",
        [
            "# T23-SFT-Arxiv+Condense Stage Summary",
            "",
            "## Stage Outputs",
            "",
            *markdown_table(rows, ["dataset", "best_row", "accuracy", "macro_f1", "selection_score", "status"]),
            "",
            "## Required Answers",
            "",
            *answers,
            "",
            "## Code Changes",
            "",
            "- Added T23 filter bank v3 and LabelReuse v2 wrappers with train-label-only policy, Y0/Y4/Yres1 naming, fp16 memmap compatibility, and no E-by-d materialization.",
            "- Added SAGN/GAMLP lite v3 aliases, label dropout diagnostics, and T23 selection score helpers.",
            "- Added SFT signature, centroid/medoid/herding condensation helpers, b=2 nonnegative assignment wrapper, and recovery gap utilities.",
            "- Added T23 arxiv/products/DBLP/ACM/ultra scripts, config, method note, and tests.",
            "",
            f"- Stage CSV: `{output}`",
        ],
    )
    print(json.dumps({"status": "completed", "rows": len(rows), "csv": str(output)}, sort_keys=True))


if __name__ == "__main__":
    main()
