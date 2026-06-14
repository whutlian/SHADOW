from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.t24_common import write_csv
from shadow_hgc.ultra.papers100m_runner import Papers100MCacheContext
from shadow_hgc.ultra.papers100m_t37_contract import T37_REQUIRED_FIELDS, make_t37_row


def main() -> None:
    parser = argparse.ArgumentParser(description="Record T37 lightweight teacher decision.")
    parser.add_argument("--cache-root", default="caches/papers100m/stt_v1")
    parser.add_argument("--tables-dir", default="experiments/tables")
    args = parser.parse_args()
    ctx = Papers100MCacheContext(Path(args.cache_root), selection_policy="stt_ratio_v2", seed=42)
    ids = ctx.cache_ids()
    row = make_t37_row(
        method="papers100m_gamlp_lite_teacher_v2",
        seed=42,
        backend="teacher",
        comparison_type="teacher_light_upgrade",
        requested_full_node_ratio=0.0,
        full_node_denominator=int(ctx.manifest["num_nodes"]),
        target_universe_size=int(ctx.manifest["target_universe_size"]),
        cache_build_id=ids["cache_build_id"],
        edge_cache_id=ids["edge_slice_cache_id"],
        sft_cache_id=ids["sft_cache_id"],
        teacher_cache_id=ids["teacher_cache_id"],
        selection_bank_id="",
        selection_bank_reused=True,
        accuracy=ctx.teacher.get("accuracy", ctx.teacher.get("test_acc", "")),
        valid_acc=ctx.teacher.get("valid_acc", ""),
        macro_f1=ctx.teacher.get("macro_f1", ""),
        predicted_classes=ctx.teacher.get("predicted_classes", ""),
        uses_teacher_probs_as_soft_targets=False,
        promotion_status="diagnostic",
        notes="T37 keeps the promoted T36 topk8_tail teacher; no teacher cache rebuild was run because SCR selection is the target iteration.",
    )
    write_csv(Path(args.tables_dir) / "t37_papers100m_teacher_light_upgrade.csv", [row], T37_REQUIRED_FIELDS)
    print("status=completed")


if __name__ == "__main__":
    main()
