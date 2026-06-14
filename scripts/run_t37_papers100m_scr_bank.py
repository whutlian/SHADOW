from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.t24_common import read_csv, write_csv
from scripts.t37_papers100m_common import ensure_t37_bank, t37_policy_for_method
from shadow_hgc.ultra.papers100m_runner import Papers100MCacheContext
from shadow_hgc.ultra.papers100m_scr_bank import SCR_POLICY_FULL_TEACHER_WEIGHT, audit_scr_bank
from shadow_hgc.ultra.papers100m_t37_contract import T37_REQUIRED_FIELDS, make_t37_row


def main() -> None:
    parser = argparse.ArgumentParser(description="Build T37 SCR selection banks and audit nested prefixes.")
    parser.add_argument("--cache-root", default="caches/papers100m/stt_v1")
    parser.add_argument("--policies", nargs="+", default=["scr_class_random", "scr_full_stochastic_coverage"])
    parser.add_argument("--bank-max-ratio", type=float, default=0.0005)
    parser.add_argument("--ratios", nargs="+", type=float, default=[0.00005, 0.0001, 0.0002, 0.0005])
    parser.add_argument("--feature-lsh-dim", type=int, default=64)
    parser.add_argument("--feature-lsh-bits", type=int, default=16)
    parser.add_argument("--degree-bucket-mode", default="log2")
    parser.add_argument("--teacher-weight-etas", nargs="+", type=float, default=[0.10])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--run-long", action="store_true")
    parser.add_argument("--tables-dir", default="experiments/tables")
    args = parser.parse_args()

    rows = read_csv(Path(args.tables_dir) / "t37_papers100m_scr_bank_audit.csv")
    cache_root = Path(args.cache_root)
    for seed in [int(value) for value in args.seeds]:
        for method in [str(value) for value in args.policies]:
            etas = [float(v) for v in args.teacher_weight_etas] if method == SCR_POLICY_FULL_TEACHER_WEIGHT else [0.0]
            for eta in etas:
                if not args.run_long:
                    continue
                policy, bank = ensure_t37_bank(
                    cache_root,
                    method=method,
                    seed=seed,
                    max_ratio=float(args.bank_max_ratio),
                    feature_lsh_dim=int(args.feature_lsh_dim),
                    feature_lsh_bits=int(args.feature_lsh_bits),
                    teacher_weight_eta=float(eta),
                    force=bool(args.force),
                )
                ctx = Papers100MCacheContext(cache_root, selection_policy=policy, seed=seed)
                for audit in audit_scr_bank(cache_root, policy=policy, seed=seed, ratios=[float(v) for v in args.ratios]):
                    row = make_t37_row(
                        method=method,
                        seed=seed,
                        backend="bank",
                        comparison_type="bank_build",
                        requested_full_node_ratio=float(audit["ratio"]),
                        full_node_denominator=int(ctx.manifest["num_nodes"]),
                        condensed_nodes=int(audit["selected_count"]),
                        target_universe_size=int(ctx.manifest["target_universe_size"]),
                        cache_build_id=ctx.cache_ids()["cache_build_id"],
                        edge_cache_id=ctx.cache_ids()["edge_slice_cache_id"],
                        sft_cache_id=ctx.cache_ids()["sft_cache_id"],
                        teacher_cache_id=ctx.cache_ids()["teacher_cache_id"],
                        selection_bank_id=bank["selection_bank_id"],
                        selection_bank_reused=False,
                        bank_policy=policy,
                        bank_max_ratio=bank.get("max_ratio_for_bank", ""),
                        candidate_universe=bank.get("candidate_universe", "train_targets"),
                        coverage_axes=bank.get("coverage_axes", ""),
                        year_bucket_available=bank.get("year_bucket_available", False),
                        degree_bucket_mode=str(args.degree_bucket_mode),
                        feature_bucket_mode=bank.get("feature_bucket_mode", ""),
                        feature_lsh_dim=int(args.feature_lsh_dim),
                        feature_lsh_bits=int(args.feature_lsh_bits),
                        teacher_weight_eta=float(eta),
                        class_floor_requested=audit.get("class_floor_requested", ""),
                        class_floor_actual_min=audit.get("class_floor_actual_min", ""),
                        class_floor_violation_count=audit.get("class_floor_violation_count", ""),
                        prefix_overlap_with_previous_ratio=audit.get("prefix_overlap_with_previous_ratio", ""),
                        prefix_violation_count=audit.get("prefix_violation_count", 0),
                        selected_count=audit.get("selected_count", ""),
                        selected_class_count=audit.get("selected_class_count", ""),
                        selected_predicted_class_count=audit.get("selected_predicted_class_count", ""),
                        selected_train_anchor_count=audit.get("selected_train_anchor_count", ""),
                        selected_soft_prior_kl=audit.get("selected_soft_prior_kl", ""),
                        selected_hard_label_prior_kl=audit.get("selected_hard_label_prior_kl", ""),
                        coverage_bucket_count=audit.get("coverage_bucket_count", ""),
                        empty_bucket_count=audit.get("empty_bucket_count", ""),
                        uses_teacher_weighting=bool(bank.get("uses_teacher_weighting", False)),
                        promotion_status="promoted",
                        notes=f"bank_build_count={bank.get('bank_build_count', 1)}",
                    )
                    rows.append(row)
                    write_csv(Path(args.tables_dir) / "t37_papers100m_scr_bank_audit.csv", rows, T37_REQUIRED_FIELDS)
    print("status=completed")


if __name__ == "__main__":
    main()
