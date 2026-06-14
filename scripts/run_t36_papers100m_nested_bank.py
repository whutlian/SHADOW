from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.t24_common import write_csv
from shadow_hgc.ultra.papers100m_memmap import read_json
from shadow_hgc.ultra.papers100m_nested_bank import NESTED_BANK_POLICY, audit_nested_bank, build_nested_bank_v2
from shadow_hgc.ultra.papers100m_runner import Papers100MCacheContext
from shadow_hgc.ultra.papers100m_teacher_upgrade import install_teacher_upgrade


def _resolve_teacher(cache_root: Path, teacher_id: str) -> str:
    if teacher_id == "best_t36_teacher":
        best_path = cache_root / "teacher_upgrade" / "best_teacher.json"
        if best_path.exists():
            best = read_json(best_path)
            install_teacher_upgrade(cache_root, str(best["teacher_id"]))
            return str(best["teacher_id"])
    return str(teacher_id)


def main() -> None:
    parser = argparse.ArgumentParser(description="T36 papers100M nested bank v2 runner.")
    parser.add_argument("--cache-root", default="caches/papers100m/stt_v1")
    parser.add_argument("--teacher-id", default="current_teacher")
    parser.add_argument("--max-ratio", type=float, default=0.002)
    parser.add_argument("--ratios", nargs="+", type=float, default=[0.00005, 0.00010, 0.00020, 0.00050, 0.001, 0.002])
    parser.add_argument("--policy", default=NESTED_BANK_POLICY)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--run-long", action="store_true")
    parser.add_argument("--tables-dir", default="experiments/tables")
    args = parser.parse_args()

    cache_root = Path(args.cache_root)
    teacher_id = _resolve_teacher(cache_root, str(args.teacher_id))
    ctx = Papers100MCacheContext(cache_root, selection_policy=str(args.policy), seed=int(args.seed))
    manifest = build_nested_bank_v2(
        ctx,
        policy=str(args.policy),
        seed=int(args.seed),
        max_ratio=float(args.max_ratio),
        force=bool(args.force),
        teacher_id=teacher_id,
    )
    rows: list[dict[str, Any]] = audit_nested_bank(cache_root, policy=str(args.policy), seed=int(args.seed), ratios=[float(v) for v in args.ratios])
    for row in rows:
        row["teacher_id"] = teacher_id
        row["teacher_cache_id"] = manifest.get("teacher_cache_id", "")
    write_csv(Path(args.tables_dir) / "t36_papers100m_nested_bank_audit.csv", rows)
    print(f"nested_bank_id={manifest.get('nested_bank_id', '')}")
    print("status=completed")


if __name__ == "__main__":
    main()
