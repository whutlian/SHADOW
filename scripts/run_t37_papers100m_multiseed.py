from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.t24_common import read_csv, write_csv
from shadow_hgc.ultra.papers100m_t37_contract import T37_REQUIRED_FIELDS, make_t37_row


def _best_method(rows: list[dict[str, str]], *, prefix: str, backend: str | None = None) -> str:
    candidates = [row for row in rows if str(row.get("method", "")).startswith(prefix) and row.get("accuracy", "") != ""]
    if backend is not None:
        candidates = [row for row in candidates if str(row.get("backend", "")).lower() == str(backend).lower()]
    if not candidates:
        return ""
    by_method: dict[str, list[float]] = {}
    for row in candidates:
        by_method.setdefault(row["method"], []).append(float(row["accuracy"]))
    return max(by_method, key=lambda method: statistics.mean(by_method[method]))


def _aggregate(rows: list[dict[str, str]], *, out_method: str) -> list[dict[str, object]]:
    groups: dict[tuple[str, str, float], list[dict[str, str]]] = {}
    for row in rows:
        if row.get("accuracy", "") == "":
            continue
        key = (row.get("method", ""), row.get("backend", ""), float(row.get("requested_full_node_ratio", 0.0) or 0.0))
        groups.setdefault(key, []).append(row)
    out: list[dict[str, object]] = []
    for (method, backend, ratio), group in sorted(groups.items()):
        acc = [float(row["accuracy"]) for row in group]
        macro = [float(row.get("macro_f1", 0.0) or 0.0) for row in group]
        gap = [float(row.get("valid_acc", 0.0) or 0.0) - float(row["accuracy"]) for row in group]
        out.append(
            make_t37_row(
                method=out_method if out_method else method,
                seed=-1,
                backend=backend,
                comparison_type="multiseed_aggregate",
                requested_full_node_ratio=ratio,
                accuracy=statistics.mean(acc),
                macro_f1=statistics.mean(macro),
                valid_acc="",
                notes=(
                    f"source_method={method}; seed_count={len(group)}; "
                    f"accuracy_mean={statistics.mean(acc)}; accuracy_std={statistics.pstdev(acc) if len(acc) > 1 else 0.0}; "
                    f"macro_f1_mean={statistics.mean(macro)}; macro_f1_std={statistics.pstdev(macro) if len(macro) > 1 else 0.0}; "
                    f"best={max(acc)}; median={statistics.median(acc)}; worst={min(acc)}; "
                    f"valid_test_gap_mean={statistics.mean(gap)}"
                ),
                promotion_status="diagnostic",
            )
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate T37 multi-seed final rows. Run row scripts separately if needed.")
    parser.add_argument("--tables-dir", default="experiments/tables")
    parser.add_argument("--methods", nargs="+", default=["random_onecache", "best_scr", "best_native_randcore"])
    parser.add_argument("--run-subcommands", action="store_true", help="Reserved; use explicit row scripts for long runs.")
    args = parser.parse_args()
    tables = Path(args.tables_dir)
    disco_seed42 = read_csv(tables / "t37_papers100m_disco_parity_scr_seed42.csv")
    disco_raw = read_csv(tables / "t37_papers100m_disco_parity_scr_multiseed_raw.csv")
    native_seed42 = read_csv(tables / "t37_papers100m_native_randcore_seed42.csv")
    native_raw = read_csv(tables / "t37_papers100m_native_randcore_multiseed_raw.csv")
    disco = disco_seed42 + disco_raw
    native = native_seed42 + native_raw
    best_scr = _best_method(disco_raw or disco, prefix="scr_", backend="sgc")
    best_native = _best_method(native_raw or native, prefix="stt_randcore", backend="gamlp_table")
    selected = [row for row in disco if row.get("method") in {"random_onecache", best_scr}] + [row for row in native if row.get("method") == best_native]
    rows = _aggregate(selected, out_method="")
    write_csv(tables / "t37_papers100m_disco_parity_scr_multiseed.csv", [row for row in rows if row.get("backend") == "sgc"], T37_REQUIRED_FIELDS)
    write_csv(tables / "t37_papers100m_native_randcore_multiseed.csv", [row for row in rows if row.get("backend") != "sgc"], T37_REQUIRED_FIELDS)
    print(f"status=completed best_scr={best_scr} best_native={best_native}")


if __name__ == "__main__":
    main()
