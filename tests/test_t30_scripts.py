from __future__ import annotations

from argparse import Namespace

import numpy as np

from scripts.run_t30_arxiv_qoc import build_arxiv_qoc_rows
from scripts.run_t30_products_maintenance import build_products_rows
from scripts.run_t30_reddit_qoc import build_reddit_qoc_rows
from scripts.run_t30_stage import build_stage_summary_rows
from shadow_hgc.sft.t30_contract import T30_REQUIRED_FIELDS, validate_t30_row


def _write_block(root, name: str, values: np.ndarray) -> dict:
    path = root / f"block_{name}.memmap"
    mm = np.memmap(path, mode="w+", dtype=values.dtype, shape=values.shape)
    mm[:] = values[:]
    mm.flush()
    (root / f"block_{name}_stats.json").write_text(
        '{"fit_scope":"train_target_rows","frozen":true,"mean":[0.0],"std":[1.0]}',
        encoding="utf-8",
    )
    return {
        "name": name,
        "path": path.name,
        "shape": list(values.shape),
        "dtype": str(values.dtype),
    }


def _write_tiny_reddit_preprop(root) -> None:
    n = 8
    labels = np.array([0, 0, 1, 1, 0, 1, 0, 1], dtype=np.int64)
    x0 = np.array(
        [[2.0, 0.0], [1.8, 0.1], [0.0, 2.0], [0.1, 1.8], [2.1, 0.0], [0.0, 2.1], [1.9, 0.2], [0.2, 1.9]],
        dtype=np.float32,
    )
    y = np.zeros((n, 41), dtype=np.float32)
    y[np.arange(n), labels] = 1.0
    blocks = [
        _write_block(root, "X0", x0),
        _write_block(root, "X1", x0 * 0.9),
        _write_block(root, "X2", x0 * 0.8),
        _write_block(root, "Y1", y),
        _write_block(root, "Y2", y * 0.9),
        _write_block(root, "structure", np.ones((n, 1), dtype=np.float32)),
    ]
    (root / "manifest.json").write_text(__import__("json").dumps({"blocks": blocks}), encoding="utf-8")


def _write_tiny_reddit_memmap(root) -> None:
    labels = np.array([0, 0, 1, 1, 0, 1, 0, 1], dtype=np.int64)
    src = np.array([0, 1, 2, 3, 4, 5, 6, 7, 0, 2, 4, 6], dtype=np.int32)
    dst = np.array([1, 0, 3, 2, 5, 4, 7, 6, 2, 0, 6, 4], dtype=np.int32)
    np.save(root / "src.npy", src)
    np.save(root / "dst.npy", dst)
    np.save(root / "y.int64.npy", labels)
    np.save(root / "train_idx.npy", np.array([0, 1, 2, 3], dtype=np.int64))
    np.save(root / "valid_idx.npy", np.array([4, 5], dtype=np.int64))
    np.save(root / "test_idx.npy", np.array([6, 7], dtype=np.int64))
    (root / "manifest.json").write_text(
        __import__("json").dumps(
            {
                "num_nodes": 8,
                "num_edges": int(src.shape[0]),
                "num_classes": 41,
                "src_path": "src.npy",
                "dst_path": "dst.npy",
                "label_path": "y.int64.npy",
                "train_idx_path": "train_idx.npy",
                "valid_idx_path": "valid_idx.npy",
                "test_idx_path": "test_idx.npy",
                "uses_processed_data_pt": False,
            }
        ),
        encoding="utf-8",
    )


def test_t30_reddit_qoc_rows_have_required_schema_and_block_without_real_cache() -> None:
    rows = build_reddit_qoc_rows(
        Namespace(
            seed=42,
            ratios=[0.001, 0.005],
            assignment_modes=["qoc_class_conditional_online_kmeans"],
            operator_topks=[8],
            quotient_build_modes=["code_row_normalized_fallback"],
            students=["operator_sft_table_head"],
            hidden_dims=[128],
            epochs=[60],
            enable_pltc=False,
            promotion_track="safe_main",
            run_long=False,
            smoke=True,
            sft_cache_dir="missing",
        )
    )
    assert {int(row["num_codewords"]) for row in rows} == {233, 1165}
    for row in rows:
        assert set(T30_REQUIRED_FIELDS).issubset(row)
        assert row["transfer_eval_type"] != "real_transfer_eval"
        assert row["status"] in {"completed_operator_smoke", "blocked"}
        assert validate_t30_row(row)["valid"]


def test_t30_reddit_qoc_can_build_real_transfer_eval_from_memmap(tmp_path) -> None:
    preprop = tmp_path / "preprop"
    raw = tmp_path / "raw_memmap"
    preprop.mkdir()
    raw.mkdir()
    _write_tiny_reddit_preprop(preprop)
    _write_tiny_reddit_memmap(raw)
    rows = build_reddit_qoc_rows(
        Namespace(
            seed=42,
            ratios=[4.0 / 232965.0],
            assignment_modes=["qoc_class_conditional_online_kmeans"],
            operator_topks=[2],
            quotient_build_modes=["original_dest_normalized"],
            students=["operator_sft_table_head"],
            hidden_dims=[8],
            epochs=[8],
            enable_pltc=False,
            promotion_track="safe_main",
            run_long=True,
            smoke=False,
            sft_cache_dir="missing",
            manifest_dir=str(preprop),
            memmap_root=str(raw),
            selected_blocks=["X0", "X1", "X2", "Y1", "Y2", "structure"],
            assignment_blocks=["X0", "X1", "structure"],
            edge_chunk_size=4,
        )
    )
    assert len(rows) == 1
    row = rows[0]
    assert row["status"] == "completed_transfer_eval"
    assert row["transfer_eval_type"] == "real_transfer_eval"
    assert row["failure_reason"] == ""
    assert row["source_table"] == str(preprop)
    assert validate_t30_row(row)["valid"]


def test_t30_arxiv_qoc_blocks_until_teacher_gate_passes() -> None:
    rows = build_arxiv_qoc_rows(Namespace(seed=42, ratios=[0.005], teacher_cache="", run_long=False))
    assert rows[0]["status"] == "blocked"
    assert rows[0]["failure_reason"] == "teacher_gate_not_passed"


def test_t30_products_rows_are_maintenance_not_promotions() -> None:
    rows = build_products_rows(Namespace(seed=42, seeds=[42], ratios=[0.0002, 0.005], methods=["products_uca_hybrid_mixup"]))
    assert len(rows) == 2
    assert all(row["promotion_status"] == "not_promoted" for row in rows)
    assert all(row["status"] == "carried_forward_reference" for row in rows)


def test_t30_stage_summary_reports_required_checks() -> None:
    rows = build_stage_summary_rows(
        arxiv_cns=[],
        semantic=[],
        reddit_qoc=[],
        arxiv_qoc=[],
        products=[],
    )
    checks = {row["requirement_check"]: row for row in rows}
    for key in [
        "promoted_safe_rows",
        "promoted_sota_chase_rows",
        "blocked_rows_by_reason",
        "best_reddit_0p10",
        "best_reddit_0p50",
        "best_arxiv_teacher",
        "arxiv_A1_A2_A3_status",
        "products_maintenance_status",
        "forbidden_guard_hits",
    ]:
        assert key in checks
