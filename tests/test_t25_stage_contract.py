from pathlib import Path

import numpy as np
import pytest

from shadow_hgc.sft.signature_cache import load_existing_sft_signature_cache, write_or_load_sft_signature_cache_from_memmap
from shadow_hgc.sft.t25_contract import (
    T25_OUTPUT_FIELDS,
    T25_FORBIDDEN_PROMOTED_FLAGS,
    apply_ultra_safe_guards,
    make_t25_row,
    validate_t25_promoted_row,
)


def test_t25_full_node_ratio_includes_shadow_nodes():
    row = make_t25_row(
        dataset="Reddit",
        method="sft_hnr_fdm_hybrid",
        requested_full_node_ratio=0.005,
        original_total_nodes=1000,
        target_prototypes=3,
        shadow_nodes=2,
        total_condensed_edges=4,
    )

    assert row["actual_full_node_ratio"] == pytest.approx(0.005)
    assert row["requested_full_node_ratio"] == pytest.approx(0.005)
    assert row["total_condensed_nodes"] == 5
    assert row["ratio_mode"] == "full_node"


def test_t25_promoted_rows_reject_forbidden_components():
    clean = make_t25_row(
        dataset="Reddit",
        method="sft_hnr_fdm_hybrid",
        requested_full_node_ratio=0.005,
        original_total_nodes=1000,
        target_prototypes=5,
        shadow_nodes=0,
        total_condensed_edges=0,
        accuracy=0.93,
        macro_f1=0.90,
        promotion_status="promoted",
    )
    assert validate_t25_promoted_row(clean)["valid"] is True
    bad = dict(clean, uses_teacher_logits=True, uses_full_edge_index_on_gpu=True)
    result = validate_t25_promoted_row(bad)
    assert result["valid"] is False
    assert "uses_teacher_logits" in result["forbidden_flags"]
    assert "uses_full_edge_index_on_gpu" in result["forbidden_flags"]


def test_t25_promoted_rows_require_metrics():
    row = make_t25_row(
        dataset="ogbn-papers100M",
        method="t25_ultra_safe_planner",
        requested_full_node_ratio=0.0001,
        original_total_nodes=1000,
        target_prototypes=1,
        shadow_nodes=0,
        total_condensed_edges=0,
        promotion_status="promoted",
    )

    assert row["promotion_status"] == "blocked_forbidden"
    assert "missing_accuracy_for_promotion" in row["failure_reason"]


def test_t25_ultra_safe_forces_lite_and_forbids_expensive_paths():
    guarded = apply_ultra_safe_guards(
        {
            "fdm_mode": "full",
            "hnr_hist_mode": "full",
            "uses_all_target_cache": True,
            "uses_exact_pairwise": True,
            "full_class_kmeans": True,
            "uses_dense_p2": True,
        }
    )

    assert guarded["fdm_mode"] == "lite"
    assert guarded["hnr_hist_mode"] == "topk"
    assert guarded["uses_all_target_cache"] is False
    assert guarded["uses_exact_pairwise"] is False
    assert guarded["full_class_kmeans"] is False
    assert guarded["uses_dense_p2"] is False


def test_t25_required_output_fields_include_forbidden_flags():
    for field in T25_FORBIDDEN_PROMOTED_FLAGS:
        assert field in T25_OUTPUT_FIELDS
    for field in ["hnr_edge_scans", "fdm_signature_dim", "fdm_candidate_pool_size", "shadow_b", "notes"]:
        assert field in T25_OUTPUT_FIELDS


def test_t25_gcrd_baseline_csv_exists_with_required_columns():
    path = Path("baselines/gcrd_tpami26.csv")
    assert path.exists()
    header = path.read_text(encoding="utf-8").splitlines()[0].split(",")
    assert header == [
        "dataset",
        "ratio_reported",
        "ratio_definition",
        "backbone",
        "accuracy_mean",
        "accuracy_std",
        "macro_f1_if_available",
        "nodes_synthetic",
        "edges_synthetic",
        "source",
        "notes",
    ]


def test_t25_signature_cache_can_be_reused_when_contract_matches(tmp_path: Path):
    cache_dir = tmp_path / "sig"
    manifest_dir = tmp_path / "manifest"
    cache_dir.mkdir()
    manifest_dir.mkdir()
    mm = np.memmap(cache_dir / "train_signature.memmap", mode="w+", dtype=np.float16, shape=(2, 3))
    mm[:] = np.ones((2, 3), dtype=np.float16)
    mm.flush()
    (cache_dir / "metadata.json").write_text(
        """
{
  "cache_bytes": 12,
  "manifest_dir": "%s",
  "block_names": ["self"],
  "dtype": "float16",
  "arrays": {
    "train_signature": {
      "path": "train_signature.memmap",
      "shape": [2, 3],
      "dtype": "float16",
      "bytes": 12
    }
  }
}
""".strip()
        % str(manifest_dir).replace("\\", "\\\\"),
        encoding="utf-8",
    )

    loaded = load_existing_sft_signature_cache(cache_dir, manifest_dir=manifest_dir, selected_blocks=["X0"], train_rows=[10, 20], dtype="float16")
    assert loaded is not None
    reused = write_or_load_sft_signature_cache_from_memmap(
        manifest_dir=manifest_dir,
        splits={},
        train_rows=[10, 20],
        out_dir=cache_dir,
        selected_blocks=["X0"],
    )
    assert reused.metadata["cache_bytes"] == 12


def test_t25_signature_cache_reuse_rejects_stale_contract(tmp_path: Path):
    cache_dir = tmp_path / "sig"
    manifest_dir = tmp_path / "manifest"
    cache_dir.mkdir()
    manifest_dir.mkdir()
    mm = np.memmap(cache_dir / "train_signature.memmap", mode="w+", dtype=np.float16, shape=(2, 3))
    mm[:] = np.ones((2, 3), dtype=np.float16)
    mm.flush()
    (cache_dir / "metadata.json").write_text(
        """
{
  "cache_bytes": 12,
  "manifest_dir": "%s",
  "block_names": ["self"],
  "dtype": "float16",
  "arrays": {
    "train_signature": {
      "path": "train_signature.memmap",
      "shape": [2, 3],
      "dtype": "float16",
      "bytes": 12
    }
  }
}
""".strip()
        % str(manifest_dir).replace("\\", "\\\\"),
        encoding="utf-8",
    )

    assert load_existing_sft_signature_cache(cache_dir, manifest_dir=manifest_dir, selected_blocks=["X1"], train_rows=[10, 20], dtype="float16") is None
    assert load_existing_sft_signature_cache(cache_dir, manifest_dir=manifest_dir, selected_blocks=["X0"], train_rows=[10], dtype="float16") is None
