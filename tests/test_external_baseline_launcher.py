from pathlib import Path

from shadow_hgc.baselines.external_full_ratio import (
    FULL_RATIO_SCHEDULES,
    DATASET_STATS,
    baseline_code_ratio,
    build_run_plan,
    detect_failure_status,
    parse_metrics_from_text,
    directory_size_bytes,
    ensure_deepcgc_device_mask_compat,
    ensure_numpy_legacy_alias_compat,
    _missing_metric_failure_reason,
    preflight_failure_reason,
    ensure_wbgc_graphsaint_sampler_sizes,
    ensure_clustgdd_induct_import_compat,
)


def test_full_ratio_schedule_matches_requested_table():
    assert FULL_RATIO_SCHEDULES["ogbn-arxiv"] == [0.0005, 0.001, 0.0025, 0.005, 0.01]
    assert FULL_RATIO_SCHEDULES["reddit"] == [0.0005, 0.001, 0.002, 0.005, 0.01]
    assert FULL_RATIO_SCHEDULES["ogbn-products"] == [0.0005, 0.0025, 0.005, 0.01]
    assert FULL_RATIO_SCHEDULES["ogbn-products-low"] == [0.0002, 0.0004, 0.0008]


def test_baseline_code_ratio_converts_full_ratio_to_train_ratio():
    stats = DATASET_STATS["ogbn-products"]
    ratio = baseline_code_ratio("ogbn-products", 0.005, denominator="train")
    assert round(ratio, 6) == round((stats.num_nodes * 0.005) / stats.train_nodes, 6)

    full_ratio = baseline_code_ratio("reddit", 0.001, denominator="full")
    assert full_ratio == 0.001


def test_build_deepcgc_plan_uses_data_root_gpu_and_metric_contract(tmp_path):
    plan = build_run_plan(
        baseline="DeepCGC",
        dataset="reddit",
        full_ratio=0.001,
        seed=3,
        gpu=2,
        data_root=Path("/data/shared"),
        baseline_root=Path("/repo/baselines"),
        output_root=tmp_path,
    )
    cmd = " ".join(plan.command)
    assert plan.status == "planned"
    assert plan.baseline_ratio == baseline_code_ratio("reddit", 0.001, denominator="train")
    assert "--dataset_name reddit" in cmd
    raw_idx = plan.command.index("--raw_data_dir") + 1
    assert Path(plan.command[raw_idx]).is_absolute()
    assert "--gpu 0" in cmd
    assert plan.env["CUDA_VISIBLE_DEVICES"] == "2"
    assert plan.metrics_path.name == "metrics.json"
    assert plan.summary_path.name == "summary.json"
    assert Path(plan.metadata["config_folder"]).is_absolute()


def test_repo_cwd_paths_are_absolute_for_generated_configs():
    plan = build_run_plan(
        baseline="TGCC",
        dataset="reddit",
        full_ratio=0.001,
        seed=0,
        gpu=2,
        data_root=Path("dataset/baseline_graphsaint"),
        baseline_root=Path("baselines/external_repos"),
        output_root=Path("experiments/logs/external_baselines/test_run"),
    )
    config_idx = plan.command.index("--config") + 1
    save_idx = plan.command.index("--save_dir") + 1
    log_idx = plan.command.index("--log_dir") + 1
    assert Path(plan.command[config_idx]).is_absolute()
    assert Path(plan.command[save_idx]).is_absolute()
    assert Path(plan.command[log_idx]).is_absolute()


def test_products_plan_marks_unsupported_for_baselines_without_products_support(tmp_path):
    plan = build_run_plan(
        baseline="TGCC",
        dataset="ogbn-products",
        full_ratio=0.005,
        seed=0,
        gpu=0,
        data_root=Path("/data/shared"),
        baseline_root=Path("/repo/baselines"),
        output_root=tmp_path,
    )
    assert plan.status == "unsupported"
    assert "ogbn-products" in plan.failure_reason
    assert plan.command == []


def test_tgcc_preflight_reports_missing_augmentation_files(tmp_path):
    plan = build_run_plan(
        baseline="TGCC",
        dataset="reddit",
        full_ratio=0.001,
        seed=0,
        gpu=0,
        data_root=tmp_path / "data",
        baseline_root=tmp_path / "baselines",
        output_root=tmp_path / "out",
    )
    reason = preflight_failure_reason(plan)
    assert "missing TGCC precomputed augmentation file" in reason
    assert "0.01_1_1tr.npz" in reason


def test_metric_parser_extracts_accuracy_f1_params_and_oom():
    text = """
    Average accuracy: 0.9411 +/- 0.0002
    micro_f1: 0.9408 macro_f1=0.9112
    Parameters: 123456
    CUDA out of memory. Tried to allocate 10.00 GiB
    """
    metrics = parse_metrics_from_text(text)
    assert metrics["accuracy"] == 0.9411
    assert metrics["micro_f1"] == 0.9408
    assert metrics["macro_f1"] == 0.9112
    assert metrics["param_count"] == 123456
    assert detect_failure_status(137, text) == ("oom", "CUDA out of memory")


def test_directory_size_bytes_counts_files(tmp_path):
    (tmp_path / "a.txt").write_bytes(b"abc")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "b.bin").write_bytes(b"12345")
    assert directory_size_bytes(tmp_path) == 8


def test_numpy_legacy_alias_patch_preserves_sized_aliases(tmp_path):
    path = tmp_path / "legacy.py"
    path.write_text(
        "a = np.zeros(3, dtype=np.int)\n"
        "b = values.astype(np.bool)\n"
        "c = values.astype(np.int32)\n",
        encoding="utf-8",
    )
    ensure_numpy_legacy_alias_compat(path)
    text = path.read_text(encoding="utf-8")
    assert "dtype=int" in text
    assert "astype(bool)" in text
    assert "np.int32" in text


def test_deepcgc_device_mask_patch_moves_index_masks_to_cpu(tmp_path):
    path = tmp_path / "utils.py"
    path.write_text(
        "idx_center = idx[cls_mask][center_mask]\n"
        "idx_center = idx[cls_mask[cls]][center_mask]\n",
        encoding="utf-8",
    )
    ensure_deepcgc_device_mask_compat(path)
    text = path.read_text(encoding="utf-8")
    assert "idx[cls_mask.cpu()][center_mask]" in text
    assert "idx[cls_mask[cls].cpu()][center_mask]" in text


def test_wbgc_sampler_size_patch_adds_three_layer_default(tmp_path):
    path = tmp_path / "utils_graphsaint.py"
    path.write_text(
        "        if args.nlayers == 2:\n"
        "            if args.dataset in ['reddit', 'flickr']:\n"
        "                if args.option == 0:\n"
        "                    sizes = [15, 8]\n"
        "            else:\n"
        "                sizes = [10, 5]\n"
        " \n"
        "        if self.class_dict2 is None:\n",
        encoding="utf-8",
    )
    ensure_wbgc_graphsaint_sampler_sizes(path)
    ensure_wbgc_graphsaint_sampler_sizes(path)
    text = path.read_text(encoding="utf-8")
    assert text.count("if args.nlayers == 3:") == 1
    assert "sizes = [15, 10, 5]" in text


def test_clustgdd_induct_import_patch_uses_local_module(tmp_path):
    path = tmp_path / "clustgdd_agent_induct.py"
    path.write_text(
        "from KDD2025_ClustGDD.utils_clustgdd import graph_analysis, ER_estimator, attaw_ER_estimator\n",
        encoding="utf-8",
    )
    ensure_clustgdd_induct_import_compat(path)
    text = path.read_text(encoding="utf-8")
    assert "from utils_clustgdd import graph_analysis, ER_estimator, attaw_ER_estimator" in text
    assert "KDD2025_ClustGDD" not in text


def test_missing_metric_failure_reason_detects_budget_mismatch():
    text = "28th (319) class: 317, [100 37 38]"
    assert _missing_metric_failure_reason(text) == "completed without metrics: class budget mismatch"
    assert _missing_metric_failure_reason("finished without score") == (
        "completed without parseable accuracy/micro-F1"
    )
