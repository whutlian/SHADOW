import csv
import json

from shadow_hgc.eval.tables import build_ratio_main_rows_from_logs, build_small_main_rows_from_logs, write_rows_csv


def test_small_table_rows_are_built_from_json_logs(tmp_path):
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    payload = {
        "dataset": "toyset",
        "method": "Shadow-HGC-R-1",
        "budget_mode": "ratio",
        "ratio": 0.01,
        "ratio_base": "train_target",
        "requested_target_budget": 8,
        "effective_target_prototypes": 12,
        "num_train_target_nodes": 800,
        "num_train_classes": 3,
        "shadow_nodes_total": 10,
        "condensed_nodes_total": 22,
        "condensed_edges_total": 44,
        "effective_target_ratio": 0.015,
        "condensed_node_ratio_to_train_target": 0.0275,
        "condensed_node_ratio_to_all_task_nodes": 0.011,
        "requested_M_tau": 8,
        "effective_M_tau": 12,
        "resolved_M_r": {"a--r-->b": 8},
        "feature_dim": 64,
        "projection_type": "raw",
        "loss_type": "clipped",
        "model": "relation_mlp",
        "accuracy": 0.5,
        "macro_f1": 0.25,
        "condensation_time": 1.0,
        "training_time": 2.0,
        "inference_time": 3.0,
        "status": "completed",
        "generated_at": "2026-06-07T00:00:00Z",
        "git_commit": "abc",
        "config_hash": "def",
        "run_id": "run",
    }
    (log_dir / "toyset_M8_seed0.json").write_text(json.dumps(payload), encoding="utf-8")

    rows = build_small_main_rows_from_logs(log_dir)

    assert rows[0]["dataset"] == "toyset"
    assert rows[0]["M_tau_requested"] == "8"
    assert rows[0]["M_tau_effective"] == "12"
    assert rows[0]["accuracy_mean"] == "0.500000"
    assert rows[0]["macro_f1_mean"] == "0.250000"
    assert rows[0]["generated_at"] == "2026-06-07T00:00:00Z"

    out = tmp_path / "table.csv"
    write_rows_csv(out, rows)
    assert list(csv.DictReader(out.open()))[0]["config_hash"] == "def"

    ratio_rows = build_ratio_main_rows_from_logs(log_dir)
    assert ratio_rows[0]["ratio"] == "0.01"
    assert ratio_rows[0]["requested_target_budget"] == 8
    assert ratio_rows[0]["effective_target_prototypes"] == 12
    assert ratio_rows[0]["shadow_nodes_total"] == 10
    assert ratio_rows[0]["condensed_nodes_total"] == 22
