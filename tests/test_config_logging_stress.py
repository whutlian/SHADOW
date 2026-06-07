import json
import pytest

from shadow_hgc.config import load_experiment_config, validate_experiment_config
from shadow_hgc.data.edge_stream import run_synthetic_streaming_stress
from shadow_hgc.eval.logging import write_json_summary


def test_default_config_contains_required_shadow_hgc_fields():
    config = load_experiment_config("configs/methods/shadow_hgc_r1_default.yaml")

    assert config["method"] == "Shadow-HGC-R-1"
    assert config["target_type"] == "paper"
    assert config["relation"]["normalization"] == "destination_row_alpha"
    assert config["io"]["train_target_only_demand"] is True


def test_config_validation_rejects_invalid_projection_loss_and_model():
    config = load_experiment_config("configs/methods/shadow_hgc_r1_default.yaml")
    config["projection_type"] = "pca"
    with pytest.raises(ValueError, match="projection_type"):
        validate_experiment_config(config)

    config = load_experiment_config("configs/methods/shadow_hgc_r1_default.yaml")
    config["loss_type"] = "bad_loss"
    with pytest.raises(ValueError, match="loss_type"):
        validate_experiment_config(config)

    config = load_experiment_config("configs/methods/shadow_hgc_r1_default.yaml")
    config["model"] = "hgt"
    with pytest.raises(ValueError, match="model"):
        validate_experiment_config(config)


def test_json_summary_writer_creates_parent_directory(tmp_path):
    path = tmp_path / "nested" / "summary.json"

    write_json_summary(path, {"method": "Shadow-HGC-R-1", "value": 3})

    assert json.loads(path.read_text())["value"] == 3


def test_synthetic_streaming_stress_uses_two_scans_without_memory_blowup(tmp_path):
    summary = run_synthetic_streaming_stress(
        output_path=tmp_path / "stress.json",
        num_edges=20_000,
        num_src_nodes=500,
        num_dst_nodes=300,
        num_train_targets=50,
        feature_dim=8,
        chunk_size=4096,
        seed=2,
    )

    assert summary["num_edges"] == 20_000
    assert summary["full_edge_scans"] == 2
    assert summary["demand_shape"] == [50, 8]
    assert summary["edge_slice_cache_bytes"] > 0
