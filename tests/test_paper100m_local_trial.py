from __future__ import annotations

import json
import subprocess
import sys

import numpy as np

from shadow_hgc.ultra.paper100m_trial import (
    build_server_commands,
    compute_trial_decision,
    inspect_memmap_manifest,
)


def test_inspect_memmap_manifest_reads_shapes(tmp_path):
    root = tmp_path / "papers100m_memmap"
    root.mkdir()
    (root / "manifest.json").write_text(
        '{"node_feat.npy":{"shape":[100,128],"dtype":"float32","bytes":51200},'
        '"edge_index.npy":{"shape":[2,200],"dtype":"int64","bytes":3200},'
        '"train_idx.npy":{"shape":[10],"dtype":"int64","bytes":80},'
        '"valid_idx.npy":{"shape":[5],"dtype":"int64","bytes":40},'
        '"test_idx.npy":{"shape":[5],"dtype":"int64","bytes":40},'
        '"node_label.npy":{"shape":[100,1],"dtype":"float32","bytes":400}}',
        encoding="utf-8",
    )

    info = inspect_memmap_manifest(root)

    assert info["dataset_present"] is True
    assert info["num_nodes"] == 100
    assert info["feature_dim"] == 128
    assert info["num_edges"] == 200
    assert info["train_nodes"] == 10


def test_compute_trial_decision_blocks_full_when_ram_exceeds_guard():
    decision = compute_trial_decision(
        peak_ram_estimate_bytes=120 * 1024**3,
        available_ram_bytes=32 * 1024**3,
        local_trial_status="completed",
    )

    assert decision["full_scale_local_status"] == "blocked_resource_guard"
    assert decision["needs_server_run"] is True


def test_build_server_commands_include_dataset_root_and_no_diffusion():
    commands = build_server_commands(
        dataset_root="D:/Shadow-HGC/dataset/paper100M",
        output_dir="experiments/logs/paper100m_local_trial_seed42",
    )

    joined = "\n".join(commands)
    assert "run_paper100m_local_trial.py" in joined
    assert "--dataset-root D:/Shadow-HGC/dataset/paper100M" in joined
    assert "--full-scale" in joined
    assert "--no-diffusion" in joined


def test_paper100m_trial_script_writes_artifacts(tmp_path):
    dataset_root = tmp_path / "paper100M"
    memmap = dataset_root / "processed" / "papers100m_memmap"
    memmap.mkdir(parents=True)
    rng = np.random.default_rng(0)
    np.save(memmap / "node_feat.npy", rng.normal(size=(40, 4)).astype("float32"))
    np.save(memmap / "node_label.npy", (np.arange(40, dtype="float32").reshape(-1, 1) % 3))
    np.save(memmap / "train_idx.npy", np.arange(0, 20, dtype="int64"))
    np.save(memmap / "valid_idx.npy", np.arange(20, 30, dtype="int64"))
    np.save(memmap / "test_idx.npy", np.arange(30, 40, dtype="int64"))
    np.save(memmap / "edge_index.npy", np.zeros((2, 10), dtype="int64"))
    manifest = {
        "node_feat.npy": {"shape": [40, 4], "dtype": "float32", "bytes": 640},
        "node_label.npy": {"shape": [40, 1], "dtype": "float32", "bytes": 160},
        "train_idx.npy": {"shape": [20], "dtype": "int64", "bytes": 160},
        "valid_idx.npy": {"shape": [10], "dtype": "int64", "bytes": 80},
        "test_idx.npy": {"shape": [10], "dtype": "int64", "bytes": 80},
        "edge_index.npy": {"shape": [2, 10], "dtype": "int64", "bytes": 160},
    }
    (memmap / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    output_dir = tmp_path / "out"

    subprocess.run(
        [
            sys.executable,
            "scripts/run_paper100m_local_trial.py",
            "--dataset-root",
            str(dataset_root),
            "--output-dir",
            str(output_dir),
            "--sample-train",
            "12",
            "--sample-valid",
            "6",
            "--epochs",
            "2",
        ],
        check=True,
    )

    assert (output_dir / "paper100m_local_trial_seed42.json").exists()
    assert (output_dir / "paper100m_local_trial_seed42.csv").exists()
    assert (output_dir / "paper100m_local_trial_seed42.md").exists()
