# Paper100M Local Trial Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a guarded local ogbn-papers100M trial that validates the local memmap dataset, runs a bounded sample classifier when possible, reports full-scale resource feasibility, and emits server commands when local execution is blocked.

**Architecture:** Keep the existing fullgraph parity stage unchanged and add an opt-in `scripts/run_paper100m_local_trial.py` plus a small utility module for paper100M manifest/resource logic. The script reads local memmaps with `mmap_mode='r'`, runs a bounded CPU/GPU-safe sample linear probe, writes CSV/JSON/Markdown artifacts, and never attempts unguarded all-node demand caching.

**Tech Stack:** Python, NumPy memmap, PyTorch, existing `shadow_hgc.demand.cache.estimate_ultra_dry_run`, pytest, local conda env `C:\Users\slian\anaconda3\envs\pytorch\python.exe`.

---

### Task 1: Paper100M Trial Utilities

**Files:**
- Create: `shadow_hgc/ultra/paper100m_trial.py`
- Test: `tests/test_paper100m_local_trial.py`

- [ ] **Step 1: Write failing tests**

```python
from pathlib import Path

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
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```powershell
& 'C:\Users\slian\anaconda3\envs\pytorch\python.exe' -m pytest tests/test_paper100m_local_trial.py -q
```

Expected: import failure for `shadow_hgc.ultra.paper100m_trial`.

- [ ] **Step 3: Implement minimal utilities**

Create `shadow_hgc/ultra/paper100m_trial.py` with:

```python
from __future__ import annotations

import json
from pathlib import Path


def inspect_memmap_manifest(memmap_root: str | Path) -> dict:
    root = Path(memmap_root)
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        return {"dataset_present": False, "memmap_root": str(root), "reason": "manifest.json missing"}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    def entry(name: str) -> dict:
        return dict(manifest.get(name, {}))
    node_feat = entry("node_feat.npy")
    edge_index = entry("edge_index.npy")
    train_idx = entry("train_idx.npy")
    valid_idx = entry("valid_idx.npy")
    test_idx = entry("test_idx.npy")
    return {
        "dataset_present": True,
        "memmap_root": str(root),
        "manifest_path": str(manifest_path),
        "num_nodes": int(node_feat.get("shape", [0, 0])[0]),
        "feature_dim": int(node_feat.get("shape", [0, 0])[1]),
        "num_edges": int(edge_index.get("shape", [0, 0])[1]),
        "train_nodes": int(train_idx.get("shape", [0])[0]),
        "valid_nodes": int(valid_idx.get("shape", [0])[0]),
        "test_nodes": int(test_idx.get("shape", [0])[0]),
        "node_feat_bytes": int(node_feat.get("bytes", 0)),
        "edge_index_bytes": int(edge_index.get("bytes", 0)),
        "label_bytes": int(entry("node_label.npy").get("bytes", 0)),
    }


def compute_trial_decision(*, peak_ram_estimate_bytes: int, available_ram_bytes: int, local_trial_status: str) -> dict:
    if local_trial_status not in {"completed", "completed_smoke"}:
        return {"full_scale_local_status": local_trial_status, "needs_server_run": True}
    if peak_ram_estimate_bytes > available_ram_bytes * 0.8:
        return {"full_scale_local_status": "blocked_resource_guard", "needs_server_run": True}
    return {"full_scale_local_status": "local_feasible_by_estimate", "needs_server_run": False}


def build_server_commands(*, dataset_root: str, output_dir: str) -> list[str]:
    py = r"C:\Users\slian\anaconda3\envs\pytorch\python.exe"
    return [
        f"& '{py}' scripts/run_paper100m_local_trial.py --dataset-root {dataset_root} --output-dir {output_dir} --seed 42 --sample-train 200000 --sample-valid 50000 --epochs 50 --full-scale --no-diffusion",
        f"& '{py}' scripts/dry_run_ultra.py --dataset ogbn-papers100M --ratios 0.001 0.0025 0.005 --output {output_dir}/paper100m_ultra_dry_run_server.json",
    ]
```

- [ ] **Step 4: Run tests to verify pass**

Run the same pytest command. Expected: pass.

### Task 2: Local Trial Script And Artifacts

**Files:**
- Create: `scripts/run_paper100m_local_trial.py`
- Modify: `experiments/reports/fullgraph_parity_condensation_recovery_summary.md` after running
- Test: `tests/test_paper100m_local_trial.py`

- [ ] **Step 1: Add failing script test**

Append to `tests/test_paper100m_local_trial.py`:

```python
import subprocess
import sys
import numpy as np


def test_paper100m_trial_script_writes_artifacts(tmp_path):
    dataset_root = tmp_path / "paper100M"
    memmap = dataset_root / "processed" / "papers100m_memmap"
    memmap.mkdir(parents=True)
    np.save(memmap / "node_feat.npy", np.random.default_rng(0).normal(size=(40, 4)).astype("float32"))
    np.save(memmap / "node_label.npy", np.arange(40, dtype="float32").reshape(-1, 1) % 3)
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
```

- [ ] **Step 2: Run script test to verify failure**

Expected: script missing.

- [ ] **Step 3: Implement script**

Create `scripts/run_paper100m_local_trial.py` that:
- opens `processed/papers100m_memmap/*.npy` with `mmap_mode='r'`;
- samples train/valid indices deterministically with seed 42;
- trains a bounded linear classifier for the requested epochs;
- writes JSON, CSV, and Markdown artifacts;
- reports `blocked_resource_guard` and server commands when full-scale estimated RAM exceeds local guard.

- [ ] **Step 4: Run script test to verify pass**

Run:

```powershell
& 'C:\Users\slian\anaconda3\envs\pytorch\python.exe' -m pytest tests/test_paper100m_local_trial.py -q
```

Expected: pass.

### Task 3: Run Local Paper100M Trial And Final Verification

**Files:**
- Generate: `experiments/logs/paper100m_local_trial_seed42/paper100m_local_trial_seed42.json`
- Generate: `experiments/tables/paper100m_local_trial_seed42.csv`
- Generate: `experiments/reports/paper100m_local_trial_seed42.md`
- Modify: `experiments/reports/fullgraph_parity_condensation_recovery_summary.md`

- [ ] **Step 1: Run full pytest before experiment**

```powershell
& 'C:\Users\slian\anaconda3\envs\pytorch\python.exe' -m pytest tests -q
```

- [ ] **Step 2: Run local guarded paper100M trial**

```powershell
& 'C:\Users\slian\anaconda3\envs\pytorch\python.exe' scripts/run_paper100m_local_trial.py --dataset-root D:\Shadow-HGC\dataset\paper100M --output-dir experiments/logs/paper100m_local_trial_seed42 --table experiments/tables/paper100m_local_trial_seed42.csv --report experiments/reports/paper100m_local_trial_seed42.md --seed 42 --sample-train 20000 --sample-valid 5000 --epochs 10 --no-diffusion
```

- [ ] **Step 3: Update final summary**

Add a paper100M section to `experiments/reports/fullgraph_parity_condensation_recovery_summary.md` with local smoke metrics, full-scale resource decision, and server commands if blocked.

- [ ] **Step 4: Run final verification**

```powershell
& 'C:\Users\slian\anaconda3\envs\pytorch\python.exe' -m pytest tests -q
git diff --check
```

- [ ] **Step 5: Commit and push**

```powershell
git add ...
git commit -m "Add paper100M local trial"
git push origin main
```
