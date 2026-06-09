# T2-SFT-NL No-Logits Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement and validate T2-SFT-NL as an opt-in scalable fullgraph table-teacher stage with logits removed from promoted graph signals.

**Architecture:** Add a `shadow_hgc.preprop` package for chunked target-side block generation, memmap manifests, and resource reports. Add no-logits `sagn_lite` and `gamlp_lite` table teachers plus validation-only safe block selection scripts that consume block manifests and emit machine-readable summaries.

**Tech Stack:** Python 3.9+, PyTorch, NumPy memmap, existing Shadow-HGC loaders/features, local conda env `C:\Users\slian\anaconda3\envs\pytorch\python.exe`.

---

### Task 1: T2 Tests And Resource Contracts

**Files:**
- Create: `tests/test_preprop_no_e_by_d_materialization.py`
- Create: `tests/test_preprop_chunked_matches_dense_tiny.py`
- Create: `tests/test_preprop_memmap_manifest.py`
- Create: `tests/test_block_stats_fit_train_freeze.py`
- Create: `tests/test_sft_no_logits_as_input.py`
- Create: `tests/test_sagn_lite_forward_shapes.py`
- Create: `tests/test_gamlp_lite_forward_shapes.py`
- Create: `tests/test_no_dense_p2_promoted.py`
- Create: `tests/test_no_bounded_edges_promoted.py`
- Create: `tests/test_t2_resource_report_schema.py`
- Modify: `tests/test_safe_block_selection_non_regression.py`

- [ ] **Step 1: Write failing tests**

Write tests that import the required T2 modules, check tiny dense equivalence, assert manifest fields, confirm train-only stats freezing, verify teacher forward shapes, and reject promoted rows with logits/dense P2/bounded edges.

- [ ] **Step 2: Run red tests**

Run:

```powershell
& 'C:\Users\slian\anaconda3\envs\pytorch\python.exe' -m pytest tests/test_preprop_no_e_by_d_materialization.py tests/test_preprop_chunked_matches_dense_tiny.py tests/test_preprop_memmap_manifest.py tests/test_block_stats_fit_train_freeze.py tests/test_sft_no_logits_as_input.py tests/test_sagn_lite_forward_shapes.py tests/test_gamlp_lite_forward_shapes.py tests/test_safe_block_selection_non_regression.py tests/test_no_dense_p2_promoted.py tests/test_no_bounded_edges_promoted.py tests/test_t2_resource_report_schema.py -q
```

Expected: fail because new modules/scripts do not exist yet.

### Task 2: Preprop Package

**Files:**
- Create: `shadow_hgc/preprop/__init__.py`
- Create: `shadow_hgc/preprop/specs.py`
- Create: `shadow_hgc/preprop/manifest.py`
- Create: `shadow_hgc/preprop/memmap_blocks.py`
- Create: `shadow_hgc/preprop/normalization.py`
- Create: `shadow_hgc/preprop/spmm_chunked.py`
- Create: `shadow_hgc/preprop/io.py`
- Create: `shadow_hgc/preprop/engine.py`
- Create: `shadow_hgc/features/block_stats.py`

- [ ] **Step 1: Implement spec and manifest dataclasses**

Define `PrepropBlockSpec`, `PrepropBlockMeta`, and `PrepropManifest` with deterministic hashes and JSON roundtrip.

- [ ] **Step 2: Implement chunked aggregation**

Implement destination-row normalized chunked SpMM that only materializes `chunk_edges x d`, logs `uses_e_by_d_materialization=false`, and supports target-row gather.

- [ ] **Step 3: Implement memmap writes and stats**

Write `block_<name>.memmap`, `block_<name>_stats.json`, and `manifest.json`; fit stats on `train_target_rows` only and freeze them for later model use.

- [ ] **Step 4: Run preprop tests**

Run the Task 1 preprop/stat tests and confirm pass.

### Task 3: No-Logits Table Teacher

**Files:**
- Create: `shadow_hgc/models/hop_attention.py`
- Create: `shadow_hgc/models/gamlp_lite.py`
- Create: `shadow_hgc/models/sft_teacher.py`
- Create: `shadow_hgc/train/train_sft_teacher.py`
- Create: `shadow_hgc/eval/sft_eval.py`

- [ ] **Step 1: Implement SAGN-lite**

Each block gets an MLP embedding; node/block attention produces a weighted sum and classifier logits. No input block name may contain or declare logits.

- [ ] **Step 2: Implement GAMLP-lite**

Process normalized blocks through gated residual/JK-style updates, log gate values, and keep final logits unactivated.

- [ ] **Step 3: Implement losses and training**

Support cross entropy, class-balanced CE, balanced softmax, logit-adjusted CE, and label smoothing; use validation accuracy with macro-F1 tie-break for early stopping.

- [ ] **Step 4: Run teacher tests**

Run T2 teacher tests and confirm pass.

### Task 4: T2 Scripts

**Files:**
- Create: `scripts/run_t2_preprop_blocks.py`
- Create: `scripts/run_t2_sft_fullgraph.py`
- Create: `scripts/run_t2_safe_block_selection.py`
- Create: `scripts/run_t2_medium_sft.py`
- Create: `scripts/run_t2_small_sft.py`
- Create: `scripts/run_t2_scalability_dry_run.py`
- Create: `scripts/run_t2_stage.py`

- [ ] **Step 1: Build preprop block runner**

Load small/medium datasets, generate block specs, write manifest index CSV and preprop summary.

- [ ] **Step 2: Build no-logits fullgraph teacher runner**

Train SAGN/GAMLP-lite from T2 blocks only, save predictions as evaluation artifacts, and set `uses_logits_as_input=false`.

- [ ] **Step 3: Build safe block selection runner**

Add candidate groups one at a time, keep only validation-improving groups, log `kept_or_dropped`, resource fields, and non-regression status.

- [ ] **Step 4: Build scalability dry-run**

Estimate arxiv/products/papers100M/MAG240M cache bytes, scans, server recommendation, and forbidden flags without allocating high-dimensional diffusion or dense P2.

### Task 5: Experiments And Final Report

**Files:**
- Create: `experiments/tables/t2_preprop_manifest_index_seed42.csv`
- Create: `experiments/tables/t2_sft_fullgraph_seed42.csv`
- Create: `experiments/tables/t2_sft_safe_block_selection_seed42.csv`
- Create: `experiments/tables/t2_sft_scalability_dry_run_seed42.csv`
- Create: `experiments/tables/t2_sft_stage_summary_seed42.csv`
- Optional Create: `experiments/tables/t2_condensation_recovery_seed42.csv`
- Create: `experiments/reports/t2_preprop_summary.md`
- Create: `experiments/reports/t2_sft_fullgraph_summary.md`
- Create: `experiments/reports/t2_sft_scalability_summary.md`
- Create: `experiments/reports/t2_sft_no_logits_stage_summary.md`

- [ ] **Step 1: Run small T2**

Run ACM/DBLP/IMDB seed 42 and record validation-selected block subsets and test metrics.

- [ ] **Step 2: Run medium T2**

Run arxiv/products only after dry-run passes; if local resources block a full row, write explicit resource-guard rows.

- [ ] **Step 3: Run ultra dry-runs**

Estimate paper100M and MAG240M cache/storage/scans and mark server recommendation where needed.

- [ ] **Step 4: Recovery gate**

Run condensation recovery only for datasets whose no-logits fullgraph SFT row improves over safe baseline.

- [ ] **Step 5: Write final summary**

Answer the 13 required report questions and include all result tables and changes.

### Task 6: Verification, Commit, Push

**Files:**
- All files changed above.

- [ ] **Step 1: Run targeted T2 tests**

Run the Task 1 command and inspect output.

- [ ] **Step 2: Run full test suite**

Run:

```powershell
& 'C:\Users\slian\anaconda3\envs\pytorch\python.exe' -m pytest tests -q
```

- [ ] **Step 3: Check requirement coverage**

Re-read `C:\Users\slian\Downloads\codex_t2_sft_no_logits_prompt.md` and confirm each required module, script, test, artifact, and final report answer exists.

- [ ] **Step 4: Commit and push**

Stage, commit with a T2 no-logits message, push `main` to GitHub, and report commit hash.
