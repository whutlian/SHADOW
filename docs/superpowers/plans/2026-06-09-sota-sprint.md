# Shadow-HGC-SOTA Sprint Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an explicit opt-in Shadow-HGC-SOTA branch, run seed-42 SOTA experiments, publish reports, and push the repository.

**Architecture:** Preserve the default Shadow-HGC-R-1 path. SOTA mode is added through explicit modules, flags, configs, and `scripts/run_sota_*.py`; meta-path and Path-LAD signals are feature blocks, not exposed graph edge types.

**Tech Stack:** Python, PyTorch, scikit-learn, pytest, local conda environment `C:\Users\slian\anaconda3\envs\pytorch\python.exe`.

---

### Task 1: Compiled Block Stats Source

**Files:**
- Modify: `shadow_hgc/models/compiled_demand.py`
- Modify: `shadow_hgc/features/compiled_table.py`
- Modify: `shadow_hgc/pipeline/core.py`
- Test: `tests/test_compiled_block_stats_fit_source.py`
- Test: `tests/test_compiled_block_stats_frozen.py`

- [ ] Add explicit `CompiledDemandMLP.fit_block_stats(table)` and `freeze_block_stats()` APIs.
- [ ] Add `fit_compiled_block_stats(model, train_full_table, schema)` helper.
- [ ] Guard lazy fitting in SOTA/compiled explicit mode.
- [ ] Fit stats from original train-target compiled rows even when the training table is prototype rows.
- [ ] Log `compiled_block_stats_source=train_full_demand_table`.
- [ ] Run the two new compiled stats tests and the existing compiled tests.

### Task 2: Meta-Path Blocks and SeHGNN-lite

**Files:**
- Create: `shadow_hgc/features/metapath_blocks.py`
- Create: `shadow_hgc/models/sehgnn_lite.py`
- Test: `tests/test_sehgnn_lite_forward.py`
- Test: `tests/test_metapath_blocks_schema_skip.py`
- Test: `tests/test_metapath_blocks_no_exposed_edge_type.py`

- [ ] Add schema-driven target meta-path feature block computation for small datasets.
- [ ] Use destination-row normalization at each relation step.
- [ ] Add `SeHGNNLite` with block normalization, gates, and concat/sum-logits fusion.
- [ ] Ensure no meta-path edge type is exposed.
- [ ] Run the three new tests.

### Task 3: Path-LAD

**Files:**
- Create: `shadow_hgc/features/path_label_affinity.py`
- Modify: `shadow_hgc/pipeline/core.py`
- Test: `tests/test_path_lad_uses_train_labels_only.py`
- Test: `tests/test_path_lad_leave_one_out.py`
- Test: `tests/test_path_lad_shapes.py`
- Test: `tests/test_path_lad_no_metapath_edges.py`

- [ ] Implement `compute_path_label_affinity(...)`.
- [ ] Add train-label-only and leave-one-out diagnostics.
- [ ] Integrate optional path LAD blocks into compiled/SOTA feature tables.
- [ ] Run all path LAD tests.

### Task 4: Coverage Medoids and Source Anchors

**Files:**
- Create: `shadow_hgc/prototype/coverage_medoids.py`
- Create: `shadow_hgc/anchors/source_anchors.py`
- Test: `tests/test_coverage_medoids_budget.py`
- Test: `tests/test_coverage_medoids_class_coverage.py`
- Test: `tests/test_coverage_medoids_no_val_test_labels.py`
- Test: `tests/test_source_anchors_schema_preserved.py`
- Test: `tests/test_source_anchors_train_label_only.py`
- Test: `tests/test_anchor_residual_decomposition.py`

- [ ] Add opt-in coverage medoid selection utilities with class-wise budgets.
- [ ] Add opt-in source anchor scoring and residual decomposition utilities.
- [ ] Keep exposed node and edge types schema-preserving.
- [ ] Run all medoid and source anchor tests.

### Task 5: Teacher and KD

**Files:**
- Create: `shadow_hgc/teacher/train_teacher.py`
- Create: `shadow_hgc/teacher/cache.py`
- Create: `shadow_hgc/models/distill_losses.py`
- Test: `tests/test_teacher_cache_shapes.py`
- Test: `tests/test_kd_loss_no_nan.py`
- Test: `tests/test_teacher_train_labels_only.py`

- [ ] Add lightweight teacher cache and train-label-only metadata.
- [ ] Add KD and optional embedding distillation losses.
- [ ] Run teacher/KD tests.

### Task 6: SOTA Configs and Scripts

**Files:**
- Create: `configs/sota/acm.yaml`
- Create: `configs/sota/dblp.yaml`
- Create: `configs/sota/imdb.yaml`
- Create: `configs/sota/ogbn_arxiv.yaml`
- Create: `configs/sota/ogbn_products.yaml`
- Create: `scripts/run_sota_common.py`
- Create: `scripts/run_sota_small.py`
- Create: `scripts/run_sota_medium.py`
- Create: `scripts/run_sota_diagnostics.py`
- Create: `scripts/run_sota_stage.py`

- [ ] Add S0-S4 variant metadata and output fields.
- [ ] Reuse existing `run_shadow_hgc_experiment` and LAD output conventions.
- [ ] Keep diffusion disabled in SOTA scripts.
- [ ] Write JSON logs, CSV tables, and Markdown reports.

### Task 7: Verification, Experiments, Report, Push

**Files:**
- Create: `experiments/tables/sota_small_seed42.csv`
- Create: `experiments/tables/sota_medium_seed42.csv`
- Create: `experiments/tables/sota_diagnostics_seed42.csv`
- Create: `experiments/reports/sota_stage_summary.md`

- [ ] Run `C:\Users\slian\anaconda3\envs\pytorch\python.exe -m pytest tests -q`.
- [ ] Run `scripts/run_sota_stage.py` with seed 42.
- [ ] Check final checklist against the prompt.
- [ ] Commit all SOTA-stage changes.
- [ ] Push to GitHub.
