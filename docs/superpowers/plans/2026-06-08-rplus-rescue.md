# Shadow-HGC-R+ Rescue Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement and evaluate the Shadow-HGC-R+ diagnostic and rescue sprint from `codex_next_stage_rplus_prompt.md`.

**Architecture:** Keep Shadow-HGC-R-1 as the default path and add R+ only through explicit configuration and script flags. The core pipeline gains optional rank diagnostics, adaptive shadow budgets/top-b assignment, multiscale target features, relation gates, coverage skeletons, logit-adjusted loss, and class-collapse diagnostics while preserving original schema and destination-row normalization.

**Tech Stack:** Python, PyTorch, pytest, existing `shadow_hgc` modules, local conda environment `C:\Users\slian\anaconda3\envs\pytorch\python.exe`.

---

### Task 1: R+ Unit-Test Surface

**Files:**
- Create: `tests/test_rank_diagnostics.py`
- Create: `tests/test_adaptive_shadow_budget.py`
- Create: `tests/test_topb_nonnegative_assignment.py`
- Create: `tests/test_diffusion_features.py`
- Create: `tests/test_metapath_features_schema_preserving.py`
- Create: `tests/test_relation_gate.py`
- Create: `tests/test_coverage_skeleton_policy.py`
- Create: `tests/test_logit_adjusted_loss.py`

- [ ] **Step 1: Add tests that define the expected R+ public APIs.**
- [ ] **Step 2: Run each new test file and confirm it fails because the API or behavior is missing.**

### Task 2: Diagnostics and Adaptive Shadows

**Files:**
- Create: `shadow_hgc/diagnostics/rank.py`
- Create: `shadow_hgc/diagnostics/reconstruction.py`
- Create: `shadow_hgc/shadows/adaptive.py`
- Modify: `shadow_hgc/shadows/assign.py`
- Modify: `shadow_hgc/graph/materialize.py`
- Modify: `shadow_hgc/pipeline/core.py`

- [ ] **Step 1: Implement finite rank diagnostics and reconstruction metrics from train-target demand matrices.**
- [ ] **Step 2: Implement deterministic rank-adaptive shadow allocation from effective ranks only.**
- [ ] **Step 3: Implement nonnegative weighted top-b assignment and materialization without negative edge weights.**
- [ ] **Step 4: Log requested/realized `M_r`, relation-wise `b_r`, and nested `diagnostics.rank`.**

### Task 3: Multiscale Features and Skeleton Policy

**Files:**
- Create: `shadow_hgc/features/diffusion.py`
- Create: `shadow_hgc/features/metapath.py`
- Create: `shadow_hgc/features/multiscale.py`
- Create: `shadow_hgc/skeleton/policy.py`
- Modify: `shadow_hgc/prototype/signatures.py`
- Modify: `shadow_hgc/skeleton/transition.py`
- Modify: `shadow_hgc/pipeline/core.py`

- [ ] **Step 1: Implement deterministic diffusion target blocks and optional high-pass target block.**
- [ ] **Step 2: Implement schema-preserving target meta-path feature blocks without adding edge types.**
- [ ] **Step 3: Append multiscale target blocks to `phi_tau` and to the prototype signature only when requested.**
- [ ] **Step 4: Add coverage skeleton selection without renormalizing retained transition mass.**

### Task 4: Model, Loss, and Collapse Diagnostics

**Files:**
- Create: `shadow_hgc/eval/class_collapse.py`
- Modify: `shadow_hgc/models/weighted_rel_linear.py`
- Modify: `shadow_hgc/models/losses.py`
- Modify: `shadow_hgc/config.py`
- Modify: `shadow_hgc/eval/tables.py`
- Modify: `scripts/run_small.py`
- Modify: `scripts/run_medium.py`
- Modify: `scripts/run_medium_ratio_sweep.py`

- [ ] **Step 1: Add positive softplus relation gates while defaulting gates off.**
- [ ] **Step 2: Add `sqrt_weighted_logit_adjusted` loss with train-label priors.**
- [ ] **Step 3: Add per-class support/accuracy, entropy, top predicted classes, and weighted-F1 diagnostics.**
- [ ] **Step 4: Make table builders ignore nested non-relation diagnostic keys.**

### Task 5: R+ Scripts and Required Artifacts

**Files:**
- Create: `scripts/run_rplus_diagnostics.py`
- Create: `scripts/run_imdb_rescue.py`
- Create: `scripts/run_medium_diffusion.py`
- Create: `scripts/run_rplus_regression.py`
- Create outputs under `experiments/tables`, `experiments/reports`, and `configs/stage5_selected.yaml`.

- [ ] **Step 1: Add seed42-only scripts for rank diagnostics, IMDB rescue, medium diffusion rescue, and ACM/DBLP regression.**
- [ ] **Step 2: Run `pytest tests -q` with the local pytorch conda Python.**
- [ ] **Step 3: Run the reduced seed42 experiment grids from the prompt, recording OOM/OOT if encountered.**
- [ ] **Step 4: Generate `rank_diagnostics_summary.md`, rescue summaries, baseline alignment, `rplus_rescue_summary.md`, `stage5_readiness_after_rplus.md`, and `configs/stage5_selected.yaml`.**
- [ ] **Step 5: Verify every checklist item in Section 14 of the prompt exists and is populated.**
