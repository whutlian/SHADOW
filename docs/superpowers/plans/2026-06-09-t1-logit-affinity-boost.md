# T1 Logit Affinity Boost Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement opt-in T1 low-dimensional logit/class-affinity boosters with strict non-regression and scalability gates.

**Architecture:** Add a `shadow_hgc.logits` subsystem for safe cache metadata/I/O, implement LogitCorrectLite, Pseudo-SCAP, and Safe Logit Ensemble over C-dimensional logits/probabilities only. The stage runner indexes available safe logits, blocks missing historical caches explicitly, runs toy/available-cache diagnostics, and never promotes rows with forbidden components or bounded-edge performance.

**Tech Stack:** Python, PyTorch, NumPy, local conda env `pytorch`, existing `HeteroGraphData`, destination-row normalization, pytest.

---

### Task 1: T1 Tests

**Files:**
- Create: `tests/test_logit_cache_roundtrip.py`
- Create: `tests/test_logit_cache_forbidden_flags.py`
- Create: `tests/test_logit_correct_no_test_leakage.py`
- Create: `tests/test_logit_correct_dense_vs_streaming.py`
- Create: `tests/test_pseudo_scap_train_override.py`
- Create: `tests/test_pseudo_scap_topk_sparse.py`
- Create: `tests/test_pseudo_scap_confidence_gate.py`
- Create: `tests/test_safe_logit_ensemble_nonnegative_weights.py`
- Create: `tests/test_safe_logit_ensemble_validation_gate.py`
- Create: `tests/test_t1_no_forbidden_promotion.py`

- [x] Write tests against wished-for APIs and verify they fail before implementation.
- [x] Keep tests small and deterministic with toy logits/edges.

### Task 2: Logit Cache Infrastructure

**Files:**
- Create: `shadow_hgc/logits/__init__.py`
- Create: `shadow_hgc/logits/metadata.py`
- Create: `shadow_hgc/logits/io.py`
- Create: `shadow_hgc/logits/cache.py`
- Create: `shadow_hgc/logits/utils.py`

- [x] Implement `LogitCacheMeta`, `save_logits_cache`, `load_logits_cache`, and promotion safety checks.
- [x] Support `.npy` split arrays and optional memmap all-target logits.

### Task 3: T1 Boosters

**Files:**
- Create: `shadow_hgc/logits/correct_lite.py`
- Create: `shadow_hgc/features/pseudo_scap.py`
- Create: `shadow_hgc/features/pseudo_scap_io.py`
- Create: `shadow_hgc/logits/ensemble.py`
- Create: `shadow_hgc/logits/calibration.py`

- [x] Implement destination-row logit/prob smoothing and train-label-only correction.
- [x] Implement confidence-gated pseudo labels, target-target Pseudo-SCAP, top-k sparse storage helpers, prior centering stats.
- [x] Implement nonnegative safe logit ensemble and validation promotion gate.

### Task 4: Scripts and Stage Runner

**Files:**
- Create: `scripts/run_t1_logit_correct.py`
- Create: `scripts/run_t1_pseudo_scap.py`
- Create: `scripts/run_t1_safe_logit_ensemble.py`
- Create: `scripts/run_t1_logit_affinity_stage.py`

- [x] Emit required CSV/MD artifacts.
- [x] Mark historical rows without safe logits as `blocked_missing_safe_logit_cache` instead of fabricating logits.
- [x] Log medium scalability fields and dry-run estimates.

### Task 5: Verification and Publish

- [x] Run `python -m pytest tests -q`.
- [x] Run `scripts/run_t1_logit_affinity_stage.py --seed 42`.
- [x] Check final report answers all 10 requested questions.
- [ ] Commit and push to `origin/main`.
