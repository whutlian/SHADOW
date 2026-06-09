# T1 Cache Entry Replay Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Regenerate replayable safe-row logit caches for DBLP, IMDB, ogbn-arxiv, and ogbn-products where feasible, then rerun T1.1 replay/boost summaries.

**Architecture:** Keep the T1 cache schema in `shadow_hgc/logits/io.py` unchanged. Add logits exposure at training outputs, add small helper functions in `scripts/t1_safe_common.py` to save historical/gate caches from existing row providers, and keep replay audit as the acceptance gate.

**Tech Stack:** Python, PyTorch, pytest, existing Shadow-HGC scripts, local conda `pytorch` environment.

---

### Task 1: Expose SeHGNNLite Target Logits

**Files:**
- Modify: `shadow_hgc/train/sehgnn_lite_target.py`
- Test: `tests/test_t1_historical_cache_entrypoints.py`

- [ ] Add a failing test that builds a tiny graph through the existing small-data loader path or asserts `SeHGNNTargetRun` has a `logits` field when constructed with summary, blocks, and logits.
- [ ] Run `python -m pytest tests/test_t1_historical_cache_entrypoints.py -q` and confirm the test fails because `SeHGNNTargetRun` has no logits field.
- [ ] Add `logits: torch.Tensor | None = None` to `SeHGNNTargetRun` and return `all_logits.detach().cpu()` from `train_fullgraph_sehgnn_lite` and `train_prototype_sehgnn_lite`.
- [ ] Rerun the test and confirm it passes.

### Task 2: Add Historical Cache Writer Helpers

**Files:**
- Modify: `scripts/t1_safe_common.py`
- Modify: `scripts/run_t1_generate_safe_logit_caches.py`
- Test: `tests/test_t1_historical_cache_entrypoints.py`

- [ ] Add failing tests for a helper that saves a replay cache and a gate-selection cache from an existing graph/logits pair using the current metadata format.
- [ ] Implement helper functions for SeHGNN caches and pipeline-summary cache rows.
- [ ] Keep forbidden component flags set from the historical source row and never mark forbidden rows as promoted.
- [ ] Rerun the helper tests.

### Task 3: Wire DBLP And IMDB Cache Generation

**Files:**
- Modify: `scripts/run_t1_generate_safe_logit_caches.py`

- [ ] For DBLP, call `run_shadow_hgc_experiment(..., return_logits=True)` on the R+ current-best ratio `0.065` and save historical/gate caches.
- [ ] For IMDB, call `train_prototype_sehgnn_lite` on S1 clean MAM/MDM/MKM ratio `0.05` and save historical/gate caches.
- [ ] Leave rows blocked only if the run fails or replay mismatch occurs later.

### Task 4: Wire Medium Cache Generation Where Feasible

**Files:**
- Modify: `shadow_hgc/pipeline/core.py`
- Modify: `scripts/run_t1_generate_safe_logit_caches.py`

- [ ] Add `return_logits` support to non-compiled pipeline output through an internal result object.
- [ ] Add compiled full-target inference only if the all-target table can be built without violating memory policy.
- [ ] If local medium generation cannot finish under resource limits, report the command and blocked reason explicitly instead of fabricating cache.

### Task 5: Rerun And Summarize

**Files:**
- Output: `experiments/tables/t1_safe_logit_cache_index_seed42.csv`
- Output: `experiments/tables/t1_cache_replay_audit_seed42.csv`
- Output: `experiments/tables/t1_safe_fullgraph_boost_summary_seed42.csv`
- Output: `experiments/reports/t1_safe_cache_and_boost_stage_summary.md`

- [ ] Run `C:\Users\slian\anaconda3\envs\pytorch\python.exe scripts\run_t1_safe_cache_and_boost_stage.py --seed 42 --epochs 80`.
- [ ] Run relevant pytest tests, then full pytest if runtime is acceptable.
- [ ] Update the summary report with available caches, replay deltas, boost outcomes, and remaining blocked rows with concrete reasons.
