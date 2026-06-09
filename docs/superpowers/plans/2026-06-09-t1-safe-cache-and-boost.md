# T1.1 Safe Cache and Boost Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Regenerate replay-auditable safe-row logit caches and run validation-only T1.1 logit-affinity boosters without promoting forbidden components.

**Architecture:** Add cache replay/index helpers and validation-only booster primitives under `shadow_hgc.logits`, then drive experiments through opt-in scripts. Only replay-verified safe caches enter boosters; missing historical all-target logits are reported as blocked.

**Tech Stack:** Python, PyTorch, NumPy, pytest, existing Shadow-HGC SFB-v2 and logit cache utilities.

---

### Task 1: Tests

- [x] Add cache replay, validation-only selection, Correct&Smooth, path correction, pseudo-label, ensemble, and forbidden-promotion tests.
- [x] Run the new tests and verify RED before implementation.
- [x] Run the new tests after implementation and verify GREEN.

### Task 2: Core Modules

- [x] Update `shadow_hgc/logits/io.py` with T1.1-compatible metadata and split file names.
- [x] Add `shadow_hgc/logits/replay.py` and `shadow_hgc/logits/index.py`.
- [x] Add `shadow_hgc/logits/correct_smooth.py`.
- [x] Add `shadow_hgc/logits/path_correct.py`.
- [x] Add `shadow_hgc/logits/pseudo_scap.py`.

### Task 3: Experiment Scripts

- [x] Add `scripts/run_t1_generate_safe_logit_caches.py`.
- [x] Add `scripts/run_t1_cache_replay_audit.py`.
- [x] Add `scripts/run_t1_logit_correct_safe.py`.
- [x] Add `scripts/run_t1_path_logit_correct_safe.py`.
- [x] Add `scripts/run_t1_pseudo_scap_safe.py`.
- [x] Add `scripts/run_t1_safe_logit_ensemble_safe.py`.
- [x] Add `scripts/run_t1_large_logit_affinity_dry_run.py`.
- [x] Update `scripts/run_t1_safe_cache_and_boost_stage.py` to run the full stage.

### Task 4: Experiments and Verification

- [x] Run `scripts/run_t1_safe_cache_and_boost_stage.py --seed 42 --epochs 80`.
- [x] Inspect replay and promoted-row artifacts.
- [x] Run `python -m pytest tests -q`.
- [x] Restore unrelated pytest log side effects.
- [ ] Commit and push.
