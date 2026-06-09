# Fullgraph Parity Condensation Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a seed-42 fullgraph parity and condensation recovery stage that diagnoses schema completeness, fullgraph ceilings, identity/oracle gaps, and valid recovery candidates without changing the frozen Shadow-HGC-R-1 default path.

**Architecture:** Add explicit opt-in scripts and small diagnostic modules that reuse the existing SeHGNNLite target trainer, LAD diagnostics, pipeline summaries, audit gates, and compiled demand head. Keep default `run_small.py`, `run_medium.py`, toy, and ultra dry-run paths unchanged.

**Tech Stack:** Python, PyTorch in `C:\Users\slian\anaconda3\envs\pytorch`, pytest, existing `shadow_hgc` data/model/pipeline modules.

---

### Task 1: Required Field And Row Validation Tests

**Files:**
- Create: `tests/test_fullgraph_parity_outputs_required_fields.py`
- Create: `tests/test_sota_row_validity_gates.py`
- Create: `tests/test_no_diffusion_promoted_path.py`

- [ ] Write a test that a fullgraph parity row contains all prompt-required fields.
- [ ] Write a test that fullgraph parity rows missing hashes are invalid.
- [ ] Write a test that promoted rows using diffusion or products P2 LAD are rejected.
- [ ] Implement reusable row utilities in `shadow_hgc/audit/parity.py`.
- [ ] Run the three tests and verify they pass.

### Task 2: Schema Alignment Audit

**Files:**
- Create: `scripts/run_schema_alignment_audit.py`
- Modify: `shadow_hgc/data/small.py`
- Create/extend: `shadow_hgc/data/schema_audit.py`
- Test: `tests/test_schema_alignment_audit_dblp_full_schema.py`

- [ ] Write a test that `full_schema` DBLP audit reports author-paper/paper-author and paper-term/paper-venue availability when present in processed data.
- [ ] Add a non-default `load_processed_small_dataset_full_schema()` helper that preserves all processed edge stores while leaving the default loader unchanged.
- [ ] Implement schema, split, feature, label, and schema hash helpers.
- [ ] Write schema alignment CSV/markdown reports for ACM/DBLP/IMDB and medium OGB rows.
- [ ] Run the schema alignment audit script with seed 42.

### Task 3: Fullgraph Parity

**Files:**
- Create: `scripts/run_fullgraph_parity.py`
- Extend: `shadow_hgc/train/sehgnn_lite_target.py`
- Test: `tests/test_fullgraph_parity_outputs_required_fields.py`

- [ ] Implement fullgraph rows for ACM/DBLP/IMDB SeHGNNLite current/tuned candidates.
- [ ] Implement medium table-teacher rows from existing no-diffusion diagnostics plus calibrated variants as guarded diagnostics.
- [ ] Log all required hashes, split counts, schema counts, runtime, and gate decisions.
- [ ] Write `experiments/tables/fullgraph_parity_seed42.csv` and report.
- [ ] Run with seed 42.

### Task 4: Identity And Oracle Condensation Gap Decomposition

**Files:**
- Create: `scripts/run_identity_condensation_audit.py`
- Create: `shadow_hgc/eval/gap_decomposition.py`
- Test: `tests/test_identity_condensation_gap_decomposition.py`

- [ ] Write a test that gap labels classify fullgraph-blocked, condensed-path inconsistent, prototype bottleneck, shadow bottleneck, and head bottleneck.
- [ ] Implement gap decomposition from existing fullgraph, LAD diagnostics, clean small, and medium recovery summaries.
- [ ] Materialize identity/prototype/oracle rows from existing exact/full-demand diagnostics when available; use status rows when missing.
- [ ] Write `experiments/tables/identity_condensation_audit_seed42.csv` and report.

### Task 5: Compiled Block Stats And KD V2 Logs

**Files:**
- Modify: `shadow_hgc/models/compiled_demand.py`
- Modify: `shadow_hgc/models/distill_losses.py`
- Modify: `shadow_hgc/teacher/kd_v2.py`
- Test: `tests/test_compiled_block_stats_fit_freeze.py`
- Test: `tests/test_compiled_block_stats_same_for_condensed_and_full.py`
- Test: `tests/test_kd_v2_gate_requires_teacher_quality.py`
- Test: `tests/test_kd_v2_logs_ce_and_kd_losses.py`
- Test: `tests/test_kd_v2_keeps_hard_ce.py`

- [ ] Add explicit `apply_block_stats` and `freeze_block_stats` helpers with richer metadata.
- [ ] Ensure stats metadata logs block names, dims, mean norms, std min/max, fit rows, and frozen state.
- [ ] Add KD v2 training-log helpers for CE warmup and separated CE/KD losses.
- [ ] Run targeted tests.

### Task 6: Candidate Recovery Scripts And Stage Runner

**Files:**
- Create: `scripts/run_fullgraph_parity_stage.py`
- Create: `experiments/reports/fullgraph_parity_condensation_recovery_summary.md`
- Optionally create candidate output scripts/tables for ACM, DBLP, IMDB, arxiv, products.

- [ ] Orchestrate schema audit, fullgraph parity, identity audit, gated candidate rows, and final summary.
- [ ] For ACM, run tuned clean SeHGNNLite ratios 0.096/0.12/0.15 if fullgraph gate passes.
- [ ] For DBLP/IMDB, mark condensation SOTA rows blocked unless fullgraph/schema gate passes.
- [ ] For arxiv/products, run only no-diffusion LAD reference/tuned rows, no P2 LAD and no diffusion.
- [ ] Write all required tables/reports and blocked/pass decisions.

### Task 7: Verification, Commit, Push

**Files:**
- Output: `experiments/tables/*parity*seed42.csv`
- Output: `experiments/reports/fullgraph_parity_condensation_recovery_summary.md`

- [ ] Run all required scripts using `C:\Users\slian\anaconda3\envs\pytorch\python.exe`.
- [ ] Run `C:\Users\slian\anaconda3\envs\pytorch\python.exe -m pytest tests -q`.
- [ ] Rebuild the final summary with pytest result.
- [ ] Verify the prompt checklist line by line.
- [ ] Commit and push to `origin/main`.
