# Non-Regression SFB Repair Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Repair the opt-in SFB/SFB-v2 path so it cannot replace or corrupt historical Shadow-HGC strong paths, and produce gated seed-42 diagnostics/reports.

**Architecture:** Keep default Shadow-HGC-R-1 unchanged. Add diagnostics and safe fusion modules around existing loaders, SFB-v2 feature blocks, clean S1/ R+/ LAD reference artifacts, and enforce non-regression gates in a stage runner. Products performance rows are only allowed after self-only OGB parity passes; bounded-edge rows stay diagnostics only.

**Tech Stack:** Python, PyTorch, local conda env `pytorch`, OGB evaluator, pytest, existing `shadow_hgc` loaders/pipeline scripts.

---

### Task 1: Regression Tests

**Files:**
- Create: `tests/test_products_self_parity_schema.py`
- Create: `tests/test_dblp_demand_equivalence.py`
- Create: `tests/test_imdb_relation_inventory.py`
- Create: `tests/test_imdb_metapath_equivalence.py`
- Create: `tests/test_safe_block_fusion_gate_init.py`
- Create: `tests/test_safe_block_selection_non_regression.py`
- Create: `tests/test_safe_block_fusion_raw_logits.py`

- [ ] Write tests for OGB products label/output/schema guards, DBLP demand equivalence under destination-row alpha, IMDB keyword relation inventory and clean metapath block equivalence, and safe fusion gate/non-regression behavior.
- [ ] Run the new tests and confirm they fail because modules/behaviors are missing.

### Task 2: Diagnostics and Safe Fusion Modules

**Files:**
- Create: `shadow_hgc/diagnostics/demand_equivalence.py`
- Create: `shadow_hgc/diagnostics/imdb_inventory.py`
- Create: `shadow_hgc/models/safe_block_fusion.py`
- Create: `shadow_hgc/training/safe_block_selection.py`

- [ ] Implement relation-demand comparison metrics and direction/alpha diagnostics.
- [ ] Implement IMDB relation inventory plus metapath equivalence helpers reusing existing full-schema loader and metapath table provider.
- [ ] Implement `SafeBlockFusionClassifier` with raw logits and non-self `raw_gate=-8.0`.
- [ ] Implement validation-gated block selection that drops noisy blocks and keeps useful blocks.

### Task 3: Stage Scripts

**Files:**
- Create: `scripts/debug_products_self_parity.py`
- Create: `scripts/reproduce_historical_safe_rows.py`
- Create: `scripts/debug_dblp_demand_equivalence.py`
- Create: `scripts/debug_imdb_relation_inventory.py`
- Create: `scripts/run_nonregression_sfb_repair_stage.py`

- [ ] Products script trains raw-feature MLP with OGB evaluator and writes required JSON/CSV diagnostics; fail loudly on schema issues.
- [ ] Historical reproduction script materializes seed-42 safe rows from existing logs/scripts and checks tolerance.
- [ ] DBLP/IMDB scripts write required equivalence/inventory artifacts.
- [ ] Stage runner executes phases in order, stops downstream gated phases, and writes final summary sections.

### Task 4: Experiments and Reports

**Files:**
- Create required `experiments/tables/*nonregression*` and related CSV files.
- Create required `experiments/reports/*nonregression*` and related MD files.

- [ ] Run `python -m pytest tests -q` in conda env before experiments.
- [ ] Run stage runner with seed 42.
- [ ] Ensure all required artifacts exist, failures are explicitly marked with blocked reasons, and no forbidden component is promoted.

### Task 5: Completion

- [ ] Re-run full pytest.
- [ ] Run `git diff --check` and inspect `git status`.
- [ ] Commit only intended files and push to `origin/main`.
