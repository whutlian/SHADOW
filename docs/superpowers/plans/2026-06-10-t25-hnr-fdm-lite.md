# T25 HNR-FDM-lite Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement and smoke-test T25 Shadow-HGC-SFT-HNR-FDM-lite without regressing the existing T24 SFT/R-1 paths.

**Architecture:** Add bounded HNR/FDM-lite modules under `shadow_hgc/sft/`, then wire them into new T25 runner scripts that reuse existing memmap SFT stores and full-node ratio accounting. All promoted rows pass forbidden-component guards; rows that miss performance/resource gates remain diagnostic and do not replace T24 references.

**Tech Stack:** Python, PyTorch, NumPy memmap, existing SFT block store, pytest, local conda env `C:\Users\slian\anaconda3\envs\pytorch\python.exe`.

---

### Task 1: Contract Tests First

**Files:**
- Create: `tests/test_t25_hnr_stats.py`
- Create: `tests/test_t25_fdm_lite.py`
- Create: `tests/test_t25_stage_contract.py`
- Create: `baselines/gcrd_tpami26.csv`

- [ ] Write failing tests for directed HNR counts, train-label-only leakage, FDM-lite budgets/candidate caps, full-node ratio accounting, forbidden promoted flags, and b=2 nonnegative shadow assignment.
- [ ] Run the new tests and confirm they fail because `shadow_hgc.sft.hnr`, `shadow_hgc.sft.fdm_lite`, and `shadow_hgc.sft.t25_contract` do not exist yet.

### Task 2: Streaming HNR Statistics

**Files:**
- Create: `shadow_hgc/sft/hnr.py`

- [ ] Implement `compute_streaming_hnr_stats()` over re-iterable edge streams using `edge_index[0]=src`, `edge_index[1]=dst`.
- [ ] Only labels for nodes in `train_rows` contribute to label histograms.
- [ ] Return degree, labeled support, same-label support, max affinity, entropy, homophily, quality, classwise robust HNR node weights, and H+/H0/H- strata.
- [ ] Run `tests/test_t25_hnr_stats.py`.

### Task 3: Bounded FDM-lite Selectors

**Files:**
- Create: `shadow_hgc/sft/fdm_lite.py`

- [ ] Implement deterministic random projection with block normalization for reduced signatures.
- [ ] Implement rare-class-protected class budgets and sqrt subclass budgets.
- [ ] Implement bounded weighted reservoir pools, subclass discovery on coreset only, and selectors: `sft_hnr_random`, `sft_hnr_fdm_herding`, `sft_hnr_fdm_kcenter`, `sft_hnr_fdm_hybrid`.
- [ ] Implement `shadow_b=1` and `shadow_b=2` assignment summaries with nonnegative weights.
- [ ] Run `tests/test_t25_fdm_lite.py`.

### Task 4: T25 Guards, Rows, and Scripts

**Files:**
- Create: `shadow_hgc/sft/t25_contract.py`
- Create: `scripts/run_t25_reddit_hnr_fdm.py`
- Create: `scripts/run_t25_products_recovery.py`
- Create: `scripts/run_t25_arxiv_sft_v4.py`
- Create: `scripts/run_t25_ultra_dryrun.py`
- Create: `scripts/run_t25_stage.py`

- [ ] Add method identifiers and required output columns.
- [ ] Enforce `--ultra-safe` forbids all-target cache, exact pairwise, dense P2, full edge_index GPU, E-by-d materialization, and full-class KMeans.
- [ ] Reuse T24 memmap SFT blocks and training loop for Reddit/products; add T25 selectors and diagnostics.
- [ ] Generate required CSV/JSON outputs and markdown summaries.
- [ ] Run `tests/test_t25_stage_contract.py`.

### Task 5: Experiments and Non-Regression

**Files:**
- Update generated files under `experiments/tables/` and `experiments/summaries/`

- [ ] Run targeted tests.
- [ ] Run Reddit T25 ratio sweep with local conda env. Promote no row unless it beats T24 reference gates.
- [ ] Run products recovery ladder or mark missing/blocked rows explicitly if local cache/data blocks prevent a true run.
- [ ] Run arxiv teacher gate table; do not run arxiv condensation unless A1 >= 0.715 is achieved.
- [ ] Run ultra dry-run planner for papers100M/MAG240M.
- [ ] Run full tests and compare T24 behavior remains intact.

### Task 6: Review, Summary, Push

**Files:**
- Create: `experiments/summaries/t25_hnr_fdm_stage_summary.md`
- Update: `handoff.md`

- [ ] Write a detailed summary with changed files, flags/methods, tests, experiment rows, gates, missing exact GCRD values, and next server commands.
- [ ] Review git diff and ensure unrelated dirty files are not staged.
- [ ] Commit T25 changes and push to GitHub on a `codex/t25-hnr-fdm-lite` branch.
