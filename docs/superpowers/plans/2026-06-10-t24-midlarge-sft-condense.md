# T24 Mid/Large SFT-Condense Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the T24 opt-in medium/large SFT condensation stage for ogbn-arxiv, ogbn-products, Reddit, fixed full-node ratio reporting, and ultra dry-run accounting.

**Architecture:** Keep Shadow-HGC-R-1 default behavior frozen and add T24 as opt-in wrappers, scripts, reports, and tests. Reuse T23/T22 memmap SFT blocks and lazy table training where possible; record blocked/unavailable data explicitly rather than fabricating promoted results.

**Tech Stack:** Python, PyTorch, fp16 memmaps, existing `shadow_hgc` preprop/SFT utilities, CSV/JSON/Markdown outputs.

---

### Task 1: Ratio Policy And Safety

**Files:**
- Create: `shadow_hgc/ratio/scale_bucket.py`
- Create: `shadow_hgc/ratio/__init__.py`
- Create: `scripts/run_t24_bucket_ratio_table.py`
- Test: `tests/test_ratio_scale_bucket_policy.py`
- Test: `tests/test_full_node_ratio_accounting.py`
- Test: `tests/test_bucket_ratio_tables.py`
- Test: `tests/test_t24_forbidden_components.py`

- [ ] Implement deterministic medium/large/ultra full-node ratio presets.
- [ ] Implement full-node ratio accounting from condensed node counts.
- [ ] Implement promoted-row safety validation for T24 forbidden flags.
- [ ] Generate fixed bucket and sweep ratio tables.

### Task 2: T24 Arxiv SFT-v4

**Files:**
- Create: `shadow_hgc/preprop/filter_bank_v4.py`
- Create: `shadow_hgc/preprop/label_reuse_v3.py`
- Modify: `shadow_hgc/models/sft_teacher_v3.py`
- Modify: `shadow_hgc/train/lazy_sft_memmap.py`
- Create: `scripts/run_t24_arxiv_sft_v4.py`
- Test: `tests/test_arxiv_filter_bank_v4_no_e_by_d.py`
- Test: `tests/test_labelreuse_v3_train_only.py`
- Test: `tests/test_sagn_v4_forward_shapes.py`

- [ ] Add v4 block names and CLI-compatible wrappers.
- [ ] Add train-label-only LabelReuse v3 aliases.
- [ ] Add `sagn_lite_v4` and `gamlp_lite_v4` model aliases with diagnostics.
- [ ] Produce arxiv seed42 table and summary.

### Task 3: Products SFT Signature And Recovery

**Files:**
- Create: `shadow_hgc/sft/signature_cache.py`
- Create: `shadow_hgc/sft/products_recovery.py`
- Create: `scripts/run_t24_products_sft_recovery.py`
- Test: `tests/test_products_sft_signature_cache.py`
- Test: `tests/test_products_recovery_no_proxy_promotion.py`

- [ ] Extract train-target SFT signatures from memmap blocks with train-row stats.
- [ ] Write required memmap signature cache metadata.
- [ ] Run streaming recovery rows for the requested full-node ratios.
- [ ] Prevent proxy or bounded-edge rows from being promoted.

### Task 4: Reddit Onboarding

**Files:**
- Create: `shadow_hgc/data/reddit.py`
- Create: `scripts/run_t24_reddit_sft.py`
- Create: `scripts/run_t24_reddit_condense.py`
- Test: `tests/test_reddit_loader_basic.py`
- Test: `tests/test_reddit_preprop_memmap.py`

- [ ] Add DGL RedditDataset loader when DGL is installed.
- [ ] Add cached processed fallback loader.
- [ ] Generate fullgraph and condensation status tables with resource fields.

### Task 5: Stage Runner And Reports

**Files:**
- Create: `scripts/run_t24_midlarge_stage.py`
- Create: `scripts/t24_common.py`
- Create: `configs/t24_midlarge_sft.yaml`
- Output: `experiments/reports/t24_stage_summary.md`
- Output: T24 CSV/JSON tables listed in the prompt.

- [ ] Orchestrate arxiv, products, Reddit, ratio tables, resources, ablations, and ultra dry-run.
- [ ] Answer all required T24 summary questions.
- [ ] Run `pytest tests -q` with local conda `pytorch`.
- [ ] Commit and push.
