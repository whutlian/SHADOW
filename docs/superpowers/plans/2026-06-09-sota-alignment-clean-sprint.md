# SOTA Alignment Clean Sprint Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Align the SOTA branch with the frozen Shadow-HGC-R-1 invariants, gate invalid historical rows, run single-seed clean experiments, push the project, and write a detailed sprint summary.

**Architecture:** Add a read-only audit layer over existing JSON/CSV logs, then add clean feature builders for schema-default meta-paths, Path-LAD v2, two-hop LAD, teacher-demand herding, and KD v2 gates. Keep the default R-1 pipeline unchanged and wire new scripts as explicit sprint diagnostics.

**Tech Stack:** Python, PyTorch in `C:\Users\slian\anaconda3\envs\pytorch`, pytest, existing `shadow_hgc` modules and experiment JSON/CSV artifacts.

---

### Task 1: Hard Audit Gates

**Files:**
- Create: `shadow_hgc/audit/config_checks.py`
- Create: `shadow_hgc/audit/reporting.py`
- Create: `shadow_hgc/audit/schema_checks.py`
- Test: `tests/test_sota_config_checks.py`
- Test: `tests/test_invalid_rows_not_in_best_summary.py`
- Test: `tests/test_sehgnn_lite_required_blocks.py`

- [ ] Write tests that invalid SeHGNN/meta-path rows require `model_type=sehgnn_lite`, non-empty feature blocks, block dims, and block norm source.
- [ ] Write tests that KD rows require teacher quality fields and separate CE/KD diagnostics.
- [ ] Write tests that invalid rows are marked `invalid_config` and excluded from best-row summaries.
- [ ] Implement `validate_variant_config`, `assert_or_mark_invalid`, and best-row filtering.
- [ ] Run `pytest tests/test_sota_config_checks.py tests/test_invalid_rows_not_in_best_summary.py tests/test_sehgnn_lite_required_blocks.py -q`.

### Task 2: Schema Defaults And DBLP Audit

**Files:**
- Create: `shadow_hgc/features/metapath_schema.py`
- Create: `shadow_hgc/data/schema_audit.py`
- Create: `scripts/audit_dblp_schema.py`
- Test: `tests/test_dblp_schema_audit.py`

- [ ] Write tests that DBLP author classification requires target type `author` and APA availability.
- [ ] Implement schema-default meta-path block names for ACM, DBLP, IMDB, and no hetero defaults for OGB medium datasets.
- [ ] Implement DBLP schema audit rows and markdown report generation.
- [ ] Run `pytest tests/test_dblp_schema_audit.py -q`.

### Task 3: Path-LAD V2 And Two-Hop LAD

**Files:**
- Create: `shadow_hgc/features/path_lad_v2.py`
- Create: `shadow_hgc/features/two_hop_lad.py`
- Modify: `shadow_hgc/pipeline/core.py`
- Test: `tests/test_path_lad_v2_train_labels_only.py`
- Test: `tests/test_path_lad_v2_leave_one_out.py`
- Test: `tests/test_two_hop_lad_no_diffusion.py`

- [ ] Write tests that Path-LAD v2 ignores validation/test labels and logs row normalization, leave-one-out, hub clipping, and train-only usage.
- [ ] Write tests that two-hop LAD uses label histograms only and never high-dimensional feature diffusion.
- [ ] Implement Path-LAD v2 as a train-label-only wrapper around existing destination-row normalized Path-LAD.
- [ ] Implement sparse O(E*C) one/two-hop LAD for target-target graphs.
- [ ] Extend `_build_path_lad_blocks` to support `P2` for medium target-target relations.
- [ ] Run `pytest tests/test_path_lad_v2_train_labels_only.py tests/test_path_lad_v2_leave_one_out.py tests/test_two_hop_lad_no_diffusion.py -q`.

### Task 4: Teacher Herding, KD V2, And Losses

**Files:**
- Create: `shadow_hgc/prototype/teacher_demand_herding.py`
- Create: `shadow_hgc/teacher/kd_v2.py`
- Modify: `shadow_hgc/models/distill_losses.py`
- Modify: `shadow_hgc/models/losses.py`
- Test: `tests/test_teacher_demand_herding_budget.py`
- Test: `tests/test_kd_v2_gate.py`

- [ ] Write tests that teacher-demand herding respects total/class budgets and does not use boundary nodes without a valid teacher.
- [ ] Write tests that KD v2 rejects weak teacher validation accuracy, class collapse, and near-zero entropy collapse.
- [ ] Implement herding selector diagnostics.
- [ ] Implement KD v2 quality gate and separate CE/KD loss output.
- [ ] Add balanced softmax and class-balanced focal loss variants.
- [ ] Run `pytest tests/test_teacher_demand_herding_budget.py tests/test_kd_v2_gate.py -q`.

### Task 5: Clean Scripts And Reports

**Files:**
- Create: `scripts/run_sota_audit.py`
- Create: `scripts/run_fullgraph_backbone_audit.py`
- Create: `scripts/run_sota_clean_small.py`
- Create: `scripts/run_medium_no_diffusion_refine.py`
- Create: `scripts/run_teacher_herding_kd_gated.py`
- Create: `scripts/build_sota_alignment_clean_summary.py`

- [ ] Implement `run_sota_audit.py` as a read-only enrichment of existing and new SOTA logs.
- [ ] Implement fullgraph SeHGNN-lite/table teacher audit with pass/fail gate fields.
- [ ] Implement clean small rows for ACM, DBLP, and IMDB without KD, coverage medoid, source anchors, or diffusion.
- [ ] Implement medium no-diffusion refine rows with LAD reference, P2 LAD, fusion head, and balanced-softmax diagnostics.
- [ ] Implement optional herding/KD script that logs skipped rows unless gates pass.

### Task 6: Experiment Execution, Verification, Push

**Files:**
- Output: `experiments/tables/*_seed42.csv`
- Output: `experiments/reports/*_seed42.md`
- Output: `experiments/reports/sota_alignment_clean_sprint_summary.md`

- [ ] Run audit, fullgraph audit, DBLP schema audit, clean small, medium refine, and optional gated herding/KD using seed 42.
- [ ] Run `C:\Users\slian\anaconda3\envs\pytorch\python.exe -m pytest tests -q`.
- [ ] Build final summary with all rows, invalid reasons, promoted/dropped branches, changes, and pytest summary.
- [ ] Verify attachment checklist line by line against produced files.
- [ ] Commit and push to `origin/main`.
