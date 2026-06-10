# T26 Condensed Training Recovery UCA Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement T26 condensed training recovery, unlabeled coverage alignment, required diagnostics, experiment tables, summaries, and no-regression guards.

**Architecture:** Extend the existing T25 SFT stage instead of replacing it. Contracts and row validation live in `shadow_hgc/sft/t26_contract.py`; product diagnostics and UCA selection are pure tensor utilities; scripts load existing memmap/signature caches and only write machine-readable stage outputs.

**Tech Stack:** Python 3.11, PyTorch, NumPy, pytest, existing `scripts.t24_common` CSV/Markdown helpers, local conda environment `C:\Users\slian\anaconda3\envs\pytorch\python.exe`.

---

### Task 1: T26 Contract And Row Guards

**Files:**
- Create: `shadow_hgc/sft/t26_contract.py`
- Test: `tests/test_t26_contract.py`

- [ ] **Step 1: Write failing tests**

```python
from shadow_hgc.sft.t26_contract import T26_REQUIRED_FIELDS, make_t26_row, validate_t26_promoted_row


def test_t26_full_node_ratio_counts_target_and_shadow_only():
    row = make_t26_row(dataset="Reddit", method="reddit_tuned", requested_full_node_ratio=0.005, original_total_nodes=1000, target_prototypes=3, shadow_nodes=2, total_condensed_edges=7, accuracy=0.93, macro_f1=0.89)
    assert row["actual_full_node_ratio"] == 0.005
    assert row["total_condensed_nodes"] == 5
    assert row["ratio_mode"] == "full_node"


def test_t26_promoted_row_blocks_forbidden_components():
    row = make_t26_row(dataset="ogbn-products", method="products_uca_hybrid", requested_full_node_ratio=0.0025, original_total_nodes=1000, target_prototypes=3, shadow_nodes=0, total_condensed_edges=3, accuracy=0.75, macro_f1=0.40, promotion_status="promoted", uses_kd=True)
    assert row["promotion_status"] == "blocked_forbidden"
    assert "uses_kd" in row["failure_reason"]
    assert validate_t26_promoted_row(row)["valid"] is False


def test_t26_required_fields_cover_stage_outputs():
    for field in ["dataset", "stage", "method", "coverage_gap_l1", "p0a_passed", "uca_uses_valid_test_labels", "promotion_status", "failure_reason"]:
        assert field in T26_REQUIRED_FIELDS
```

- [ ] **Step 2: Verify red**

Run: `C:\Users\slian\anaconda3\envs\pytorch\python.exe -m pytest tests/test_t26_contract.py -q`

Expected: FAIL because `shadow_hgc.sft.t26_contract` is missing.

- [ ] **Step 3: Implement minimal contract**

Add constants for method IDs, forbidden flags, required fields, `_truthy`, `validate_t26_promoted_row`, `make_t26_row`, and `summarize_requirement_status`. Reuse `account_full_node_ratio` so T26 preserves full-node ratio semantics.

- [ ] **Step 4: Verify green**

Run: `C:\Users\slian\anaconda3\envs\pytorch\python.exe -m pytest tests/test_t26_contract.py -q`

Expected: PASS.

### Task 2: Product Budget, Diagnostics, And Self-Fit Utilities

**Files:**
- Create: `shadow_hgc/sft/products_recovery_t26.py`
- Test: `tests/test_t26_products_recovery.py`

- [ ] **Step 1: Write failing tests**

```python
import torch

from shadow_hgc.sft.products_recovery_t26 import (
    compute_p0_recovery_diagnostics,
    mixed_class_budget,
    nearest_prototype_oracle,
    per_class_collapse_report,
)


def test_mixed_class_budget_uses_floor_and_exact_total():
    labels = torch.tensor([0] * 200 + [1] * 20 + [2] * 5)
    rows = torch.arange(labels.numel())
    budget = mixed_class_budget(labels, rows, total_budget=30, ratio=0.0005, num_classes=3, seed=42)
    assert sum(budget.values()) == 30
    assert all(value >= 1 for value in budget.values())
    assert budget[2] >= 1


def test_nearest_prototype_oracle_reports_accuracy_without_training():
    train_sig = torch.tensor([[0.0], [0.1], [3.0], [3.2]])
    train_labels = torch.tensor([0, 0, 1, 1])
    selected_pos = torch.tensor([0, 2])
    eval_sig = torch.tensor([[0.05], [3.1]])
    eval_labels = torch.tensor([0, 1])
    out = nearest_prototype_oracle(train_sig, train_labels, selected_pos, eval_sig, eval_labels, metric="euclidean")
    assert out["prototype_oracle_acc"] == 1.0
    assert out["centroid_oracle_acc"] == 1.0


def test_per_class_report_detects_predicted_collapse():
    labels = torch.tensor([0, 0, 1, 1, 2, 2])
    selected = torch.tensor([0, 1, 2])
    pred = torch.tensor([0, 0, 0, 0, 1, 1])
    report = per_class_collapse_report(labels, selected, pred, num_classes=3)
    assert report[2]["selected_count"] == 0
    assert report[2]["predicted_count"] == 0
    assert report[2]["collapsed"] is True


def test_p0_diagnostics_encode_required_gates():
    diag = compute_p0_recovery_diagnostics(alltrain_acc=0.75, self_fit_acc=0.96, normalization_match=True, predicted_class_count=46, num_classes=47)
    assert diag["p0a_passed"] is True
    assert diag["p0b_passed"] is True
    assert diag["p0f_normalization_parity"] is True
    assert diag["p0e_predicted_class_collapse"] is False
```

- [ ] **Step 2: Verify red**

Run: `C:\Users\slian\anaconda3\envs\pytorch\python.exe -m pytest tests/test_t26_products_recovery.py -q`

Expected: FAIL because module is missing.

- [ ] **Step 3: Implement minimal utilities**

Implement deterministic mixed budget, vectorized nearest prototype oracle, per-class report rows, and P0 diagnostic booleans.

- [ ] **Step 4: Verify green**

Run: `C:\Users\slian\anaconda3\envs\pytorch\python.exe -m pytest tests/test_t26_products_recovery.py -q`

Expected: PASS.

### Task 3: Balanced Condensed Trainer Utilities

**Files:**
- Create: `shadow_hgc/sft/balanced_condensed_trainer.py`
- Test: `tests/test_t26_balanced_condensed_trainer.py`

- [ ] **Step 1: Write failing tests**

```python
import torch

from shadow_hgc.sft.balanced_condensed_trainer import balanced_batch_order, condensed_training_loss, within_class_sft_mixup


def test_balanced_batch_order_interleaves_classes():
    labels = torch.tensor([0, 0, 0, 1, 1, 2])
    rows = torch.arange(labels.numel())
    order = balanced_batch_order(rows, labels, seed=1)
    prefix = labels[order[:3]].tolist()
    assert sorted(prefix) == [0, 1, 2]


def test_condensed_training_loss_supports_label_smoothing_and_logit_adjustment():
    logits = torch.tensor([[4.0, 0.0], [0.0, 4.0]], requires_grad=True)
    labels = torch.tensor([0, 1])
    loss = condensed_training_loss(logits, labels, train_labels=labels, label_smoothing=0.05, logit_adjustment=True)
    loss.backward()
    assert torch.isfinite(loss)
    assert logits.grad is not None


def test_within_class_mixup_never_crosses_labels():
    x = torch.tensor([[0.0], [2.0], [10.0], [12.0]])
    y = torch.tensor([0, 0, 1, 1])
    mixed_x, mixed_y = within_class_sft_mixup(x, y, alpha=0.4, seed=4)
    assert mixed_x.shape == x.shape
    assert torch.equal(mixed_y, y)
    assert torch.all(mixed_x[:2] <= 2.0)
    assert torch.all(mixed_x[2:] >= 10.0)
```

- [ ] **Step 2: Verify red**

Run: `C:\Users\slian\anaconda3\envs\pytorch\python.exe -m pytest tests/test_t26_balanced_condensed_trainer.py -q`

Expected: FAIL because module is missing.

- [ ] **Step 3: Implement utilities**

Implement class-balanced ordering, CE loss with label smoothing and optional logit adjustment, and within-class mixup on SFT feature tensors.

- [ ] **Step 4: Verify green**

Run: `C:\Users\slian\anaconda3\envs\pytorch\python.exe -m pytest tests/test_t26_balanced_condensed_trainer.py -q`

Expected: PASS.

### Task 4: UCA Coverage Alignment

**Files:**
- Create: `shadow_hgc/sft/uca.py`
- Test: `tests/test_t26_uca.py`

- [ ] **Step 1: Write failing tests**

```python
import torch

from shadow_hgc.sft.uca import coverage_gap_metrics, select_uca_labeled_nearest


def test_uca_selection_uses_unlabeled_features_but_not_unlabeled_labels():
    signatures = torch.tensor([[0.0], [0.1], [5.0], [5.2], [9.0], [9.1]])
    labels_a = torch.tensor([0, 0, 1, 1, 0, 1])
    labels_b = torch.tensor([0, 0, 1, 1, 1, 0])
    train_rows = torch.tensor([0, 2])
    target_rows = torch.arange(6)
    sel_a, stats_a = select_uca_labeled_nearest(signatures, labels_a, train_rows, target_rows, budget=2, num_domains=3, seed=7)
    sel_b, stats_b = select_uca_labeled_nearest(signatures, labels_b, train_rows, target_rows, budget=2, num_domains=3, seed=7)
    assert torch.equal(sel_a, sel_b)
    assert stats_a["uca_uses_valid_test_labels"] is False
    assert stats_b["domain_hist_all"] == stats_a["domain_hist_all"]


def test_coverage_gap_metrics_report_l1_l2_and_unsupported_domains():
    out = coverage_gap_metrics(torch.tensor([2, 0, 2]), torch.tensor([1, 0, 3]))
    assert out["coverage_gap_l1"] > 0
    assert out["coverage_gap_l2"] > 0
    assert out["domains_without_train_support"] == 1
```

- [ ] **Step 2: Verify red**

Run: `C:\Users\slian\anaconda3\envs\pytorch\python.exe -m pytest tests/test_t26_uca.py -q`

Expected: FAIL because module is missing.

- [ ] **Step 3: Implement UCA**

Implement deterministic farthest-center domain sketch, all-target histogram, train-support histogram, labeled-nearest selection per domain, and gap metrics. The function must not read labels outside `train_rows`.

- [ ] **Step 4: Verify green**

Run: `C:\Users\slian\anaconda3\envs\pytorch\python.exe -m pytest tests/test_t26_uca.py -q`

Expected: PASS.

### Task 5: T26 Experiment Scripts And Summaries

**Files:**
- Create: `scripts/run_t26_products_recovery.py`
- Create: `scripts/run_t26_reddit_trainer_sweep.py`
- Create: `scripts/run_t26_arxiv_teacher_sweep.py`
- Create: `scripts/run_t26_ultra_contract_regression.py`
- Create: `scripts/run_t26_stage.py`
- Test: `tests/test_t26_scripts.py`

- [ ] **Step 1: Write failing tests**

```python
from pathlib import Path

from scripts.run_t26_arxiv_teacher_sweep import build_rows as build_arxiv_rows
from scripts.run_t26_ultra_contract_regression import build_rows as build_ultra_rows
from scripts.run_t26_stage import REQUIRED_OUTPUTS


def test_t26_required_outputs_are_declared():
    for path in [
        "experiments/tables/t26_stage_summary_seed42.csv",
        "experiments/summaries/t26_stage_summary.md",
        "experiments/tables/t26_products_recovery_diagnostics_seed42.csv",
        "experiments/tables/t26_products_uca_sweep_seed42.csv",
        "experiments/tables/t26_reddit_seed_trainer_mixup_sweep.csv",
        "experiments/tables/t26_arxiv_teacher_sweep_seed42.csv",
        "experiments/tables/t26_ultra_contract_regression_seed42.csv",
    ]:
        assert Path(path) in REQUIRED_OUTPUTS


def test_t26_arxiv_teacher_rows_block_condensation_until_a1():
    rows = build_arxiv_rows(seed=42)
    assert rows
    assert all(row["stage"] == "t26" for row in rows)
    assert any(row["condensation_status"] == "blocked_by_teacher_gate" for row in rows)


def test_t26_ultra_rows_keep_forbidden_flags_false():
    rows = build_ultra_rows(seed=42)
    assert rows
    for row in rows:
        assert row["stage"] == "t26"
        assert row["uses_all_target_cache"] is False
        assert row["uses_exact_pairwise"] is False
        assert row["uses_e_by_d_materialization"] is False
```

- [ ] **Step 2: Verify red**

Run: `C:\Users\slian\anaconda3\envs\pytorch\python.exe -m pytest tests/test_t26_scripts.py -q`

Expected: FAIL because scripts are missing.

- [ ] **Step 3: Implement scripts**

Products script writes diagnostics, per-class report, UCA sweep, and notes. Reddit script runs seed/trainer/mixup rows or diagnostic not-trained rows. Arxiv script wraps T25 teacher rows and blocks condensation until A1. Ultra script reuses T25 planner with T26 forbidden flags. Stage script aggregates all outputs and writes the final checklist summary.

- [ ] **Step 4: Verify green**

Run: `C:\Users\slian\anaconda3\envs\pytorch\python.exe -m pytest tests/test_t26_scripts.py -q`

Expected: PASS.

### Task 6: Stage Execution, Regression Verification, Commit, Push

**Files:**
- Generated: `experiments/tables/t26_stage_summary_seed42.csv`
- Generated: `experiments/summaries/t26_stage_summary.md`
- Generated: every T26 table and summary required by the prompt.

- [ ] **Step 1: Run T26 stage with local pytorch**

Run: `C:\Users\slian\anaconda3\envs\pytorch\python.exe scripts\run_t26_stage.py`

Expected: JSON line with `"status": "completed"` and all required output paths present.

- [ ] **Step 2: Run focused T26 tests**

Run: `C:\Users\slian\anaconda3\envs\pytorch\python.exe -m pytest tests/test_t26_contract.py tests/test_t26_products_recovery.py tests/test_t26_balanced_condensed_trainer.py tests/test_t26_uca.py tests/test_t26_scripts.py -q`

Expected: PASS.

- [ ] **Step 3: Run full regression suite**

Run: `C:\Users\slian\anaconda3\envs\pytorch\python.exe -m pytest -q`

Expected: PASS; no performance-promoted row may regress below its explicit gate.

- [ ] **Step 4: Check worktree and stage only T26 files**

Run: `git status --short`

Expected: T26 files plus pre-existing unrelated dirty files. Stage only T26 implementation, tests, docs, and generated T26 outputs.

- [ ] **Step 5: Commit and push main**

Run: `git commit -m "Add T26 condensed recovery stage"` then `git push origin main`.

Expected: `origin/main` advances with the T26 commit; unrelated dirty files remain unstaged.
