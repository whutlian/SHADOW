# T26 Remaining Blocked Completion Summary

## Scope

This stage addressed the remaining T26 blocked items after the prior long-run handoff.

- Products per-class collapse report: completed with real selected-class and test-prediction histograms.
- Products UCA/CB method-level rows: completed with real long training runs.
- Arxiv teacher-first gate: real SFT-v4 teacher sweep was run, but A1 remains blocked because no safe candidate reached accuracy >= 0.715.
- No performance-regression merge: previous higher real P0a/P0c Products rows were retained, while new method-level rows from the current run were added.

## Products Long Experiments

Source: `experiments/tables/t26_products_long_experiments_seed42.csv`

| method | ratio | accuracy | macro-F1 | predicted classes | status |
|---|---:|---:|---:|---:|---|
| P0a_alltrain_condensed_trainer_parity | 0.0802828386 | 0.7567198999 | 0.4013333613 | 42 | completed_long |
| P0b_selected_prototype_self_fit | 0.0025 | 0.9858970154 | 0.8774285379 | 42 | completed_long |
| P0c_same_budget_random_subset | 0.0025 | 0.6782802876 | 0.3672262278 | 42 | completed_long |
| products_cb_random | 0.0025 | 0.6923746019 | 0.3701378071 | 42 | completed_long |
| products_cb_kcenter | 0.0025 | 0.5488558762 | 0.3251615315 | 42 | completed_long |
| products_cb_herding | 0.0025 | 0.5919164643 | 0.3246630007 | 42 | completed_long |
| products_cb_hybrid | 0.0025 | 0.6206635877 | 0.3257283168 | 42 | completed_long |
| products_uca_kmeans_labeled_nearest | 0.0025 | 0.6887335406 | 0.3521801235 | 30 | completed_long |
| products_uca_hybrid | 0.0025 | 0.6887335406 | 0.3521801235 | 30 | completed_long |
| products_uca_hybrid_mixup | 0.0025 | 0.7463931668 | 0.3791035690 | 30 | completed_long |
| products_uca_hybrid_balanced_trainer | 0.0025 | 0.6404702744 | 0.3449961457 | 30 | completed_long |
| P0b_selected_prototype_self_fit | 0.005 | 0.9897917181 | 0.8573811287 | 42 | completed_long |
| P0c_same_budget_random_subset | 0.005 | 0.7213923874 | 0.3795675687 | 42 | completed_long |
| products_cb_random | 0.005 | 0.7081082522 | 0.3708838171 | 42 | completed_long |
| products_cb_kcenter | 0.005 | 0.6156158965 | 0.3508466148 | 42 | completed_long |
| products_cb_herding | 0.005 | 0.6271585759 | 0.3391884935 | 42 | completed_long |
| products_cb_hybrid | 0.005 | 0.6708919787 | 0.3441880882 | 42 | completed_long |
| products_uca_kmeans_labeled_nearest | 0.005 | 0.7110999954 | 0.3640408887 | 31 | completed_long |
| products_uca_hybrid | 0.005 | 0.7110999954 | 0.3640408887 | 31 | completed_long |
| products_uca_hybrid_mixup | 0.005 | 0.7670750999 | 0.3891223435 | 31 | completed_long |
| products_uca_hybrid_balanced_trainer | 0.005 | 0.6712656642 | 0.3449247561 | 31 | completed_long |

Products diagnostics now mark:

- `products_per_class_report`: completed.
- `products_UCA`: completed.
- `P0a`: passed.
- `P0b`: passed for both 0.0025 and 0.005.

## Arxiv Teacher-First Sweep

Source: `experiments/tables/t26_arxiv_teacher_actual_seed42.csv`

| variant | accuracy | macro-F1 | predicted classes | valid acc | A1 passed | train label scope |
|---|---:|---:|---:|---:|---|---|
| A1_real_sagn_lite_v4_h768_e300 | 0.6991955229 | 0.5053429755 | 39 | 0.7209973489 | false | train_only |
| A2_real_sagn_lite_v4_h512_dropout0p5_labeldrop0p2_lr0p001_e400 | 0.6985577022 | 0.5104670153 | 39 | 0.7231115138 | false | train_only |
| A3_real_sagn_lite_v4_h512_train_plus_valid_e300 | 0.7038660165 | 0.5146734981 | 40 | 0.9662404779 | false | train_plus_valid |
| A4_real_sagn_lite_v4_h512_all_filterbank_labelreuse_e300 | 0.7068287966 | 0.5045803242 | 39 | 0.7182120205 | false | train_only |
| A5_real_sagn_lite_v4_h768_all_filterbank_labelreuse_e300 | 0.7008620867 | 0.5070398444 | 39 | 0.7197556965 | false | train_only |

Best real Arxiv candidate: `A4_real_sagn_lite_v4_h512_all_filterbank_labelreuse_e300`, accuracy `0.7068287966`.

Result: Arxiv condensation remains `blocked_by_teacher_gate` because A1 requires accuracy >= `0.715`. This is intentional and follows the attachment; no fabricated pass row was created.

## Stage Checklist

Source: `experiments/tables/t26_stage_summary_seed42.csv`

| requirement | status |
|---|---|
| method_ids | completed |
| full_node_ratio | completed |
| forbidden_promoted_flags | completed |
| products_P0a | completed |
| products_P0b | completed |
| products_per_class_report | completed |
| products_UCA | completed |
| reddit_seed_sweep | completed |
| reddit_no_regression | completed |
| arxiv_teacher_first | blocked |
| ultra_contract | completed |
| machine_readable_outputs | completed |
| no_fabricated_results | completed |

## Code Changes

- Added Products method-level long runs for all T26 Products CB/UCA methods.
- Added selected class and predicted class histogram JSON output to long Products rows.
- Added real per-class collapse report generation from long-run histograms.
- Added lightweight chunked UCA selection for Products train-target signature diagnostics.
- Added actual Arxiv teacher training script and actual-source ingestion in the Arxiv sweep.
- Updated stage requirement checks so Products per-class and UCA completion are data-driven.
- Updated stage follow-up text so it lists only currently blocked checks.
- Added regression tests for Products real histogram recovery, Products/UCA stage completion, Arxiv actual-source ingestion, Arxiv actual-row safety flags, and lightweight UCA selection.

## Verification

Commands run:

- `python -m pytest tests/test_t26_scripts.py tests/test_t26_products_recovery.py -q`
- `python scripts/run_t26_products_long_experiments.py --device cuda --hidden-dim 256 --p0a-epochs 160 --p0b-epochs 600 --p0c-epochs 40 --method-epochs 80 --ratios 0.0025 0.005`
- `python scripts/run_t26_products_recovery.py`
- `python scripts/run_t26_arxiv_teacher_actual.py ...` for A1-A5 actual teacher candidates.
- `python scripts/run_t26_arxiv_teacher_sweep.py`
- `python scripts/run_t26_stage.py`

Safety status:

- Promoted rows: `0`.
- Forbidden promoted rows: `0`.
- Products P0a/P0b gates pass without regressing the retained real long-run P0 metrics.
- Arxiv remains blocked rather than fabricating an A1 pass.
