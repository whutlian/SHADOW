# T2.1-SFT-NL++ Stage Summary

## What Changed

- Added true chunked/memmap preprop API with `X0/X1/X2/Xres`, typed demand, structure, manifest schema, and forbidden-signal flags.
- Added `SFTTableTeacherV2` with `sagn_lite`, `gamlp_lite`, and `residual_block_gated` modes plus class-aware losses including focal and sqrt-weighted CE.
- Added robust block-selection scoring (`acc + 0.2 * macro_f1 + 0.05 * class_coverage`) and products full-run guards.
- Added SFT block-signature recovery helpers and DBLP recovery-start table rows.

## Final Rows

| dataset | status | accuracy | macro_f1 | predicted_class_count | target_accuracy | target_passed | recovery_status | reason |
|---|---|---|---|---|---|---|---|---|
| acm | promoted | 0.9206798672676086 | 0.9211405118306478 | 3 | 0.93 | False | completed_diagnostic | validation_selected_and_safe_improved |
| dblp | promoted | 0.9426056146621704 | 0.9387593418359756 | 4 | 0.85 | True | started_diagnostic | validation_selected_and_safe_improved |
| imdb | promoted | 0.47158026695251465 | 0.3900045245885849 | 5 | 0.5 | False | completed_diagnostic | validation_selected_and_safe_improved |
| ogbn-arxiv | blocked_class_collapse | 0.6105796098709106 | 0.32606273433193567 | 27 | 0.64 | False | blocked_by_t21_fullgraph_gate | predicted_class_count<35 |
| ogbn-products | preprop_completed |  |  |  | 0.7 | False | blocked_by_t21_fullgraph_gate | full_edge_products_preprop_completed |

## Products Execution

| dataset | status | run_mode | accuracy | macro_f1 | full_edge_scans | total_cache_bytes | reason |
|---|---|---|---|---|---|---|---|
| ogbn-products | preprop_completed | full_edges |  |  | 2 | 940427136 | full_edge_products_preprop_completed |

## Scalability Dry Run

| dataset | cache_mode | total_cache_bytes | full_edge_scans | wall_time_category | server_recommended |
|---|---|---|---|---|---|
| ogbn-arxiv | all_target_rows | 143602864 | 6 | local_short | False |
| ogbn-products | all_target_rows | 2111062998 | 6 | local_long | False |
| ogbn-papers100M | train_target_only | 123498671072 | 6 | server_recommended | True |
| MAG240M | train_target_only | 130761289284 | 6 | server_recommended | True |

## Required Answers

1. Did ACM reach 0.93? No; current accuracy=0.9206798672676086.
2. Did DBLP move to recovery? Yes; fullgraph accuracy=0.9426056146621704.
3. What remains as DBLP gap? Fullgraph SFT is strong; compressed prototype/shadow SFT block-signature accuracy is not yet promoted.
4. Was IMDB B3 robustly retained? False; selected=["B0_self", "B1_typed", "B2_metapath", "B4_structure"].
5. Did IMDB reach 0.50? No; current accuracy=0.47158026695251465.
6. Was arxiv class collapse fixed? No; predicted_class_count=27.
7. Did arxiv reach 0.64 without forbidden signals? No; current accuracy=0.6105796098709106.
8. Did products full execution complete? No; status=preprop_completed. Full-edge preprop completed if status is `preprop_completed`, but SFT training/eval is still not a completed products full execution.
9. Did products beat 0.6689/macro baseline? No; accuracy=, macro_f1=.
10. Any promoted bounded/logit/KD/E*d rows? No in T2.1 generated tables; forbidden flags are explicitly false for promoted/reported rows.
11. paper100M dry-run: cache=123498671072, scans=6, server_recommended=True.
12. MAG240M dry-run: cache=130761289284, scans=6, server_recommended=True.
13. Eligible datasets for recovery: ACM, DBLP, IMDB by current fullgraph gate; DBLP is the immediate started diagnostic target.
14. Are all attachment gates satisfied? No: ACM 0.93, IMDB 0.50, arxiv class coverage/0.64, and full products training are not yet achieved locally.

## Artifacts

- `experiments/tables/t21_preprop_manifest_index_seed42.csv`
- `experiments/tables/t21_sft_block_selection_seed42.csv`
- `experiments/tables/t21_sft_fullgraph_seed42.csv`
- `experiments/tables/t21_products_full_execution_seed42.csv`
- `experiments/tables/t21_sft_condensation_recovery_seed42.csv`
- `experiments/tables/t21_scalability_dry_run_seed42.csv`
- `experiments/tables/t21_stage_summary_seed42.csv`
- CSV summary: `experiments\tables\t21_stage_summary_seed42.csv`
