# T2-SFT-NL: No-Logits Scalable Fullgraph Teacher Stage Summary

## What Changed

- Added `shadow_hgc/preprop/*` chunked/memmap preprop modules with manifest, block stats, and resource schema.
- Added no-logits `SFTTableTeacher` with `sagn_lite` and `gamlp_lite` modes.
- Added validation-only T2 safe block selection and runner scripts.
- Added promoted-row guards for logits, teacher logits/KD, dense P2, bounded edges, diffusion legacy, fullgraph edge backprop, and E x d materialization.
- Kept T1 logit code as historical artifact only; T2 scripts do not consume logit caches or propagated logits.

## Final Dataset Results

| dataset | status | accuracy | macro_f1 | predicted_class_count | primary_target | primary_target_passed | delta_acc_vs_safe | selected_blocks | reason |
|---|---|---|---|---|---|---|---|---|---|
| acm | promoted | 0.9206798672676086 | 0.9211405118306478 | 3 | 0.93 | False | 0.005193867267608621 | ["B0_self", "B1_typed", "B2_metapath", "B3_lad_scap"] | validation_selected_and_safe_improved |
| dblp | promoted | 0.9426056146621704 | 0.9387593418359756 | 4 | 0.85 | True | 0.10563361466217036 | ["B0_self", "B1_typed", "B2_metapath", "B3_lad_scap"] | validation_selected_and_safe_improved |
| imdb | promoted | 0.47158026695251465 | 0.3900045245885849 | 5 | 0.45 | True | 0.04747026695251466 | ["B0_self", "B1_typed", "B2_metapath", "B4_structure"] | validation_selected_and_safe_improved |
| ogbn-arxiv | blocked_class_collapse | 0.6105796098709106 | 0.32606273433193567 | 27 | 0.66 | False | 0.013805609870910618 | ["B0_self", "B1_typed", "B3_lad_scap"] | predicted_class_count<35 |
| ogbn-products | blocked_resource_guard |  |  |  | 0.7 | False |  | [] | products full T2 SFT skipped locally; use --run-products-full after dry-run |

## Kept Blocks

| dataset | block_group | branch_valid_acc | branch_test_acc_debug | gate_value |
|---|---|---|---|---|
| acm | B1_typed | 0.8618784546852112 | 0.8578848242759705 | 0.5135835409164429 |
| acm | B2_metapath | 0.90055251121521 | 0.8923512697219849 | 0.5075643658638 |
| acm | B3_lad_scap | 0.939226508140564 | 0.9206798672676086 | 0.5415382385253906 |
| dblp | B1_typed | 0.7448559403419495 | 0.7387323975563049 | 0.48465263843536377 |
| dblp | B2_metapath | 0.9423868060112 | 0.9492957592010498 | 0.5103060603141785 |
| dblp | B3_lad_scap | 0.9506173133850098 | 0.9426056146621704 | 0.5124279260635376 |
| imdb | B1_typed | 0.45255473256111145 | 0.446908175945282 | 0.35110411047935486 |
| imdb | B2_metapath | 0.4781021773815155 | 0.46002498269081116 | 0.17308862507343292 |
| imdb | B4_structure | 0.48905110359191895 | 0.47158026695251465 | 0.14528152346611023 |
| ogbn-arxiv | B1_typed | 0.5429041385650635 | 0.5235685110092163 | 0.33655062317848206 |
| ogbn-arxiv | B3_lad_scap | 0.6250545382499695 | 0.6105796098709106 | 0.20645399391651154 |

## Dropped Blocks

| dataset | block_group | branch_valid_acc | branch_test_acc_debug | drop_reason |
|---|---|---|---|---|
| acm | B4_structure | 0.9337016344070435 | 0.9093484282493591 | dropped_by_validation |
| dblp | B4_structure | 0.9506173133850098 | 0.9352112412452698 | dropped_by_validation |
| imdb | B3_lad_scap | 0.45985400676727295 | 0.497813880443573 | dropped_by_validation |
| ogbn-arxiv | B4_structure | 0.6170005798339844 | 0.5929057598114014 | dropped_by_validation |

## Preprop Manifest Status

| dataset | status | num_blocks | total_cache_bytes | full_edge_scans | uses_logits_as_input | reason |
|---|---|---|---|---|---|---|
| acm | completed | 8 | 6195200 | 10 | False | completed |
| dblp | completed | 5 | 5192960 | 11 | False | completed |
| imdb | completed | 6 | 7575552 | 8 | False | completed |
| ogbn-arxiv | completed | 3 | 65027712 | 2 | False | completed |
| ogbn-products | blocked_resource_guard | 0 | 0 | 0 | False | products full preprop skipped locally; dry-run covers resources |

## Scalability Dry-Run

| dataset | cache_mode | total_cache_bytes | full_edge_scans | wall_time_category | server_recommended |
|---|---|---|---|---|---|
| ogbn-arxiv | all_target_rows | 121926960 | 6 | local_short | False |
| ogbn-products | all_target_rows | 1797587286 | 6 | local_long | False |
| ogbn-papers100M | train_target_only | 109282996704 | 6 | server_recommended | True |
| MAG240M | train_target_only | 115177076036 | 6 | server_recommended | True |

## Condensation Recovery Gate

| dataset | recovery_row | fullgraph_accuracy | status | reason |
|---|---|---|---|---|
| acm | identity_condensed_sft_replay | 0.9206798672676086 | completed_diagnostic | identity replay of the validation-selected T2 SFT table teacher; diagnostic only |
| acm | prototype_oracle_sft_block_signature | 0.9206798672676086 | eligible_not_run | fullgraph T2 gate passed; compressed SFT block-signature recovery must be launched separately |
| acm | shadow_condensed_sft_block_signature | 0.9206798672676086 | eligible_not_run | fullgraph T2 gate passed; compressed SFT block-signature recovery must be launched separately |
| dblp | identity_condensed_sft_replay | 0.9426056146621704 | completed_diagnostic | identity replay of the validation-selected T2 SFT table teacher; diagnostic only |
| dblp | prototype_oracle_sft_block_signature | 0.9426056146621704 | eligible_not_run | fullgraph T2 gate passed; compressed SFT block-signature recovery must be launched separately |
| dblp | shadow_condensed_sft_block_signature | 0.9426056146621704 | eligible_not_run | fullgraph T2 gate passed; compressed SFT block-signature recovery must be launched separately |
| imdb | identity_condensed_sft_replay | 0.47158026695251465 | completed_diagnostic | identity replay of the validation-selected T2 SFT table teacher; diagnostic only |
| imdb | prototype_oracle_sft_block_signature | 0.47158026695251465 | eligible_not_run | fullgraph T2 gate passed; compressed SFT block-signature recovery must be launched separately |
| imdb | shadow_condensed_sft_block_signature | 0.47158026695251465 | eligible_not_run | fullgraph T2 gate passed; compressed SFT block-signature recovery must be launched separately |
| ogbn-arxiv | recovery_gate | 0.6105796098709106 | blocked_by_t2_fullgraph_gate | predicted_class_count<35 |
| ogbn-products | recovery_gate |  | blocked_by_t2_fullgraph_gate | products full T2 SFT skipped locally; use --run-products-full after dry-run |

## Required Answers

1. Were logits completely removed from promoted signals? Yes. T2 rows set `uses_logits_as_input=false`, `uses_teacher_logits=false`, `uses_kd=false`; no promoted T2 row consumed logit caches or propagated logits.
2. Did T2 improve ACM beyond 0.93? No.
3. Did T2 recover DBLP beyond 0.85? Yes.
4. Did T2 improve IMDB beyond 0.45? Yes.
5. Did T2 improve arxiv beyond 0.66 without logits/diffusion/P2? No.
6. Did T2 improve products beyond 0.70 or macro-F1 beyond LAD baseline? No.
7. Which blocks were kept/dropped by validation? See `Kept Blocks` and `Dropped Blocks` tables above.
8. Which blocks hurt and why? Dropped rows are marked `dropped_by_validation`; the current script treats validation accuracy/macro-F1 regression as the reason.
9. Did any promoted row use bounded edges? No; T2 promotion guard rejects `uses_bounded_edges=true`.
10. Did any promoted row materialize `E x d`? No; T2 preprop and promotion rows record `uses_e_by_d_materialization=false`.
11. Are paper100M/MAG240M dry-runs still feasible? paper100M server_recommended=True; MAG240M server_recommended=True.
12. Which datasets are eligible for condensation recovery? acm, dblp, imdb See `t2_condensation_recovery_seed42.csv` for identity replay and eligible-not-run prototype/shadow rows.
13. If no dataset improves, is the bottleneck data/schema, feature strength, or model capacity? For rows below safe baselines, the immediate bottleneck is feature-strength/model-capacity of the no-logits table teacher; products is additionally gated by local scalability.

## Artifacts

- `experiments/tables/t2_preprop_manifest_index_seed42.csv`
- `experiments/tables/t2_sft_safe_block_selection_seed42.csv`
- `experiments/tables/t2_sft_fullgraph_seed42.csv`
- `experiments/tables/t2_sft_scalability_dry_run_seed42.csv`
- `experiments/tables/t2_condensation_recovery_seed42.csv`
- `experiments/tables/t2_sft_stage_summary_seed42.csv`
