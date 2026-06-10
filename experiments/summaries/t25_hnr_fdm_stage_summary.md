# T25 Shadow-HGC-SFT-HNR-FDM-lite Stage Summary

Last updated: 2026-06-10

## Implementation And Review Fixes

- Implemented streaming train-label-only HNR statistics in `shadow_hgc/sft/hnr.py`.
- Implemented bounded FDM-lite reducers, class/subclass budgets, candidate pools, selectors, and nonnegative b=2 assignment helper in `shadow_hgc/sft/fdm_lite.py`.
- Implemented T25 row schema, full-node ratio accounting, promoted-row forbidden guards, ultra-safe guard forcing, and metric-required promotion checks in `shadow_hgc/sft/t25_contract.py`.
- Added reusable SFT signature cache loading in `shadow_hgc/sft/signature_cache.py`, with compatibility checks for manifest directory, selected block list, train row count, and dtype.
- Added T25 experiment runners and summaries for Reddit, products, arxiv, ultra dry-run, and aggregate stage reporting.
- Added placeholder GCRD baseline CSV in `baselines/gcrd_tpami26.csv`; exact TPAMI 2026 values were not available locally and were not fabricated.

Reviewer issues addressed before commit:

- Shadow b1/b2 rows are now marked `shadow_materialization_not_trained` and `completed_streaming_diagnostic`; they are not promoted and no longer claim a trained shadow graph result.
- Ultra dry-run rows are not promoted as performance rows; they report resource-gate behavior only.
- Selector-only rows now use full target prototype budgets so `actual_full_node_ratio` matches the requested full-node ratio instead of mixing target-only and target+shadow accounting.
- Products row-order and mask-alignment diagnostics are computed from store/split properties instead of hardcoded as true.
- Promoted rows now require accuracy and macro-F1 metrics.

## Verification

Commands run with local conda env `C:\Users\slian\anaconda3\envs\pytorch\python.exe`:

```powershell
& 'C:\Users\slian\anaconda3\envs\pytorch\python.exe' -m pytest tests/test_t25_hnr_stats.py tests/test_t25_fdm_lite.py tests/test_t25_stage_contract.py -q
```

Result: `18 passed in 1.85s`.

```powershell
& 'C:\Users\slian\anaconda3\envs\pytorch\python.exe' -m pytest tests -q
```

Result before the reviewer fixes: `315 passed in 76.08s`. The T25 contract tests were rerun after the fixes.

## Key Experiment Results

Reddit best rows by ratio:

| Ratio | Best row | Accuracy | Macro-F1 | Predicted classes | Status |
|---:|---|---:|---:|---:|---|
| 0.10% | `sft_hnr_fdm_hybrid` | 0.9215841157567815 | 0.884890777869315 | 41 | not promoted |
| 0.25% | `current_sft_signature_shadow_b1` | 0.9212968780855609 | 0.8824121238133515 | 41 | diagnostic, shadow not materialized |
| 0.50% | `current_sft_signature_random` | 0.9233254941385562 | 0.885149317831537 | 41 | not promoted, below no-regression/T25 gate |
| 1.00% | `sft_hnr_fdm_shadow_b1` | 0.924564206595695 | 0.8881542321172001 | 41 | diagnostic, shadow not materialized |

Reddit 0.50% target was `>= 0.928` accuracy and `>= 0.890` macro-F1. No row met both, so no Reddit performance row was promoted.

Products best non-P0 rows by ratio:

| Ratio | Best row | Accuracy | Macro-F1 | Predicted classes | Status |
|---:|---|---:|---:|---:|---|
| 0.05% | `P1_selected_real_prototypes_replay` | 0.10049609347288475 | 0.05600118653478825 | 38 | not promoted |
| 0.10% | `P1_selected_real_prototypes_replay` | 0.06170690676524372 | 0.0571249905020672 | 38 | not promoted |
| 0.25% | `P3_hnr_fdm_shadow_b1` | 0.2046097517002238 | 0.12613426122985633 | 31 | diagnostic, shadow not materialized |
| 0.50% | `P2_hnr_fdm_prototype_oracle` | 0.28839031020414435 | 0.17021558791081592 | 37 | below recovery target |

Products P0 identity replay retained the known fullgraph reference: `0.7555780580193042` accuracy / `0.4046991170720907` macro-F1. P1/P2/P3 did not meet recovery gates.

Arxiv teacher-first gate:

- Current replay accuracy: `0.7016645063061951`.
- A1 gate `>= 0.715`: failed.
- Arxiv HNR-FDM condensation remains blocked by the teacher-first rule.

Ultra dry-run:

| Dataset | Ratio | Planned nodes | Target prototypes | Shadow nodes | Estimated train-target cache bytes | Resource gates |
|---|---:|---:|---:|---:|---:|---|
| `ogbn-papers100M` | 0.01% | 11106 | 7774 | 3332 | 1313410752 | S1/S2/S3 pass |
| `MAG240M` | 0.01% | 12175 | 8522 | 3653 | 1125740704 | S1/S2/S3 pass |

No T25 row is promoted as a performance replacement. This preserves the no-performance-regression rule.

## Scope

- Added scalable HNR/FDM-lite contracts, selectors, guards, and stage runners.
- Existing T24/R-1 paths are not replaced; T25 rows are promoted only when explicit gates pass.
- Exact GCRD TPAMI 2026 numbers are not fabricated; placeholder baseline rows are kept in `baselines/gcrd_tpami26.csv`.

## Aggregated Rows

| dataset | method | requested_full_node_ratio | status | accuracy | macro_f1 | promotion_status | failure_reason | source_table |
|---|---|---|---|---|---|---|---|---|
| ogbn-products | P0_identity_replay | 0.0005 | completed_fullgraph_replay | 0.7555780580193042 | 0.4046991170720907 | not_promoted | identity_reference_only | experiments\tables\t25_products_recovery_ladder_seed42.csv |
| ogbn-products | P1_selected_real_prototypes_replay | 0.0005 | completed_streaming | 0.10049609347288475 | 0.05600118653478825 | not_promoted | products_t25_gate_not_met | experiments\tables\t25_products_recovery_ladder_seed42.csv |
| ogbn-products | P2_hnr_fdm_prototype_oracle | 0.0005 | completed_streaming | 0.0006931481805312118 | 0.005752349819509904 | not_promoted | products_t25_gate_not_met | experiments\tables\t25_products_recovery_ladder_seed42.csv |
| ogbn-products | P3_hnr_fdm_shadow_b1 | 0.0005 | completed_streaming | 0.007570859038331456 | 0.02498478202578895 | not_promoted | shadow_materialization_not_trained | experiments\tables\t25_products_recovery_ladder_seed42.csv |
| ogbn-products | P3_hnr_fdm_shadow_b2 | 0.0005 | completed_streaming | 0.0286151812103524 | 0.037082679156536756 | not_promoted | shadow_materialization_not_trained | experiments\tables\t25_products_recovery_ladder_seed42.csv |
| ogbn-products | P0_identity_replay | 0.001 | completed_fullgraph_replay | 0.7555780580193042 | 0.4046991170720907 | not_promoted | identity_reference_only | experiments\tables\t25_products_recovery_ladder_seed42.csv |
| ogbn-products | P1_selected_real_prototypes_replay | 0.001 | completed_streaming | 0.06170690676524372 | 0.0571249905020672 | not_promoted | products_t25_gate_not_met | experiments\tables\t25_products_recovery_ladder_seed42.csv |
| ogbn-products | P2_hnr_fdm_prototype_oracle | 0.001 | completed_streaming | 0.04874087870765369 | 0.0313827694162128 | not_promoted | products_t25_gate_not_met | experiments\tables\t25_products_recovery_ladder_seed42.csv |
| ogbn-products | P3_hnr_fdm_shadow_b1 | 0.001 | completed_streaming | 0.03653894033277438 | 0.03821091279641932 | not_promoted | shadow_materialization_not_trained | experiments\tables\t25_products_recovery_ladder_seed42.csv |
| ogbn-products | P3_hnr_fdm_shadow_b2 | 0.001 | completed_streaming | 0.03389467491395519 | 0.03317839398667312 | not_promoted | shadow_materialization_not_trained | experiments\tables\t25_products_recovery_ladder_seed42.csv |
| ogbn-products | P0_identity_replay | 0.0025 | completed_fullgraph_replay | 0.7555780580193042 | 0.4046991170720907 | not_promoted | identity_reference_only | experiments\tables\t25_products_recovery_ladder_seed42.csv |
| ogbn-products | P1_selected_real_prototypes_replay | 0.0025 | completed_streaming | 0.08198804296795749 | 0.04286723753284472 | not_promoted | products_t25_gate_not_met | experiments\tables\t25_products_recovery_ladder_seed42.csv |
| ogbn-products | P2_hnr_fdm_prototype_oracle | 0.0025 | completed_streaming | 0.1261082350432043 | 0.09777620150207107 | not_promoted | products_t25_gate_not_met | experiments\tables\t25_products_recovery_ladder_seed42.csv |
| ogbn-products | P3_hnr_fdm_shadow_b1 | 0.0025 | completed_streaming | 0.2046097517002238 | 0.12613426122985633 | not_promoted | shadow_materialization_not_trained | experiments\tables\t25_products_recovery_ladder_seed42.csv |
| ogbn-products | P3_hnr_fdm_shadow_b2 | 0.0025 | completed_streaming | 0.16920632725902368 | 0.1183319577852288 | not_promoted | shadow_materialization_not_trained | experiments\tables\t25_products_recovery_ladder_seed42.csv |
| ogbn-products | P0_identity_replay | 0.005 | completed_fullgraph_replay | 0.7555780580193042 | 0.4046991170720907 | not_promoted | identity_reference_only | experiments\tables\t25_products_recovery_ladder_seed42.csv |
| ogbn-products | P1_selected_real_prototypes_replay | 0.005 | completed_streaming | 0.1519146749952894 | 0.0967926371510277 | not_promoted | products_t25_gate_not_met | experiments\tables\t25_products_recovery_ladder_seed42.csv |
| ogbn-products | P2_hnr_fdm_prototype_oracle | 0.005 | completed_streaming | 0.28839031020414435 | 0.17021558791081592 | not_promoted | products_t25_gate_not_met | experiments\tables\t25_products_recovery_ladder_seed42.csv |
| ogbn-products | P3_hnr_fdm_shadow_b1 | 0.005 | completed_streaming | 0.26935810592515175 | 0.20213225646425712 | not_promoted | shadow_materialization_not_trained | experiments\tables\t25_products_recovery_ladder_seed42.csv |
| ogbn-products | P3_hnr_fdm_shadow_b2 | 0.005 | completed_streaming | 0.16941915176556227 | 0.14921068889307768 | not_promoted | shadow_materialization_not_trained | experiments\tables\t25_products_recovery_ladder_seed42.csv |
| Reddit | current_sft_signature_random | 0.001 | completed_streaming | 0.8983896738057197 | 0.8433886102863218 | not_promoted | acceptance_gate_not_met | experiments\tables\t25_reddit_hnr_fdm_ratio_sweep_seed42.csv |
| Reddit | current_sft_signature_medoid | 0.001 | completed_streaming | 0.9127515573667486 | 0.8714283868911621 | not_promoted | acceptance_gate_not_met | experiments\tables\t25_reddit_hnr_fdm_ratio_sweep_seed42.csv |
| Reddit | current_sft_signature_kcenter | 0.001 | completed_streaming | 0.8673680053138969 | 0.8233612044131783 | not_promoted | acceptance_gate_not_met | experiments\tables\t25_reddit_hnr_fdm_ratio_sweep_seed42.csv |
| Reddit | current_sft_signature_shadow_b1 | 0.001 | completed_streaming_diagnostic | 0.9153905534710878 | 0.8746059467852552 | not_promoted | shadow_materialization_not_trained | experiments\tables\t25_reddit_hnr_fdm_ratio_sweep_seed42.csv |
| Reddit | sft_hnr_random | 0.001 | completed_streaming | 0.8959302012458934 | 0.8454848241770149 | not_promoted | acceptance_gate_not_met | experiments\tables\t25_reddit_hnr_fdm_ratio_sweep_seed42.csv |
| Reddit | sft_hnr_fdm_herding | 0.001 | completed_streaming | 0.9117103208085741 | 0.8713230962144167 | not_promoted | acceptance_gate_not_met | experiments\tables\t25_reddit_hnr_fdm_ratio_sweep_seed42.csv |
| Reddit | sft_hnr_fdm_kcenter | 0.001 | completed_streaming | 0.9110101789849738 | 0.8723440088839591 | not_promoted | acceptance_gate_not_met | experiments\tables\t25_reddit_hnr_fdm_ratio_sweep_seed42.csv |
| Reddit | sft_hnr_fdm_hybrid | 0.001 | completed_streaming | 0.9215841157567815 | 0.884890777869315 | not_promoted | acceptance_gate_not_met | experiments\tables\t25_reddit_hnr_fdm_ratio_sweep_seed42.csv |
| Reddit | sft_hnr_fdm_shadow_b1 | 0.001 | completed_streaming_diagnostic | 0.914780173419744 | 0.87557238881603 | not_promoted | shadow_materialization_not_trained | experiments\tables\t25_reddit_hnr_fdm_ratio_sweep_seed42.csv |
| Reddit | sft_hnr_fdm_shadow_b2 | 0.001 | completed_streaming_diagnostic | 0.911333321365097 | 0.8705033164012371 | not_promoted | shadow_materialization_not_trained | experiments\tables\t25_reddit_hnr_fdm_ratio_sweep_seed42.csv |
| Reddit | current_sft_signature_random | 0.0025 | completed_streaming | 0.9163958853203598 | 0.8805489216328221 | not_promoted | acceptance_gate_not_met | experiments\tables\t25_reddit_hnr_fdm_ratio_sweep_seed42.csv |
| Reddit | current_sft_signature_medoid | 0.0025 | completed_streaming | 0.9099868947812506 | 0.8773157884466924 | not_promoted | acceptance_gate_not_met | experiments\tables\t25_reddit_hnr_fdm_ratio_sweep_seed42.csv |
| Reddit | current_sft_signature_kcenter | 0.0025 | completed_streaming | 0.8960379153726011 | 0.8423379438366947 | not_promoted | acceptance_gate_not_met | experiments\tables\t25_reddit_hnr_fdm_ratio_sweep_seed42.csv |
| Reddit | current_sft_signature_shadow_b1 | 0.0025 | completed_streaming_diagnostic | 0.9212968780855609 | 0.8824121238133515 | not_promoted | shadow_materialization_not_trained | experiments\tables\t25_reddit_hnr_fdm_ratio_sweep_seed42.csv |
| Reddit | sft_hnr_random | 0.0025 | completed_streaming | 0.9180654542843294 | 0.8795818615448815 | not_promoted | acceptance_gate_not_met | experiments\tables\t25_reddit_hnr_fdm_ratio_sweep_seed42.csv |
| Reddit | sft_hnr_fdm_herding | 0.0025 | completed_streaming | 0.9179577401576217 | 0.8764673025046401 | not_promoted | acceptance_gate_not_met | experiments\tables\t25_reddit_hnr_fdm_ratio_sweep_seed42.csv |
| Reddit | sft_hnr_fdm_kcenter | 0.0025 | completed_streaming | 0.9113512737195483 | 0.8742284765588461 | not_promoted | acceptance_gate_not_met | experiments\tables\t25_reddit_hnr_fdm_ratio_sweep_seed42.csv |
| Reddit | sft_hnr_fdm_hybrid | 0.0025 | completed_streaming | 0.9140441268872412 | 0.8730428739959687 | not_promoted | acceptance_gate_not_met | experiments\tables\t25_reddit_hnr_fdm_ratio_sweep_seed42.csv |
| Reddit | sft_hnr_fdm_shadow_b1 | 0.0025 | completed_streaming_diagnostic | 0.9127695097212 | 0.8767842467500291 | not_promoted | shadow_materialization_not_trained | experiments\tables\t25_reddit_hnr_fdm_ratio_sweep_seed42.csv |
| Reddit | sft_hnr_fdm_shadow_b2 | 0.0025 | completed_streaming_diagnostic | 0.9209019262876327 | 0.8860039685317115 | not_promoted | shadow_materialization_not_trained | experiments\tables\t25_reddit_hnr_fdm_ratio_sweep_seed42.csv |
| Reddit | current_sft_signature_random | 0.005 | completed_streaming | 0.9233254941385562 | 0.885149317831537 | not_promoted | no_regression_gate_not_met | experiments\tables\t25_reddit_hnr_fdm_ratio_sweep_seed42.csv |
| Reddit | current_sft_signature_medoid | 0.005 | completed_streaming | 0.9228228282139203 | 0.8831134685348982 | not_promoted | no_regression_gate_not_met | experiments\tables\t25_reddit_hnr_fdm_ratio_sweep_seed42.csv |
| Reddit | current_sft_signature_kcenter | 0.005 | completed_streaming | 0.9084788970073425 | 0.8472429744875686 | not_promoted | no_regression_gate_not_met | experiments\tables\t25_reddit_hnr_fdm_ratio_sweep_seed42.csv |
| Reddit | current_sft_signature_shadow_b1 | 0.005 | completed_streaming_diagnostic | 0.9153546487621852 | 0.8774389012986487 | not_promoted | shadow_materialization_not_trained | experiments\tables\t25_reddit_hnr_fdm_ratio_sweep_seed42.csv |
| Reddit | sft_hnr_random | 0.005 | completed_streaming | 0.9170601224350574 | 0.8803582097707346 | not_promoted | no_regression_gate_not_met | experiments\tables\t25_reddit_hnr_fdm_ratio_sweep_seed42.csv |
| Reddit | sft_hnr_fdm_herding | 0.005 | completed_streaming | 0.9206685456797659 | 0.8863924816220402 | not_promoted | no_regression_gate_not_met | experiments\tables\t25_reddit_hnr_fdm_ratio_sweep_seed42.csv |
| Reddit | sft_hnr_fdm_kcenter | 0.005 | completed_streaming | 0.9231280182395921 | 0.882975902096398 | not_promoted | no_regression_gate_not_met | experiments\tables\t25_reddit_hnr_fdm_ratio_sweep_seed42.csv |
| Reddit | sft_hnr_fdm_hybrid | 0.005 | completed_streaming | 0.9217097822379405 | 0.8817167425644433 | not_promoted | no_regression_gate_not_met | experiments\tables\t25_reddit_hnr_fdm_ratio_sweep_seed42.csv |
| Reddit | sft_hnr_fdm_shadow_b1 | 0.005 | completed_streaming_diagnostic | 0.9197529756027503 | 0.8794423975650453 | not_promoted | shadow_materialization_not_trained | experiments\tables\t25_reddit_hnr_fdm_ratio_sweep_seed42.csv |
| Reddit | sft_hnr_fdm_shadow_b2 | 0.005 | completed_streaming_diagnostic | 0.9196452614760425 | 0.8798299763214216 | not_promoted | shadow_materialization_not_trained | experiments\tables\t25_reddit_hnr_fdm_ratio_sweep_seed42.csv |
| Reddit | current_sft_signature_random | 0.01 | completed_streaming | 0.9233793512019102 | 0.886217047127179 | not_promoted | acceptance_gate_not_met | experiments\tables\t25_reddit_hnr_fdm_ratio_sweep_seed42.csv |
| Reddit | current_sft_signature_medoid | 0.01 | completed_streaming | 0.9222663052259303 | 0.8814878728371921 | not_promoted | acceptance_gate_not_met | experiments\tables\t25_reddit_hnr_fdm_ratio_sweep_seed42.csv |
| Reddit | current_sft_signature_kcenter | 0.01 | completed_streaming | 0.9219431628458072 | 0.8801208622952813 | not_promoted | acceptance_gate_not_met | experiments\tables\t25_reddit_hnr_fdm_ratio_sweep_seed42.csv |
| Reddit | current_sft_signature_shadow_b1 | 0.01 | completed_streaming_diagnostic | 0.920237689172935 | 0.8745109862626925 | not_promoted | shadow_materialization_not_trained | experiments\tables\t25_reddit_hnr_fdm_ratio_sweep_seed42.csv |
| Reddit | sft_hnr_random | 0.01 | completed_streaming | 0.9205787839075095 | 0.8852204113072069 | not_promoted | acceptance_gate_not_met | experiments\tables\t25_reddit_hnr_fdm_ratio_sweep_seed42.csv |
| Reddit | sft_hnr_fdm_herding | 0.01 | completed_streaming | 0.9127695097212 | 0.87526898568368 | not_promoted | acceptance_gate_not_met | experiments\tables\t25_reddit_hnr_fdm_ratio_sweep_seed42.csv |
| Reddit | sft_hnr_fdm_kcenter | 0.01 | completed_streaming | 0.923558874746423 | 0.8865678667855177 | not_promoted | acceptance_gate_not_met | experiments\tables\t25_reddit_hnr_fdm_ratio_sweep_seed42.csv |
| Reddit | sft_hnr_fdm_hybrid | 0.01 | completed_streaming | 0.9236127318097769 | 0.8881558607412497 | not_promoted | acceptance_gate_not_met | experiments\tables\t25_reddit_hnr_fdm_ratio_sweep_seed42.csv |
| Reddit | sft_hnr_fdm_shadow_b1 | 0.01 | completed_streaming_diagnostic | 0.924564206595695 | 0.8881542321172001 | not_promoted | shadow_materialization_not_trained | experiments\tables\t25_reddit_hnr_fdm_ratio_sweep_seed42.csv |
| Reddit | sft_hnr_fdm_shadow_b2 | 0.01 | completed_streaming_diagnostic | 0.9180654542843294 | 0.8800353437565743 | not_promoted | shadow_materialization_not_trained | experiments\tables\t25_reddit_hnr_fdm_ratio_sweep_seed42.csv |
| ogbn-arxiv | arxiv_sft_v4_teacher | 0.005 | completed_replay | 0.7016645063061951 | 0.5048992808650066 | not_promoted | A1_teacher_gate_not_met | experiments\tables\t25_arxiv_sft_v4_teacher_seed42.csv |
| ogbn-arxiv | arxiv_sft_v4_teacher | 0.005 | ready_not_rerun | 0.7016645063061951 | 0.5048992808650066 | not_promoted | A1_teacher_gate_not_met | experiments\tables\t25_arxiv_sft_v4_teacher_seed42.csv |
| ogbn-arxiv | arxiv_sft_v4_teacher | 0.005 | ready_not_rerun | 0.7016645063061951 | 0.5048992808650066 | not_promoted | A1_teacher_gate_not_met | experiments\tables\t25_arxiv_sft_v4_teacher_seed42.csv |
| ogbn-arxiv | arxiv_sft_v4_teacher | 0.005 | ready_not_rerun | 0.7016645063061951 | 0.5048992808650066 | not_promoted | A1_teacher_gate_not_met | experiments\tables\t25_arxiv_sft_v4_teacher_seed42.csv |
| ogbn-arxiv | arxiv_sft_v4_teacher | 0.005 | ready_not_rerun | 0.7016645063061951 | 0.5048992808650066 | not_promoted | A1_teacher_gate_not_met | experiments\tables\t25_arxiv_sft_v4_teacher_seed42.csv |
| ogbn-arxiv | arxiv_sft_v4_teacher | 0.005 | ready_not_rerun | 0.7016645063061951 | 0.5048992808650066 | not_promoted | A1_teacher_gate_not_met | experiments\tables\t25_arxiv_sft_v4_teacher_seed42.csv |
| ogbn-arxiv | arxiv_sft_v4_teacher | 0.005 | ready_not_rerun | 0.7016645063061951 | 0.5048992808650066 | not_promoted | A1_teacher_gate_not_met | experiments\tables\t25_arxiv_sft_v4_teacher_seed42.csv |
| ogbn-arxiv | arxiv_sft_v4_teacher | 0.005 | ready_not_rerun | 0.7016645063061951 | 0.5048992808650066 | not_promoted | A1_teacher_gate_not_met | experiments\tables\t25_arxiv_sft_v4_teacher_seed42.csv |
| ogbn-arxiv | arxiv_sft_v4_teacher | 0.005 | ready_not_rerun | 0.7016645063061951 | 0.5048992808650066 | not_promoted | A1_teacher_gate_not_met | experiments\tables\t25_arxiv_sft_v4_teacher_seed42.csv |
| ogbn-papers100M | t25_ultra_safe_planner | 0.0001 | completed_ultra_dryrun |  |  | not_promoted |  | experiments\tables\t25_ultra_dryrun_seed42.csv |
| MAG240M | t25_ultra_safe_planner | 0.0001 | completed_ultra_dryrun |  |  | not_promoted |  | experiments\tables\t25_ultra_dryrun_seed42.csv |

## Promoted Rows

_No rows._

## Rows With Gates Not Met Or Diagnostics

| dataset | method | requested_full_node_ratio | failure_reason |
|---|---|---|---|
| ogbn-products | P0_identity_replay | 0.0005 | identity_reference_only |
| ogbn-products | P1_selected_real_prototypes_replay | 0.0005 | products_t25_gate_not_met |
| ogbn-products | P2_hnr_fdm_prototype_oracle | 0.0005 | products_t25_gate_not_met |
| ogbn-products | P3_hnr_fdm_shadow_b1 | 0.0005 | shadow_materialization_not_trained |
| ogbn-products | P3_hnr_fdm_shadow_b2 | 0.0005 | shadow_materialization_not_trained |
| ogbn-products | P0_identity_replay | 0.001 | identity_reference_only |
| ogbn-products | P1_selected_real_prototypes_replay | 0.001 | products_t25_gate_not_met |
| ogbn-products | P2_hnr_fdm_prototype_oracle | 0.001 | products_t25_gate_not_met |
| ogbn-products | P3_hnr_fdm_shadow_b1 | 0.001 | shadow_materialization_not_trained |
| ogbn-products | P3_hnr_fdm_shadow_b2 | 0.001 | shadow_materialization_not_trained |
| ogbn-products | P0_identity_replay | 0.0025 | identity_reference_only |
| ogbn-products | P1_selected_real_prototypes_replay | 0.0025 | products_t25_gate_not_met |
| ogbn-products | P2_hnr_fdm_prototype_oracle | 0.0025 | products_t25_gate_not_met |
| ogbn-products | P3_hnr_fdm_shadow_b1 | 0.0025 | shadow_materialization_not_trained |
| ogbn-products | P3_hnr_fdm_shadow_b2 | 0.0025 | shadow_materialization_not_trained |
| ogbn-products | P0_identity_replay | 0.005 | identity_reference_only |
| ogbn-products | P1_selected_real_prototypes_replay | 0.005 | products_t25_gate_not_met |
| ogbn-products | P2_hnr_fdm_prototype_oracle | 0.005 | products_t25_gate_not_met |
| ogbn-products | P3_hnr_fdm_shadow_b1 | 0.005 | shadow_materialization_not_trained |
| ogbn-products | P3_hnr_fdm_shadow_b2 | 0.005 | shadow_materialization_not_trained |
| Reddit | current_sft_signature_random | 0.001 | acceptance_gate_not_met |
| Reddit | current_sft_signature_medoid | 0.001 | acceptance_gate_not_met |
| Reddit | current_sft_signature_kcenter | 0.001 | acceptance_gate_not_met |
| Reddit | current_sft_signature_shadow_b1 | 0.001 | shadow_materialization_not_trained |
| Reddit | sft_hnr_random | 0.001 | acceptance_gate_not_met |
| Reddit | sft_hnr_fdm_herding | 0.001 | acceptance_gate_not_met |
| Reddit | sft_hnr_fdm_kcenter | 0.001 | acceptance_gate_not_met |
| Reddit | sft_hnr_fdm_hybrid | 0.001 | acceptance_gate_not_met |
| Reddit | sft_hnr_fdm_shadow_b1 | 0.001 | shadow_materialization_not_trained |
| Reddit | sft_hnr_fdm_shadow_b2 | 0.001 | shadow_materialization_not_trained |
| Reddit | current_sft_signature_random | 0.0025 | acceptance_gate_not_met |
| Reddit | current_sft_signature_medoid | 0.0025 | acceptance_gate_not_met |
| Reddit | current_sft_signature_kcenter | 0.0025 | acceptance_gate_not_met |
| Reddit | current_sft_signature_shadow_b1 | 0.0025 | shadow_materialization_not_trained |
| Reddit | sft_hnr_random | 0.0025 | acceptance_gate_not_met |
| Reddit | sft_hnr_fdm_herding | 0.0025 | acceptance_gate_not_met |
| Reddit | sft_hnr_fdm_kcenter | 0.0025 | acceptance_gate_not_met |
| Reddit | sft_hnr_fdm_hybrid | 0.0025 | acceptance_gate_not_met |
| Reddit | sft_hnr_fdm_shadow_b1 | 0.0025 | shadow_materialization_not_trained |
| Reddit | sft_hnr_fdm_shadow_b2 | 0.0025 | shadow_materialization_not_trained |
| Reddit | current_sft_signature_random | 0.005 | no_regression_gate_not_met |
| Reddit | current_sft_signature_medoid | 0.005 | no_regression_gate_not_met |
| Reddit | current_sft_signature_kcenter | 0.005 | no_regression_gate_not_met |
| Reddit | current_sft_signature_shadow_b1 | 0.005 | shadow_materialization_not_trained |
| Reddit | sft_hnr_random | 0.005 | no_regression_gate_not_met |
| Reddit | sft_hnr_fdm_herding | 0.005 | no_regression_gate_not_met |
| Reddit | sft_hnr_fdm_kcenter | 0.005 | no_regression_gate_not_met |
| Reddit | sft_hnr_fdm_hybrid | 0.005 | no_regression_gate_not_met |
| Reddit | sft_hnr_fdm_shadow_b1 | 0.005 | shadow_materialization_not_trained |
| Reddit | sft_hnr_fdm_shadow_b2 | 0.005 | shadow_materialization_not_trained |
| Reddit | current_sft_signature_random | 0.01 | acceptance_gate_not_met |
| Reddit | current_sft_signature_medoid | 0.01 | acceptance_gate_not_met |
| Reddit | current_sft_signature_kcenter | 0.01 | acceptance_gate_not_met |
| Reddit | current_sft_signature_shadow_b1 | 0.01 | shadow_materialization_not_trained |
| Reddit | sft_hnr_random | 0.01 | acceptance_gate_not_met |
| Reddit | sft_hnr_fdm_herding | 0.01 | acceptance_gate_not_met |
| Reddit | sft_hnr_fdm_kcenter | 0.01 | acceptance_gate_not_met |
| Reddit | sft_hnr_fdm_hybrid | 0.01 | acceptance_gate_not_met |
| Reddit | sft_hnr_fdm_shadow_b1 | 0.01 | shadow_materialization_not_trained |
| Reddit | sft_hnr_fdm_shadow_b2 | 0.01 | shadow_materialization_not_trained |
| ogbn-arxiv | arxiv_sft_v4_teacher | 0.005 | A1_teacher_gate_not_met |
| ogbn-arxiv | arxiv_sft_v4_teacher | 0.005 | A1_teacher_gate_not_met |
| ogbn-arxiv | arxiv_sft_v4_teacher | 0.005 | A1_teacher_gate_not_met |
| ogbn-arxiv | arxiv_sft_v4_teacher | 0.005 | A1_teacher_gate_not_met |
| ogbn-arxiv | arxiv_sft_v4_teacher | 0.005 | A1_teacher_gate_not_met |
| ogbn-arxiv | arxiv_sft_v4_teacher | 0.005 | A1_teacher_gate_not_met |
| ogbn-arxiv | arxiv_sft_v4_teacher | 0.005 | A1_teacher_gate_not_met |
| ogbn-arxiv | arxiv_sft_v4_teacher | 0.005 | A1_teacher_gate_not_met |
| ogbn-arxiv | arxiv_sft_v4_teacher | 0.005 | A1_teacher_gate_not_met |

## Required Next Server Commands

```powershell
& 'C:\Users\slian\anaconda3\envs\pytorch\python.exe' scripts\run_t25_reddit_hnr_fdm.py --train --epochs 30 --device cuda
& 'C:\Users\slian\anaconda3\envs\pytorch\python.exe' scripts\run_t25_products_recovery.py --train --epochs 4 --device cuda
& 'C:\Users\slian\anaconda3\envs\pytorch\python.exe' scripts\run_t25_arxiv_sft_v4.py
& 'C:\Users\slian\anaconda3\envs\pytorch\python.exe' scripts\run_t25_ultra_dryrun.py --ultra-safe
```

- CSV: `experiments\tables\t25_hnr_fdm_summary_seed42.csv`
