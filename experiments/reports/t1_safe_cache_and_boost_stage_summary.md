# T1.1 Safe Cache and Boost Stage Summary

This stage implements safe-row logit cache generation, replay audit, validation-only T1.1 boosters, and large-scale dry-run estimates.

## Code Changes

- Exposed all-target logits from SeHGNNLite and pipeline graph/compiled inference for replay-safe cache generation.
- Extended T1.1 safe-row cache generation to DBLP, IMDB, ogbn-arxiv, and ogbn-products LAD; products R++ 500-epoch base is opt-in because it was too slow locally.
- Wired validation-only PathLogitCorrectLite for replay-verified DBLP/IMDB caches while keeping sparse path steps and no exposed meta-path edge types.
- Made cache reuse explicit and validated replay/gate cache metadata before reuse.
- Refreshed replay audit, booster tables, large dry-run, and this stage summary.

## Cache Generation

| dataset | base_variant | cache_status | train_nodes | valid_nodes | test_nodes | all_target_nodes | blocked_reason |
| --- | --- | --- | --- | --- | --- | --- | --- |
| acm | SFB-v2 B3_scap_v2 retained | available_unreplayed | 907 | 0 | 2118 | 3025 |  |
| dblp | R+ relation-linear current-best | available_unreplayed | 1217 | 0 | 2840 | 4057 |  |
| imdb | clean S1 MAM/MDM/MKM | available_unreplayed | 1371 | 0 | 3202 | 4932 |  |
| ogbn-arxiv | LAD_reference | available_unreplayed | 90941 | 29799 | 48603 | 169343 |  |
| ogbn-products | R++ base shadow-fusion | blocked_cache_generation_failed |  |  |  |  | products R++ base 500-epoch cache generation was dropped as too slow on local machine; rerun with --run-products-rpp-base to force it |
| ogbn-products | LAD_reference | available_unreplayed | 196615 | 39323 | 2213091 | 2449029 |  |

## Cache Replay

| dataset | base_variant | cache_status | historical_test_acc | replay_test_acc | delta_replay | blocked_reason |
| --- | --- | --- | --- | --- | --- | --- |
| acm | SFB-v2 B3_scap_v2 retained | available_verified | 0.915486 | 0.9154863357543945 | 3.3575439450928e-07 |  |
| dblp | R+ relation-linear current-best | available_verified | 0.836972 | 0.8369718194007874 | -1.805992126957534e-07 |  |
| imdb | clean S1 MAM/MDM/MKM | invalid_replay_mismatch | 0.42411 | 0.420362263917923 | -0.0037477360820770134 | replay accuracy mismatch |
| ogbn-arxiv | LAD_reference | invalid_replay_mismatch | 0.596774 | 0.6059914231300354 | 0.009217423130035374 | replay accuracy mismatch |
| ogbn-products | R++ base shadow-fusion | blocked_cache_generation_failed | 0.668908 |  |  | products R++ base 500-epoch cache generation was dropped as too slow on local machine; rerun with --run-products-rpp-base to force it |
| ogbn-products | LAD_reference | invalid_replay_mismatch | 0.658674 | 0.6399619579315186 | -0.018712042068481427 | replay accuracy mismatch |

## Replay Classification

### Verified

| dataset | base_variant | historical_test_acc | replay_test_acc | delta_replay |
| --- | --- | --- | --- | --- |
| acm | SFB-v2 B3_scap_v2 retained | 0.915486 | 0.9154863357543945 | 3.3575439450928e-07 |
| dblp | R+ relation-linear current-best | 0.836972 | 0.8369718194007874 | -1.805992126957534e-07 |

### Mismatch

| dataset | base_variant | historical_test_acc | replay_test_acc | delta_replay | blocked_reason |
| --- | --- | --- | --- | --- | --- |
| imdb | clean S1 MAM/MDM/MKM | 0.42411 | 0.420362263917923 | -0.0037477360820770134 | replay accuracy mismatch |
| ogbn-arxiv | LAD_reference | 0.596774 | 0.6059914231300354 | 0.009217423130035374 | replay accuracy mismatch |
| ogbn-products | LAD_reference | 0.658674 | 0.6399619579315186 | -0.018712042068481427 | replay accuracy mismatch |

### Not Generated

| dataset | base_variant | cache_status | blocked_reason |
| --- | --- | --- | --- |
| ogbn-products | R++ base shadow-fusion | blocked_cache_generation_failed | products R++ base 500-epoch cache generation was dropped as too slow on local machine; rerun with --run-products-rpp-base to force it |

## Promoted Rows

| dataset | promoted_variant | base_variant | valid_acc_after | accuracy | macro_f1 | predicted_class_count | promotion_status | promotion_reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dblp | path_logit_correct | R+ relation-linear current-best | 0.8196721076965332 | 0.8376760482788086 | 0.8305231779813766 | 4 | promoted | validation_selected |
| acm | pseudo_scap | SFB-v2 B3_scap_v2 retained | 0.9340659379959106 | 0.9159584641456604 | 0.9172853231430054 | 3 | promoted | validation_selected |

## Blocked Rows

| dataset | base_variant | cache_status | promotion_status | blocked_reason | promotion_reason |
| --- | --- | --- | --- | --- | --- |
| ogbn-products | R++ base shadow-fusion | blocked_cache_generation_failed |  | products R++ base 500-epoch cache generation was dropped as too slow on local machine; rerun with --run-products-rpp-base to force it |  |
| imdb | clean S1 MAM/MDM/MKM | invalid_replay_mismatch |  | replay accuracy mismatch |  |
| ogbn-arxiv | LAD_reference | invalid_replay_mismatch |  | replay accuracy mismatch |  |
| ogbn-products | LAD_reference | invalid_replay_mismatch |  | replay accuracy mismatch |  |
| acm | SFB-v2 B3_scap_v2 retained |  | blocked |  | validation_no_improvement |
| dblp | R+ relation-linear current-best |  | blocked |  | no_target_target_relation |
| acm | SFB-v2 B3_scap_v2 retained |  | blocked |  | not_enough_validated_component_logits |
| ogbn-products |  |  | blocked |  | not_enough_validated_component_logits |

## Required Questions

1. Were the correct historical safe rows regenerated with logits cache? Generated: acm:SFB-v2 B3_scap_v2 retained, dblp:R+ relation-linear current-best, imdb:clean S1 MAM/MDM/MKM, ogbn-arxiv:LAD_reference, ogbn-products:LAD_reference. Slow/failed: ogbn-products:R++ base shadow-fusion (products R++ base 500-epoch cache generation was dropped as too slow on local machine; rerun with --run-products-rpp-base to force it).
2. Did cache replay match the historical metrics? Verified: acm:SFB-v2 B3_scap_v2 retained, dblp:R+ relation-linear current-best. Mismatch: imdb:clean S1 MAM/MDM/MKM delta=-0.0037477360820770134, ogbn-arxiv:LAD_reference delta=0.009217423130035374, ogbn-products:LAD_reference delta=-0.018712042068481427.
3. Did LogitCorrectLite improve any dataset? No promoted LogitCorrectLite row.
4. Did Correct&Smooth-lite improve arxiv/products? No promoted arxiv/products Correct&Smooth row; their replay caches did not pass the historical gate.
5. Did PathLogitCorrectLite improve DBLP/IMDB? Yes: dblp 0.8369718194007874->0.8376760482788086
6. Did Pseudo-SCAP improve any dataset? Yes: acm 0.9154863357543945->0.9159584641456604
7. Did Safe Logit Ensemble improve products or ACM? No promoted safe ensemble row.
8. Which rows were promoted? dblp path_logit_correct acc=0.8376760482788086 macro_f1=0.8305231779813766; acm pseudo_scap acc=0.9159584641456604 macro_f1=0.9172853231430054.
9. Which rows were blocked and why? See `Blocked Rows`.
10. Did any promoted row use forbidden components? No; promotion validator rejects forbidden flags.
11. Did any promoted medium row use bounded edges? No.
12. Did macro-F1 or predicted class count collapse? No for promoted rows.
13. Which datasets are now eligible for condensation recovery? ['acm', 'dblp'].
14. If no improvement occurred, what is the next bottleneck: cache mismatch, base signal ceiling, or validation overfit? Current blockers are replay mismatch (imdb:clean S1 MAM/MDM/MKM delta=-0.0037477360820770134, ogbn-arxiv:LAD_reference delta=0.009217423130035374, ogbn-products:LAD_reference delta=-0.018712042068481427) and slow local products R++ generation (ogbn-products:R++ base shadow-fusion (products R++ base 500-epoch cache generation was dropped as too slow on local machine; rerun with --run-products-rpp-base to force it)); verified rows can proceed to condensation recovery.

## Artifacts

- `experiments\tables\t1_safe_logit_cache_index_seed42.csv`
- `experiments\tables\t1_cache_replay_audit_seed42.csv`
- `experiments\tables\t1_safe_logit_correct_seed42.csv`
- `experiments\tables\t1_path_logit_correct_seed42.csv`
- `experiments\tables\t1_pseudo_scap_safe_seed42.csv`
- `experiments\tables\t1_safe_logit_ensemble_safe_seed42.csv`
- `experiments\tables\t1_safe_fullgraph_boost_summary_seed42.csv`
- `experiments\tables\t1_large_logit_affinity_dry_run_seed42.csv`
