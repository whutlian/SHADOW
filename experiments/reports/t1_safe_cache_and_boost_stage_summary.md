# T1.1 Safe Cache and Boost Stage Summary

This stage implements safe-row logit cache generation, replay audit, validation-only T1.1 boosters, and large-scale dry-run estimates.

## Code Changes

- Added cache replay/index helpers and T1.1 cache filename compatibility.
- Added validation-only Correct&Smooth-lite, path logit correction primitives, and T1 pseudo-label helpers.
- Added safe-row cache generation, replay audit, booster scripts, large dry-run, and this stage runner.

## Cache Replay

| dataset | base_variant | cache_status | historical_test_acc | replay_test_acc | delta_replay | blocked_reason |
| --- | --- | --- | --- | --- | --- | --- |
| acm | SFB-v2 B3_scap_v2 retained | available_verified | 0.915486 | 0.9154863357543945 | 3.3575439450928e-07 |  |
| dblp | R+ relation-linear current-best | blocked_missing_replayable_logit_path | 0.836972 |  |  | current historical safe-row script records metrics but does not expose replayable all-target logits |
| imdb | clean S1 MAM/MDM/MKM | blocked_missing_replayable_logit_path | 0.42411 |  |  | current historical safe-row script records metrics but does not expose replayable all-target logits |
| ogbn-arxiv | LAD_reference | blocked_missing_replayable_logit_path | 0.596774 |  |  | current historical safe-row script records metrics but does not expose replayable all-target logits |
| ogbn-products | R++ base shadow-fusion | blocked_missing_replayable_logit_path | 0.668908 |  |  | current historical safe-row script records metrics but does not expose replayable all-target logits |
| ogbn-products | LAD_reference | blocked_missing_replayable_logit_path | 0.658674 |  |  | current historical safe-row script records metrics but does not expose replayable all-target logits |

## Promoted Rows

| dataset | promoted_variant | base_variant | valid_acc_after | accuracy | macro_f1 | predicted_class_count | promotion_status | promotion_reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| acm | pseudo_scap | SFB-v2 B3_scap_v2 retained | 0.9340659379959106 | 0.9159584641456604 | 0.9172853231430054 | 3 | promoted | validation_selected |

## Blocked Rows

| dataset | base_variant | cache_status | promotion_status | blocked_reason | promotion_reason |
| --- | --- | --- | --- | --- | --- |
| acm | SFB-v2 B3_scap_v2 retained | available_unreplayed |  |  |  |
| dblp | R+ relation-linear current-best | blocked_missing_replayable_logit_path |  | current historical safe-row script records metrics but does not expose replayable all-target logits |  |
| imdb | clean S1 MAM/MDM/MKM | blocked_missing_replayable_logit_path |  | current historical safe-row script records metrics but does not expose replayable all-target logits |  |
| ogbn-arxiv | LAD_reference | blocked_missing_replayable_logit_path |  | current historical safe-row script records metrics but does not expose replayable all-target logits |  |
| ogbn-products | R++ base shadow-fusion | blocked_missing_replayable_logit_path |  | current historical safe-row script records metrics but does not expose replayable all-target logits |  |
| ogbn-products | LAD_reference | blocked_missing_replayable_logit_path |  | current historical safe-row script records metrics but does not expose replayable all-target logits |  |
| dblp | R+ relation-linear current-best | blocked_missing_replayable_logit_path |  | current historical safe-row script records metrics but does not expose replayable all-target logits |  |
| imdb | clean S1 MAM/MDM/MKM | blocked_missing_replayable_logit_path |  | current historical safe-row script records metrics but does not expose replayable all-target logits |  |
| ogbn-arxiv | LAD_reference | blocked_missing_replayable_logit_path |  | current historical safe-row script records metrics but does not expose replayable all-target logits |  |
| ogbn-products | R++ base shadow-fusion | blocked_missing_replayable_logit_path |  | current historical safe-row script records metrics but does not expose replayable all-target logits |  |
| ogbn-products | LAD_reference | blocked_missing_replayable_logit_path |  | current historical safe-row script records metrics but does not expose replayable all-target logits |  |
| acm | SFB-v2 B3_scap_v2 retained |  | blocked |  | validation_no_improvement |
| dblp | R+ relation-linear current-best |  | blocked |  | current historical safe-row script records metrics but does not expose replayable all-target logits |
| imdb | clean S1 MAM/MDM/MKM |  | blocked |  | current historical safe-row script records metrics but does not expose replayable all-target logits |
| ogbn-arxiv | LAD_reference |  | blocked |  | current historical safe-row script records metrics but does not expose replayable all-target logits |
| ogbn-products | R++ base shadow-fusion |  | blocked |  | current historical safe-row script records metrics but does not expose replayable all-target logits |
| ogbn-products | LAD_reference |  | blocked |  | current historical safe-row script records metrics but does not expose replayable all-target logits |
| dblp | R+ relation-linear current-best |  | blocked |  | current historical safe-row script records metrics but does not expose replayable all-target logits |
| imdb | clean S1 MAM/MDM/MKM |  | blocked |  | current historical safe-row script records metrics but does not expose replayable all-target logits |
| dblp | R+ relation-linear current-best |  | blocked |  | current historical safe-row script records metrics but does not expose replayable all-target logits |
| imdb | clean S1 MAM/MDM/MKM |  | blocked |  | current historical safe-row script records metrics but does not expose replayable all-target logits |
| ogbn-arxiv | LAD_reference |  | blocked |  | current historical safe-row script records metrics but does not expose replayable all-target logits |
| ogbn-products | R++ base shadow-fusion |  | blocked |  | current historical safe-row script records metrics but does not expose replayable all-target logits |
| ogbn-products | LAD_reference |  | blocked |  | current historical safe-row script records metrics but does not expose replayable all-target logits |
| acm | SFB-v2 B3_scap_v2 retained |  | blocked |  | not_enough_validated_component_logits |
| ogbn-products |  |  | blocked |  | not_enough_validated_component_logits |

## Required Questions

1. Were the correct historical safe rows regenerated with logits cache? ACM SFB-v2 B3 was regenerated with historical replay and gate-selection caches; DBLP, IMDB, arxiv, and products safe rows are blocked because current historical scripts do not expose replayable all-target logits.
2. Did cache replay match the historical metrics? ACM matched within tolerance; blocked rows have no replay cache.
3. Did LogitCorrectLite improve any dataset? No. ACM validation did not improve; arxiv/products were blocked by missing replay-verified historical caches.
4. Did Correct&Smooth-lite improve arxiv/products? No, because their historical safe caches were not replay-verified locally.
5. Did PathLogitCorrectLite improve DBLP/IMDB? No, both are blocked by missing replay-verified historical caches.
6. Did Pseudo-SCAP improve any dataset? Yes: acm pseudo_scap acc=0.9159584641456604 macro_f1=0.9172853231430054.
7. Did Safe Logit Ensemble improve products or ACM? No; ensemble is blocked unless component logits are persisted.
8. Which rows were promoted? acm pseudo_scap acc=0.9159584641456604 macro_f1=0.9172853231430054.
9. Which rows were blocked and why? See `Blocked Rows`.
10. Did any promoted row use forbidden components? No; promotion validator rejects forbidden flags.
11. Did any promoted medium row use bounded edges? No.
12. Did macro-F1 or predicted class count collapse? No for promoted rows.
13. Which datasets are now eligible for condensation recovery? ['acm'].
14. If no improvement occurred, what is the next bottleneck: cache mismatch, base signal ceiling, or validation overfit? For blocked datasets the bottleneck is cache mismatch/missing replayable logits; for ACM, inspect promoted row deltas for possible validation overfit before condensation recovery.

## Artifacts

- `experiments\tables\t1_safe_logit_cache_index_seed42.csv`
- `experiments\tables\t1_cache_replay_audit_seed42.csv`
- `experiments\tables\t1_safe_logit_correct_seed42.csv`
- `experiments\tables\t1_path_logit_correct_seed42.csv`
- `experiments\tables\t1_pseudo_scap_safe_seed42.csv`
- `experiments\tables\t1_safe_logit_ensemble_safe_seed42.csv`
- `experiments\tables\t1_safe_fullgraph_boost_summary_seed42.csv`
- `experiments\tables\t1_large_logit_affinity_dry_run_seed42.csv`
