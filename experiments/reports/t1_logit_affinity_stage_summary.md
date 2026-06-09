# T1 Logit-Affinity Fullgraph Boost Summary

This stage implements opt-in low-dimensional T1 boosters while leaving the default Shadow-HGC-R-1 path frozen.

## Code Changes

- Added `shadow_hgc.logits` cache metadata/I/O with forbidden-promotion flags for diffusion, dense P2, bounded edges, source anchors, CoverageMedoid, and old KD.
- Added LogitCorrectLite over destination-row normalized target-target edges using only C-dimensional logits/probabilities and train labels for error correction.
- Added confidence-gated Pseudo-SCAP helpers with train-node one-hot override, top-k class sparse storage, prior centering helpers, and destination-row affinity aggregation.
- Added safe nonnegative logit ensemble utilities with validation improvement and test non-regression gates.
- Added T1 runners and stage artifact generation; historical safe rows without split/all-target logits are blocked explicitly instead of promoted.

## Promoted Rows

| dataset | promoted_variant | base_variant | accuracy | macro_f1 | predicted_class_count | base_accuracy | base_macro_f1 | delta_accuracy | delta_macro_f1 | uses_logit_correct | uses_pseudo_scap | uses_ensemble | uses_diffusion | uses_dense_p2 | uses_bounded_edges | promotion_status | promotion_reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

## Cache Index

| dataset | base_variant | cache_status | accuracy | macro_f1 | blocked_reason |
| --- | --- | --- | --- | --- | --- |
| acm | SFB-v2 B3_scap_v2 retained | blocked_missing_safe_logit_cache | 0.915486 | 0.91658 | historical safe row metrics exist but split/all-target logits were not stored in current artifacts |
| dblp | R+ relation-linear current-best | blocked_missing_safe_logit_cache | 0.836972 | 0.829937 | historical safe row metrics exist but split/all-target logits were not stored in current artifacts |
| imdb | clean S1 MAM/MDM/MKM | blocked_missing_safe_logit_cache | 0.42411 | 0.353932 | historical safe row metrics exist but split/all-target logits were not stored in current artifacts |
| ogbn-arxiv | LAD_reference | blocked_missing_safe_logit_cache | 0.596774 | 0.415452 | historical safe row metrics exist but split/all-target logits were not stored in current artifacts |
| ogbn-products | P0b_Rpp_base_shadow_fusion_reference | blocked_missing_safe_logit_cache | 0.668908 | 0.307981 | historical safe row metrics exist but split/all-target logits were not stored in current artifacts |
| ogbn-products | P0_LAD_reference | blocked_missing_safe_logit_cache | 0.658674 | 0.338064 | historical safe row metrics exist but split/all-target logits were not stored in current artifacts |

## T1 Dataset Summary

| dataset | best_base_variant | best_base_accuracy | best_base_macro_f1 | status | eligible_for_condensation_recovery | reason |
| --- | --- | --- | --- | --- | --- | --- |
| acm | SFB-v2 B3_scap_v2 retained | 0.915486 | 0.91658 | blocked_missing_safe_logit_cache | False | T1 requires split/all-target logits from safe base rows; current artifacts contain metrics but not logits |
| dblp | R+ relation-linear current-best | 0.836972 | 0.829937 | blocked_missing_safe_logit_cache | False | T1 requires split/all-target logits from safe base rows; current artifacts contain metrics but not logits |
| imdb | clean S1 MAM/MDM/MKM | 0.42411 | 0.353932 | blocked_missing_safe_logit_cache | False | T1 requires split/all-target logits from safe base rows; current artifacts contain metrics but not logits |
| ogbn-arxiv | LAD_reference | 0.596774 | 0.415452 | blocked_missing_safe_logit_cache | False | T1 requires split/all-target logits from safe base rows; current artifacts contain metrics but not logits |
| ogbn-products | P0b_Rpp_base_shadow_fusion_reference | 0.668908 | 0.307981 | blocked_missing_safe_logit_cache | False | T1 requires split/all-target logits from safe base rows; current artifacts contain metrics but not logits |

## Large Dry-Run Estimates

| dataset | num_target_nodes | num_classes | logit_cache_gb | pseudo_scap_cache_gb | edge_scans | uses_memmap | uses_topk_sparse | uses_bounded_edges | expected_wall_time_category |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ogbn-arxiv | 169343 | 40 | 0.0135 | 0.0156 | 2 | True | True | False | minutes |
| ogbn-products | 2449029 | 47 | 0.2302 | 0.2253 | 2 | True | True | False | tens_of_minutes |
| ogbn-papers100M | 111059956 | 172 | 38.2046 | 10.2175 | 2 | True | True | False | hours_server_recommended |
| MAG240M | 121751666 | 153 | 37.256 | 11.2012 | 2 | True | True | False | hours_server_recommended |

## Required Final Questions

1. Did LogitCorrectLite improve any dataset? No. It is implemented, but all historical safe base rows are missing split/all-target logit caches in the current artifacts, so all LogitCorrectLite experiment rows are `blocked_missing_safe_logit_cache`.
2. Did Pseudo-SCAP improve any dataset? No. It is implemented, but no safe base logit cache is available to construct validation-gated pseudo labels.
3. Did Safe Logit Ensemble improve any dataset? No. Ensemble components require valid safe logit caches; current artifacts contain metrics only.
4. Which rows are promoted? None.
5. Which rows are blocked and why? All planned T1 rows are blocked because the previous safe rows did not persist train/valid/test/all-target logits; no row is blocked by forbidden component use in this stage.
6. Did any promoted row use forbidden components? No promoted rows exist, and the promotion validator rejects diffusion, dense P2, bounded edges, source anchors, CoverageMedoid, and old KD.
7. Did any promoted medium row use bounded_edges? No. There are no promoted medium rows, and bounded-edge rows are invalid for promotion.
8. Did macro-F1 regress? No promoted row regressed macro-F1; promoted macro-regression count is 0.
9. Is any dataset eligible for condensation recovery? No. The condensation recovery rule requires at least one improved fullgraph T1 row, and none improved.
10. What is the next recommended step? Re-run the safe base models with `save_logits_cache` enabled, then rerun `scripts/run_t1_logit_affinity_stage.py --seed 42`; do not add another high-dimensional feature block.

## Artifacts

- `experiments\tables\t1_logit_cache_index_seed42.csv`
- `experiments\tables\t1_logit_correct_seed42.csv`
- `experiments\tables\t1_pseudo_scap_seed42.csv`
- `experiments\tables\t1_safe_logit_ensemble_seed42.csv`
- `experiments\tables\t1_fullgraph_boost_summary_seed42.csv`
- `experiments\tables\t1_large_dry_run_seed42.csv`
