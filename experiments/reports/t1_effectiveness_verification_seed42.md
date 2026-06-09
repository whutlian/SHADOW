# T1 Effectiveness Verification Seed 42

This verification uses freshly generated split/all-target logits where the repository can produce them locally.
For small datasets without a native validation split, the original train split is deterministically partitioned into train-fit and validation-gate rows.

## Cache Rows

| dataset | base_variant | cache_status | train_fit_nodes | validation_gate_nodes | base_valid_acc | base_test_acc | base_test_macro_f1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| acm | B3_scap_v2 | available_verified | 725 | 182 | 0.9285714030265808 | 0.8956562876701355 | 0.8962585926055908 |
| dblp | B2_metapath | available_verified | 973 | 244 | 0.6065573692321777 | 0.5845070481300354 | 0.5760394930839539 |
| imdb | B2_metapath | available_verified | 1097 | 274 | 0.35036495327949524 | 0.3244847059249878 | 0.26115010529756544 |

## Best Dataset Summary

| dataset | base_variant | base_valid_acc | base_test_acc | best_validated_variant | best_validated_test_acc | delta_test_acc | promotion_status | verification_scope |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| acm | B3_scap_v2 | 0.9285714030265808 | 0.8956562876701355 | smooth_prob | 0.906987726688385 | 0.011331439018249512 | promoted | available_sfb_v2_logit_cache_matches_acm_t1_base_family |
| dblp | B2_metapath | 0.6065573692321777 | 0.5845070481300354 |  |  |  | no_promoted_t1_row | available_sfb_v2_logit_cache_not_historical_safe_row |
| imdb | B2_metapath | 0.35036495327949524 | 0.3244847059249878 |  |  |  | no_promoted_t1_row | available_sfb_v2_logit_cache_not_historical_safe_row |

## Promoted Rows

| dataset | base_variant | mode | ensemble_mode | valid_acc_after | valid_acc | test_acc_after | test_acc | macro_f1_after | macro_f1 | promotion_status | promotion_reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| acm | B3_scap_v2 | smooth_prob |  | 0.9450549483299255 |  | 0.906987726688385 |  | 0.9076114495595297 |  | promoted | validation_and_test_gate_passed |
| acm | B3_scap_v2 | pseudo_scap |  | 0.9340659379959106 |  | 0.8956562876701355 |  | 0.8962492942810059 |  | promoted | validation_and_test_gate_passed |

## Method Best Rows

| dataset | base_variant | mode | valid_acc_before | valid_acc_after | test_acc_before | test_acc_after | promotion_status | promotion_reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| acm | B3_scap_v2 | smooth_prob | 0.9285714030265808 | 0.9450549483299255 | 0.8956562876701355 | 0.906987726688385 | promoted | validation_and_test_gate_passed |

| dataset | base_variant | threshold | pseudo_weight | temperature | affinity_lambda | valid_acc_before | valid_acc_after | test_acc_before | test_acc_after | promotion_status | promotion_reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| acm | B3_scap_v2 | 0.7 | 0.25 | 1.0 | 1.0 | 0.9285714030265808 | 0.9340659379959106 | 0.8956562876701355 | 0.8956562876701355 | promoted | validation_and_test_gate_passed |

| dataset | base_variant | candidate_components | weights | valid_acc | test_acc | macro_f1 | promotion_status | promotion_reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| acm | B3_scap_v2 | ["base", "best_logit_correct", "best_pseudo_scap"] | [0.0, 1.0, 0.0] | 0.9450549483299255 | 0.906987726688385 | 0.9076114495595297 | blocked | validation_no_improvement |

## Interpretation

- This is an effectiveness check on available SFB-v2 logits, not a replacement for the attachment's historical safe-row verification.
- ACM has real target-target relations and therefore exercises LogitCorrectLite and target-target Pseudo-SCAP.
- DBLP and IMDB local full schemas have no target-target relation, so target-target T1 correction/SCAP rows are blocked for those datasets in this verification.
- CSV artifacts: `experiments\tables\t1_available_logit_cache_index_seed42.csv`, `experiments\tables\t1_effective_logit_correct_seed42.csv`, `experiments\tables\t1_effective_pseudo_scap_seed42.csv`, `experiments\tables\t1_effective_safe_logit_ensemble_seed42.csv`, `experiments\tables\t1_effectiveness_verification_seed42.csv`.
