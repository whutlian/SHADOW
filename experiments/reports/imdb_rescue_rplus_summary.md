# IMDB R+ Rescue Summary

## Scope

- Dataset: IMDB.
- Seed: 42 only.
- Ratios: 0.5%, 2.5%, 5.0%.
- Variants: base and full R+ at all ratios; component variants at 2.5%.

## Results

| Variant | Loss | Ratio | Acc | Macro-F1 | Pred classes | Entropy |
|---|---|---:|---:|---:|---:|---:|
| base | clipped | 0.5000% | 0.2926 | 0.2849 | 5 | 1.5600 |
| base | sqrt_weighted | 0.5000% | 0.3260 | 0.2880 | 5 | 1.4626 |
| base | class_balanced | 0.5000% | 0.2826 | 0.2606 | 5 | 1.4761 |
| base | clipped | 2.5000% | 0.3507 | 0.2956 | 5 | 1.4574 |
| base | sqrt_weighted | 2.5000% | 0.3351 | 0.2895 | 5 | 1.4997 |
| base | class_balanced | 2.5000% | 0.3307 | 0.2906 | 5 | 1.5383 |
| base | clipped | 5.0000% | 0.2945 | 0.2771 | 5 | 1.5863 |
| base | sqrt_weighted | 5.0000% | 0.3079 | 0.2818 | 5 | 1.5576 |
| base | class_balanced | 5.0000% | 0.2961 | 0.2788 | 5 | 1.5957 |
| full_rplus | clipped | 0.5000% | 0.3810 | 0.3403 | 5 | 1.4872 |
| full_rplus | sqrt_weighted | 0.5000% | 0.3660 | 0.3335 | 5 | 1.4901 |
| full_rplus | class_balanced | 0.5000% | 0.3716 | 0.3370 | 5 | 1.5093 |
| full_rplus | clipped | 2.5000% | 0.3666 | 0.3363 | 5 | 1.5258 |
| full_rplus | sqrt_weighted | 2.5000% | 0.3576 | 0.3370 | 5 | 1.5629 |
| full_rplus | class_balanced | 2.5000% | 0.3648 | 0.3207 | 5 | 1.4584 |
| full_rplus | clipped | 5.0000% | 0.3485 | 0.3349 | 5 | 1.5699 |
| full_rplus | sqrt_weighted | 5.0000% | 0.3504 | 0.3303 | 5 | 1.5494 |
| full_rplus | class_balanced | 5.0000% | 0.3713 | 0.3424 | 5 | 1.5352 |
| metapath | clipped | 2.5000% | 0.3585 | 0.3098 | 5 | 1.5314 |
| metapath | sqrt_weighted | 2.5000% | 0.3513 | 0.3024 | 5 | 1.5310 |
| metapath | class_balanced | 2.5000% | 0.3554 | 0.3067 | 5 | 1.5260 |
| rank_adaptive | clipped | 2.5000% | 0.3751 | 0.3569 | 5 | 1.5594 |
| rank_adaptive | sqrt_weighted | 2.5000% | 0.3242 | 0.3205 | 5 | 1.5579 |
| rank_adaptive | class_balanced | 2.5000% | 0.3517 | 0.3370 | 5 | 1.5859 |
| adaptive_b | clipped | 2.5000% | 0.3367 | 0.2884 | 5 | 1.4576 |
| adaptive_b | sqrt_weighted | 2.5000% | 0.3154 | 0.2771 | 5 | 1.5317 |
| adaptive_b | class_balanced | 2.5000% | 0.3463 | 0.2977 | 5 | 1.4868 |
| relation_gate | clipped | 2.5000% | 0.3310 | 0.2870 | 5 | 1.5001 |
| relation_gate | sqrt_weighted | 2.5000% | 0.3292 | 0.2846 | 5 | 1.5030 |
| relation_gate | class_balanced | 2.5000% | 0.3538 | 0.3100 | 5 | 1.5152 |

## Best Point

- Best accuracy: `0.3810` from `full_rplus` with `clipped` at `0.5000%`.
- Rescue is successful only if accuracy and macro-F1 improve without predicted-class collapse.

## Files

- CSV: `experiments\tables\imdb_rescue_rplus_seed42.csv`
- Report: `experiments\reports\imdb_rescue_rplus_summary.md`
