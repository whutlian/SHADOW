# IMDB R++ Rescue Summary

## Completed / OOM / Failed Rows

| Dataset | Variant | Ratio | Model | Feature | Loss | Acc | Macro-F1 | Pred Classes | Status |
|---|---|---:|---|---|---|---:|---:|---:|---|
| imdb | full_rplus_current | 0.5% | relation_linear | metapath | clipped | 0.3810 | 0.3403 | 5 | completed |
| imdb | full_rplus_current | 0.5% | relation_linear | metapath | class_balanced | 0.3716 | 0.3370 | 5 | completed |
| imdb | full_rplus_current | 2.5% | relation_linear | metapath | clipped | 0.3666 | 0.3363 | 5 | completed |
| imdb | full_rplus_current | 2.5% | relation_linear | metapath | class_balanced | 0.3648 | 0.3207 | 5 | completed |
| imdb | full_rplus_current | 5.0% | relation_linear | metapath | clipped | 0.3485 | 0.3349 | 5 | completed |
| imdb | full_rplus_current | 5.0% | relation_linear | metapath | class_balanced | 0.3713 | 0.3424 | 5 | completed |
| imdb | full_rplus_blocknorm | 0.5% | relation_linear | metapath | clipped | 0.3866 | 0.3615 | 5 | completed |
| imdb | full_rplus_blocknorm | 0.5% | relation_linear | metapath | class_balanced | 0.3894 | 0.3546 | 5 | completed |
| imdb | full_rplus_blocknorm | 2.5% | relation_linear | metapath | clipped | 0.3485 | 0.3283 | 5 | completed |
| imdb | full_rplus_blocknorm | 2.5% | relation_linear | metapath | class_balanced | 0.3832 | 0.3430 | 5 | completed |
| imdb | full_rplus_blocknorm | 5.0% | relation_linear | metapath | clipped | 0.3791 | 0.3506 | 5 | completed |
| imdb | full_rplus_blocknorm | 5.0% | relation_linear | metapath | class_balanced | 0.3685 | 0.3506 | 5 | completed |
| imdb | full_rplus_shadow_fusion | 0.5% | shadow_fusion | metapath | clipped | 0.3407 | 0.3326 | 5 | completed |
| imdb | full_rplus_shadow_fusion | 0.5% | shadow_fusion | metapath | class_balanced | 0.3339 | 0.3256 | 5 | completed |
| imdb | full_rplus_shadow_fusion | 2.5% | shadow_fusion | metapath | clipped | 0.3848 | 0.3669 | 5 | completed |
| imdb | full_rplus_shadow_fusion | 2.5% | shadow_fusion | metapath | class_balanced | 0.4076 | 0.3842 | 5 | completed |
| imdb | full_rplus_shadow_fusion | 5.0% | shadow_fusion | metapath | clipped | 0.3819 | 0.3365 | 5 | completed |
| imdb | full_rplus_shadow_fusion | 5.0% | shadow_fusion | metapath | class_balanced | 0.3351 | 0.3297 | 5 | completed |
| imdb | full_rplus_shadow_fusion_adaptive_b | 0.5% | shadow_fusion | metapath | clipped | 0.3407 | 0.3326 | 5 | completed |
| imdb | full_rplus_shadow_fusion_adaptive_b | 0.5% | shadow_fusion | metapath | class_balanced | 0.3339 | 0.3256 | 5 | completed |
| imdb | full_rplus_shadow_fusion_adaptive_b | 2.5% | shadow_fusion | metapath | clipped | 0.3848 | 0.3669 | 5 | completed |
| imdb | full_rplus_shadow_fusion_adaptive_b | 2.5% | shadow_fusion | metapath | class_balanced | 0.4076 | 0.3842 | 5 | completed |
| imdb | full_rplus_shadow_fusion_adaptive_b | 5.0% | shadow_fusion | metapath | clipped | 0.3819 | 0.3365 | 5 | completed |
| imdb | full_rplus_shadow_fusion_adaptive_b | 5.0% | shadow_fusion | metapath | class_balanced | 0.3351 | 0.3297 | 5 | completed |

## Best Rows

- Best accuracy: `0.4076` from `imdb / full_rplus_shadow_fusion` at `2.5%`.
- Best macro-F1: `0.3842` from `imdb / full_rplus_shadow_fusion` at `2.5%`.

## Comparison To R+ Best

- imdb: R++ best `0.4076` vs R+ `0.3810`.

## Compression And Resource Accounting

| Dataset | Variant | Eff target ratio | Total node ratio | Edge ratio | Byte ratio | CPU RAM | GPU RAM | Disk bytes |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| imdb | full_rplus_current | 0.0146 | 0.0059 | 0.0014 | 0.0058 | 2629566464 | 0 | 0 |
| imdb | full_rplus_current | 0.0146 | 0.0059 | 0.0014 | 0.0058 | 2984988672 | 0 | 0 |
| imdb | full_rplus_current | 0.0248 | 0.0111 | 0.0024 | 0.0110 | 2990084096 | 0 | 0 |
| imdb | full_rplus_current | 0.0248 | 0.0111 | 0.0024 | 0.0110 | 2987089920 | 0 | 0 |
| imdb | full_rplus_current | 0.0503 | 0.0225 | 0.0048 | 0.0222 | 2993119232 | 0 | 0 |
| imdb | full_rplus_current | 0.0503 | 0.0225 | 0.0048 | 0.0222 | 2994163712 | 0 | 0 |
| imdb | full_rplus_blocknorm | 0.0146 | 0.0059 | 0.0014 | 0.0058 | 2989223936 | 0 | 0 |
| imdb | full_rplus_blocknorm | 0.0146 | 0.0059 | 0.0014 | 0.0058 | 2990383104 | 0 | 0 |
| imdb | full_rplus_blocknorm | 0.0248 | 0.0111 | 0.0024 | 0.0109 | 2991304704 | 0 | 0 |
| imdb | full_rplus_blocknorm | 0.0248 | 0.0111 | 0.0024 | 0.0109 | 2990260224 | 0 | 0 |
| imdb | full_rplus_blocknorm | 0.0503 | 0.0225 | 0.0048 | 0.0222 | 2995609600 | 0 | 0 |
| imdb | full_rplus_blocknorm | 0.0503 | 0.0225 | 0.0048 | 0.0222 | 2997190656 | 0 | 0 |
| imdb | full_rplus_shadow_fusion | 0.0146 | 0.0059 | 0.0014 | 0.0058 | 3003023360 | 0 | 0 |
| imdb | full_rplus_shadow_fusion | 0.0146 | 0.0059 | 0.0014 | 0.0058 | 3003101184 | 0 | 0 |
| imdb | full_rplus_shadow_fusion | 0.0248 | 0.0111 | 0.0024 | 0.0109 | 2999914496 | 0 | 0 |
| imdb | full_rplus_shadow_fusion | 0.0248 | 0.0111 | 0.0024 | 0.0109 | 3015536640 | 0 | 0 |
| imdb | full_rplus_shadow_fusion | 0.0503 | 0.0225 | 0.0048 | 0.0222 | 3018715136 | 0 | 0 |
| imdb | full_rplus_shadow_fusion | 0.0503 | 0.0225 | 0.0048 | 0.0222 | 3020460032 | 0 | 0 |
| imdb | full_rplus_shadow_fusion_adaptive_b | 0.0146 | 0.0059 | 0.0014 | 0.0058 | 3017203712 | 0 | 0 |
| imdb | full_rplus_shadow_fusion_adaptive_b | 0.0146 | 0.0059 | 0.0014 | 0.0058 | 3017359360 | 0 | 0 |
| imdb | full_rplus_shadow_fusion_adaptive_b | 0.0248 | 0.0111 | 0.0024 | 0.0109 | 3000668160 | 0 | 0 |
| imdb | full_rplus_shadow_fusion_adaptive_b | 0.0248 | 0.0111 | 0.0024 | 0.0109 | 3015716864 | 0 | 0 |
| imdb | full_rplus_shadow_fusion_adaptive_b | 0.0503 | 0.0225 | 0.0048 | 0.0222 | 3006943232 | 0 | 0 |
| imdb | full_rplus_shadow_fusion_adaptive_b | 0.0503 | 0.0225 | 0.0048 | 0.0222 | 3018960896 | 0 | 0 |

## Diagnostics

| Dataset | Variant | Entropy | Relation gates | Block gates | Skel cov | Residual energy | Recon err |
|---|---|---:|---|---|---:|---:|---:|
| imdb | full_rplus_current | 1.4872 | `{"actor--acts_in-->movie": 1.4038727283477783, "director--directs-->movie": 1.4090012311935425, "keyword--keyword_in-->movie": 0.7894238233566284}` | `{}` | 0.0000 | 1.0000 | 0.0105 |
| imdb | full_rplus_current | 1.5093 | `{"actor--acts_in-->movie": 1.3705196380615234, "director--directs-->movie": 1.3743420839309692, "keyword--keyword_in-->movie": 0.7741811275482178}` | `{}` | 0.0000 | 1.0000 | 0.0105 |
| imdb | full_rplus_current | 1.5258 | `{"actor--acts_in-->movie": 1.2796980142593384, "director--directs-->movie": 1.2828339338302612, "keyword--keyword_in-->movie": 0.8077306151390076}` | `{}` | 0.0000 | 1.0000 | 0.0104 |
| imdb | full_rplus_current | 1.4584 | `{"actor--acts_in-->movie": 1.2804796695709229, "director--directs-->movie": 1.3120495080947876, "keyword--keyword_in-->movie": 0.8037439584732056}` | `{}` | 0.0000 | 1.0000 | 0.0104 |
| imdb | full_rplus_current | 1.5699 | `{"actor--acts_in-->movie": 1.33412504196167, "director--directs-->movie": 1.3289875984191895, "keyword--keyword_in-->movie": 0.7906262874603271}` | `{}` | 0.0000 | 1.0000 | 0.0148 |
| imdb | full_rplus_current | 1.5352 | `{"actor--acts_in-->movie": 1.312472939491272, "director--directs-->movie": 1.3163669109344482, "keyword--keyword_in-->movie": 0.7926567196846008}` | `{}` | 0.0000 | 1.0000 | 0.0148 |
| imdb | full_rplus_blocknorm | 1.5480 | `{"actor--acts_in-->movie": 1.3821098804473877, "director--directs-->movie": 1.3943910598754883, "keyword--keyword_in-->movie": 0.7720992565155029}` | `{}` | 0.0000 | 1.0000 | 0.0116 |
| imdb | full_rplus_blocknorm | 1.5231 | `{"actor--acts_in-->movie": 1.3590890169143677, "director--directs-->movie": 1.3755561113357544, "keyword--keyword_in-->movie": 0.7650600671768188}` | `{}` | 0.0000 | 1.0000 | 0.0116 |
| imdb | full_rplus_blocknorm | 1.5342 | `{"actor--acts_in-->movie": 1.4115772247314453, "director--directs-->movie": 1.4191718101501465, "keyword--keyword_in-->movie": 0.7603440880775452}` | `{}` | 0.0000 | 1.0000 | 0.0090 |
| imdb | full_rplus_blocknorm | 1.4753 | `{"actor--acts_in-->movie": 1.3832594156265259, "director--directs-->movie": 1.3993034362792969, "keyword--keyword_in-->movie": 0.7526576519012451}` | `{}` | 0.0000 | 1.0000 | 0.0090 |
| imdb | full_rplus_blocknorm | 1.5507 | `{"actor--acts_in-->movie": 1.3707966804504395, "director--directs-->movie": 1.362994909286499, "keyword--keyword_in-->movie": 0.7836158275604248}` | `{}` | 0.0000 | 1.0000 | 0.0139 |
| imdb | full_rplus_blocknorm | 1.5829 | `{"actor--acts_in-->movie": 1.4117615222930908, "director--directs-->movie": 1.4143145084381104, "keyword--keyword_in-->movie": 0.7546306848526001}` | `{}` | 0.0000 | 1.0000 | 0.0139 |
| imdb | full_rplus_shadow_fusion | 1.5601 | `{"actor--acts_in-->movie": 1.4926128387451172, "director--directs-->movie": 1.9088774919509888, "keyword--keyword_in-->movie": 0.7292864918708801}` | `{}` | 0.0000 | 1.0000 | 0.0116 |
| imdb | full_rplus_shadow_fusion | 1.5710 | `{"actor--acts_in-->movie": 2.370387554168701, "director--directs-->movie": 2.2202494144439697, "keyword--keyword_in-->movie": 0.6786882877349854}` | `{}` | 0.0000 | 1.0000 | 0.0116 |
| imdb | full_rplus_shadow_fusion | 1.5921 | `{"actor--acts_in-->movie": 2.8378758430480957, "director--directs-->movie": 2.8633270263671875, "keyword--keyword_in-->movie": 0.732187032699585}` | `{}` | 0.0000 | 1.0000 | 0.0090 |
| imdb | full_rplus_shadow_fusion | 1.5576 | `{"actor--acts_in-->movie": 4.178904056549072, "director--directs-->movie": 4.289737701416016, "keyword--keyword_in-->movie": 0.5868607759475708}` | `{}` | 0.0000 | 1.0000 | 0.0090 |
| imdb | full_rplus_shadow_fusion | 1.4795 | `{"actor--acts_in-->movie": 2.5026094913482666, "director--directs-->movie": 2.4385318756103516, "keyword--keyword_in-->movie": 0.7401094436645508}` | `{}` | 0.0000 | 1.0000 | 0.0139 |
| imdb | full_rplus_shadow_fusion | 1.5863 | `{"actor--acts_in-->movie": 2.203131914138794, "director--directs-->movie": 2.309816837310791, "keyword--keyword_in-->movie": 0.7293660044670105}` | `{}` | 0.0000 | 1.0000 | 0.0139 |
| imdb | full_rplus_shadow_fusion_adaptive_b | 1.5601 | `{"actor--acts_in-->movie": 1.4926128387451172, "director--directs-->movie": 1.9088774919509888, "keyword--keyword_in-->movie": 0.7292864918708801}` | `{}` | 0.0000 | 1.0000 | 0.0116 |
| imdb | full_rplus_shadow_fusion_adaptive_b | 1.5710 | `{"actor--acts_in-->movie": 2.370387554168701, "director--directs-->movie": 2.2202494144439697, "keyword--keyword_in-->movie": 0.6786882877349854}` | `{}` | 0.0000 | 1.0000 | 0.0116 |
| imdb | full_rplus_shadow_fusion_adaptive_b | 1.5921 | `{"actor--acts_in-->movie": 2.8378758430480957, "director--directs-->movie": 2.8633270263671875, "keyword--keyword_in-->movie": 0.732187032699585}` | `{}` | 0.0000 | 1.0000 | 0.0090 |
| imdb | full_rplus_shadow_fusion_adaptive_b | 1.5576 | `{"actor--acts_in-->movie": 4.178904056549072, "director--directs-->movie": 4.289737701416016, "keyword--keyword_in-->movie": 0.5868607759475708}` | `{}` | 0.0000 | 1.0000 | 0.0090 |
| imdb | full_rplus_shadow_fusion_adaptive_b | 1.4795 | `{"actor--acts_in-->movie": 2.5026094913482666, "director--directs-->movie": 2.4385318756103516, "keyword--keyword_in-->movie": 0.7401094436645508}` | `{}` | 0.0000 | 1.0000 | 0.0139 |
| imdb | full_rplus_shadow_fusion_adaptive_b | 1.5863 | `{"actor--acts_in-->movie": 2.203131914138794, "director--directs-->movie": 2.309816837310791, "keyword--keyword_in-->movie": 0.7293660044670105}` | `{}` | 0.0000 | 1.0000 | 0.0139 |

## Interpretation

- R++ rows are single seed 42 and should be interpreted as sprint diagnostics, not final multi-seed claims.
- A row is considered scalable only when it reports completion rather than OOM/OOT.
- Next recommendation is to keep R-1 defaults frozen and promote only opt-in R++ settings that improve accuracy without class collapse.

## Files

- CSV: `experiments/tables/imdb_rpp_rescue_seed42.csv`
- Report: `experiments\reports\imdb_rpp_rescue_summary.md`
