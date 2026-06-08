# Small R++ Non-Regression Summary

## Completed / OOM / Failed Rows

| Dataset | Variant | Ratio | Model | Feature | Loss | Acc | Macro-F1 | Pred Classes | Status |
|---|---|---:|---|---|---|---:|---:|---:|---|
| acm | current_best | 9.6% | relation_linear | metapath | clipped | 0.8432 | 0.8462 | 3 | completed |
| acm | shadow_fusion_blocknorm | 9.6% | shadow_fusion | metapath | clipped | 0.8187 | 0.8192 | 3 | completed |
| dblp | current_best | 0.5% | relation_linear | metapath | clipped | 0.8282 | 0.8214 | 4 | completed |
| dblp | shadow_fusion_blocknorm | 0.5% | shadow_fusion | metapath | clipped | 0.7158 | 0.7127 | 4 | completed |
| dblp | current_best | 6.5% | relation_linear | metapath | clipped | 0.8370 | 0.8299 | 4 | completed |
| dblp | shadow_fusion_blocknorm | 6.5% | shadow_fusion | metapath | clipped | 0.7092 | 0.7033 | 4 | completed |

## Best Rows

- Best accuracy: `0.8432` from `acm / current_best` at `9.6%`.
- Best macro-F1: `0.8462` from `acm / current_best` at `9.6%`.

## Comparison To R+ Best

- acm: R++ best `0.8432` vs R+ `0.8432`.
- dblp: R++ best `0.8370` vs R+ `0.8370`.

## Compression And Resource Accounting

| Dataset | Variant | Eff target ratio | Total node ratio | Edge ratio | Byte ratio | CPU RAM | GPU RAM | Disk bytes |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| acm | current_best | 0.0959 | 0.0246 | 0.0023 | 0.0235 | 8207478784 | 0 | 0 |
| acm | shadow_fusion_blocknorm | 0.0959 | 0.0268 | 0.0031 | 0.0255 | 12232429568 | 0 | 0 |
| dblp | current_best | 0.0131 | 0.0018 | 0.0008 | 0.0023 | 12949049344 | 0 | 0 |
| dblp | shadow_fusion_blocknorm | 0.0131 | 0.0018 | 0.0008 | 0.0023 | 13619945472 | 0 | 0 |
| dblp | current_best | 0.0649 | 0.0091 | 0.0040 | 0.0113 | 13620387840 | 0 | 0 |
| dblp | shadow_fusion_blocknorm | 0.0649 | 0.0091 | 0.0040 | 0.0113 | 13632843776 | 0 | 0 |

## Diagnostics

| Dataset | Variant | Entropy | Relation gates | Block gates | Skel cov | Residual energy | Recon err |
|---|---|---:|---|---|---:|---:|---:|
| acm | current_best | 1.0950 | `{"author--writes-->paper": 1.7146743535995483, "paper--cite_ref-->paper": 0.5746864676475525, "paper--cited_by-->paper": 0.701059877872467, "subject--subject_of-->paper": 0.7823005318641663, "term--term_in-->paper": 0.6962363719940186}` | `{}` | 0.3056 | 0.9960 | 0.1334 |
| acm | shadow_fusion_blocknorm | 1.0782 | `{"author--writes-->paper": 0.9656726121902466, "paper--cite_ref-->paper": 0.8171812891960144, "paper--cited_by-->paper": 0.941821813583374, "subject--subject_of-->paper": 0.6679457426071167, "term--term_in-->paper": 0.7949249744415283}` | `{}` | 0.3368 | 1.0000 | 0.1016 |
| dblp | current_best | 1.3741 | `{"paper--written_by-->author": 0.9680579900741577}` | `{}` | 0.0000 | 1.0000 | 0.0055 |
| dblp | shadow_fusion_blocknorm | 1.3847 | `{"paper--written_by-->author": 2.877674102783203}` | `{}` | 0.0000 | 1.0000 | 0.0139 |
| dblp | current_best | 1.3668 | `{"paper--written_by-->author": 1.0525171756744385}` | `{}` | 0.0000 | 1.0000 | 0.0097 |
| dblp | shadow_fusion_blocknorm | 1.3615 | `{"paper--written_by-->author": 2.3655858039855957}` | `{}` | 0.0000 | 1.0000 | 0.0201 |

## Interpretation

- R++ rows are single seed 42 and should be interpreted as sprint diagnostics, not final multi-seed claims.
- A row is considered scalable only when it reports completion rather than OOM/OOT.
- Next recommendation is to keep R-1 defaults frozen and promote only opt-in R++ settings that improve accuracy without class collapse.

## Files

- CSV: `experiments/tables/small_rpp_nonregression_seed42.csv`
- Report: `experiments\reports\small_rpp_nonregression_summary.md`
