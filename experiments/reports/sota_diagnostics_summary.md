# Shadow-HGC-SOTA Diagnostics Summary

Seed `42`; diffusion is disabled and remains diagnostic-only.

## Best Rows
| dataset | variant | requested_ratio | requested_full_condensed_node_ratio | accuracy | macro_f1 | prototype_mode | teacher_type | use_kd |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| acm | Medoid-vs-Mean | 0.096 |  | 0.8682719469070435 | 0.8684150377909342 | kmeans_mean | none | False |
| imdb | PathLAD-off | 0.048 |  | 0.30387258529663086 | 0.23062760829925538 | coverage_medoid | none | False |

## All Rows
| dataset | variant | requested_ratio | requested_full_condensed_node_ratio | accuracy | macro_f1 | predicted_class_count | total_condensed_node_ratio | status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| acm | Medoid-vs-Mean | 0.096 |  | 0.8682719469070435 | 0.8684150377909342 | 3 | 0.024584171083896912 | completed |
| acm | CoverageMedoid | 0.096 |  | 0.7455146312713623 | 0.7218442956606547 | 3 | 0.027051727289343814 | completed |
| imdb | PathLAD-off | 0.048 |  | 0.30387258529663086 | 0.23062760829925538 | 5 | 0.021568627450980392 | completed |
| imdb | PathLAD-on | 0.048 |  | 0.28107431530952454 | 0.23565983027219772 | 5 | 0.021568627450980392 | completed |
| imdb | KD-off | 0.048 |  | 0.28107431530952454 | 0.23565983027219772 | 5 | 0.021568627450980392 | completed |
| imdb | KD-on | 0.048 |  | 0.25515303015708923 | 0.20060989558696746 | 5 | 0.021568627450980392 | completed |

## Failed / OOM / Timeout Rows
None.
