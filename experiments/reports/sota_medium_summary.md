# Shadow-HGC-SOTA Medium Summary

Seed `42`; diffusion is disabled and remains diagnostic-only.

## Best Rows
| dataset | variant | requested_ratio | requested_full_condensed_node_ratio | accuracy | macro_f1 | prototype_mode | teacher_type | use_kd |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ogbn-arxiv | S2_coverage_medoids |  | 0.005 | 0.4671933948993683 | 0.2052860957570374 | coverage_medoid | none | False |
| ogbn-products | S0_current_best |  | 0.0005 | 0.5135102868080139 | 0.18708413856183279 | kmeans_mean | none | False |

## All Rows
| dataset | variant | requested_ratio | requested_full_condensed_node_ratio | accuracy | macro_f1 | predicted_class_count | total_condensed_node_ratio | status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ogbn-arxiv | S0_current_best |  | 0.0005 | 0.3409666121006012 | 0.20036885540466756 | 40 | 0.000501939849890459 | completed |
| ogbn-arxiv | S2_coverage_medoids |  | 0.0005 | 0.24883237481117249 | 0.03835016243392601 | 23 | 0.000501939849890459 | completed |
| ogbn-arxiv | S4_teacher_kd |  | 0.0005 | 0.1873341202735901 | 0.0315606064224994 | 5 | 0.000501939849890459 | completed |
| ogbn-arxiv | S0_current_best |  | 0.0025 | 0.3779190480709076 | 0.22030348842963576 | 38 | 0.002497888900043108 | completed |
| ogbn-arxiv | S2_coverage_medoids |  | 0.0025 | 0.38330966234207153 | 0.1439939347677864 | 29 | 0.002497888900043108 | completed |
| ogbn-arxiv | S4_teacher_kd |  | 0.0025 | 0.3736394941806793 | 0.09214521762914955 | 17 | 0.002497888900043108 | completed |
| ogbn-arxiv | S0_current_best |  | 0.005 | 0.44456103444099426 | 0.25993386530317364 | 39 | 0.005001682974790809 | completed |
| ogbn-arxiv | S2_coverage_medoids |  | 0.005 | 0.4671933948993683 | 0.2052860957570374 | 34 | 0.005001682974790809 | completed |
| ogbn-arxiv | S4_teacher_kd |  | 0.005 | 0.2010781168937683 | 0.059561424615094435 | 16 | 0.005001682974790809 | completed |
| ogbn-products | S0_current_best |  | 0.0005 | 0.5135102868080139 | 0.18708413856183279 | 36 | 0.0005001982418338044 | completed |
| ogbn-products | S2_coverage_medoids |  | 0.0005 |  |  |  |  | timeout_dropped |
| ogbn-products | S4_teacher_kd |  | 0.0005 |  |  |  |  | timeout_dropped |
| ogbn-products | S0_current_best |  | 0.0025 |  |  |  |  | timeout_dropped |
| ogbn-products | S2_coverage_medoids |  | 0.0025 |  |  |  |  | timeout_dropped |
| ogbn-products | S4_teacher_kd |  | 0.0025 |  |  |  |  | timeout_dropped |
| ogbn-products | S0_current_best |  | 0.005 |  |  |  |  | timeout_dropped |
| ogbn-products | S2_coverage_medoids |  | 0.005 |  |  |  |  | timeout_dropped |
| ogbn-products | S4_teacher_kd |  | 0.005 |  |  |  |  | timeout_dropped |

## Failed / OOM / Timeout Rows
| dataset | variant | status | reason | source_log |
| --- | --- | --- | --- | --- |
| ogbn-products | S2_coverage_medoids | timeout_dropped | products SOTA medium row was dropped after no new JSON progress during the timeout window | experiments\logs\sota_medium_seed42\ogbn-products_S2_coverage_medoids_fullnode_r0p0005_seed42.json |
| ogbn-products | S4_teacher_kd | timeout_dropped | products SOTA medium row was dropped after no new JSON progress during the timeout window | experiments\logs\sota_medium_seed42\ogbn-products_S4_teacher_kd_fullnode_r0p0005_seed42.json |
| ogbn-products | S0_current_best | timeout_dropped | products SOTA medium row was dropped after no new JSON progress during the timeout window | experiments\logs\sota_medium_seed42\ogbn-products_S0_current_best_fullnode_r0p0025_seed42.json |
| ogbn-products | S2_coverage_medoids | timeout_dropped | products SOTA medium row was dropped after no new JSON progress during the timeout window | experiments\logs\sota_medium_seed42\ogbn-products_S2_coverage_medoids_fullnode_r0p0025_seed42.json |
| ogbn-products | S4_teacher_kd | timeout_dropped | products SOTA medium row was dropped after no new JSON progress during the timeout window | experiments\logs\sota_medium_seed42\ogbn-products_S4_teacher_kd_fullnode_r0p0025_seed42.json |
| ogbn-products | S0_current_best | timeout_dropped | products SOTA medium row was dropped after no new JSON progress during the timeout window | experiments\logs\sota_medium_seed42\ogbn-products_S0_current_best_fullnode_r0p005_seed42.json |
| ogbn-products | S2_coverage_medoids | timeout_dropped | products SOTA medium row was dropped after no new JSON progress during the timeout window | experiments\logs\sota_medium_seed42\ogbn-products_S2_coverage_medoids_fullnode_r0p005_seed42.json |
| ogbn-products | S4_teacher_kd | timeout_dropped | products SOTA medium row was dropped after no new JSON progress during the timeout window | experiments\logs\sota_medium_seed42\ogbn-products_S4_teacher_kd_fullnode_r0p005_seed42.json |
