# T2-SFT-NL Fullgraph Teacher Summary

Rows are materialized from the validation-selected safe block selection run; no duplicate training is performed by the stage driver.

| dataset | status | accuracy | macro_f1 | predicted_class_count | selected_blocks | reason |
|---|---|---|---|---|---|---|
| acm | promoted | 0.9206798672676086 | 0.9211405118306478 | 3 | ["B0_self", "B1_typed", "B2_metapath", "B3_lad_scap"] | validation_selected_and_safe_improved |
| dblp | promoted | 0.9426056146621704 | 0.9387593418359756 | 4 | ["B0_self", "B1_typed", "B2_metapath", "B3_lad_scap"] | validation_selected_and_safe_improved |
| imdb | promoted | 0.47158026695251465 | 0.3900045245885849 | 5 | ["B0_self", "B1_typed", "B2_metapath", "B4_structure"] | validation_selected_and_safe_improved |
| ogbn-arxiv | blocked_class_collapse | 0.6105796098709106 | 0.32606273433193567 | 27 | ["B0_self", "B1_typed", "B3_lad_scap"] | predicted_class_count<35 |
| ogbn-products | blocked_resource_guard |  |  |  | [] | products full T2 SFT skipped locally; use --run-products-full after dry-run |

- CSV: `experiments\tables\t2_sft_fullgraph_seed42.csv`
