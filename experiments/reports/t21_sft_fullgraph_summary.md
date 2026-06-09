# T2.1 Fullgraph SFT Table

This table contains no-logits fullgraph SFT rows plus the products execution row when available.

| dataset | status | accuracy | macro_f1 | predicted_class_count | selected_blocks | reason |
|---|---|---|---|---|---|---|
| acm | promoted | 0.9206798672676086 | 0.9211405118306478 | 3 | ["B0_self", "B1_typed", "B2_metapath", "B3_lad_scap"] | validation_selected_and_safe_improved |
| dblp | promoted | 0.9426056146621704 | 0.9387593418359756 | 4 | ["B0_self", "B1_typed", "B2_metapath", "B3_lad_scap"] | validation_selected_and_safe_improved |
| imdb | promoted | 0.47158026695251465 | 0.3900045245885849 | 5 | ["B0_self", "B1_typed", "B2_metapath", "B4_structure"] | validation_selected_and_safe_improved |
| ogbn-arxiv | blocked_class_collapse | 0.6105796098709106 | 0.32606273433193567 | 27 | ["B0_self", "B1_typed", "B3_lad_scap"] | predicted_class_count<35 |
| ogbn-products | preprop_completed |  |  |  | [] | full_edge_products_preprop_completed |

- CSV: `experiments\tables\t21_sft_fullgraph_seed42.csv`
