# T2.1 Robust Block Selection Audit

Rows reuse the T2 no-logits candidate outcomes and add the T2.1 coverage-aware selection score. No row uses logits, KD, dense P2, bounded edges, or E x d materialization.

| dataset | status | accuracy | macro_f1 | predicted_class_count | selection_score | selected_blocks | reason |
|---|---|---|---|---|---|---|---|
| acm | promoted | 0.9206798672676086 | 0.9211405118306478 | 3 | 1.176967434088389 | ["B0_self", "B1_typed", "B2_metapath", "B3_lad_scap"] | validation_selected_and_safe_improved |
| dblp | promoted | 0.9426056146621704 | 0.9387593418359756 | 4 | 1.1904118627309799 | ["B0_self", "B1_typed", "B2_metapath", "B3_lad_scap"] | validation_selected_and_safe_improved |
| imdb | promoted | 0.47158026695251465 | 0.3900045245885849 | 5 | 0.616212739944458 | ["B0_self", "B1_typed", "B2_metapath", "B4_structure"] | validation_selected_and_safe_improved |
| ogbn-arxiv | blocked_class_collapse | 0.6105796098709106 | 0.32606273433193567 | 27 | 0.7328903097552912 | ["B0_self", "B1_typed", "B3_lad_scap"] | predicted_class_count<35 |

- CSV: `experiments\tables\t21_sft_block_selection_seed42.csv`
