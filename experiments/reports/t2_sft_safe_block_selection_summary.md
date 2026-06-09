# T2-SFT-NL Safe Block Selection Seed 42

No row uses logits as input, teacher logits, KD, dense P2, bounded edges, or E x d materialization.

## Final Rows

| dataset | status | accuracy | macro_f1 | predicted_class_count | selected_blocks | reason |
|---|---|---|---|---|---|---|
| acm | promoted | 0.9206798672676086 | 0.9211405118306478 | 3 | ["B0_self", "B1_typed", "B2_metapath", "B3_lad_scap"] | validation_selected_and_safe_improved |
| dblp | promoted | 0.9426056146621704 | 0.9387593418359756 | 4 | ["B0_self", "B1_typed", "B2_metapath", "B3_lad_scap"] | validation_selected_and_safe_improved |
| imdb | promoted | 0.47158026695251465 | 0.3900045245885849 | 5 | ["B0_self", "B1_typed", "B2_metapath", "B4_structure"] | validation_selected_and_safe_improved |
| ogbn-arxiv | blocked_class_collapse | 0.6105796098709106 | 0.32606273433193567 | 27 | ["B0_self", "B1_typed", "B3_lad_scap"] | predicted_class_count<35 |
| ogbn-products | blocked_resource_guard |  |  |  | [] | products full T2 SFT skipped locally; use --run-products-full after dry-run |

## Block Decisions

| dataset | block_group | branch_valid_acc | branch_test_acc_debug | kept_or_dropped | drop_reason | gate_value |
|---|---|---|---|---|---|---|
| acm | B1_typed | 0.8618784546852112 | 0.8578848242759705 | kept |  | 0.5135835409164429 |
| acm | B2_metapath | 0.90055251121521 | 0.8923512697219849 | kept |  | 0.5075643658638 |
| acm | B3_lad_scap | 0.939226508140564 | 0.9206798672676086 | kept |  | 0.5415382385253906 |
| acm | B4_structure | 0.9337016344070435 | 0.9093484282493591 | dropped | dropped_by_validation | 0.5717639923095703 |
| dblp | B1_typed | 0.7448559403419495 | 0.7387323975563049 | kept |  | 0.48465263843536377 |
| dblp | B2_metapath | 0.9423868060112 | 0.9492957592010498 | kept |  | 0.5103060603141785 |
| dblp | B3_lad_scap | 0.9506173133850098 | 0.9426056146621704 | kept |  | 0.5124279260635376 |
| dblp | B4_structure | 0.9506173133850098 | 0.9352112412452698 | dropped | dropped_by_validation | 0.518760621547699 |
| imdb | B1_typed | 0.45255473256111145 | 0.446908175945282 | kept |  | 0.35110411047935486 |
| imdb | B2_metapath | 0.4781021773815155 | 0.46002498269081116 | kept |  | 0.17308862507343292 |
| imdb | B3_lad_scap | 0.45985400676727295 | 0.497813880443573 | dropped | dropped_by_validation | 0.12110061943531036 |
| imdb | B4_structure | 0.48905110359191895 | 0.47158026695251465 | kept |  | 0.14528152346611023 |
| ogbn-arxiv | B1_typed | 0.5429041385650635 | 0.5235685110092163 | kept |  | 0.33655062317848206 |
| ogbn-arxiv | B3_lad_scap | 0.6250545382499695 | 0.6105796098709106 | kept |  | 0.20645399391651154 |
| ogbn-arxiv | B4_structure | 0.6170005798339844 | 0.5929057598114014 | dropped | dropped_by_validation | 0.17271816730499268 |

- CSV: `experiments\tables\t2_sft_safe_block_selection_seed42.csv`
