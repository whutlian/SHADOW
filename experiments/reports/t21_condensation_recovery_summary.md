# T2.1 Condensation Recovery Summary

Identity replay is diagnostic and uses frozen SFT block signatures, not logits as input. DBLP prototype/shadow recovery is marked as started because DBLP is the immediate recovery target.

| dataset | recovery_row | fullgraph_accuracy | status | reason |
|---|---|---|---|---|
| acm | identity_condensed_sft_replay | 0.9206798672676086 | completed_diagnostic | identity replay of frozen SFT block signature |
| acm | prototype_oracle_sft_block_signature | 0.9206798672676086 | eligible_not_run | diagnostic recovery eligible after fullgraph gate |
| acm | shadow_condensed_sft_block_signature | 0.9206798672676086 | eligible_not_run | diagnostic recovery eligible after fullgraph gate |
| dblp | identity_condensed_sft_replay | 0.9426056146621704 | completed_diagnostic | identity replay of frozen SFT block signature |
| dblp | prototype_oracle_sft_block_signature | 0.9426056146621704 | started_diagnostic | DBLP SFT block-signature prototype recovery started; compressed accuracy not yet promoted |
| dblp | shadow_condensed_sft_block_signature | 0.9426056146621704 | started_diagnostic | DBLP SFT block-signature prototype recovery started; compressed accuracy not yet promoted |
| imdb | identity_condensed_sft_replay | 0.47158026695251465 | completed_diagnostic | identity replay of frozen SFT block signature |
| imdb | prototype_oracle_sft_block_signature | 0.47158026695251465 | eligible_not_run | diagnostic recovery eligible after fullgraph gate |
| imdb | shadow_condensed_sft_block_signature | 0.47158026695251465 | eligible_not_run | diagnostic recovery eligible after fullgraph gate |
| ogbn-arxiv | recovery_gate | 0.6105796098709106 | blocked_by_t21_fullgraph_gate | predicted_class_count<35 |
| ogbn-products | recovery_gate |  | blocked_by_t21_fullgraph_gate | full_edge_products_preprop_completed |

- CSV: `experiments\tables\t21_sft_condensation_recovery_seed42.csv`
