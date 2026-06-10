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
| ogbn-arxiv | recovery_gate | 0.6544040491327696 | not_recovery_target_medium | medium lazy SFT fullgraph row completed; condensation recovery is not in the current T2.1 small-dataset recovery scope |
| ogbn-products | recovery_gate | 0.7029715452279188 | not_recovery_target_medium | medium lazy SFT fullgraph row completed; condensation recovery is not in the current T2.1 small-dataset recovery scope |

- CSV: `experiments\tables\t21_sft_condensation_recovery_seed42.csv`
