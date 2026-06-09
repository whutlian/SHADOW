# T0-S Condensation Recovery Seed 42

| dataset | fullgraph_variant | fullgraph_accuracy | fullgraph_gate_passed | condensation_status | promoted | reason |
| --- | --- | --- | --- | --- | --- | --- |
| acm | t0s_sfb_scap | 0.9211520552635193 | False | blocked_by_t0s_fullgraph_gate | False | completed |
| dblp | t0s_sfb_scap_full_schema | 0.7133802771568298 | False | blocked_by_t0s_fullgraph_gate | False | completed |
| imdb | t0s_sfb_scap | 0.36851966381073 | False | blocked_by_t0s_fullgraph_gate | False | completed |
| ogbn-arxiv | t0s_sfb_scap_resource_guard |  | False | blocked_by_t0s_fullgraph_gate | False | medium fullgraph SFB+SCAP is resource-guarded on this local desktop; no diffusion, dense P2, or full-graph backprop path was executed |
| ogbn-products | t0s_sfb_scap_resource_guard |  | False | blocked_by_t0s_fullgraph_gate | False | medium fullgraph SFB+SCAP is resource-guarded on this local desktop; no diffusion, dense P2, or full-graph backprop path was executed |

No compressed result is promoted unless the corresponding T0-S fullgraph row passes accuracy and scalability gates.

- CSV: `experiments\tables\t0s_condensation_recovery_seed42.csv`
