# T0-S Fullgraph Parity Seed 42

Promoted rows must pass both the dataset accuracy gate and the scalability gate.

| dataset | variant | status | accuracy | gate_acc | gate_acc_passed | gate_scalability_passed | blocked_reason |
| --- | --- | --- | --- | --- | --- | --- | --- |
| acm | t0s_sfb_scap | completed | 0.9211520552635193 | 0.93 | False | True |  |
| dblp | t0s_sfb_scap_full_schema | completed | 0.7133802771568298 | 0.91 | False | True |  |
| imdb | t0s_sfb_scap | completed | 0.36851966381073 | 0.6 | False | True |  |
| ogbn-arxiv | t0s_sfb_scap_resource_guard | skipped_resource_guard |  | 0.7 | False | True |  |
| ogbn-products | t0s_sfb_scap_resource_guard | skipped_resource_guard |  | 0.72 | False | True |  |

Scalability policy: no diffusion, no dense P2, no full-graph backprop, train-label-only SCAP, and no all-target demand cache.

- CSV: `experiments\tables\t0s_fullgraph_parity_seed42.csv`
