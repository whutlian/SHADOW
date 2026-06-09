# T0-S Scalable Fullgraph Parity Summary

This document summarizes only the T0-S stage. All rows are seed 42 and were run from the local conda `pytorch` environment.

## What Changed

- Added opt-in SCAP blocks under `shadow_hgc/features/` for train-label-only target-target and non-target source class-affinity propagation.
- Added small/medium-safe Path-SCAP diagnostic blocks for available two-hop target-source-target schema paths; longer unavailable paths are logged as skipped instead of synthesized with dense P2.
- Added opt-in SFB under `shadow_hgc/fullgraph/` with residual-logit block fusion, positive gates, train-row-fitted frozen block stats, and no final ReLU.
- Added T0-S gate/resource helpers and scripts for fullgraph parity, scalability stress dry-runs, paper100M/MAG240M dry-runs, and gated condensation recovery.
- The default Shadow-HGC-R-1 condensation scripts are unchanged; T0-S is an explicit fullgraph diagnostic/recovery stage.

## Fullgraph Parity

| dataset | variant | status | accuracy | gate_acc | gate_acc_passed | gate_scalability_passed | blocked_reason |
| --- | --- | --- | --- | --- | --- | --- | --- |
| acm | t0s_sfb_scap | completed | 0.9211520552635193 | 0.93 | False | True |  |
| dblp | t0s_sfb_scap_full_schema | completed | 0.7133802771568298 | 0.91 | False | True |  |
| imdb | t0s_sfb_scap | completed | 0.36851966381073 | 0.6 | False | True |  |
| ogbn-arxiv | t0s_sfb_scap_resource_guard | skipped_resource_guard |  | 0.7 | False | True |  |
| ogbn-products | t0s_sfb_scap_resource_guard | skipped_resource_guard |  | 0.72 | False | True |  |

## Fullgraph Metrics

| dataset | accuracy | macro_f1 | weighted_f1 | predicted_class_count | training_time_s | wall_time_s | peak_cpu_ram_gb |
| --- | --- | --- | --- | --- | --- | --- | --- |
| acm | 0.9211520552635193 | 0.9218372305234274 | 0.9210340206141246 | 3 | 11.793390599999999 | 12.9362687 | 0.79034423828125 |
| dblp | 0.7133802771568298 | 0.7066751569509506 | 0.7129611080484841 | 4 | 5.0727504 | 5.306747999999999 | 1.04327392578125 |
| imdb | 0.36851966381073 | 0.2857880204916 | 0.33428405251370086 | 5 | 15.727901 | 16.059436199999997 | 1.5378074645996094 |
| ogbn-arxiv |  |  |  |  |  | 0.0 | 1.5023002624511719 |
| ogbn-products |  |  |  |  |  | 0.0 | 1.5023002624511719 |

## Feature Blocks

| dataset | feature_blocks | scap_blocks | path_scap_blocks | cache_bytes | full_edge_scans |
| --- | --- | --- | --- | --- | --- |
| acm | ["self", "scap:cite_ref", "scap:cited_by", "scap:writes", "scap:subject_of", "scap:term_in", "path_scap:PAP", "path_scap:PSP", "path_scap:PTP"] | ["scap:cite_ref", "scap:cited_by", "scap:writes", "scap:subject_of", "scap:term_in"] | ["path_scap:PAP", "path_scap:PSP", "path_scap:PTP"] | 145200 | 2 |
| dblp | ["self", "scap:written_by", "path_scap:APA"] | ["scap:written_by"] | ["path_scap:APA"] | 64912 | 2 |
| imdb | ["self", "scap:directs", "scap:acts_in", "scap:keyword_in", "path_scap:MAM", "path_scap:MDM", "path_scap:MKM"] | ["scap:directs", "scap:acts_in", "scap:keyword_in"] | ["path_scap:MAM", "path_scap:MDM", "path_scap:MKM"] | 295920 | 2 |
| ogbn-arxiv | ["self", "scap:incoming_target_relations"] | ["resource_guarded"] | [] | 0 | 2 |
| ogbn-products | ["self", "scap:incoming_target_relations"] | ["resource_guarded"] | [] | 0 | 2 |

## Scalability Stress

| dataset | status | num_nodes | num_edges | scap_topk | full_edge_scans | disk_cache_gb | valid | reasons |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ogbn-arxiv | dry_run_estimate | 169343 | 2332486 | 8 | 2 | 0.09429831802845001 | True | [] |
| ogbn-products | dry_run_estimate | 2449029 | 123718280 | 8 | 2 | 0.25596971809864044 | True | [] |
| ogbn-papers100M | dry_run_estimate | 111059956 | 1615685872 | 8 | 2 | 6.116022527217865 | True | [] |
| mag240m | dry_run_estimate | 121751666 | 1728364232 | 8 | 2 | 11.80788168311119 | True | [] |

## Condensation Recovery Gate

| dataset | fullgraph_variant | fullgraph_accuracy | fullgraph_gate_passed | condensation_status | promoted |
| --- | --- | --- | --- | --- | --- |
| acm | t0s_sfb_scap | 0.9211520552635193 | False | blocked_by_t0s_fullgraph_gate | False |
| dblp | t0s_sfb_scap_full_schema | 0.7133802771568298 | False | blocked_by_t0s_fullgraph_gate | False |
| imdb | t0s_sfb_scap | 0.36851966381073 | False | blocked_by_t0s_fullgraph_gate | False |
| ogbn-arxiv | t0s_sfb_scap_resource_guard |  | False | blocked_by_t0s_fullgraph_gate | False |
| ogbn-products | t0s_sfb_scap_resource_guard |  | False | blocked_by_t0s_fullgraph_gate | False |

## Dry-Run Artifacts

- paper100M: `experiments\tables\t0s_paper100m_dry_run_seed42.json`
- MAG240M: `experiments\tables\t0s_mag240m_dry_run_seed42.json`

## Artifact Index

- `experiments/tables/t0s_fullgraph_parity_seed42.csv`
- `experiments/reports/t0s_fullgraph_parity_summary.md`
- `experiments/tables/t0s_scalability_stress_seed42.csv`
- `experiments/tables/t0s_scalability_stress_seed42.json`
- `experiments/reports/t0s_scalability_stress_summary.md`
- `experiments/tables/t0s_condensation_recovery_seed42.csv`
- `experiments/reports/t0s_condensation_recovery_summary.md`
- `experiments/logs/t0s_fullgraph_parity_seed42/*.json`

## Completion Check

- `shadow_hgc/features/scap.py`, `scap_blocks.py`, and `scap_io.py` exist and are used by the T0-S parity script.
- `shadow_hgc/fullgraph/sfb.py`, `sfb_model.py`, `sfb_train.py`, `sfb_infer.py`, `sfb_logging.py`, and `t0s_gates.py` exist.
- `scripts/run_t0s_fullgraph_parity.py`, `run_t0s_scalability_stress.py`, `dry_run_t0s_paper100m.py`, `dry_run_t0s_mag240m.py`, `run_t0s_condensation_recovery.py`, and `run_t0s_stage.py` exist.
- SCAP train-label-only tests: passed.
- SFB raw-logit/gate/stat-freeze tests: passed.
- T0-S no diffusion/dense-P2/fullgraph-backprop gate tests: passed.
- Scalability resource schema tests: passed.
- Medium and ultra rows are resource-guarded/dry-run on this desktop to avoid the previous OOM/reboot failure mode; this is reported explicitly instead of hidden.
- No T0-S condensation row is promoted because no dataset passed both the fullgraph accuracy and scalability gates.
