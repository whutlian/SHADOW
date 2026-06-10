# T23 Ultra SFT Dry-Run

| dataset | cache_mode | total_cache_bytes | full_edge_scans | peak_cpu_ram_estimate_gb | peak_gpu_ram_estimate_gb | server_recommended | ultra_policy |
|---|---|---|---|---|---|---|---|
| ogbn-arxiv | all_target_rows | 216759040 | 6 | 2.030280888080597 | 1.0 | False | local_feasible |
| ogbn-arxiv | train_target_only | 116404480 | 6 | 2.0162615180015564 | 1.0 | False | local_feasible |
| ogbn-products | all_target_rows | 3237616338 | 6 | 2.4522897775284944 | 1.0 | False | local_feasible |
| ogbn-products | train_target_only | 259925030 | 6 | 2.036311107221991 | 1.0 | False | local_feasible |
| ogbn-papers100M | all_target_rows | 173253531360 | 6 | 26.203238733112812 | 1.0 | True | train_target_only_required |
| ogbn-papers100M | train_target_only | 1883199240 | 6 | 2.263079894706607 | 1.0 | True | train_target_only_required |
| MAG240M | all_target_rows | 176052909036 | 6 | 26.594307276792822 | 1.0 | True | train_target_only_required |
| MAG240M | train_target_only | 1608518832 | 6 | 2.2247074849903585 | 1.0 | True | train_target_only_required |

- CSV: `experiments\tables\t23_scalability_dry_run_seed42.csv`
