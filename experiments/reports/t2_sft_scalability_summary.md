# T2-SFT-NL Scalability Dry-Run

Dry-runs estimate memmap cache size and scans only; they do not allocate dense P2, logits, bounded edges, or E x d tensors.

| dataset | cache_mode | total_cache_bytes | full_edge_scans | wall_time_category | server_recommended | uses_logits_as_input | uses_dense_p2 |
|---|---|---|---|---|---|---|---|
| ogbn-arxiv | all_target_rows | 121926960 | 6 | local_short | False | False | False |
| ogbn-products | all_target_rows | 1797587286 | 6 | local_long | False | False | False |
| ogbn-papers100M | train_target_only | 109282996704 | 6 | server_recommended | True | False | False |
| MAG240M | train_target_only | 115177076036 | 6 | server_recommended | True | False | False |

- CSV: `experiments\tables\t2_sft_scalability_dry_run_seed42.csv`
