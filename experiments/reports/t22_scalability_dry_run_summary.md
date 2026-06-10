# T2.2 Scalability Dry-Run

| dataset | cache_mode | block_set | total_cache_bytes | peak_cpu_ram_estimate_gb | peak_gpu_ram_estimate_gb | server_recommended |
|---|---|---|---|---|---|---|
| ogbn-arxiv | all_target_rows | X0,X1,X2,X3,Y1,Y2,Y3,structure | 130055424 | 2.018168532848358 | 1.0 | False |
| ogbn-arxiv | train_target_only | X0,X1,X2,X3,Y1,Y2,Y3,structure | 69842688 | 2.0097569108009337 | 1.0 | False |
| ogbn-products | all_target_rows | X0,X1,X2,X3,Y1,Y2,Y3,structure | 1983713490 | 2.2771215732209384 | 1.0 | False |
| ogbn-products | train_target_only | X0,X1,X2,X3,Y1,Y2,Y3,structure | 159258150 | 2.0222481065429747 | 1.0 | False |
| ogbn-papers100M | all_target_rows | X0,X1,X2,Y1,Y2,structure | 120833232128 | 18.880207526683808 | 1.0 | True |
| ogbn-papers100M | train_target_only | X0,X1,X2,Y1,Y2,structure | 1313410752 | 2.183481362462044 | 1.0 | True |
| MAG240M | all_target_rows | X0,X1,X2,Y1,Y2,structure | 123212685992 | 19.21261339150369 | 1.0 | True |
| MAG240M | train_target_only | X0,X1,X2,Y1,Y2,structure | 1125740704 | 2.157264159619808 | 1.0 | True |

- CSV: `experiments\tables\t22_scalability_dry_run_seed42.csv`
