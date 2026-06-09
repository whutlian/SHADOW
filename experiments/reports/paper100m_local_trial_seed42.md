# paper100M Local Trial Seed 42

| Field | Value |
|---|---|
| status | completed_smoke |
| dataset_root | D:\Shadow-HGC\dataset\paper100M |
| memmap_root | D:\Shadow-HGC\dataset\paper100M\processed\papers100m_memmap |
| sample_train | 20000 |
| sample_valid | 5000 |
| train_accuracy | 0.3817499876022339 |
| valid_accuracy | 0.33799999952316284 |
| full_scale_local_status | blocked_resource_guard |
| needs_server_run | True |
| peak_ram_estimate_gb | 115.367639968 |
| available_ram_gb | 23.427371008 |
| full_edge_scans | 4 |
| disk_spill_used | False |

## Server Commands

```powershell
& 'C:\Users\slian\anaconda3\envs\pytorch\python.exe' scripts/run_paper100m_local_trial.py --dataset-root D:/Shadow-HGC/dataset/paper100M --output-dir experiments/logs/paper100m_local_trial_seed42 --seed 42 --sample-train 200000 --sample-valid 50000 --epochs 50 --full-scale --no-diffusion
```
```powershell
& 'C:\Users\slian\anaconda3\envs\pytorch\python.exe' scripts/dry_run_ultra.py --dataset ogbn-papers100M --ratios 0.001 0.0025 0.005 --output experiments/logs/paper100m_local_trial_seed42/paper100m_ultra_dry_run_server.json
```
```powershell
python scripts/run_paper100m_local_trial.py --dataset-root /path/to/paper100M --output-dir experiments/logs/paper100m_server_seed42 --seed 42 --sample-train 200000 --sample-valid 50000 --epochs 50 --full-scale --no-diffusion
```
