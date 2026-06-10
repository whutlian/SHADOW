# T2.1 ogbn-arxiv Lazy SFT

This row uses CPU/memmap-resident T2.1 preprop blocks and GPU mini-batch SFT. It does not load full edge_index during training/eval.

| dataset | status | run_mode | model_type | loss_type | hidden_dim | accuracy | macro_f1 | predicted_class_count | peak_cpu_ram_gb | peak_gpu_ram_gb |
|---|---|---|---|---|---|---|---|---|---|---|
| ogbn-arxiv | completed | lazy_memmap_cuda | gamlp_lite | sqrt_weighted_ce | 256 | 0.6315865275806021 | 0.43814341149277886 | 40 | 1.3625717163085938 | 0.49083709716796875 |

- CSV: `experiments\tables\t21_arxiv_lazy_perf_gamlp_h256_sqrt_e100_seed42.csv`
