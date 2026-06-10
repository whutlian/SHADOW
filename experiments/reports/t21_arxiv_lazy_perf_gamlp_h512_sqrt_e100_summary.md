# T2.1 ogbn-arxiv Lazy SFT

This row uses CPU/memmap-resident T2.1 preprop blocks and GPU mini-batch SFT. It does not load full edge_index during training/eval.

| dataset | status | run_mode | model_type | loss_type | hidden_dim | accuracy | macro_f1 | predicted_class_count | peak_cpu_ram_gb | peak_gpu_ram_gb |
|---|---|---|---|---|---|---|---|---|---|---|
| ogbn-arxiv | completed | lazy_memmap_cuda | gamlp_lite | sqrt_weighted_ce | 512 | 0.6383145073349382 | 0.4420411939228622 | 40 | 1.3615455627441406 | 0.9140005111694336 |

- CSV: `experiments\tables\t21_arxiv_lazy_perf_gamlp_h512_sqrt_e100_seed42.csv`
