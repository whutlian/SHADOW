# T2.1 ogbn-arxiv Lazy SFT

This row uses CPU/memmap-resident T2.1 preprop blocks and GPU mini-batch SFT. It does not load full edge_index during training/eval.

| dataset | status | run_mode | model_type | loss_type | hidden_dim | accuracy | macro_f1 | predicted_class_count | peak_cpu_ram_gb | peak_gpu_ram_gb |
|---|---|---|---|---|---|---|---|---|---|---|
| ogbn-arxiv | completed | lazy_memmap_cuda | gamlp_lite | cross_entropy | 512 | 0.6544040491327696 | 0.420481437217901 | 39 | 1.2698135375976562 | 0.9140005111694336 |

- CSV: `experiments\tables\t21_arxiv_lazy_perf_gamlp_h512_ce_e100_seed42.csv`
