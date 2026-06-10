# T2.1 ogbn-products Full Execution

This row never uses bounded edges, logits, KD, dense P2, legacy diffusion, or E x d materialization. If `--run-full` is not supplied, the row is explicitly blocked rather than promoted.

| dataset | status | run_mode | accuracy | macro_f1 | full_edge_scans | total_cache_bytes | reason |
|---|---|---|---|---|---|---|---|
| ogbn-products | completed | lazy_memmap_cuda | 0.7029715452279188 | 0.3420856155760991 | 2 | 940427136 | lazy_memmap_gpu_sft_completed |

- CSV: `experiments\tables\t21_products_lazy_perf_gamlp_h512_sqrt_e100_seed42.csv`
