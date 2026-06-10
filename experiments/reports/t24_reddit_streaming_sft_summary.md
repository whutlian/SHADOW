# T24 Reddit Streaming SFT

This row trains/evaluates on CPU/memmap-resident streaming preprop blocks with GPU mini-batches. It does not load the full edge index during training/eval.

| dataset | status | run_mode | model_type | loss_type | hidden_dim | training_epochs | accuracy | macro_f1 | predicted_class_count | peak_cpu_ram_gb | peak_gpu_ram_gb |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Reddit | completed_streaming_sft | lazy_memmap_cuda | sagn_lite_v4 | sqrt_weighted_ce | 128 | 30 | 0.9400570884871551 | 0.9110379599379667 | 41 | 1.4455795288085938 | 0.9631681442260742 |

- CSV: `experiments\tables\t24_reddit_streaming_sft_seed42.csv`
