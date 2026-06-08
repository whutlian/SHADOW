# ogbn-products R++ Streaming Diffusion Summary

## Drop Decision

This products path is **dropped from the recommended R++ configuration** for this stage.

Reason:

- The second-stage destination/edge chunking fixed the immediate inference OOM, but the full products run took about 102 minutes.
- Completed `streaming_diffusion_X0X1X2` rows match the `base` rows exactly because the fp16 memmap diffusion blocks were precomputed but not yet wired into the products `phi` lazy feature path.
- `streaming_diffusion_X0X1` and `streaming_diffusion_X0X1X2_highpass` rows failed on Windows memmap path handling (`Errno 22`).
- The stage therefore treats products as a scalability diagnostic only, not as evidence of streaming diffusion accuracy improvement.

## Completed / OOM / Failed Rows

| Dataset | Variant | Ratio | Model | Feature | Loss | Acc | Macro-F1 | Pred Classes | Status |
|---|---|---:|---|---|---|---:|---:|---:|---|
| ogbn-products | base | 2.0% | shadow_fusion | base | sqrt_weighted | 0.5808 | 0.2396 | 41 | completed |
| ogbn-products | base | 2.0% | shadow_fusion | base | sqrt_weighted_logit_adjusted | 0.5830 | 0.2356 | 38 | completed |
| ogbn-products | base | 6.0% | shadow_fusion | base | sqrt_weighted | 0.6390 | 0.2833 | 41 | completed |
| ogbn-products | base | 6.0% | shadow_fusion | base | sqrt_weighted_logit_adjusted | 0.6366 | 0.2700 | 33 | completed |
| ogbn-products | base | 12.0% | shadow_fusion | base | sqrt_weighted | 0.6689 | 0.3080 | 41 | completed |
| ogbn-products | base | 12.0% | shadow_fusion | base | sqrt_weighted_logit_adjusted | 0.6673 | 0.2851 | 32 | completed |
| ogbn-products | streaming_diffusion_X0X1 | 2.0% |  |  | sqrt_weighted |  |  |  | experiment_failed |
| ogbn-products | streaming_diffusion_X0X1 | 2.0% |  |  | sqrt_weighted_logit_adjusted |  |  |  | experiment_failed |
| ogbn-products | streaming_diffusion_X0X1 | 6.0% |  |  | sqrt_weighted |  |  |  | experiment_failed |
| ogbn-products | streaming_diffusion_X0X1 | 6.0% |  |  | sqrt_weighted_logit_adjusted |  |  |  | experiment_failed |
| ogbn-products | streaming_diffusion_X0X1 | 12.0% |  |  | sqrt_weighted |  |  |  | experiment_failed |
| ogbn-products | streaming_diffusion_X0X1 | 12.0% |  |  | sqrt_weighted_logit_adjusted |  |  |  | experiment_failed |
| ogbn-products | streaming_diffusion_X0X1X2 | 2.0% | shadow_fusion | streaming_diffusion_X0X1X2 | sqrt_weighted | 0.5808 | 0.2396 | 41 | completed |
| ogbn-products | streaming_diffusion_X0X1X2 | 2.0% | shadow_fusion | streaming_diffusion_X0X1X2 | sqrt_weighted_logit_adjusted | 0.5830 | 0.2356 | 38 | completed |
| ogbn-products | streaming_diffusion_X0X1X2 | 6.0% | shadow_fusion | streaming_diffusion_X0X1X2 | sqrt_weighted | 0.6390 | 0.2833 | 41 | completed |
| ogbn-products | streaming_diffusion_X0X1X2 | 6.0% | shadow_fusion | streaming_diffusion_X0X1X2 | sqrt_weighted_logit_adjusted | 0.6366 | 0.2700 | 33 | completed |
| ogbn-products | streaming_diffusion_X0X1X2 | 12.0% | shadow_fusion | streaming_diffusion_X0X1X2 | sqrt_weighted | 0.6689 | 0.3080 | 41 | completed |
| ogbn-products | streaming_diffusion_X0X1X2 | 12.0% | shadow_fusion | streaming_diffusion_X0X1X2 | sqrt_weighted_logit_adjusted | 0.6673 | 0.2851 | 32 | completed |
| ogbn-products | streaming_diffusion_X0X1X2_highpass | 2.0% |  |  | sqrt_weighted |  |  |  | experiment_failed |
| ogbn-products | streaming_diffusion_X0X1X2_highpass | 2.0% |  |  | sqrt_weighted_logit_adjusted |  |  |  | experiment_failed |
| ogbn-products | streaming_diffusion_X0X1X2_highpass | 6.0% |  |  | sqrt_weighted |  |  |  | experiment_failed |
| ogbn-products | streaming_diffusion_X0X1X2_highpass | 6.0% |  |  | sqrt_weighted_logit_adjusted |  |  |  | experiment_failed |
| ogbn-products | streaming_diffusion_X0X1X2_highpass | 12.0% |  |  | sqrt_weighted |  |  |  | experiment_failed |
| ogbn-products | streaming_diffusion_X0X1X2_highpass | 12.0% |  |  | sqrt_weighted_logit_adjusted |  |  |  | experiment_failed |

## Best Rows

- Best accuracy: `0.6689` from `ogbn-products / base` at `12.0%`.
- Best macro-F1: `0.3080` from `ogbn-products / base` at `12.0%`.

## Comparison To R+ Best

- ogbn-products: R++ best `0.6689` vs R+ `0.5891`.

## Compression And Resource Accounting

| Dataset | Variant | Eff target ratio | Total node ratio | Edge ratio | Byte ratio | CPU RAM | GPU RAM | Disk bytes |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| ogbn-products | base | 0.0199 | 0.0024 | 0.0001 | 0.0006 | 19605073920 | 0 | 0 |
| ogbn-products | base | 0.0199 | 0.0024 | 0.0001 | 0.0006 | 19689820160 | 0 | 0 |
| ogbn-products | base | 0.0592 | 0.0071 | 0.0003 | 0.0018 | 14387466240 | 0 | 0 |
| ogbn-products | base | 0.0592 | 0.0071 | 0.0003 | 0.0018 | 13795573760 | 0 | 0 |
| ogbn-products | base | 0.1159 | 0.0140 | 0.0005 | 0.0036 | 13697396736 | 0 | 0 |
| ogbn-products | base | 0.1159 | 0.0140 | 0.0005 | 0.0036 | 13722189824 | 0 | 0 |
| ogbn-products | streaming_diffusion_X0X1X2 | 0.0199 | 0.0024 | 0.0001 | 0.0006 | 13687681024 | 0 | 626951424 |
| ogbn-products | streaming_diffusion_X0X1X2 | 0.0199 | 0.0024 | 0.0001 | 0.0006 | 13669380096 | 0 | 626951424 |
| ogbn-products | streaming_diffusion_X0X1X2 | 0.0592 | 0.0071 | 0.0003 | 0.0018 | 13687304192 | 0 | 626951424 |
| ogbn-products | streaming_diffusion_X0X1X2 | 0.0592 | 0.0071 | 0.0003 | 0.0018 | 13654212608 | 0 | 626951424 |
| ogbn-products | streaming_diffusion_X0X1X2 | 0.1159 | 0.0140 | 0.0005 | 0.0036 | 13722529792 | 0 | 626951424 |
| ogbn-products | streaming_diffusion_X0X1X2 | 0.1159 | 0.0140 | 0.0005 | 0.0036 | 13707759616 | 0 | 626951424 |

## Diagnostics

| Dataset | Variant | Entropy | Relation gates | Block gates | Skel cov | Residual energy | Recon err |
|---|---|---:|---|---|---:|---:|---:|
| ogbn-products | base | 2.7695 | `{"product--co_purchase-->product": 2.3003549575805664, "product--co_purchased_by-->product": 2.244420289993286}` | `{}` | 0.3911 | 0.8592 | 0.5359 |
| ogbn-products | base | 2.4574 | `{"product--co_purchase-->product": 1.4078665971755981, "product--co_purchased_by-->product": 1.4636675119400024}` | `{}` | 0.3911 | 0.8592 | 0.5359 |
| ogbn-products | base | 2.7816 | `{"product--co_purchase-->product": 3.3687937259674072, "product--co_purchased_by-->product": 3.4204843044281006}` | `{}` | 0.3380 | 0.8830 | 0.5045 |
| ogbn-products | base | 2.4655 | `{"product--co_purchase-->product": 1.5781244039535522, "product--co_purchased_by-->product": 1.6826430559158325}` | `{}` | 0.3380 | 0.8830 | 0.5045 |
| ogbn-products | base | 2.7525 | `{"product--co_purchase-->product": 3.848414182662964, "product--co_purchased_by-->product": 4.026067733764648}` | `{}` | 0.3026 | 0.8793 | 0.3699 |
| ogbn-products | base | 2.4818 | `{"product--co_purchase-->product": 1.592192530632019, "product--co_purchased_by-->product": 1.7517385482788086}` | `{}` | 0.3026 | 0.8793 | 0.3699 |
| ogbn-products | streaming_diffusion_X0X1X2 | 2.7695 | `{"product--co_purchase-->product": 2.3003549575805664, "product--co_purchased_by-->product": 2.244420289993286}` | `{}` | 0.3911 | 0.8592 | 0.5359 |
| ogbn-products | streaming_diffusion_X0X1X2 | 2.4574 | `{"product--co_purchase-->product": 1.4078665971755981, "product--co_purchased_by-->product": 1.4636675119400024}` | `{}` | 0.3911 | 0.8592 | 0.5359 |
| ogbn-products | streaming_diffusion_X0X1X2 | 2.7816 | `{"product--co_purchase-->product": 3.3687937259674072, "product--co_purchased_by-->product": 3.4204843044281006}` | `{}` | 0.3380 | 0.8830 | 0.5045 |
| ogbn-products | streaming_diffusion_X0X1X2 | 2.4655 | `{"product--co_purchase-->product": 1.5781244039535522, "product--co_purchased_by-->product": 1.6826430559158325}` | `{}` | 0.3380 | 0.8830 | 0.5045 |
| ogbn-products | streaming_diffusion_X0X1X2 | 2.7525 | `{"product--co_purchase-->product": 3.848414182662964, "product--co_purchased_by-->product": 4.026067733764648}` | `{}` | 0.3026 | 0.8793 | 0.3699 |
| ogbn-products | streaming_diffusion_X0X1X2 | 2.4818 | `{"product--co_purchase-->product": 1.592192530632019, "product--co_purchased_by-->product": 1.7517385482788086}` | `{}` | 0.3026 | 0.8793 | 0.3699 |

## Failed Rows

- `ogbn-products/streaming_diffusion_X0X1` status `experiment_failed`: [Errno 22] Invalid argument: 'experiments\\logs\\products_streaming_diffusion_seed42\\streaming_diffusion_X0X1_memmap\\X1.float16.mmap'
- `ogbn-products/streaming_diffusion_X0X1` status `experiment_failed`: [Errno 22] Invalid argument: 'experiments\\logs\\products_streaming_diffusion_seed42\\streaming_diffusion_X0X1_memmap\\X1.float16.mmap'
- `ogbn-products/streaming_diffusion_X0X1` status `experiment_failed`: [Errno 22] Invalid argument: 'experiments\\logs\\products_streaming_diffusion_seed42\\streaming_diffusion_X0X1_memmap\\X1.float16.mmap'
- `ogbn-products/streaming_diffusion_X0X1` status `experiment_failed`: [Errno 22] Invalid argument: 'experiments\\logs\\products_streaming_diffusion_seed42\\streaming_diffusion_X0X1_memmap\\X1.float16.mmap'
- `ogbn-products/streaming_diffusion_X0X1` status `experiment_failed`: [Errno 22] Invalid argument: 'experiments\\logs\\products_streaming_diffusion_seed42\\streaming_diffusion_X0X1_memmap\\X1.float16.mmap'
- `ogbn-products/streaming_diffusion_X0X1` status `experiment_failed`: [Errno 22] Invalid argument: 'experiments\\logs\\products_streaming_diffusion_seed42\\streaming_diffusion_X0X1_memmap\\X1.float16.mmap'
- `ogbn-products/streaming_diffusion_X0X1X2_highpass` status `experiment_failed`: [Errno 22] Invalid argument: 'experiments\\logs\\products_streaming_diffusion_seed42\\streaming_diffusion_X0X1X2_highpass_memmap\\X1.float16.mmap'
- `ogbn-products/streaming_diffusion_X0X1X2_highpass` status `experiment_failed`: [Errno 22] Invalid argument: 'experiments\\logs\\products_streaming_diffusion_seed42\\streaming_diffusion_X0X1X2_highpass_memmap\\X1.float16.mmap'
- `ogbn-products/streaming_diffusion_X0X1X2_highpass` status `experiment_failed`: [Errno 22] Invalid argument: 'experiments\\logs\\products_streaming_diffusion_seed42\\streaming_diffusion_X0X1X2_highpass_memmap\\X1.float16.mmap'
- `ogbn-products/streaming_diffusion_X0X1X2_highpass` status `experiment_failed`: [Errno 22] Invalid argument: 'experiments\\logs\\products_streaming_diffusion_seed42\\streaming_diffusion_X0X1X2_highpass_memmap\\X1.float16.mmap'
- `ogbn-products/streaming_diffusion_X0X1X2_highpass` status `experiment_failed`: [Errno 22] Invalid argument: 'experiments\\logs\\products_streaming_diffusion_seed42\\streaming_diffusion_X0X1X2_highpass_memmap\\X1.float16.mmap'
- `ogbn-products/streaming_diffusion_X0X1X2_highpass` status `experiment_failed`: [Errno 22] Invalid argument: 'experiments\\logs\\products_streaming_diffusion_seed42\\streaming_diffusion_X0X1X2_highpass_memmap\\X1.float16.mmap'

## Interpretation

- R++ rows are single seed 42 and should be interpreted as sprint diagnostics, not final multi-seed claims.
- A row is considered scalable only when it reports completion rather than OOM/OOT.
- Next recommendation is to keep R-1 defaults frozen and promote only opt-in R++ settings that improve accuracy without class collapse.

## Files

- CSV: `experiments/tables/products_streaming_diffusion_seed42.csv`
- Report: `experiments\reports\products_streaming_diffusion_summary.md`
