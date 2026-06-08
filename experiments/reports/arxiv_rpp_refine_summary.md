# ogbn-arxiv R++ Refinement Summary

## Completed / OOM / Failed Rows

| Dataset | Variant | Ratio | Model | Feature | Loss | Acc | Macro-F1 | Pred Classes | Status |
|---|---|---:|---|---|---|---:|---:|---:|---|
| ogbn-arxiv | diffusion_X0X1X2_highpass_relation_linear_no_final_relu | 2.0% | relation_linear | diffusion | sqrt_weighted_logit_adjusted | 0.5279 | 0.3475 | 39 | completed |
| ogbn-arxiv | diffusion_X0X1X2_highpass_relation_linear_no_final_relu | 6.0% | relation_linear | diffusion | sqrt_weighted_logit_adjusted | 0.5248 | 0.3370 | 39 | completed |
| ogbn-arxiv | diffusion_X0X1X2_highpass_relation_linear_no_final_relu | 12.0% | relation_linear | diffusion | sqrt_weighted_logit_adjusted | 0.5286 | 0.3344 | 39 | completed |
| ogbn-arxiv | diffusion_X0X1X2_highpass_shadow_fusion | 2.0% | shadow_fusion | diffusion | sqrt_weighted_logit_adjusted | 0.5361 | 0.3226 | 39 | completed |
| ogbn-arxiv | diffusion_X0X1X2_highpass_shadow_fusion | 6.0% | shadow_fusion | diffusion | sqrt_weighted_logit_adjusted | 0.5719 | 0.3643 | 39 | completed |
| ogbn-arxiv | diffusion_X0X1X2_highpass_shadow_fusion | 12.0% | shadow_fusion | diffusion | sqrt_weighted_logit_adjusted | 0.5989 | 0.3998 | 40 | completed |
| ogbn-arxiv | diffusion_X0X1X2_highpass_blocknorm_relation_linear_no_final_relu | 2.0% | relation_linear | diffusion | sqrt_weighted_logit_adjusted | 0.5475 | 0.3593 | 39 | completed |
| ogbn-arxiv | diffusion_X0X1X2_highpass_blocknorm_relation_linear_no_final_relu | 6.0% | relation_linear | diffusion | sqrt_weighted_logit_adjusted | 0.5305 | 0.3427 | 39 | completed |
| ogbn-arxiv | diffusion_X0X1X2_highpass_blocknorm_relation_linear_no_final_relu | 12.0% | relation_linear | diffusion | sqrt_weighted_logit_adjusted | 0.5221 | 0.3268 | 39 | completed |
| ogbn-arxiv | diffusion_X0X1X2_highpass_blocknorm_shadow_fusion | 2.0% | shadow_fusion | diffusion | sqrt_weighted_logit_adjusted | 0.5430 | 0.3344 | 40 | completed |
| ogbn-arxiv | diffusion_X0X1X2_highpass_blocknorm_shadow_fusion | 6.0% | shadow_fusion | diffusion | sqrt_weighted_logit_adjusted | 0.6029 | 0.3778 | 39 | completed |
| ogbn-arxiv | diffusion_X0X1X2_highpass_blocknorm_shadow_fusion | 12.0% | shadow_fusion | diffusion | sqrt_weighted_logit_adjusted | 0.6172 | 0.4147 | 40 | completed |

## Best Rows

- Best accuracy: `0.6172` from `ogbn-arxiv / diffusion_X0X1X2_highpass_blocknorm_shadow_fusion` at `12.0%`.
- Best macro-F1: `0.4147` from `ogbn-arxiv / diffusion_X0X1X2_highpass_blocknorm_shadow_fusion` at `12.0%`.

## Comparison To R+ Best

- ogbn-arxiv: R++ best `0.6172` vs R+ `0.5369`.

## Compression And Resource Accounting

| Dataset | Variant | Eff target ratio | Total node ratio | Edge ratio | Byte ratio | CPU RAM | GPU RAM | Disk bytes |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| ogbn-arxiv | diffusion_X0X1X2_highpass_relation_linear_no_final_relu | 0.0200 | 0.0161 | 0.0095 | 0.0154 | 7079178240 | 0 | 0 |
| ogbn-arxiv | diffusion_X0X1X2_highpass_relation_linear_no_final_relu | 0.0599 | 0.0483 | 0.0213 | 0.0452 | 8062431232 | 0 | 0 |
| ogbn-arxiv | diffusion_X0X1X2_highpass_relation_linear_no_final_relu | 0.1191 | 0.0959 | 0.0349 | 0.0890 | 8225718272 | 0 | 0 |
| ogbn-arxiv | diffusion_X0X1X2_highpass_shadow_fusion | 0.0200 | 0.0161 | 0.0095 | 0.0154 | 8208748544 | 0 | 0 |
| ogbn-arxiv | diffusion_X0X1X2_highpass_shadow_fusion | 0.0599 | 0.0483 | 0.0213 | 0.0452 | 8235220992 | 0 | 0 |
| ogbn-arxiv | diffusion_X0X1X2_highpass_shadow_fusion | 0.1191 | 0.0959 | 0.0349 | 0.0890 | 8240472064 | 0 | 0 |
| ogbn-arxiv | diffusion_X0X1X2_highpass_blocknorm_relation_linear_no_final_relu | 0.0200 | 0.0161 | 0.0091 | 0.0153 | 8264949760 | 0 | 0 |
| ogbn-arxiv | diffusion_X0X1X2_highpass_blocknorm_relation_linear_no_final_relu | 0.0599 | 0.0483 | 0.0206 | 0.0451 | 8243806208 | 0 | 0 |
| ogbn-arxiv | diffusion_X0X1X2_highpass_blocknorm_relation_linear_no_final_relu | 0.1192 | 0.0960 | 0.0341 | 0.0889 | 8242454528 | 0 | 0 |
| ogbn-arxiv | diffusion_X0X1X2_highpass_blocknorm_shadow_fusion | 0.0200 | 0.0161 | 0.0091 | 0.0153 | 8255770624 | 0 | 0 |
| ogbn-arxiv | diffusion_X0X1X2_highpass_blocknorm_shadow_fusion | 0.0599 | 0.0483 | 0.0206 | 0.0451 | 8237334528 | 0 | 0 |
| ogbn-arxiv | diffusion_X0X1X2_highpass_blocknorm_shadow_fusion | 0.1192 | 0.0960 | 0.0341 | 0.0889 | 5715345408 | 0 | 0 |

## Diagnostics

| Dataset | Variant | Entropy | Relation gates | Block gates | Skel cov | Residual energy | Recon err |
|---|---|---:|---|---|---:|---:|---:|
| ogbn-arxiv | diffusion_X0X1X2_highpass_relation_linear_no_final_relu | 2.8727 | `{"paper--cite_ref-->paper": 1.870389461517334, "paper--cited_by-->paper": 1.9711090326309204}` | `{}` | 0.6009 | 0.6795 | 0.5454 |
| ogbn-arxiv | diffusion_X0X1X2_highpass_relation_linear_no_final_relu | 2.8875 | `{"paper--cite_ref-->paper": 1.1977360248565674, "paper--cited_by-->paper": 1.2669662237167358}` | `{}` | 0.7020 | 0.7490 | 0.6360 |
| ogbn-arxiv | diffusion_X0X1X2_highpass_relation_linear_no_final_relu | 2.9140 | `{"paper--cite_ref-->paper": 2.148832321166992, "paper--cited_by-->paper": 2.0632920265197754}` | `{}` | 0.7662 | 0.7556 | 0.6099 |
| ogbn-arxiv | diffusion_X0X1X2_highpass_shadow_fusion | 2.7816 | `{"paper--cite_ref-->paper": 0.7493817806243896, "paper--cited_by-->paper": 0.7603793144226074}` | `{}` | 0.6009 | 0.6795 | 0.5454 |
| ogbn-arxiv | diffusion_X0X1X2_highpass_shadow_fusion | 2.7417 | `{"paper--cite_ref-->paper": 1.0831061601638794, "paper--cited_by-->paper": 1.1292097568511963}` | `{}` | 0.7020 | 0.7490 | 0.6360 |
| ogbn-arxiv | diffusion_X0X1X2_highpass_shadow_fusion | 2.8082 | `{"paper--cite_ref-->paper": 1.361948013305664, "paper--cited_by-->paper": 1.327901005744934}` | `{}` | 0.7662 | 0.7556 | 0.6099 |
| ogbn-arxiv | diffusion_X0X1X2_highpass_blocknorm_relation_linear_no_final_relu | 2.8422 | `{"paper--cite_ref-->paper": 1.2342594861984253, "paper--cited_by-->paper": 1.310025930404663}` | `{}` | 0.6171 | 0.6338 | 0.5514 |
| ogbn-arxiv | diffusion_X0X1X2_highpass_blocknorm_relation_linear_no_final_relu | 2.8708 | `{"paper--cite_ref-->paper": 1.4003136157989502, "paper--cited_by-->paper": 1.4247221946716309}` | `{}` | 0.7135 | 0.7044 | 0.6152 |
| ogbn-arxiv | diffusion_X0X1X2_highpass_blocknorm_relation_linear_no_final_relu | 2.9470 | `{"paper--cite_ref-->paper": 1.7702040672302246, "paper--cited_by-->paper": 1.6556204557418823}` | `{}` | 0.7706 | 0.7090 | 0.5883 |
| ogbn-arxiv | diffusion_X0X1X2_highpass_blocknorm_shadow_fusion | 2.7325 | `{"paper--cite_ref-->paper": 0.6571879982948303, "paper--cited_by-->paper": 0.6691007614135742}` | `{}` | 0.6171 | 0.6338 | 0.5514 |
| ogbn-arxiv | diffusion_X0X1X2_highpass_blocknorm_shadow_fusion | 2.6843 | `{"paper--cite_ref-->paper": 0.9122979044914246, "paper--cited_by-->paper": 0.9344524145126343}` | `{}` | 0.7135 | 0.7044 | 0.6152 |
| ogbn-arxiv | diffusion_X0X1X2_highpass_blocknorm_shadow_fusion | 2.7792 | `{"paper--cite_ref-->paper": 1.112012267112732, "paper--cited_by-->paper": 1.1015856266021729}` | `{}` | 0.7706 | 0.7090 | 0.5883 |

## Interpretation

- R++ rows are single seed 42 and should be interpreted as sprint diagnostics, not final multi-seed claims.
- A row is considered scalable only when it reports completion rather than OOM/OOT.
- Next recommendation is to keep R-1 defaults frozen and promote only opt-in R++ settings that improve accuracy without class collapse.

## Files

- CSV: `experiments/tables/arxiv_rpp_refine_seed42.csv`
- Report: `experiments\reports\arxiv_rpp_refine_summary.md`
