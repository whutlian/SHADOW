# T24 Mid/Large SFT-Condense Stage Summary

## Main Results

| dataset | best | accuracy | macro_f1 | status |
|---|---|---|---|---|
| ogbn-arxiv | A0_current_A3_true_sagn_lite_v3_replay | 0.7016645063061951 | 0.5048992808650066 | completed_replay |
| ogbn-products | P3_shadow_condensed_medoid_b1 | 0.174593814714352 | 0.11342605750338396 | completed_streaming |
| Reddit | sagn_lite_v4_x0_smoke | 0.32427337765693665 | 0.0982786856773423 | completed_x0_sft_smoke |

## Required Answers

1. Did arxiv improve beyond 0.7017? `False`; best=`0.7016645063061951`.
2. Did arxiv pass 0.715 / 0.725 / 0.740? `False` / `False` / `False`.
3. Which arxiv blocks were selected? `["X0", "X1_cite_ref", "X1_cited_by", "X2_cite_ref", "X2_cited_by", "X3_mix", "Xres1_cite_ref", "Xres1_cited_by", "structure", "Y1_cite_ref", "Y1_cited_by", "Y2_cite_ref", "Y2_cited_by", "Y3_mix"]`.
4. Did products run full streaming SFT recovery rather than proxy rows? `True`.
5. Products 0.25% shadow-condensed accuracy: `0.174593814714352`.
6. Products 0.50% shadow-condensed accuracy: `0.2740235263710349`.
7. Did Reddit fullgraph SFT complete? `True`.
8. Did Reddit condensation complete at 0.50% full-node ratio? `False`.
9. Fixed bucket ratios: arxiv=0.50%, Reddit=0.50%, products=0.25%.
10. All T24 main ratios are reported as full-node ratios.
11. Any promoted row used forbidden components? `False`.
12. Fullgraph-to-condensed gap: arxiv=`not_run_until_gate_A`, Reddit=``, products_0.25=`0.5809842433049522`.
13. Biggest bottleneck: products/reddit full streaming training resource; arxiv fullgraph teacher gate remains below 0.715.
14. Next dataset after arxiv/products/Reddit: ogbn-papers100M train-target-only dry-run to server execution.

## Stage Changes

- Added T24 scale-bucket full-node ratio policy and safety validation.
- Added arxiv filter-bank v4 / LabelReuse v3 wrappers and SFT v4 model aliases.
- Added products SFT signature cache and memmap recovery script with proxy promotion blocked.
- Added Reddit processed-cache loader and T24 Reddit SFT/condense entrypoints.
- Added unified T24 tables, ultra dry-run, tests, and configuration.

## Post-Stage Reddit Streaming Check

- Added raw Reddit `.npz` to stream-friendly memmap conversion and a chunked edge reader that avoids `processed/data.pt`.
- Completed full Reddit streaming preprop for `X0/X1/X2/X3/Xres1/Y1/Y2/Y3/structure` at block dim 64.
- Full run scanned 114,615,892 edges for 13 total edge scans in 84.03s, wrote 207,804,780 bytes of preprop cache, and reported peak CPU RSS 0.53GB.
- The run logged `uses_processed_data_pt=False`, `materialized_stacked_edge_index=False`, and `uses_e_by_d_materialization=False`.
- Completed actual Reddit lazy SFT training/eval from those full streaming-preprop blocks with `sagn_lite_v4`, 30 epochs, hidden dim 128, and CUDA mini-batches.
- Reddit streaming SFT test accuracy is 0.9400570884871551, macro-F1 is 0.9110379599379667, training time is 18.04s, peak CPU RSS is 1.45GB, and peak GPU allocation is 0.96GB.
- The SFT run logged `loads_edge_index=False`, `uses_lazy_memmap=True`, `uses_logits_as_input=False`, `uses_teacher_logits=False`, `uses_kd=False`, and `uses_e_by_d_materialization=False`.
- Reddit condensation training is now completed in a post-stage run over the same full streaming-preprop blocks.

## Post-Stage Reddit Condensation Check

- Ran Reddit condensed training at full-node ratios 0.10%, 0.25%, 0.50%, and 1.00% with `SFT-signature random`, `medoid`, `kcenter`, and `shadow condensed b=1`.
- All rows used CPU/memmap-resident full Reddit streaming preprop blocks and CUDA mini-batch condensed training; all rows logged `loads_edge_index=False`, `uses_lazy_memmap=True`, `uses_logits_as_input=False`, `uses_teacher_logits=False`, `uses_kd=False`, and `uses_e_by_d_materialization=False`.
- At the T24 medium default 0.50% full-node ratio, best row is `SFT-signature random` with accuracy 0.9244564924689873 and macro-F1 0.8862562817528249 over 1,165 condensed nodes.
- At 0.50% full-node ratio, the main `SFT-signature shadow condensed b=1` row has accuracy 0.9215841157567815 and macro-F1 0.8840176339405728 over 1,165 condensed nodes.
- Best ratio-sweep rows: 0.10% b=1 accuracy 0.9097894188822864; 0.25% medoid accuracy 0.9179577401576217; 0.50% random accuracy 0.9244564924689873; 1.00% random accuracy 0.9245283018867925.
- Full Reddit streaming SFT reference is accuracy 0.9400570884871551, so the best 0.50% condensed row is 0.015600596018167851 absolute accuracy below the full-preprop SFT reference.

- Stage CSV: `experiments\tables\t24_stage_summary_seed42.csv`
