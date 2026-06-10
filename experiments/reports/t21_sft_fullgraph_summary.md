# T2.1 Fullgraph SFT Table

This table contains no-logits fullgraph SFT rows plus the products execution row when available.

| dataset | status | accuracy | macro_f1 | predicted_class_count | selected_blocks | reason |
|---|---|---|---|---|---|---|
| acm | promoted | 0.9206798672676086 | 0.9211405118306478 | 3 | ["B0_self", "B1_typed", "B2_metapath", "B3_lad_scap"] | validation_selected_and_safe_improved |
| dblp | promoted | 0.9426056146621704 | 0.9387593418359756 | 4 | ["B0_self", "B1_typed", "B2_metapath", "B3_lad_scap"] | validation_selected_and_safe_improved |
| imdb | promoted | 0.47158026695251465 | 0.3900045245885849 | 5 | ["B0_self", "B1_typed", "B2_metapath", "B4_structure"] | validation_selected_and_safe_improved |
| ogbn-arxiv | completed | 0.6544040491327696 | 0.420481437217901 | 39 | ["self", "x1_cite_ref", "x1_cited_by", "x2_cite_ref", "x2_cited_by", "xres", "typed_demand", "structure"] | lazy_memmap_gpu_sft_completed |
| ogbn-products | completed | 0.7029715452279188 | 0.3420856155760991 | 40 | ["self", "x1_co_purchase", "x1_co_purchased_by"] | lazy_memmap_gpu_sft_completed |

- CSV: `experiments\tables\t21_sft_fullgraph_seed42.csv`
