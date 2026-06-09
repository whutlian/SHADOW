# T2.1 True Preprop Manifest Index

Completed rows use chunked destination-row SpMM and memmap output. No row uses logits, KD, dense P2, bounded edges, or E x d materialization.

| dataset | status | num_blocks | block_names | total_cache_bytes | full_edge_scans | uses_e_by_d_materialization | reason |
|---|---|---|---|---|---|---|---|
| acm | completed | 8 | ["X0", "X1_cite_ref", "X1_cited_by", "Xres", "typed_demand", "structure", "metapath", "lad_scap"] | 3944600 | 10 | False | true_chunked_memmap_preprop_completed |
| dblp | completed | 5 | ["X0", "typed_demand", "structure", "metapath", "lad_scap"] | 2109640 | 2 | False | true_chunked_memmap_preprop_completed |
| imdb | completed | 5 | ["X0", "typed_demand", "structure", "metapath", "lad_scap"] | 2604096 | 6 | False | true_chunked_memmap_preprop_completed |
| ogbn-arxiv | completed | 8 | ["X0", "X1_cite_ref", "X1_cited_by", "X2_cite_ref", "X2_cited_by", "Xres", "typed_demand", "structure"] | 131748854 | 6 | False | true_chunked_memmap_preprop_completed |

- CSV: `experiments\tables\t21_preprop_manifest_index_seed42.csv`
