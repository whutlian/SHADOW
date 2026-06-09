# Medium No-Diffusion Refine Seed 42

All rows keep diffusion disabled. Two-hop LAD rows request `P1` and `P2` Path-LAD blocks in the existing compiled demand head.

| Dataset | Variant | Ratio | Acc | Macro-F1 | Pred classes | Two-hop blocks | Status |
|---|---|---:|---:|---:|---:|---|---|
| ogbn-arxiv | LAD_reference | 0.06 | 0.590210497379303 | 0.40769136818125845 | 39 | [] | completed |
| ogbn-arxiv | LAD_plus_two_hop_LAD | 0.06 | 0.33508220314979553 | 0.1428937314078212 | 34 | ["P2"] | completed |
| ogbn-arxiv | LAD_plus_two_hop_LAD_plus_lad_fusion_head | 0.06 | 0.3188897669315338 | 0.09305294038495049 | 24 | ["P2"] | completed |
| ogbn-arxiv | LAD_reference | 0.12 | 0.5967738628387451 | 0.4154518236406147 | 40 | [] | completed |
| ogbn-arxiv | LAD_plus_two_hop_LAD | 0.12 | 0.22848384082317352 | 0.05985004580579698 | 21 | ["P2"] | completed |
| ogbn-arxiv | LAD_plus_two_hop_LAD_plus_lad_fusion_head | 0.12 | 0.22665268182754517 | 0.019331736117601396 | 2 | ["P2"] | completed |
| ogbn-products | LAD_reference | 0.06 | 0.6223331689834595 | 0.3307438173598828 | 32 | [] | completed |
| ogbn-products | LAD_plus_two_hop_LAD | 0.06 |  |  |  | ["P2"] | oom |
| ogbn-products | LAD_plus_two_hop_LAD_plus_lad_fusion_head | 0.06 |  |  |  | ["P2"] | oom |
| ogbn-products | LAD_plus_balanced_softmax | 0.06 |  |  |  | ["P2"] | oom |
| ogbn-products | LAD_reference | 0.12 | 0.6586742401123047 | 0.3380637136387064 | 31 | [] | completed |
| ogbn-products | LAD_plus_two_hop_LAD | 0.12 |  |  |  | ["P2"] | oom |
| ogbn-products | LAD_plus_two_hop_LAD_plus_lad_fusion_head | 0.12 |  |  |  | ["P2"] | oom |
| ogbn-products | LAD_plus_balanced_softmax | 0.12 |  |  |  | ["P2"] | timeout_dropped |

- CSV: `experiments\tables\medium_no_diffusion_refine_seed42.csv`
