# Safe Block Fusion Diagnostics Seed 42

- Selected blocks: `['self', 'useful']`
- Self validation accuracy: `0.5`
- Final validation accuracy: `1.0`

| Block | Val Acc | Gate Initial | Gate Final | Decision | Drop Reason |
|---|---:|---:|---:|---|---|
| useful | 1.0 | 0.00033540636650286615 | 5.755476539803794e-09 | kept |  |
| noise | 0.0 | 0.00033540636650286615 | 17.65916633605957 | dropped | validation_non_regression_gate |

- CSV: `experiments\tables\safe_block_fusion_diagnostics_seed42.csv`
