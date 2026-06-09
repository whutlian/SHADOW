# DBLP Schema Audit Seed 42

- Target type: `author`
- Label node type: `author`
- APA available: `True`
- Hard requirements passed: `True`
- Available target meta-path source types: `paper`
- Computed meta-path blocks: `APA`
- Skipped meta-path blocks: `APVPA, APTPA, APCPA`

## Edge Types

| Source | Relation | Destination | Edges |
|---|---|---|---:|
| paper | written_by | author | 19645 |

## Label Distribution

- Train: `{"0": 368, "1": 238, "2": 317, "3": 294}`
- Valid: `{}`
- Test: `{"0": 829, "1": 507, "2": 792, "3": 712}`

## Note

APVPA/APTPA require non-target paper-venue/paper-term edges; current small loader keeps incoming-to-target relations only.

- CSV: `experiments\tables\dblp_schema_audit_seed42.csv`
