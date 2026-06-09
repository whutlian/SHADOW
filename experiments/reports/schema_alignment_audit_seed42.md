# Schema Alignment Audit Seed 42

| Dataset | Loader | Target | Available | Missing | Status | Notes |
|---|---|---|---|---|---|---|
| acm | current_processed | paper | ["PAP", "PSP", "PTP"] | [] | aligned | full_schema audit only; default condensation path remains incoming-to-target |
| acm | full_schema | paper | ["PAP", "PSP", "PTP"] | [] | aligned | full_schema audit only; default condensation path remains incoming-to-target |
| dblp | current_processed | author | ["APA"] | ["APVPA", "APTPA", "APCPA"] | partial | full_schema audit only; default condensation path remains incoming-to-target |
| dblp | full_schema | author | ["APA", "APVPA", "APTPA", "APCPA"] | [] | aligned | full_schema audit only; default condensation path remains incoming-to-target |
| imdb | current_processed | movie | ["MAM", "MDM", "MKM"] | [] | aligned | full_schema audit only; default condensation path remains incoming-to-target |
| imdb | full_schema | movie | ["MAM", "MDM", "MKM"] | [] | aligned | full_schema audit only; default condensation path remains incoming-to-target |
| ogbn-arxiv | ogb_homogeneous | paper | [] | [] | not_applicable | homogeneous OGB schema; full feature hashing is skipped by resource guard |
| ogbn-products | ogb_homogeneous | product | [] | [] | not_applicable | homogeneous OGB schema; full feature hashing is skipped by resource guard |

- CSV: `experiments\tables\schema_alignment_audit_seed42.csv`
