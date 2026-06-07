# Shadow-HGC-R-1 Invariants

## Direction

Every relation is represented as `(source_type, relation_name, destination_type)`. In code, `edge_index[0]` is the message source and `edge_index[1]` is the message destination.

For target-type node classification, condensation uses only directed incoming relations to the target type.

## Alpha Normalization

For every edge `(v -> u)` in relation `r`, the coefficient is:

```text
alpha_uv^r = w_uv^r / sum_{v' in N_r(u)} w_uv'^r
```

This is destination-row normalization. The same alpha is used for demand aggregation, target-target transition mass, original graph inference edge weights, and diagnostics. Source-degree normalization is not used.

## Feature Schema

`psi_t(v)` is the base feature. Feature-bearing node types use fixed random projection plus standardization. Featureless source types use leakage-safe neighbor means.

`phi_t(v)` is the model input feature. For the target type:

```text
phi_tau(v) = [psi_tau(v) || g_v]
```

For non-target source types:

```text
phi_sigma(v) = psi_sigma(v)
```

All relation demands use `phi_source(v)`. For target-target relations, source target nodes send `phi_tau(v)`, including degree encodings.

## Schema Preservation

Shadow pools are implementation partitions. The exposed condensed graph uses only original node types and original edge types. It must not expose node types such as `author_shadow` or edge types such as `shadow_writes`.

## Weighted Relation-Linear Layer

The model layer performs:

```text
message = edge_weight * W_r(x_src)
aggregate = scatter_add(message, dst)
```

There is no hidden library normalization. Pre-normalized alpha weights must not be passed through layers that normalize again.
