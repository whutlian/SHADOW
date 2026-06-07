# AGENT.md — Shadow-HGC-R-1 Research Coding Agent

## Project mission

Implement and experimentally validate **Shadow-HGC-R-1**, a schema-preserving typed message condensation method for **target-type node classification** on large and ultra-large heterogeneous graphs.

The project is not a generic heterogeneous graph condensation framework. The main method is fixed as:

> **Shadow-HGC-R-1: Degree-Calibrated Residual Message-Shadow Condensation for Target-Type Node Classification**

Core claim:

> Condense target-type node classification by factorizing **target-side directed typed relation demands** into sparse schema-preserving shadow nodes, while decomposing target-target relations into a prototype-level residual skeleton plus signed residual shadow features.

Do not broaden the claim to arbitrary graph tasks, arbitrary HGNNs, or full higher-order topology preservation.

---

## Non-negotiable method invariants

### 1. Main method is frozen

The main method is **Shadow-HGC-R-1**.

Main components only:

1. leakage-safe typed feature provider;
2. directed degree-calibrated relation demand;
3. class-wise target prototyping;
4. target-target residual skeleton;
5. sparse typed shadow factorization with `b = 1` nearest shadow assignment;
6. custom demand-compatible weighted relation-linear HGNN layer.

Optional variants are allowed only as ablations or appendix experiments:

- weighted shadow assignment `b = 2`;
- meta-path sketch used only as an auxiliary clustering signature;
- institution-aware initializer;
- PCA;
- full-transductive feature initializer;
- HGT/HAN/SimpleHGN transfer backbones.

Do not introduce new main modules without an explicit project decision.

---

### 2. Direction convention is fixed

Every relation is represented as:

```text
r = (source_type, relation_name, destination_type)
```

Code convention:

```python
edge_index[0] = message_source_node
edge_index[1] = message_destination_node
```

For target-type node classification, only directed incoming relations to the target type are condensed:

```text
R_tau_in = {r = (sigma, rho, tau)}
```

Citation relations must be explicitly directed. Prefer names such as:

```text
paper --cite_ref--> paper
paper --cited_by--> paper
```

Avoid ambiguous names like `paper-cites-paper` unless the source and destination message direction are documented in config and tests.

---

### 3. All normalization uses the same destination-row coefficient

For every edge `(v -> u)` under relation `r`, define:

```text
alpha_uv^r = w_uv^r / sum_{v' in N_r(u)} w_{uv'}^r
```

Default raw edge weight is `w_uv^r = 1`.

The same `alpha_uv^r` must be used consistently for:

- train-target relation demand `mu_u^r`;
- target-target transition mass `S_ij^r`;
- original graph inference edge weights;
- any debugging or diagnostics.

Do not mix destination degree normalization and source degree normalization.

---

### 4. Feature schema is unified

There are two feature symbols.

#### Base feature

```text
psi_t(v)
```

This is the base feature for node type `t`.

For feature-bearing node types:

```text
psi_t(v) = Std(X_t[v] R_t)
```

where `R_t` is a fixed random projection in the main method. PCA is not the default main method.

For featureless source types, use leakage-safe neighbor-mean initialization.

#### Model input feature

```text
phi_t(v)
```

This is the actual feature fed into the HGNN.

For target type `tau`:

```text
phi_tau(v) = [psi_tau(v) || g_v]
```

where `g_v` is the concatenation of per-directed-relation degree encodings.

For non-target source type `sigma != tau`:

```text
phi_sigma(v) = psi_sigma(v)
```

All relation demands must use the actual source model feature:

```text
mu_u^r = sum_{v in N_r(u)} alpha_uv^r * phi_source(v)
```

For target-target relations, the source target node must send `phi_tau(v)`, not only `psi_tau(v)`.

---

### 5. Degree encoding is part of the target model input

For each train target node `u` and directed incoming relation `r`:

```text
deg_r(u) = |N_r(u)|
```

Degree encoding:

```text
g_u^r = [log(1 + deg_r(u)), Bucket(deg_r(u)), 1_{deg_r(u)=0}]
```

Use log-scale buckets:

```text
0, 1, 2, 3-4, 5-8, 9-16, 17-32, 33-64, >64
```

Target degree feature:

```text
g_u = concat_{r in R_tau_in} g_u^r
```

Degree features must be used in:

1. target clustering signature;
2. condensed target prototype input feature;
3. original graph inference input feature.

Do not use degree only for clustering.

---

### 6. Target prototypes are class-wise supervised coresets

For every train target node `u`, build clustering signature:

```text
s_u = BlockNorm([psi_tau(u) || {mu_u^r}_r || eta * g_u || optional_high_order_sketch])
```

Default:

```text
eta = 0.1
```

For each class `c`, run class-wise MiniBatchKMeans or FAISS KMeans on `{s_u : y_u = c}`.

Class-wise budget:

```text
m_c = max(1, round(M_tau * sqrt(n_c) / sum_{c'} sqrt(n_{c'})))
```

For each target cell `C_i`:

```text
x_i' = mean_{u in C_i} phi_tau(u)
y_i' = c
w_i = |C_i|
```

Training loss on the condensed graph:

```text
L = (1 / sum_i |C_i|) * sum_i |C_i| * CE(f(G')_i, y_i')
```

Also implement loss variants for ablation:

- unweighted prototype loss;
- clipped-weight loss;
- class-balanced prototype loss.

---

### 7. Target-target relation uses residual skeleton

For target-target relation:

```text
r = (tau, rho, tau)
```

Cell-level original demand:

```text
D_i^r = (1 / |C_i|) * sum_{u in C_i} mu_u^r
```

Prototype-to-prototype transition mass:

```text
S_ij^r = (1 / |C_i|) * sum_{u in C_i} sum_{v in N_r(u) ∩ C_j} alpha_uv^r
```

Here:

- `i` is the destination target prototype;
- `j` is the source target prototype;
- `S_ij^r` has the same normalization scale as `D_i^r`.

For each destination prototype `i`, keep top-`k_s` source prototypes by `S_ij^r`.

Default:

```text
k_s = 2
```

Ablation must include:

```text
k_s in {0, 1, 2, 4, 8}
```

Add skeleton edges:

```text
p_j -> p_i
edge_type = original relation r
edge_weight = S_ij^r
```

Important: **do not renormalize** the top-`k_s` skeleton weights. Uncovered transition mass should remain in the residual.

Skeleton message:

```text
D_skel_i^r = sum_{j in TopK_i} S_ij^r * x_j'
```

Residual demand:

```text
B_i^r = D_i^r - D_skel_i^r
```

The target-target relation is therefore represented as:

```text
target-target demand = sparse prototype skeleton message + residual shadow message
```

---

### 8. Residual shadow features may be signed; edge weights remain non-negative

Residual vector `B_i^r` can contain negative values. This is valid because residual shadows are virtual correction carriers, not real source nodes.

Main assignment uses `b = 1`:

```text
pi(i) = argmin_k ||B_i^r - z_k^r||_2^2
```

Add residual shadow edge:

```text
s_{r, pi(i)} -> p_i
edge_type = original relation r
edge_weight = 1
```

Rules:

- shadow features `z_k^r` may be signed;
- edge weights remain non-negative;
- do not use negative edge weights;
- do not use dense synthetic adjacency;
- NNLS / weighted `b = 2` is an ablation only.

Implementation safeguard:

```text
clip ||z_k^r||_2 by the 99.5% quantile of residual norms for relation r
```

This is a stability guard, not a main contribution.

---

### 9. Non-target relation uses typed shadow factorization

For non-target incoming relation:

```text
r = (sigma, rho, tau), sigma != tau
```

Compute:

```text
D_i^r = (1 / |C_i|) * sum_{u in C_i} sum_{v in N_r(u)} alpha_uv^r * phi_sigma(v)
B_i^r = D_i^r
```

Run weighted KMeans on rows of `B^r` to obtain shadow features `Z^r`.

Main assignment uses `b = 1` nearest shadow.

Schema rule:

> Shadow pools are implementation-level partitions, not new node types. The exposed condensed graph preserves the original task schema.

Example:

```text
internal pool: author-writing-shadow-pool
exposed node type: author
exposed edge type: author --writes--> paper
```

Do not expose:

```text
node type: author_shadow
edge type: shadow_writes
```

---

### 10. Relation norm calibration is allowed but constrained

After factorization for relation `r`, compute reconstruction:

```text
B_hat_i^r = z_{pi(i)}^r
```

Robust scale ratio:

```text
gamma_r = clip_[0.5, 2.0](median_i ||B_i^r||_2 / (||B_hat_i^r||_2 + eps))
```

Scale shadow features:

```text
Z^r <- gamma_r * Z^r
```

Rules:

- use train target cells only;
- do not use validation or test statistics;
- do not scale original graph inference features;
- include calibration on/off in ablation or diagnostics.

---

### 11. Use a custom demand-compatible weighted relation-linear layer

Do not rely on library layers that perform hidden normalization by default.

Layer definition:

```text
h_u^{l+1} = sigma(
    W_self^{type(u)} h_u^l
    + sum_{r=(sigma,rho,type(u))} sum_{(v,u) in E_r} e_vu^r W_r h_v^l
)
```

Implementation principle:

```python
message = edge_weight * W_r(x_src)
aggregate = scatter_add(message, dst)
```

Forbidden pattern:

```python
edge_weight = pre_normalized_weight
conv = GCNConv(normalize=True)  # double normalization risk
```

Original graph inference edge weights:

```text
e_vu^r = alpha_uv^r
```

Condensed graph edge weights:

| Edge kind | Edge weight |
|---|---:|
| skeleton edge `p_j -> p_i` | `S_ij^r` |
| residual shadow edge | `1` for main `b=1` |
| non-target shadow edge | `1` for main `b=1` |
| weighted `b=2` variant | learned non-negative coefficient, ablation only |

Theory guarantee applies only to one-layer relation-linear HGNN. Deeper and attention-based HGNNs are empirical transfer experiments only.

---

### 12. Leakage-safe feature provider

For ultra-scale temporal benchmarks such as MAG240M:

- main method uses fixed random projection, not PCA;
- standardization statistics come from train-period paper features;
- author features use temporal-safe train-period neighboring paper features;
- full-transductive author initializer is appendix / upper-bound only;
- institution-aware initializer is ablation only.

Author initializer for MAG240M main setting:

```text
psi_author(a) = mean_{p in P_{<=T_train}(a)} psi_paper(p)
```

If the set is empty, use type-level mean fallback.

Distinguish clearly between:

- feature leakage;
- structural / edge leakage;
- label leakage.

Main method primarily prevents future-feature leakage in featureless source initialization.

---

### 13. I/O policy for paper100M and MAG240M

Ultra-scale implementation must be train-target-only.

Do not cache relation demand for all target nodes.

Cache only:

```text
N_train_target × |R_tau_in| × d
```

Per relation, at most two full edge scans:

1. degree / active source collection;
2. train-target message aggregation.

For target-target skeleton, during the aggregation pass cache compact train-target edge slice:

```text
(dst_train_pos, src_train_pos, alpha)
```

Only cache edges where both source and destination are train target nodes.

After KMeans, scan the compact edge-slice cache to compute `S_ij^r`.

Report:

- number of full edge scans;
- edge-slice cache bytes;
- cache build time;
- cache aggregation time;
- whether disk spill is used;
- peak CPU RAM;
- peak GPU RAM;
- disk bytes.

Avoid memmap random-access explosions. Prefer source-id block partition, CSR/CSC layouts, chunked feature gathers, and fp16/fp32 memmaps.

---

## Repository structure guideline

Use a simple modular structure:

```text
shadow-hgc-r1/
  AGENT.md
  GOAL.md
  README.md
  pyproject.toml or environment.yml
  configs/
    datasets/
    methods/
    experiments/
  shadow_hgc/
    __init__.py
    data/
      loaders.py
      schemas.py
      edge_stream.py
      memmap.py
      splits.py
    features/
      base.py
      projection.py
      temporal_author.py
      degree.py
    demand/
      normalize.py
      aggregate.py
      cache.py
    prototype/
      signatures.py
      budgets.py
      cluster.py
    skeleton/
      transition.py
      residual.py
    shadows/
      factorize.py
      assign.py
      calibrate.py
    graph/
      materialize.py
      schema.py
    models/
      weighted_rel_linear.py
      losses.py
    train/
      train_condensed.py
      infer_original.py
    eval/
      metrics.py
      diagnostics.py
      tables.py
  scripts/
    run_toy.py
    run_small.py
    run_medium.py
    run_paper100m.py
    run_mag240m.py
  tests/
    test_direction.py
    test_alpha_normalization.py
    test_weighted_layer_no_double_norm.py
    test_target_target_skeleton.py
    test_schema_preservation.py
    test_residual_shadow.py
  experiments/
    logs/
    tables/
    figures/
  docs/
    method_notes.md
    experiment_protocol.md
```

Keep file names stable. Prefer small, testable modules over large scripts.

---

## Required tests before experiments

Implement and keep these tests passing:

1. **Direction test**: `edge_index[0]` is source and `edge_index[1]` is destination for all relations.
2. **Alpha normalization test**: destination-row alpha sums to 1 for nonzero-degree destination nodes.
3. **Demand consistency test**: hand-computed `mu_u^r` matches aggregator output on a tiny graph.
4. **Skeleton scale test**: `S_ij^r` uses exactly the same `alpha_uv^r` as demand.
5. **No renormalized top-k test**: top-`k_s` skeleton weights are not renormalized after truncation.
6. **Residual sign test**: residual shadow features may be signed while edge weights are non-negative.
7. **Schema preservation test**: exposed condensed graph uses only original node and edge types.
8. **Layer test**: weighted relation-linear layer equals explicit weighted scatter-add; no hidden normalization.
9. **Loss test**: cell-weighted loss matches manual empirical-risk formula.
10. **I/O policy test**: large-mode code paths refuse all-node demand caching unless explicitly in debug mode.

---

## Diagnostics to log for every run

Always log:

- dataset name, split, target type, directed relations;
- target prototype budget `M_tau`;
- shadow budget per relation `M_r`;
- `k_s`;
- feature dimension;
- random seed;
- number of condensed nodes and edges by type/relation;
- condensation time;
- training time;
- inference time;
- peak CPU RAM;
- peak GPU RAM;
- disk bytes;
- number of full edge scans;
- edge-slice cache bytes;
- accuracy / macro-F1 if applicable.

Required method diagnostics:

```text
SkeletonMassCoverage_r(k_s) = sum_i sum_{j in TopK_i} S_ij^r / (sum_i sum_j S_ij^r + eps)
ResidualEnergy_r = ||B^r||_F / (||D^r||_F + eps)
ShadowReconErr_r = ||B^r - B_hat^r||_F / (||B^r||_F + eps)
```

Also log shadow feature norm distribution and compare it to real source feature norms when possible.

---

## Baseline policy

Small heterogeneous datasets:

```text
ACM, DBLP, IMDB
```

Baselines:

```text
Random-HG, Herding-HG, K-Center-HG, HGCond, FreeHGC, HGC-Herd if available
```

Medium homogeneous / near-homogeneous datasets:

```text
ogbn-arxiv, ogbn-products
```

Baselines:

```text
Random, Herding, K-Center, GCond, DosCond, SFGC, GCPA, CGC, DisCo, Bonsai where feasible
```

Ultra-scale feasibility datasets:

```text
ogbn-papers100M, MAG240M
```

Baselines:

```text
MLP, SIGN/SAGN/NARS-style preprocessing, neighbor-sampling GraphSAGE, simple target coreset, Graph-Skeleton on MAG240M if feasible, OOM/OOT for methods that cannot run
```

Do not hide OOM/OOT. Report them as scalability results under a fixed resource budget.

---

## Paper positioning to preserve

Use this framing:

> Shadow-HGC-R-1 reformulates ultra-scale heterogeneous graph condensation for target-type node classification as target-side directed typed relation-demand condensation, rather than source-node selection or dense synthetic adjacency learning.

Do not claim:

- exact preservation of full graph topology;
- exact preservation for HGT/HAN attention;
- arbitrary downstream task support;
- all-HGNN model-agnostic guarantees;
- no synthetic edges at all.

Correct wording:

> We avoid dense synthetic adjacency learning and construct sparse typed assignments from relation-demand factorization.

---

## Interaction policy for coding agents

When implementing:

1. Prefer the smallest end-to-end runnable version before optimizing.
2. Add tests for every mathematical invariant before running large experiments.
3. Do not silently change edge direction, feature schema, normalization, or exposed graph schema.
4. Do not introduce external baselines into the main code path before the main method is stable.
5. Every experiment script must write a machine-readable JSON summary.
6. Every large-data script must have a dry-run mode that estimates memory, disk, and edge-slice cache size.
7. When uncertain, preserve correctness and explicitness over speed.
