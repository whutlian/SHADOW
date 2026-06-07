# GOAL.md — Execution Plan for Shadow-HGC-R-1

Use this file as the project `/goal` for Codex.

## Final objective

Build a reproducible research codebase for **Shadow-HGC-R-1**, a schema-preserving typed message condensation method for target-type node classification on large heterogeneous graphs.

The final project must demonstrate:

1. the method runs end-to-end on small heterogeneous graphs;
2. the target-target residual skeleton works on citation / co-purchase graphs;
3. the implementation scales through train-target-only demand caching and edge streaming;
4. the ultra-scale path can run on ogbn-papers100M and MAG240M under fixed resource constraints;
5. ablations validate the three core mechanisms:
   - degree-calibrated demand;
   - target-target residual skeleton;
   - virtual demand shadows rather than real source centroids.

Main method is fixed as **Shadow-HGC-R-1**. Do not add new main modules. Weighted `b=2`, meta-path sketch, institution initializer, PCA, and HGT/HAN are optional ablations or transfer studies only.

---

## Stage 0 — Repository bootstrap and invariant tests

### Purpose

Create the project skeleton and lock down the mathematical / engineering invariants before writing dataset-specific code.

### Tasks

1. Create repository structure:

```text
shadow-hgc-r1/
  configs/
  shadow_hgc/
  scripts/
  tests/
  experiments/
  docs/
```

2. Add configuration system with fields for:

```text
dataset
target_type
directed_relations
feature_dim
projection_type
standardization_scope
M_tau
M_r per relation
k_s
shadow_assignment_b
loss_type
seed
resource_budget
```

3. Implement tiny utilities:

- directed relation schema object;
- edge direction checker;
- destination-row alpha normalization;
- weighted scatter-add helper;
- JSON logging helper.

4. Implement unit tests for:

- source/destination edge convention;
- destination-row normalization;
- no source-degree normalization leakage;
- weighted scatter-add equivalence;
- schema-preserving relation names.

### Acceptance criteria

- `pytest tests/test_direction.py` passes.
- `pytest tests/test_alpha_normalization.py` passes.
- A tiny hand-written graph produces exactly the manually computed normalized relation messages.
- No dataset download is required at this stage.

### Deliverables

- `README.md` with method one-liner and setup instructions.
- `docs/invariants.md` explaining feature schema, alpha normalization, and edge direction.
- Machine-readable default config template.

---

## Stage 1 — Toy end-to-end Shadow-HGC-R-1 pipeline

### Purpose

Build the smallest complete version of the method on a synthetic heterogeneous graph. This stage is about correctness, not performance.

### Toy graph requirements

Construct at least one synthetic graph with:

```text
target type: paper
source type: author
relations:
  paper --cite_ref--> paper
  paper --cited_by--> paper
  author --writes--> paper
labels on paper only
```

The graph must contain:

- target-target edges between labeled train nodes;
- target-target edges from unlabeled / validation / test nodes into train nodes;
- zero-degree target nodes for at least one relation;
- featureless author nodes;
- at least two classes with imbalance.

### Tasks

1. Implement base feature provider:

```text
psi_t(v)
```

For toy author nodes, initialize author features from neighboring paper features.

2. Implement degree encoding:

```text
g_u^r = [log(1 + deg_r(u)), Bucket(deg_r(u)), zero_degree_indicator]
```

3. Implement model input feature:

```text
phi_tau(u) = [psi_tau(u) || g_u]
phi_sigma(v) = psi_sigma(v) for non-target source types
```

4. Implement relation demand aggregation:

```text
mu_u^r = sum_{v in N_r(u)} alpha_uv^r * phi_source(v)
```

5. Implement class-wise target signatures:

```text
s_u = BlockNorm([psi_tau(u) || {mu_u^r}_r || eta * g_u])
```

6. Implement class-wise KMeans target prototyping.

7. Implement target-target residual skeleton:

```text
D_i^r
S_ij^r
top-k_s skeleton edges
B_i^r = D_i^r - sum_j S_ij^r x_j'
```

8. Implement non-target typed shadow factorization.

9. Implement `b=1` nearest shadow assignment.

10. Materialize a schema-preserving condensed graph.

11. Implement custom `WeightedRelationLinearConv` with explicit edge weights and no library auto-normalization.

12. Train on the condensed graph and run inference on the original toy graph using the same `phi_t` and same `alpha_uv^r`.

### Acceptance criteria

- End-to-end script runs in under one minute on CPU.
- Condensed graph exposes only original node types and edge types.
- Residual shadows may have signed features, but all edge weights are non-negative.
- Target-target top-`k_s` skeleton weights are not renormalized after truncation.
- Weighted relation-linear layer output matches manual matrix/scatter computation.
- Logs include skeleton mass coverage, residual energy, and shadow reconstruction error.

### Deliverables

- `scripts/run_toy.py`
- `tests/test_target_target_skeleton.py`
- `tests/test_residual_shadow.py`
- `tests/test_schema_preservation.py`
- `tests/test_weighted_layer_no_double_norm.py`
- `experiments/logs/toy/*.json`

---

## Stage 2 — Small heterogeneous datasets: ACM / DBLP / IMDB

### Purpose

Validate that schema-preserving typed shadow condensation works on standard small heterogeneous node classification benchmarks.

### Datasets

Use:

```text
ACM
DBLP
IMDB
```

Target tasks should follow common heterogeneous benchmark conventions:

- ACM: paper classification;
- DBLP: author classification;
- IMDB: movie classification.

### Tasks

1. Add dataset loaders and schema configs.
2. Convert every relation into explicit message direction.
3. Build degree-calibrated target features.
4. Run Shadow-HGC-R-1 over multiple condensation ratios.
5. Train primary backbones:

```text
1-layer WeightedRelationLinear
1-layer / 2-layer weighted R-GCN-style relation-linear variant
weighted HeteroGraphSAGE-style variant
```

6. Add simple baselines:

```text
Random-HG
Herding-HG
K-Center-HG
real source centroids baseline
```

7. Integrate external baselines if feasible:

```text
HGCond
FreeHGC
HGC-Herd if code is available
```

8. Run core ablations:

- mean-only demand vs degree-calibrated demand;
- `k_s = {0, 1, 2, 4, 8}` when target-target relations exist;
- residual shadow on/off;
- virtual demand shadows vs real source centroids;
- `b=1` vs weighted `b=2` if implemented;
- one-layer vs two-layer backbone;
- weighted loss vs unweighted / clipped / class-balanced loss.

### Acceptance criteria

- One command can reproduce the small-dataset table.
- Results include mean ± std over at least 3 seeds.
- Logs include accuracy, condensation time, training time, graph size, and diagnostics.
- Method beats random/herding/k-center in at least the main compression settings, or failures are diagnosed with reconstruction metrics.
- Ablations show that at least one of degree calibration, residual skeleton, or virtual shadows contributes nontrivially.

### Deliverables

- `scripts/run_small.py`
- `configs/datasets/{acm,dblp,imdb}.yaml`
- `experiments/tables/small_main.csv`
- `experiments/tables/small_ablation.csv`
- `docs/small_results_notes.md`

---

## Stage 3 — Medium homogeneous / near-homogeneous datasets: ogbn-arxiv and ogbn-products

### Purpose

Validate scalability beyond small heterogeneous graphs and specifically test target-target residual skeleton on citation / co-purchase style graphs.

### Datasets

```text
ogbn-arxiv
ogbn-products
```

Treat them as homogeneous special cases of the same framework:

```text
target type: node or paper/product
relations:
  forward relation
  reverse relation
```

### Tasks

1. Add OGB loaders and standard splits.
2. Build explicit directed relations for both forward and reverse edges.
3. Run Shadow-HGC-R-1 at multiple condensation ratios.
4. Evaluate target-target skeleton with:

```text
k_s = 0, 1, 2, 4, 8
```

5. Compare against simple and scalable baselines where feasible:

```text
Random
Herding
K-Center
GCond / DosCond if feasible
SFGC
GCPA
CGC
DisCo
Bonsai
```

6. Report OOM/OOT under fixed resources for baselines that do not run.

### Acceptance criteria

- Method runs on ogbn-arxiv and ogbn-products without full dense adjacency or teacher training.
- Target-target residual skeleton improves over `k_s = 0` full-residual-shadow setting on at least one graph or provides a clear topology/size trade-off.
- Resource logs are complete.
- No default library normalization is used in the main weighted relation-linear layer.

### Deliverables

- `scripts/run_medium.py`
- `configs/datasets/ogbn_arxiv.yaml`
- `configs/datasets/ogbn_products.yaml`
- `experiments/tables/medium_main.csv`
- `experiments/tables/medium_ablation.csv`
- `experiments/figures/skeleton_coverage_vs_accuracy.*`

---

## Stage 4 — I/O-aware scaling infrastructure

### Purpose

Build the ultra-scale data path before running paper100M or MAG240M.

### Tasks

1. Implement memmap-backed feature storage.
2. Implement fixed random projection with reproducible seeds.
3. Implement standardization statistics with explicit scope:

```text
small datasets: documented transductive or train-only setting
MAG240M main: train-period paper features only
```

4. Implement edge streaming with at most two full edge scans per relation:

```text
Pass 1: degree / active source collection
Pass 2: message aggregation and optional compact edge-slice cache
```

5. Implement active source materialization:

```text
only source nodes incident to train target nodes are materialized for condensation
```

6. Implement source-id block feature gather to reduce random memmap access.

7. Implement target-target edge-slice cache:

```text
(dst_train_pos, src_train_pos, alpha)
```

8. Implement dry-run estimator for:

- demand cache bytes;
- edge-slice cache bytes;
- active source feature bytes;
- expected number of full edge scans;
- peak RAM estimate;
- disk spill estimate.

9. Add stress tests on synthetic large edge lists.

### Acceptance criteria

- Large-mode code path never attempts to cache all-node relation demand by default.
- Per relation full edge scans are counted and logged.
- Edge-slice cache has explicit size logs.
- Dry-run estimates are emitted before any ultra-scale run.
- Synthetic stress test can process tens or hundreds of millions of dummy edges in streaming mode without memory blowup.

### Deliverables

- `shadow_hgc/data/edge_stream.py`
- `shadow_hgc/data/memmap.py`
- `shadow_hgc/demand/cache.py`
- `scripts/dry_run_ultra.py`
- `tests/test_large_mode_no_all_node_cache.py`
- `experiments/logs/scaling_stress/*.json`

---

## Stage 5 — ogbn-papers100M ultra-scale sanity check

### Purpose

Demonstrate hundred-million-scale homogeneous scalability before MAG240M.

### Schema

```text
target type: paper
relations:
  paper --cite_ref--> paper
  paper --cited_by--> paper
```

Both relations are target-target relations and should use residual skeleton + residual paper shadows.

### Tasks

1. Implement ogbn-papers100M loader with memmap feature access.
2. Use fixed random projection as main feature reduction.
3. Use train-target-only relation demand cache.
4. Use edge-slice cache only for train-target-to-train-target edges.
5. Run dry-run estimate first.
6. Run Shadow-HGC-R-1 at at least one small condensation ratio.
7. Train weighted relation-linear / HeteroGraphSAGE-style backbone on condensed graph.
8. Run original graph inference with same directed relations and `alpha_uv^r` edge weights.
9. Compare against feasible baselines:

```text
Random target coreset
Herding / K-Center target coreset if feasible
MLP
SIGN/SAGN-style preprocessing if available
Neighbor-sampling GraphSAGE if available
OOM/OOT for methods that cannot run under fixed budget
```

### Acceptance criteria

- Full run completes under the agreed fixed resource setting.
- Logs include full edge scans, condensation time, train time, inference time, RAM, GPU RAM, disk bytes, edge-slice cache bytes.
- Condensed graph size is reported by node type and relation.
- Results demonstrate training-cost reduction even if accuracy is below full-graph SOTA.

### Deliverables

- `scripts/run_paper100m.py`
- `configs/datasets/ogbn_papers100m.yaml`
- `experiments/tables/paper100m_feasibility.csv`
- `docs/paper100m_runbook.md`

---

## Stage 6 — MAG240M ultra-scale heterogeneous feasibility

### Purpose

Validate the main target use case: ultra-scale heterogeneous target-type paper classification.

### Main schema

Use exactly this main schema first:

```text
target type: paper
relations:
  paper --cite_ref--> paper
  paper --cited_by--> paper
  author --writes--> paper
```

Do not include institution in the main schema. Institution-aware author initialization is an ablation only.

Do not materialize meta-path edges such as:

```text
paper --PAP--> paper
```

Meta-path features may be signature-only auxiliary sketches in ablation.

### Feature policy

Main method:

- fixed random projection;
- train-period standardization statistics;
- temporal-safe author feature initializer;
- active author materialization only;
- type-level mean fallback for authors with no train-period papers.

Appendix / sensitivity only:

- full-transductive author initializer;
- PCA;
- institution-aware initializer;
- strict temporal graph edge filtering.

### Tasks

1. Implement MAG240M schema loader.
2. Implement train-period paper feature projection and standardization.
3. Implement temporal-safe author initializer with active-author materialization.
4. Implement train-target-only demand cache for the three main relations.
5. Implement edge-slice cache for the two paper-paper relations.
6. Run dry-run estimate and inspect cache size before full run.
7. Run Shadow-HGC-R-1 at one feasible budget.
8. Train weighted relation-linear / HeteroGraphSAGE-style backbone.
9. Run full-graph or chunked original graph inference.
10. Report feasibility and resource table.

### Acceptance criteria

- The run completes without all-node demand caching.
- The number of full edge scans per relation is logged and no relation exceeds two full scans unless explicitly documented.
- Edge-slice cache bytes and active author feature bytes are reported.
- Accuracy, train time, condensation time, inference time, peak memory, disk bytes, and compressed graph size are reported.
- The paper narrative emphasizes feasibility and training-cost reduction, not beating full-graph SOTA.

### Deliverables

- `scripts/run_mag240m.py`
- `configs/datasets/mag240m.yaml`
- `docs/mag240m_runbook.md`
- `experiments/tables/mag240m_feasibility.csv`
- `experiments/logs/mag240m/*.json`

---

## Stage 7 — Ablation and diagnostics package

### Purpose

Produce KDD-quality evidence that the method is not a bag of heuristics and that its novelty is not ordinary node selection or source coarsening.

### Required ablations

Run the following where dataset scale allows:

1. **Mean-only demand vs degree-calibrated demand**
   - Purpose: prove degree calibration is not decoration.

2. **Skeleton top-k**
   - Values: `k_s = {0, 1, 2, 4, 8}`.
   - Purpose: prove target-target residual skeleton preserves topology and identify topology/size trade-off.

3. **Residual shadow on/off**
   - Purpose: separate skeleton contribution from residual correction contribution.

4. **Virtual demand shadows vs real source centroids**
   - Purpose: prove shadows are target-side demand carriers, not real source node coarsening or Graph-Skeleton variants.

5. **b=1 vs weighted b=2**
   - Purpose: compare simple deployable main version with stronger reconstruction variant.

6. **One-layer vs two-layer R-GCN/HeteroSAGE-style backbones**
   - Purpose: quantify shadow-node hidden-state mismatch.

7. **Temporal-safe vs full-transductive author initializer**
   - Purpose: show leakage risk and upper-bound gap.

8. **Prototype loss variants**
   - Values: `|C_i|`-weighted, unweighted, clipped-weight, class-balanced.
   - Purpose: prevent class-balancing-trick criticism.

9. **Relation norm calibration on/off**
   - Purpose: show scale calibration is not artificially tuned to validation data.

10. **Optional high-order signature sketch on/off**
   - Purpose: evaluate meta-path information without making meta-paths the main novelty.

### Required diagnostics

For each relation `r`, log:

```text
SkeletonMassCoverage_r(k_s)
ResidualEnergy_r
ShadowReconErr_r
```

Also log:

- shadow feature norm distribution;
- real source feature norm distribution when available;
- condensed edge count by relation;
- target prototype count by class;
- cluster-size distribution.

### Acceptance criteria

- Main ablation table can be reproduced from scripts.
- At least one figure shows `k_s` vs skeleton coverage / accuracy / edge count.
- Real source centroids baseline is clearly worse or qualitatively different; if not, analyze failure honestly.
- Diagnostics explain major accuracy changes.

### Deliverables

- `scripts/run_ablation.py`
- `experiments/tables/ablation_main.csv`
- `experiments/figures/diagnostics_*`
- `docs/ablation_notes.md`

---

## Stage 8 — External baselines and final experiment tables

### Purpose

Assemble final comparison tables and OOM/OOT evidence under a fixed resource budget.

### Small hetero baseline table

Datasets:

```text
ACM, DBLP, IMDB
```

Baselines:

```text
Random-HG
Herding-HG
K-Center-HG
HGCond
FreeHGC
HGC-Herd if available
Shadow-HGC-R-1
```

### Medium table

Datasets:

```text
ogbn-arxiv, ogbn-products
```

Baselines:

```text
Random
Herding
K-Center
GCond / DosCond if feasible
SFGC
GCPA
CGC
DisCo
Bonsai
Shadow-HGC-R-1
```

### Ultra-scale table

Datasets:

```text
ogbn-papers100M, MAG240M
```

Baselines:

```text
MLP
SIGN/SAGN/NARS-style preprocessing
Neighbor-sampling GraphSAGE
Simple target coreset
Graph-Skeleton on MAG240M if feasible
Shadow-HGC-R-1
OOM/OOT entries for methods that cannot run
```

### Metrics

Report:

```text
accuracy
macro-F1 if applicable
condensation time
training time
full-graph inference time
number of full edge scans
edge-slice cache bytes
peak CPU RAM
peak GPU RAM
disk bytes
condensed nodes / edges
byte-size compression ratio
cross-backbone transfer
OOM/OOT status
```

### Acceptance criteria

- Tables have fixed resource budget annotations.
- Baseline failures are reported as OOM/OOT, not omitted.
- Accuracy is not the only metric; training cost and feasibility are central.
- Final result narrative does not claim MAG240M full-graph SOTA unless actually achieved.

### Deliverables

- `experiments/tables/final_small.csv`
- `experiments/tables/final_medium.csv`
- `experiments/tables/final_ultra.csv`
- `docs/baseline_protocol.md`
- `docs/resource_budget.md`

---

## Stage 9 — Paper artifacts and reproducibility package

### Purpose

Prepare the method, experiment evidence, and reproducibility materials for a KDD-style submission.

### Tasks

1. Write method notes with final formulas:

- directed relation definition;
- unified feature schema `psi_t` and `phi_t`;
- alpha normalization;
- degree-calibrated demand;
- class-wise target prototypes;
- target-target residual skeleton;
- signed residual shadows;
- non-target typed shadow factorization;
- custom weighted relation-linear layer;
- train-target-only I/O.

2. Create diagrams:

- overview pipeline;
- target-target residual skeleton decomposition;
- schema-preserving shadow graph materialization;
- ultra-scale edge streaming pipeline.

3. Write theory note:

- one-layer relation-linear HGNN pre-activation error bound;
- clearly state that deeper and attention-based HGNNs are empirical transfer only.

4. Write reproducibility instructions:

- environment;
- data preparation;
- dry-run memory estimator;
- one-command scripts;
- expected logs;
- resource requirements.

5. Freeze configs and random seeds.

### Acceptance criteria

- A new collaborator can run toy and small experiments from README.
- Dry-run estimator is documented before ultra-scale commands.
- All main tables are generated from scripts, not manually edited.
- The paper text never overclaims full higher-order topology or HGT/HAN exact preservation.

### Deliverables

- `docs/method_section_draft.md`
- `docs/theory_note.md`
- `docs/reproducibility.md`
- `docs/figures_plan.md`
- frozen configs under `configs/experiments/final/`

---

## Final definition of done

The project is considered ready for paper-writing when all of the following hold:

1. Toy end-to-end tests pass.
2. Small heterogeneous experiments run with at least 3 seeds.
3. Medium ogbn-arxiv / ogbn-products experiments run and include target-target skeleton ablations.
4. Ultra-scale dry-run estimates are available before full runs.
5. At least one of ogbn-papers100M or MAG240M completes under fixed resource constraints.
6. Main ablations validate degree calibration, residual skeleton, and virtual demand shadows.
7. Resource metrics are logged for every major run.
8. No code path uses default library normalization on pre-normalized edge weights.
9. No main code path caches all-node relation demand for ultra-scale datasets.
10. Exposed condensed graphs preserve original node types and edge types.

---

## Default main configuration

```yaml
method: Shadow-HGC-R-1
claim_scope: target-type node classification
feature:
  projection: fixed_random_projection
  ultra_dim: 128
  standardization: train_period_for_MAG240M
  pca: appendix_only
relation:
  direction: source_to_destination
  normalization: destination_row_alpha
prototype:
  clustering: class_wise_minibatch_kmeans
  budget_exponent: 0.5
  loss: cell_weighted_ce
skeleton:
  target_target: true
  k_s: 2
  renormalize_topk: false
shadow:
  assignment_b: 1
  weighted_b2: ablation_only
  residual_features_signed: true
  edge_weights_nonnegative: true
schema:
  preserve_original_node_types: true
  preserve_original_edge_types: true
model:
  main_backbone: weighted_relation_linear
  two_layer: empirical_only
  hgt_han: transfer_only
io:
  train_target_only_demand: true
  max_full_edge_scans_per_relation: 2
  active_source_materialization: true
  target_target_edge_slice_cache: true
MAG240M:
  relations:
    - paper--cite_ref-->paper
    - paper--cited_by-->paper
    - author--writes-->paper
  author_initializer: temporal_safe
  institution: ablation_only
meta_path:
  materialize_edges: false
  signature_only: optional_ablation
```

---

## First command sequence Codex should implement

Start here before touching real datasets:

```text
1. Create package skeleton.
2. Implement directed schema and alpha normalization.
3. Implement tiny synthetic heterogeneous graph.
4. Implement demand aggregation.
5. Implement class-wise prototype construction.
6. Implement target-target residual skeleton.
7. Implement b=1 shadow factorization.
8. Implement schema-preserving condensed graph materialization.
9. Implement custom WeightedRelationLinearConv.
10. Run toy training and original-graph inference.
11. Add tests for all invariants.
```

Do not start with MAG240M. Do not start with external baselines. Do not start with HGT.
