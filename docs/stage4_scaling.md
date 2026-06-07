# Stage 4 Scaling Infrastructure

Implemented Stage 4 infrastructure:

- `.npy` memmap-backed feature stores;
- deterministic fixed random projection;
- train-scoped standardization utilities;
- re-iterable chunked edge streams;
- two-pass train-target-only relation demand cache;
- active source tracking;
- source-id block feature gather;
- compact target-target edge-slice cache with `(dst_train_pos, src_train_pos, alpha)`;
- relation-specific dry-run byte estimator;
- synthetic streaming stress helper.

Large-mode safeguards:

- all-node demand caching is rejected unless debug mode is explicit;
- id-to-position maps use a dense int32 representation only when the configured
  memory budget allows it, otherwise sorted train ids plus `searchsorted`;
- full edge scans are counted;
- edge-slice edge counts, bytes, dtype, build/aggregation time, and disk-spill
  status are logged;
- demand cache shape is `N_train_target x d`, not `N_all_target x d`.
