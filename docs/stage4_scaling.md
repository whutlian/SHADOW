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
- dry-run byte estimator;
- synthetic streaming stress helper.

Large-mode safeguards:

- all-node demand caching is rejected unless debug mode is explicit;
- full edge scans are counted;
- edge-slice bytes are logged;
- demand cache shape is `N_train_target x d`, not `N_all_target x d`.
