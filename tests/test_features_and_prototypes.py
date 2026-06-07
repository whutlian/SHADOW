import torch

from shadow_hgc.data.schemas import DirectedRelation
from shadow_hgc.features.degree import degree_encoding_dim, encode_target_degrees
from shadow_hgc.features.projection import fixed_random_projection, fit_standardizer, standardize
from shadow_hgc.prototype.budgets import class_wise_budget
from shadow_hgc.prototype.cluster import class_wise_prototypes
from shadow_hgc.prototype.signatures import block_normalize, build_target_signature


def test_degree_encoding_uses_log_bucket_and_zero_indicator_per_relation():
    relations = [
        DirectedRelation("author", "writes", "paper"),
        DirectedRelation("paper", "cite_ref", "paper"),
    ]
    degree_by_relation = {
        relations[0]: torch.tensor([0, 1, 5]),
        relations[1]: torch.tensor([2, 33, 65]),
    }

    encoded = encode_target_degrees(degree_by_relation, relations)

    assert encoded.shape == (3, 2 * degree_encoding_dim())
    assert torch.allclose(encoded[0, 0], torch.tensor(0.0))
    assert encoded[0, 10] == 1.0
    assert encoded[1, 1 + 1] == 1.0
    assert encoded[2, 1 + 4] == 1.0
    offset = degree_encoding_dim()
    assert encoded[0, offset + 1 + 2] == 1.0
    assert encoded[1, offset + 1 + 7] == 1.0
    assert encoded[2, offset + 1 + 8] == 1.0


def test_fixed_random_projection_is_reproducible_and_standardizer_is_train_scoped():
    x = torch.arange(20, dtype=torch.float32).reshape(5, 4)
    projected_a = fixed_random_projection(x, out_dim=3, seed=11)
    projected_b = fixed_random_projection(x, out_dim=3, seed=11)
    projected_c = fixed_random_projection(x, out_dim=3, seed=12)

    assert torch.allclose(projected_a, projected_b)
    assert not torch.allclose(projected_a, projected_c)

    stats = fit_standardizer(projected_a, rows=torch.tensor([0, 1, 2]))
    standardized = standardize(projected_a, stats)
    assert torch.allclose(standardized[:3].mean(dim=0), torch.zeros(3), atol=1e-6)


def test_block_normalized_signature_and_class_wise_prototype_means():
    labels = torch.tensor([0, 0, 0, 1, 1])
    train_idx = torch.arange(5)
    budgets = class_wise_budget(labels, train_idx, M_tau=3)

    assert budgets[0] >= budgets[1]
    assert sum(budgets.values()) == 3

    psi = torch.eye(5, 3)
    demand = {DirectedRelation("author", "writes", "paper"): psi * 2}
    degree = torch.ones(5, 2)
    signature = build_target_signature(psi, demand, degree, eta=0.1)
    manual = block_normalize([psi, psi * 2, 0.1 * degree])
    assert torch.allclose(signature, manual)

    phi = torch.arange(20, dtype=torch.float32).reshape(5, 4)
    result = class_wise_prototypes(
        phi_target=phi,
        signatures=signature,
        labels=labels,
        train_idx=train_idx,
        M_tau=3,
        seed=3,
    )

    assert result.prototype_features.shape[0] == 3
    assert torch.allclose(result.prototype_weights.sum(), torch.tensor(5.0))
    for cell_id, members in enumerate(result.cell_members):
        expected = phi[members].mean(dim=0)
        assert torch.allclose(result.prototype_features[cell_id], expected)
