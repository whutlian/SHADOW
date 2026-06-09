from __future__ import annotations

from shadow_hgc.audit.parity import validate_promoted_row


def test_no_diffusion_or_products_p2_in_promoted_rows():
    diffusion = validate_promoted_row(
        {"dataset": "ogbn-arxiv", "variant": "LAD_reference_tuned", "status": "completed", "use_diffusion": True}
    )
    products_p2 = validate_promoted_row(
        {"dataset": "ogbn-products", "variant": "LAD_plus_two_hop_LAD", "status": "completed", "path_lad_blocks": ["P1", "P2"]}
    )

    assert "diffusion_not_allowed_in_promoted_path" in diffusion["reasons"]
    assert "products_p2_lad_not_allowed_in_promoted_path" in products_p2["reasons"]
