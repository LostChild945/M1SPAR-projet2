"""Tests pour le module ab_testing."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "api"))


def test_get_variant_deterministic():
    """Le même user_id donne toujours la même variante."""
    from ab_testing import get_variant

    variant1 = get_variant("USER_001")
    variant2 = get_variant("USER_001")
    assert variant1 == variant2
    assert variant1 in ("control", "treatment")


def test_get_variant_distribution():
    """La distribution est à peu près 50/50 sur un grand échantillon."""
    from ab_testing import get_variant

    variants = [get_variant(f"user_{i}") for i in range(1000)]
    control_pct = variants.count("control") / len(variants)
    assert 0.40 < control_pct < 0.60


def test_different_users_can_get_different_variants():
    """Au moins deux users ont des variantes différentes."""
    from ab_testing import get_variant

    variants = {get_variant(f"user_{i}") for i in range(100)}
    assert len(variants) == 2
