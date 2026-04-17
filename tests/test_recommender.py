"""Tests pour le module recommender."""

import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "api"))


def test_warm_user_gets_als_recommendations(mock_state):
    """Un user warm reçoit des recommandations ALS."""
    import store
    store.state = mock_state

    from recommender import get_recommendations

    items, segment, is_cold = get_recommendations("USER_000", n=5, category=None, exclude_purchased=False)
    assert len(items) == 5
    assert segment == "power_user"
    assert not is_cold
    assert all(item.source in ("als", "bridge") for item in items)


def test_cold_user_gets_popular_recommendations(mock_state):
    """Un user inconnu reçoit les produits populaires."""
    import store
    store.state = mock_state

    from recommender import get_recommendations

    items, segment, is_cold = get_recommendations("UNKNOWN_USER", n=3, category=None, exclude_purchased=False)
    assert len(items) == 3
    assert is_cold
    assert all(item.source == "popular" for item in items)


def test_exclude_purchased(mock_state):
    """Les produits déjà achetés sont exclus."""
    import store
    store.state = mock_state

    from recommender import get_recommendations

    items, _, _ = get_recommendations("USER_000", n=10, category=None, exclude_purchased=True)
    reco_ids = {item.product_id for item in items}
    assert "PROD_000" not in reco_ids
    assert "PROD_001" not in reco_ids


def test_category_filter(mock_state):
    """Le filtre catégorie fonctionne."""
    import store
    store.state = mock_state

    from recommender import get_recommendations

    items, _, _ = get_recommendations("USER_001", n=5, category="Appliances", exclude_purchased=False)
    for item in items:
        assert item.category == "Appliances"


def test_get_similar_returns_items(mock_state):
    """get_similar retourne des produits similaires."""
    import store
    store.state = mock_state

    from recommender import get_similar

    items = get_similar("PROD_000", n=3)
    assert items is not None
    assert len(items) == 3
    assert all(item.product_id != "PROD_000" for item in items)


def test_get_similar_unknown_product(mock_state):
    """get_similar retourne None pour un produit inconnu."""
    import store
    store.state = mock_state

    from recommender import get_similar

    result = get_similar("UNKNOWN_PROD", n=3)
    assert result is None
