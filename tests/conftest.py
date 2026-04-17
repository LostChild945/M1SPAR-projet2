"""
Fixtures partagées pour les tests.

Crée un AppState synthétique (10 produits, 5 users, rank=16) pour
tester le recommender sans charger les données réelles.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Ajouter src/api au path pour les imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "api"))


@pytest.fixture
def mock_state():
    """AppState synthétique pour les tests."""
    from store import AppState

    state = AppState()
    rng = np.random.default_rng(42)

    n_products = 10
    n_users = 5
    rank = 16

    state.product_ids = [f"PROD_{i:03d}" for i in range(n_products)]
    state.product_index = {pid: i for i, pid in enumerate(state.product_ids)}
    state.product_categories = ["All_Beauty"] * 5 + ["Appliances"] * 5
    state.product_is_cold = np.array([False] * 8 + [True] * 2)
    state.product_matrix = rng.standard_normal((n_products, rank)).astype(np.float32)
    norms = np.linalg.norm(state.product_matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    state.product_matrix_norm = (state.product_matrix / norms).astype(np.float32)

    user_ids = [f"USER_{i:03d}" for i in range(n_users)]
    state.user_index = {uid: i for i, uid in enumerate(user_ids)}
    state.user_matrix = rng.standard_normal((n_users, rank)).astype(np.float32)
    state.user_segments = {
        "USER_000": "power_user",
        "USER_001": "regular_user",
        "USER_002": "casual_user",
        "USER_003": "regular_user",
        "USER_004": "casual_user",
    }

    state.purchased = {
        "USER_000": {"PROD_000", "PROD_001"},
        "USER_001": {"PROD_002"},
    }
    state.popular_product_ids = state.product_ids[:5]

    return state
