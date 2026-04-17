"""Tests end-to-end de l'API FastAPI.

Utilise le TestClient de FastAPI avec un AppState synthétique
pour tester tous les endpoints sans données réelles ni Redis.
"""

import sys
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

# Ajouter src/api au path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "api"))

# Configurer DB_PATH vers un fichier temporaire AVANT d'importer les modules
_tmp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
_tmp_db.close()
os.environ["DB_PATH"] = _tmp_db.name


@pytest.fixture(autouse=True)
def _setup_state(mock_state):
    """Injecte le mock_state et désactive le chargement réel + le cache."""
    import store
    import cache
    import db

    store.state = mock_state

    # Désactiver Redis pour les tests
    cache._client = False

    # Réinitialiser la connexion DB pour partir d'une base propre à chaque test
    if hasattr(db._local, "conn"):
        db._local.conn.close()
        del db._local.conn

    # Patch state.load() pour ne pas charger les données réelles
    with patch.object(store.AppState, "load"):
        yield


@pytest.fixture()
def client():
    """Client HTTP de test FastAPI."""
    from fastapi.testclient import TestClient
    from main import app

    with TestClient(app) as c:
        yield c


# ═══════════════════════════════════════════════════════════════════════════
# GET /health
# ═══════════════════════════════════════════════════════════════════════════

class TestHealth:
    def test_health_returns_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["products"] == 10
        assert data["users"] == 5
        assert data["popular_products"] == 5


# ═══════════════════════════════════════════════════════════════════════════
# GET /recommend/{user_id}
# ═══════════════════════════════════════════════════════════════════════════

class TestRecommend:
    def test_warm_user_returns_recommendations(self, client):
        """Un utilisateur connu reçoit des recommandations ALS."""
        resp = client.get("/recommend/USER_000?n=5")
        assert resp.status_code == 200
        data = resp.json()
        assert data["user_id"] == "USER_000"
        assert data["segment"] == "power_user"
        assert len(data["recommendations"]) == 5
        assert data["cache_hit"] is False
        assert "latency_ms" in data
        for item in data["recommendations"]:
            assert "product_id" in item
            assert "score" in item
            assert item["source"] in ("als", "bridge")

    def test_cold_user_returns_popular(self, client):
        """Un utilisateur inconnu reçoit les produits populaires."""
        resp = client.get("/recommend/UNKNOWN_USER?n=3")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["recommendations"]) == 3
        for item in data["recommendations"]:
            assert item["source"] == "popular"

    def test_exclude_purchased(self, client):
        """Les produits déjà achetés par USER_000 sont exclus."""
        resp = client.get("/recommend/USER_000?n=10&exclude_purchased=true")
        assert resp.status_code == 200
        reco_ids = {r["product_id"] for r in resp.json()["recommendations"]}
        assert "PROD_000" not in reco_ids
        assert "PROD_001" not in reco_ids

    def test_category_filter(self, client):
        """Le filtre catégorie ne retourne que la catégorie demandée."""
        resp = client.get("/recommend/USER_001?n=5&category=Appliances")
        assert resp.status_code == 200
        for item in resp.json()["recommendations"]:
            assert item["category"] == "Appliances"

    def test_n_parameter_limits(self, client):
        """Le paramètre n respecte les bornes [1, 100]."""
        resp = client.get("/recommend/USER_000?n=0")
        assert resp.status_code == 422

        resp = client.get("/recommend/USER_000?n=101")
        assert resp.status_code == 422

    def test_default_parameters(self, client):
        """Les paramètres par défaut fonctionnent (n=10, exclude_purchased=true)."""
        resp = client.get("/recommend/USER_002")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["recommendations"]) <= 10


# ═══════════════════════════════════════════════════════════════════════════
# GET /similar/{product_id}
# ═══════════════════════════════════════════════════════════════════════════

class TestSimilar:
    def test_similar_returns_items(self, client):
        """Retourne des produits similaires pour un produit connu."""
        resp = client.get("/similar/PROD_000?n=3")
        assert resp.status_code == 200
        data = resp.json()
        assert data["product_id"] == "PROD_000"
        assert len(data["similar_products"]) == 3
        assert "latency_ms" in data
        # Le produit source ne doit pas apparaître dans les résultats
        for item in data["similar_products"]:
            assert item["product_id"] != "PROD_000"
            assert "score" in item
            assert "category" in item

    def test_similar_unknown_product_404(self, client):
        """Retourne 404 pour un produit inconnu."""
        resp = client.get("/similar/UNKNOWN_PROD?n=3")
        assert resp.status_code == 404
        assert "detail" in resp.json()

    def test_similar_n_limits(self, client):
        """Le paramètre n respecte les bornes [1, 50]."""
        resp = client.get("/similar/PROD_000?n=0")
        assert resp.status_code == 422

        resp = client.get("/similar/PROD_000?n=51")
        assert resp.status_code == 422


# ═══════════════════════════════════════════════════════════════════════════
# POST /feedback
# ═══════════════════════════════════════════════════════════════════════════

class TestFeedback:
    def test_feedback_click(self, client):
        """Enregistre un feedback de type click."""
        resp = client.post("/feedback", json={
            "user_id": "USER_000",
            "product_id": "PROD_000",
            "interaction_type": "click",
            "ab_variant": "control",
        })
        assert resp.status_code == 201
        assert resp.json()["status"] == "recorded"

    def test_feedback_purchase(self, client):
        """Enregistre un feedback de type purchase."""
        resp = client.post("/feedback", json={
            "user_id": "USER_001",
            "product_id": "PROD_003",
            "interaction_type": "purchase",
            "ab_variant": "treatment",
        })
        assert resp.status_code == 201

    def test_feedback_skip(self, client):
        """Enregistre un feedback de type skip."""
        resp = client.post("/feedback", json={
            "user_id": "USER_002",
            "product_id": "PROD_005",
            "interaction_type": "skip",
            "ab_variant": "control",
        })
        assert resp.status_code == 201

    def test_feedback_invalid_interaction_type(self, client):
        """Rejette un type d'interaction invalide."""
        resp = client.post("/feedback", json={
            "user_id": "USER_000",
            "product_id": "PROD_000",
            "interaction_type": "invalid",
            "ab_variant": "control",
        })
        assert resp.status_code == 422

    def test_feedback_invalid_variant(self, client):
        """Rejette une variante A/B invalide."""
        resp = client.post("/feedback", json={
            "user_id": "USER_000",
            "product_id": "PROD_000",
            "interaction_type": "click",
            "ab_variant": "invalid",
        })
        assert resp.status_code == 422

    def test_feedback_missing_fields(self, client):
        """Rejette un body incomplet."""
        resp = client.post("/feedback", json={
            "user_id": "USER_000",
        })
        assert resp.status_code == 422


# ═══════════════════════════════════════════════════════════════════════════
# GET /ab_results
# ═══════════════════════════════════════════════════════════════════════════

class TestABResults:
    def test_ab_results_default(self, client):
        """Retourne les résultats A/B avec les paramètres par défaut."""
        resp = client.get("/ab_results")
        assert resp.status_code == 200
        data = resp.json()
        assert data["experiment_id"] == "hybrid_vs_als_only"
        assert "control_ctr" in data
        assert "treatment_ctr" in data
        assert "lift_pct" in data
        assert "p_value" in data
        assert "significant" in data
        assert "n_users_control" in data
        assert "n_users_treatment" in data

    def test_ab_results_after_feedback(self, client):
        """Les résultats A/B reflètent le feedback enregistré."""
        # Envoyer du feedback
        for i in range(10):
            client.post("/feedback", json={
                "user_id": f"ab_user_{i}",
                "product_id": f"PROD_00{i % 5}",
                "interaction_type": "click",
                "ab_variant": "control" if i % 2 == 0 else "treatment",
            })

        resp = client.get("/ab_results")
        assert resp.status_code == 200
        data = resp.json()
        assert data["n_users_control"] >= 0
        assert data["n_users_treatment"] >= 0


# ═══════════════════════════════════════════════════════════════════════════
# Flux complet end-to-end
# ═══════════════════════════════════════════════════════════════════════════

class TestE2EFlow:
    def test_full_recommendation_flow(self, client):
        """Scénario complet : health -> recommend -> similar -> feedback -> ab_results."""
        # 1. Vérifier que l'API est healthy
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

        # 2. Obtenir des recommandations pour un user warm
        resp = client.get("/recommend/USER_000?n=3")
        assert resp.status_code == 200
        recos = resp.json()["recommendations"]
        assert len(recos) == 3
        first_product = recos[0]["product_id"]

        # 3. Trouver des produits similaires au premier recommandé
        resp = client.get(f"/similar/{first_product}?n=3")
        assert resp.status_code == 200
        assert len(resp.json()["similar_products"]) == 3

        # 4. Envoyer un feedback (click sur le premier produit)
        resp = client.post("/feedback", json={
            "user_id": "USER_000",
            "product_id": first_product,
            "interaction_type": "click",
            "ab_variant": "control",
        })
        assert resp.status_code == 201

        # 5. Vérifier les résultats A/B
        resp = client.get("/ab_results")
        assert resp.status_code == 200
        assert "p_value" in resp.json()

    def test_cold_user_flow(self, client):
        """Scénario cold user : recommandations populaires -> feedback."""
        # 1. Recommandations pour un user inconnu
        resp = client.get("/recommend/NEW_USER_999?n=5")
        assert resp.status_code == 200
        data = resp.json()
        assert all(r["source"] == "popular" for r in data["recommendations"])

        # 2. Le user donne du feedback
        product = data["recommendations"][0]["product_id"]
        resp = client.post("/feedback", json={
            "user_id": "NEW_USER_999",
            "product_id": product,
            "interaction_type": "purchase",
            "ab_variant": "treatment",
        })
        assert resp.status_code == 201
