"""
API de recommandation Amazon — M1SPAR projet 2

Endpoints :
  GET  /recommend/{user_id}   Recommandations personnalisées
  GET  /similar/{product_id}  Produits similaires (item-to-item)
  POST /feedback              Signal utilisateur pour A/B testing
  GET  /ab_results            CTR + test Z par variant
  GET  /health                Santé + stats de démarrage
"""
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from prometheus_client import Counter
from prometheus_fastapi_instrumentator import Instrumentator

import cache
import db
from ab_testing import get_ab_results, get_variant
from recommender import get_recommendations, get_similar
from schemas import (
    ABResultsResponse,
    Error,
    FeedbackRequest,
    RecommendResponse,
    SimilarResponse,
)
from store import state

CACHE_HITS = Counter("cache_hits_total", "Total cache hits for /recommend")
CACHE_MISSES = Counter("cache_misses_total", "Total cache misses for /recommend")


@asynccontextmanager
async def lifespan(app: FastAPI):
    state.load()
    yield


app = FastAPI(
    title="P2 Recommandation Amazon",
    version="1.0.0",
    description=(
        "Combine ALS (warm users/items), Bridge MLP (cold items) "
        "et popularité (cold users). Latence cible P95 < 50 ms via cache Redis."
    ),
    lifespan=lifespan,
)

Instrumentator().instrument(app).expose(app)


# ── Recommandations personnalisées ────────────────────────────────────────────

@app.get(
    "/recommend/{user_id}",
    response_model=RecommendResponse,
    responses={404: {"model": Error}, 422: {"model": Error}},
    tags=["recommend"],
    summary="Recommandations personnalisées pour un utilisateur",
)
def recommend(
    user_id: str,
    n: int = Query(default=10, ge=1, le=100, description="Nombre de recommandations"),
    category: Optional[str] = Query(default=None, description="Filtre catégorie Amazon"),
    exclude_purchased: bool = Query(default=True, description="Exclure les produits déjà achetés"),
):
    t0 = time.monotonic()

    # Attribution A/B + log impression
    variant = get_variant(user_id)
    db.insert_impression(user_id, variant)

    # Cache Redis
    cache_key = f"reco:{user_id}:{n}:{category}:{exclude_purchased}"
    cached = cache.get(cache_key)
    if cached:
        CACHE_HITS.inc()
        cached["latency_ms"] = int((time.monotonic() - t0) * 1000)
        cached["cache_hit"] = True
        return cached

    CACHE_MISSES.inc()
    items, segment, is_cold_user = get_recommendations(user_id, n, category, exclude_purchased)

    if not items:
        raise HTTPException(
            status_code=404,
            detail=f"Aucune recommandation disponible pour '{user_id}'.",
        )

    result = RecommendResponse(
        user_id=user_id,
        segment=segment,
        recommendations=items,
        latency_ms=int((time.monotonic() - t0) * 1000),
        cache_hit=False,
    )
    cache.set(cache_key, result.model_dump())
    return result


# ── Produits similaires ───────────────────────────────────────────────────────

@app.get(
    "/similar/{product_id}",
    response_model=SimilarResponse,
    responses={404: {"model": Error}},
    tags=["similar"],
    summary="Produits similaires (item-to-item cosine sur P1-P16)",
)
def similar(
    product_id: str,
    n: int = Query(default=5, ge=1, le=50, description="Nombre de produits similaires"),
):
    t0 = time.monotonic()
    items = get_similar(product_id, n)

    if items is None:
        raise HTTPException(status_code=404, detail=f"Produit inconnu : {product_id}")

    return SimilarResponse(
        product_id=product_id,
        similar_products=items,
        latency_ms=int((time.monotonic() - t0) * 1000),
    )


# ── Feedback ──────────────────────────────────────────────────────────────────

@app.post(
    "/feedback",
    status_code=201,
    responses={422: {"model": Error}},
    tags=["feedback"],
    summary="Collecte de signal utilisateur pour A/B testing",
)
def feedback(body: FeedbackRequest):
    ts = body.timestamp or datetime.now(timezone.utc)
    db.insert_feedback(body.user_id, body.product_id, body.interaction_type, body.ab_variant, ts)
    return {"status": "recorded"}


# ── Résultats A/B ─────────────────────────────────────────────────────────────

@app.get(
    "/ab_results",
    response_model=ABResultsResponse,
    responses={404: {"model": Error}},
    tags=["ab"],
    summary="Résultats de l'expérience A/B en cours (CTR + test Z)",
)
def ab_results(
    experiment_id: str = Query(default="hybrid_vs_als_only"),
    segment: Optional[str] = Query(
        default=None,
        enum=["power_user", "regular_user", "casual_user"],
        description="Filtrer par segment utilisateur",
    ),
):
    return get_ab_results(experiment_id, segment)


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/health", tags=["health"])
def health():
    return {
        "status": "ok",
        "products": len(state.product_ids),
        "users": len(state.user_index),
        "popular_products": len(state.popular_product_ids),
    }
