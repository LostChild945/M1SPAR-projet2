"""
cold_start.py
-------------
Stratégie cold start : classification warm/cold + popularité par catégorie.

Ce module expose la logique de classification et de fallback utilisée
par le pipeline (analyse/03_cold_start_bridge.py) et l'API de serving.

Stratégie :
  - Warm (≥ MIN_INTERACTIONS)  → facteurs ALS (P1-P16 / U1-U16)
  - Cold produit               → bridge model (ST → ALS)
  - Cold user                  → popularité par catégorie

Usage :
    python -m src.ml.cold_start
"""

import os

import numpy as np
import pandas as pd

BASE = os.environ.get("DATA_DIR", "./analyse/data")
MIN_INTERACTIONS = 5


def classify_warm_cold(interactions: pd.DataFrame, id_col: str) -> tuple[set, set]:
    """
    Classifie les entités en warm (≥ MIN_INTERACTIONS) et cold (< MIN_INTERACTIONS).

    Returns:
        (warm_ids, cold_ids)
    """
    counts = interactions.groupby(id_col).size()
    warm = set(counts[counts >= MIN_INTERACTIONS].index)
    cold = set(counts[counts < MIN_INTERACTIONS].index)
    return warm, cold


def popularity_by_category(interactions: pd.DataFrame, top_k: int = 10) -> dict[str, list[str]]:
    """
    Calcule les top-K produits les plus populaires par catégorie.

    Returns:
        {category: [product_id_1, ..., product_id_k]}
    """
    pop = (
        interactions.groupby(["category", "product_id"])
        .size()
        .reset_index(name="count")
        .sort_values(["category", "count"], ascending=[True, False])
    )
    result = {}
    for cat, group in pop.groupby("category"):
        result[cat] = group["product_id"].head(top_k).tolist()
    return result


def recommend_cold_user(
    user_id: str,
    category: str | None,
    popular_by_cat: dict[str, list[str]],
    popular_global: list[str],
    n: int = 10,
) -> list[str]:
    """
    Recommandations pour un cold user : popularité par catégorie si
    catégorie connue, sinon popularité globale.
    """
    if category and category in popular_by_cat:
        return popular_by_cat[category][:n]
    return popular_global[:n]


def run_cold_start_analysis(base=BASE):
    """Analyse et rapport sur le cold start dans le dataset."""
    print("=" * 60)
    print("COLD START — Classification warm/cold")
    print("=" * 60)

    interactions = pd.read_parquet(
        os.path.join(base, "interactions"),
        columns=["user_id", "product_id", "category"],
    )

    warm_products, cold_products = classify_warm_cold(interactions, "product_id")
    warm_users, cold_users = classify_warm_cold(interactions, "user_id")

    total_p = len(warm_products) + len(cold_products)
    total_u = len(warm_users) + len(cold_users)

    print(f"\n  Produits warm : {len(warm_products):>8,}  ({len(warm_products)/total_p*100:.1f}%)")
    print(f"  Produits cold : {len(cold_products):>8,}  ({len(cold_products)/total_p*100:.1f}%)")
    print(f"  Users warm    : {len(warm_users):>8,}  ({len(warm_users)/total_u*100:.1f}%)")
    print(f"  Users cold    : {len(cold_users):>8,}  ({len(cold_users)/total_u*100:.1f}%)")

    pop_by_cat = popularity_by_category(interactions)
    print(f"\n  Catégories avec top-10 populaire : {len(pop_by_cat)}")
    for cat, prods in list(pop_by_cat.items())[:3]:
        print(f"    {cat} : {prods[:3]} …")

    print("\nCold start analysis terminée.")


if __name__ == "__main__":
    run_cold_start_analysis()
