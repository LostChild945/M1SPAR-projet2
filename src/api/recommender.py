"""
Logique de recommandation et de similarité produit.

  get_recommendations → personnalisé par user (ALS dot product + cold start fallback)
  get_similar         → item-to-item cosine similarity sur P1-P16
"""
import numpy as np

from schemas import RecommendItem, SimilarItem
from store import state


def _minmax(arr: np.ndarray) -> np.ndarray:
    mn, mx = arr.min(), arr.max()
    return (arr - mn) / (mx - mn + 1e-8)


def get_recommendations(
    user_id: str,
    n: int,
    category: str | None,
    exclude_purchased: bool,
) -> tuple[list[RecommendItem], str, bool]:
    """
    Retourne (items, segment, is_cold_user).

    Stratégie :
      - user connu  → dot product U1-U16 × P1-P16 (score ALS)
      - user inconnu → produits populaires (fallback)
    Source des items :
      - is_cold=False → "als"   (facteurs ALS directs)
      - is_cold=True  → "bridge" (embeddings via MLP bridge)
      - fallback      → "popular"
    """
    n_products = len(state.product_ids)

    # ── Masque catégorie ──────────────────────────────────────────────────────
    if category:
        cat_mask = np.array(
            [c == category for c in state.product_categories], dtype=bool
        )
    else:
        cat_mask = np.ones(n_products, dtype=bool)

    # ── Masque produits déjà achetés ─────────────────────────────────────────
    if exclude_purchased and user_id in state.purchased:
        seen = state.purchased[user_id]
        seen_mask = np.array(
            [pid in seen for pid in state.product_ids], dtype=bool
        )
        candidate_mask = cat_mask & ~seen_mask
    else:
        candidate_mask = cat_mask

    candidate_indices = np.where(candidate_mask)[0]
    if len(candidate_indices) == 0:
        return [], "casual_user", True

    # ── Utilisateur connu ─────────────────────────────────────────────────────
    if user_id in state.user_index:
        idx = state.user_index[user_id]
        user_emb = state.user_matrix[idx]                       # (16,)
        segment = state.user_segments.get(user_id, "regular_user")

        # Scores ALS : dot product sur les candidats uniquement
        candidate_matrix = state.product_matrix[candidate_indices]  # (k, 16)
        raw_scores = candidate_matrix @ user_emb                    # (k,)
        norm_scores = _minmax(raw_scores)

        top_k = min(n, len(candidate_indices))
        top_local = np.argpartition(norm_scores, -top_k)[-top_k:]
        top_local = top_local[np.argsort(norm_scores[top_local])[::-1]]

        items = [
            RecommendItem(
                product_id=state.product_ids[candidate_indices[li]],
                score=float(norm_scores[li]),
                source="bridge" if state.product_is_cold[candidate_indices[li]] else "als",
                category=state.product_categories[candidate_indices[li]],
            )
            for li in top_local
        ]
        return items, segment, False

    # ── Utilisateur inconnu → fallback popularité ────────────────────────────
    candidates = [
        pid for pid in state.popular_product_ids
        if candidate_mask[state.product_index[pid]]
    ][:n]

    items = [
        RecommendItem(
            product_id=pid,
            score=round(1.0 - i / max(len(candidates), 1), 4),
            source="popular",
            category=state.product_categories[state.product_index[pid]],
        )
        for i, pid in enumerate(candidates)
    ]
    return items, "casual_user", True


def get_similar(product_id: str, n: int) -> list[SimilarItem] | None:
    """
    Similarité cosinus sur les embeddings P1-P16 normalisés.
    Retourne None si product_id inconnu.
    """
    if product_id not in state.product_index:
        return None

    idx = state.product_index[product_id]
    query = state.product_matrix_norm[idx]              # (16,)

    # Cosine similarity = dot product sur vecteurs normalisés
    scores = state.product_matrix_norm @ query          # (n_products,)
    scores[idx] = -1.0                                  # exclure le produit lui-même

    top_k = min(n, len(state.product_ids) - 1)
    top_indices = np.argpartition(scores, -top_k)[-top_k:]
    top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

    return [
        SimilarItem(
            product_id=state.product_ids[i],
            score=float(np.clip(scores[i], 0.0, 1.0)),
            category=state.product_categories[i],
        )
        for i in top_indices
    ]
