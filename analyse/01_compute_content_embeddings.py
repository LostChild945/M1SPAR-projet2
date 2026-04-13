"""
01_compute_content_embeddings.py
---------------------------------
Calcule les embeddings de contenu (384 dims) pour tous les produits
via Sentence Transformers (all-MiniLM-L6-v2).

Ces embeddings encodent la sémantique du titre et de la description
produit — ils serviront de base au bridge model pour le cold start.

Output : data/content_embeddings/content_embeddings.parquet
         colonnes : product_id | emb_0 … emb_383

Usage :
    python 01_compute_content_embeddings.py
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer

DATA_DIR   = Path("./data")
OUT_PATH   = DATA_DIR / "content_embeddings" / "content_embeddings.parquet"
MODEL_NAME = "all-MiniLM-L6-v2"
BATCH_SIZE = 256

# =============================================================================
# FILTRE DE CATÉGORIES — optionnel
# -----------------------------------------------------------------------------
# Liste les catégories à inclure, ou None pour tout traiter.
# Exemples :
#   None                             → toutes les catégories disponibles
#   ["All_Beauty", "Appliances"]     → uniquement ces 2 catégories
#   TEST_5 (défini ci-dessous)       → 5 catégories légères pour les tests
# =============================================================================

TEST_5 = ["All_Beauty", "Amazon_Fashion", "Appliances", "Arts_Crafts_and_Sewing", "Automotive"]

FILTER_CATEGORIES = TEST_5   # ← mettre None pour traiter toutes les catégories

# =============================================================================

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)


def build_corpus(products: pd.DataFrame) -> list[str]:
    """Concatène titre + description pour enrichir l'embedding."""
    title = products["title"].fillna("")
    if "description" in products.columns:
        desc = products["description"].fillna("").apply(
            lambda x: x[0] if isinstance(x, list) and x else str(x)
        )
        return (title + " " + desc).str.strip().tolist()
    return title.tolist()


if __name__ == "__main__":
    print("Chargement des produits …")
    products = pd.read_parquet(DATA_DIR / "products")
    if FILTER_CATEGORIES is not None:
        products = products[products["category"].isin(FILTER_CATEGORIES)]
        print(f"  → Filtre actif : {FILTER_CATEGORIES}")
    print(f"  → {len(products):,} produits chargés")

    print(f"\nChargement du modèle {MODEL_NAME} …")
    model = SentenceTransformer(MODEL_NAME)

    corpus = build_corpus(products)
    print(f"\nCalcul des embeddings ({len(corpus):,} textes) …")
    embeddings = model.encode(
        corpus,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,   # norme L2 = 1 → cosine similarity = dot product
    )
    print(f"  → Shape : {embeddings.shape}")  # (n_products, 384)

    print("\nSauvegarde …")
    emb_cols = [f"emb_{i}" for i in range(embeddings.shape[1])]
    df_emb = pd.DataFrame(embeddings.astype(np.float32), columns=emb_cols)
    df_emb.insert(0, "product_id", products["product_id"].values)
    df_emb.to_parquet(OUT_PATH, index=False)
    print(f"  → Sauvegardé : {OUT_PATH}")
    print(f"  → Taille     : {OUT_PATH.stat().st_size / 1e6:.1f} MB")
