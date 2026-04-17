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

FILTER_CATEGORIES = ["All_Beauty", "Amazon_Fashion"]   # 2 catégories pour les tests rapides

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

    # Extraire les IDs avant de libérer le DataFrame produits
    product_ids = products["product_id"].values
    corpus = build_corpus(products)
    del products  # libère ~1-2 GB avant l'encodage

    print(f"\nCalcul des embeddings ({len(corpus):,} textes) …")
    embeddings = model.encode(
        corpus,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,   # norme L2 = 1 → cosine similarity = dot product
    )
    del corpus  # libère la liste de textes dès que l'encodage est terminé
    print(f"  → Shape : {embeddings.shape}")  # (n_products, 384)

    print("\nSauvegarde …")
    emb_cols = [f"emb_{i}" for i in range(embeddings.shape[1])]
    # copy=False évite une 2e copie si l'array est déjà en float32 (cas SentenceTransformer)
    df_emb = pd.DataFrame(embeddings.astype(np.float32, copy=False), columns=emb_cols)
    del embeddings  # libère le numpy array dès que le DataFrame est construit
    df_emb.insert(0, "product_id", product_ids)
    df_emb.to_parquet(OUT_PATH, index=False)
    print(f"  → Sauvegardé : {OUT_PATH}")
    print(f"  → Taille     : {OUT_PATH.stat().st_size / 1e6:.1f} MB")
