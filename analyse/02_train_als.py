"""
02_train_als.py
---------------
Entraîne ALS (Alternating Least Squares) sur les interactions Amazon
et extrait les facteurs latents :

  P1-P16  →  item factors  (embeddings comportementaux des produits)
  U1-U16  →  user factors  (embeddings comportementaux des utilisateurs)

Ces facteurs encodent les préférences implicites apprises depuis
les interactions réelles — ils sont la base du cold start bridge.

Utilise la librairie `implicit` (scipy sparse + BLAS/CUDA) au lieu de
PySpark ALS — beaucoup plus efficace sur machine unique.

Output :
  data/als_item_factors/    product_id | P1 … P16
  data/als_user_factors/    user_id    | U1 … U16
  data/id_mappings/         user_id_map.parquet | product_id_map.parquet

Usage :
    python 02_train_als.py
"""

import shutil
import numpy as np
import pandas as pd
import mlflow
from pathlib import Path
from scipy.sparse import csr_matrix
import implicit

DATA_DIR = Path("./data")

# =============================================================================
# FILTRE DE CATÉGORIES — optionnel
# -----------------------------------------------------------------------------
# Liste les catégories à inclure dans l'entraînement ALS, ou None pour tout.
# Exemples :
#   None    → toutes les catégories disponibles
#   TEST_5  → 5 catégories légères pour les tests
# =============================================================================

TEST_5 = ["All_Beauty", "Amazon_Fashion", "Appliances", "Arts_Crafts_and_Sewing", "Automotive"]

FILTER_CATEGORIES = ["All_Beauty", "Amazon_Fashion"]   # 2 catégories pour les tests rapides

# =============================================================================

# ── Hyperparamètres ALS ─────────────────────────────────────────────────────
ALS_RANK         = 16      # dimensions des embeddings → P1-P16 / U1-U16
ALS_MAX_ITER     = 15      # itérations d'optimisation
ALS_REG_PARAM    = 0.1     # régularisation L2 (évite le sur-apprentissage)
MIN_INTERACTIONS = 5       # nb minimum d'interactions pour être considéré "warm"

for sub in ("als_item_factors", "als_user_factors", "id_mappings"):
    (DATA_DIR / sub).mkdir(parents=True, exist_ok=True)

# Nettoie les éventuels dossiers parquet laissés par PySpark
for legacy in (
    DATA_DIR / "id_mappings" / "user_id_map.parquet",
    DATA_DIR / "id_mappings" / "product_id_map.parquet",
    DATA_DIR / "als_item_factors",
    DATA_DIR / "als_user_factors",
):
    if legacy.is_dir():
        shutil.rmtree(legacy)
        legacy.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    mlflow.set_tracking_uri(f"file:{DATA_DIR / 'mlruns'}")
    mlflow.set_experiment("als_training")

    print("Chargement des interactions …")
    interactions = pd.read_parquet(DATA_DIR / "interactions")
    if FILTER_CATEGORIES is not None:
        interactions = interactions[interactions["category"].isin(FILTER_CATEGORIES)]
        print(f"  → Filtre actif : {FILTER_CATEGORIES}")
    print(f"  → {len(interactions):,} interactions chargées")

    # ── Filtre MIN_INTERACTIONS ──────────────────────────────────────────────
    print(f"\nFiltrage des users/produits avec < {MIN_INTERACTIONS} interactions …")
    user_counts = interactions["user_id"].value_counts()
    item_counts = interactions["product_id"].value_counts()
    interactions = interactions[
        interactions["user_id"].isin(user_counts[user_counts >= MIN_INTERACTIONS].index) &
        interactions["product_id"].isin(item_counts[item_counts >= MIN_INTERACTIONS].index)
    ]
    print(f"  → {len(interactions):,} interactions après filtrage")

    # ── Indexation string → int ──────────────────────────────────────────────
    print("\nIndexation user_id et product_id …")
    user_ids  = interactions["user_id"].unique()
    item_ids  = interactions["product_id"].unique()

    user_to_idx = {u: i for i, u in enumerate(user_ids)}
    item_to_idx = {p: i for i, p in enumerate(item_ids)}

    interactions["user_idx"]    = interactions["user_id"].map(user_to_idx).astype(np.int32)
    interactions["product_idx"] = interactions["product_id"].map(item_to_idx).astype(np.int32)
    print(f"  → {len(user_ids):,} users | {len(item_ids):,} produits indexés")

    # Sauvegarder les mappings id_string ↔ id_int
    pd.DataFrame({"user_id_int": range(len(user_ids)), "user_id": user_ids}).to_parquet(
        DATA_DIR / "id_mappings" / "user_id_map.parquet", index=False
    )
    pd.DataFrame({"product_id_int": range(len(item_ids)), "product_id": item_ids}).to_parquet(
        DATA_DIR / "id_mappings" / "product_id_map.parquet", index=False
    )

    # ── Construction de la matrice sparse user × item ────────────────────────
    print("\nConstruction de la matrice sparse …")
    signal = (
        interactions.groupby(["user_idx", "product_idx"])
        .size()
        .reset_index(name="count")
    )
    del interactions  # libère la RAM

    user_item = csr_matrix(
        (signal["count"].values.astype(np.float32),
         (signal["user_idx"].values, signal["product_idx"].values)),
        shape=(len(user_ids), len(item_ids))
    )
    del signal
    print(f"  → Matrice sparse : {user_item.shape}, {user_item.nnz:,} valeurs non-nulles")

    # ── Entraînement ALS avec MLflow ─────────────────────────────────────────
    with mlflow.start_run(run_name="als_default"):
        mlflow.log_param("rank", ALS_RANK)
        mlflow.log_param("iterations", ALS_MAX_ITER)
        mlflow.log_param("regularization", ALS_REG_PARAM)
        mlflow.log_param("min_interactions", MIN_INTERACTIONS)
        mlflow.log_metric("n_users", len(user_ids))
        mlflow.log_metric("n_items", len(item_ids))
        mlflow.log_metric("nnz", user_item.nnz)

        print(f"\nEntraînement ALS (rank={ALS_RANK}, iter={ALS_MAX_ITER}, reg={ALS_REG_PARAM}) …")
        model = implicit.als.AlternatingLeastSquares(
            factors=ALS_RANK,
            iterations=ALS_MAX_ITER,
            regularization=ALS_REG_PARAM,
            use_gpu=False,
        )
        model.fit(user_item)
        print("  ALS entraîné.")

        mlflow.set_tag("model_stage", "Production")

    # ── Extraction et sauvegarde des facteurs latents ────────────────────────
    print("\nExtraction des facteurs latents …")
    factor_cols_P = [f"P{i+1}" for i in range(ALS_RANK)]
    factor_cols_U = [f"U{i+1}" for i in range(ALS_RANK)]

    # Item factors P1-P16
    item_factors_df = pd.DataFrame(model.item_factors, columns=factor_cols_P)
    item_factors_df.insert(0, "product_id", item_ids)
    item_factors_df.to_parquet(DATA_DIR / "als_item_factors" / "item_factors.parquet", index=False)

    # User factors U1-U16
    user_factors_df = pd.DataFrame(model.user_factors, columns=factor_cols_U)
    user_factors_df.insert(0, "user_id", user_ids)
    user_factors_df.to_parquet(DATA_DIR / "als_user_factors" / "user_factors.parquet", index=False)

    print(f"  → Item factors : {len(item_ids):,} produits  (colonnes P1-P{ALS_RANK})")
    print(f"  → User factors : {len(user_ids):,} users     (colonnes U1-U{ALS_RANK})")
    print("\nALS terminé. Lancer maintenant : python 03_cold_start_bridge.py")
