"""
02_train_als.py
---------------
Entraîne ALS (Alternating Least Squares) sur les interactions Amazon
et extrait les facteurs latents :

  P1-P16  →  item factors  (embeddings comportementaux des produits)
  U1-U16  →  user factors  (embeddings comportementaux des utilisateurs)

Ces facteurs encodent les préférences implicites apprises depuis
les interactions réelles — ils sont la base du cold start bridge.

Output :
  data/als_item_factors/    product_id | P1 … P16
  data/als_user_factors/    user_id    | U1 … U16
  data/id_mappings/         user_id_map.parquet | product_id_map.parquet

Usage :
    python 02_train_als.py
"""

import pandas as pd
from pathlib import Path
from pyspark.sql import SparkSession, functions as F
from pyspark.ml.feature import StringIndexer
from pyspark.ml.recommendation import ALS

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

FILTER_CATEGORIES = TEST_5   # ← mettre None pour toutes les catégories

# =============================================================================

# ── Hyperparamètres ALS ─────────────────────────────────────────────────────
ALS_RANK         = 16      # dimensions des embeddings → P1-P16 / U1-U16
ALS_MAX_ITER     = 15      # itérations d'optimisation
ALS_REG_PARAM    = 0.1     # régularisation L2 (évite le sur-apprentissage)
ALS_IMPLICIT     = True    # True = implicit feedback (nb interactions), False = ratings
MIN_INTERACTIONS = 5       # nb minimum d'interactions pour être considéré "warm"

for sub in ("als_item_factors", "als_user_factors", "id_mappings"):
    (DATA_DIR / sub).mkdir(parents=True, exist_ok=True)

spark = (
    SparkSession.builder
    .appName("M1SPAR-ALS-Training")
    .config("spark.driver.memory", "6g")
    .config("spark.sql.shuffle.partitions", "100")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("ERROR")


if __name__ == "__main__":
    print("Chargement des interactions …")
    interactions = spark.read.parquet(str(DATA_DIR / "interactions"))
    if FILTER_CATEGORIES is not None:
        interactions = interactions.filter(F.col("category").isin(FILTER_CATEGORIES))
        print(f"  → Filtre actif : {FILTER_CATEGORIES}")
    n_inter = interactions.count()
    print(f"  → {n_inter:,} interactions")

    # ── Indexation string → int (ALS requiert des entiers) ──────────────────
    print("\nIndexation user_id et product_id …")
    user_indexer = StringIndexer(inputCol="user_id",    outputCol="user_id_int")
    item_indexer = StringIndexer(inputCol="product_id", outputCol="product_id_int")

    ui_model = user_indexer.fit(interactions)
    ii_model = item_indexer.fit(interactions)

    interactions = ui_model.transform(interactions)
    interactions = ii_model.transform(interactions)

    interactions = interactions.withColumn("user_id_int",    F.col("user_id_int").cast("int"))
    interactions = interactions.withColumn("product_id_int", F.col("product_id_int").cast("int"))

    # Sauvegarder les mappings id_string ↔ id_int (nécessaires pour la suite)
    user_map = spark.createDataFrame(
        list(enumerate(ui_model.labels)), ["user_id_int", "user_id"]
    )
    item_map = spark.createDataFrame(
        list(enumerate(ii_model.labels)), ["product_id_int", "product_id"]
    )
    user_map.write.mode("overwrite").parquet(str(DATA_DIR / "id_mappings" / "user_id_map.parquet"))
    item_map.write.mode("overwrite").parquet(str(DATA_DIR / "id_mappings" / "product_id_map.parquet"))
    print(f"  → {len(ui_model.labels):,} users | {len(ii_model.labels):,} produits indexés")

    # ── Construction du signal d'entraînement ───────────────────────────────
    if ALS_IMPLICIT:
        # Signal implicite : compter les interactions par paire (confiance)
        train_df = (
            interactions
            .groupBy("user_id_int", "product_id_int")
            .agg(F.count("*").alias("signal"))
        )
        rating_col = "signal"
        print(f"\nMode implicit feedback — signal = nb d'interactions par paire")
    else:
        train_df = interactions.select(
            "user_id_int", "product_id_int",
            F.col("rating").alias("signal")
        ).dropna()
        rating_col = "signal"
        print(f"\nMode explicit feedback — signal = rating")

    # ── Entraînement ALS ────────────────────────────────────────────────────
    print(f"Entraînement ALS (rank={ALS_RANK}, iter={ALS_MAX_ITER}, reg={ALS_REG_PARAM}) …")
    als = ALS(
        rank=ALS_RANK,
        maxIter=ALS_MAX_ITER,
        regParam=ALS_REG_PARAM,
        implicitPrefs=ALS_IMPLICIT,
        userCol="user_id_int",
        itemCol="product_id_int",
        ratingCol=rating_col,
        coldStartStrategy="drop",
        seed=42,
    )
    model = als.fit(train_df)
    print("  ALS entraîné.")

    # ── Extraction et sauvegarde des facteurs latents ───────────────────────
    print("\nExtraction des facteurs latents …")

    # Item factors P1-P16
    item_factors = model.itemFactors.withColumnRenamed("id", "product_id_int")
    item_factors = item_factors.join(item_map, on="product_id_int", how="left")
    for i in range(ALS_RANK):
        item_factors = item_factors.withColumn(f"P{i+1}", F.col("features")[i])
    item_factors = item_factors.drop("features")
    item_factors.write.mode("overwrite").parquet(str(DATA_DIR / "als_item_factors"))

    # User factors U1-U16
    user_factors = model.userFactors.withColumnRenamed("id", "user_id_int")
    user_factors = user_factors.join(user_map, on="user_id_int", how="left")
    for i in range(ALS_RANK):
        user_factors = user_factors.withColumn(f"U{i+1}", F.col("features")[i])
    user_factors = user_factors.drop("features")
    user_factors.write.mode("overwrite").parquet(str(DATA_DIR / "als_user_factors"))

    n_items = item_factors.count()
    n_users = user_factors.count()
    print(f"  → Item factors : {n_items:,} produits  (colonnes P1-P{ALS_RANK})")
    print(f"  → User factors : {n_users:,} users     (colonnes U1-U{ALS_RANK})")

    spark.stop()
    print("\nALS terminé. Lancer maintenant : python 03_cold_start_bridge.py")
