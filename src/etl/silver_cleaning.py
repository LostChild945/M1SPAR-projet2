"""
silver_cleaning.py
------------------
Couche Silver : nettoyage, déduplication et feature engineering
sur les données Bronze (interactions).

Transformations :
  - Suppression des lignes avec user_id ou product_id null
  - Clamping des ratings entre 0 et 5
  - Déduplication sur (user_id, product_id, timestamp)
  - Window functions :
      recency_rank   — rang par user trié par timestamp desc (1 = plus récent)
      rf_score       — score récence-fréquence (interaction_count / recency_rank)

Output :
  data/silver/interactions/   (Parquet partitionné par category)

Usage :
    python -m src.etl.silver_cleaning
"""

import os

from pyspark.sql import SparkSession, functions as F
from pyspark.sql.window import Window

BASE = os.environ.get("DATA_DIR", "./analyse/data")


def create_spark():
    return (
        SparkSession.builder
        .appName("M1SPAR-P2-Silver")
        .config("spark.driver.memory", "4g")
        .config("spark.sql.shuffle.partitions", "20")
        .config("spark.ui.enabled", "true")
        .config("spark.ui.port", "4040")
        .getOrCreate()
    )


def run_silver(spark=None, base=BASE):
    """Nettoie et enrichit les interactions Bronze → Silver."""
    own_spark = spark is None
    if own_spark:
        spark = create_spark()

    bronze_path = os.path.join(base, "interactions")
    silver_path = os.path.join(base, "silver", "interactions")

    print("=" * 60)
    print("SILVER — Nettoyage + Feature Engineering")
    print("=" * 60)

    # ── Chargement Bronze ────────────────────────────────────────────
    df = spark.read.parquet(bronze_path)
    n_before = df.count()
    print(f"  Bronze chargé : {n_before:,} lignes")

    # ── Nulls ────────────────────────────────────────────────────────
    df = df.filter(F.col("user_id").isNotNull() & F.col("product_id").isNotNull())

    # ── Rating clamp [0, 5] ──────────────────────────────────────────
    df = df.withColumn(
        "rating",
        F.when(F.col("rating") < 0, 0)
        .when(F.col("rating") > 5, 5)
        .otherwise(F.col("rating")),
    )

    # ── Déduplication ────────────────────────────────────────────────
    df = df.dropDuplicates(["user_id", "product_id", "timestamp"])

    # ── Window functions ─────────────────────────────────────────────
    w_recency = Window.partitionBy("user_id").orderBy(F.col("timestamp").desc())
    w_count = Window.partitionBy("user_id")

    df = df.withColumn("recency_rank", F.row_number().over(w_recency))
    df = df.withColumn("user_interaction_count", F.count("*").over(w_count))
    df = df.withColumn(
        "rf_score",
        F.col("user_interaction_count").cast("double")
        / F.col("recency_rank").cast("double"),
    )

    n_after = df.count()
    print(f"  Silver produit : {n_after:,} lignes (supprimé {n_before - n_after:,})")

    # ── Écriture ─────────────────────────────────────────────────────
    df.write.mode("overwrite").partitionBy("category").parquet(silver_path)
    print(f"  Sauvegardé : {silver_path}")

    if own_spark:
        spark.stop()

    print("\nSilver terminé.\n")
    return df


if __name__ == "__main__":
    run_silver()
