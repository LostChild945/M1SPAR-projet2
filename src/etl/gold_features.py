"""
gold_features.py
----------------
Couche Gold : features agrégées prêtes pour le ML et le serving.

Features produit :
  - conversion_rate   = sum(converted) / count(*) par produit
  - popularity_score  = log(1 + nb_interactions) normalisé [0, 1]

Features utilisateur :
  - cold start tagging : is_cold = True si < MIN_INTERACTIONS interactions

Utilise broadcast join pour les produits (table petite ~500K lignes).

Output :
  data/gold/interactions/   (Parquet enrichi)
  data/gold/products/       (avec conversion_rate, popularity_score)

Usage :
    python -m src.etl.gold_features
"""

import os

from pyspark.sql import SparkSession, functions as F

BASE = os.environ.get("DATA_DIR", "./analyse/data")
MIN_INTERACTIONS = 5


def create_spark():
    return (
        SparkSession.builder
        .appName("M1SPAR-P2-Gold")
        .config("spark.driver.memory", "4g")
        .config("spark.sql.shuffle.partitions", "20")
        .config("spark.sql.autoBroadcastJoinThreshold", str(100 * 1024 * 1024))
        .config("spark.ui.enabled", "true")
        .config("spark.ui.port", "4040")
        .getOrCreate()
    )


def run_gold(spark=None, base=BASE):
    """Construit les features Gold depuis Silver."""
    own_spark = spark is None
    if own_spark:
        spark = create_spark()

    silver_path = os.path.join(base, "silver", "interactions")
    products_path = os.path.join(base, "products")
    gold_inter_path = os.path.join(base, "gold", "interactions")
    gold_prod_path = os.path.join(base, "gold", "products")

    print("=" * 60)
    print("GOLD — Features agrégées")
    print("=" * 60)

    # ── Chargement Silver ────────────────────────────────────────────
    silver = spark.read.parquet(silver_path)
    products = spark.read.parquet(products_path)
    print(f"  Silver : {silver.count():,} interactions")
    print(f"  Products : {products.count():,} produits")

    # ── Features produit ─────────────────────────────────────────────
    product_stats = silver.groupBy("product_id").agg(
        F.count("*").alias("nb_interactions"),
        F.sum("converted").alias("nb_conversions"),
    )
    product_stats = product_stats.withColumn(
        "conversion_rate",
        F.col("nb_conversions").cast("double") / F.col("nb_interactions").cast("double"),
    )

    # popularity_score = log(1 + nb_interactions), normalisé [0, 1]
    max_log = product_stats.agg(
        F.max(F.log1p(F.col("nb_interactions").cast("double")))
    ).collect()[0][0]

    product_stats = product_stats.withColumn(
        "popularity_score",
        F.log1p(F.col("nb_interactions").cast("double")) / F.lit(max_log),
    )

    # ── Broadcast join avec products ─────────────────────────────────
    gold_products = F.broadcast(products).join(
        product_stats.select("product_id", "conversion_rate", "popularity_score", "nb_interactions"),
        on="product_id",
        how="left",
    )
    gold_products = gold_products.fillna({"conversion_rate": 0.0, "popularity_score": 0.0, "nb_interactions": 0})

    # Cold start tagging
    gold_products = gold_products.withColumn(
        "is_cold",
        F.col("nb_interactions") < MIN_INTERACTIONS,
    )

    gold_products.write.mode("overwrite").parquet(gold_prod_path)
    print(f"  Gold produits sauvegardés : {gold_prod_path}")

    # ── Enrichir interactions Silver avec features produit ────────────
    gold_interactions = silver.join(
        F.broadcast(product_stats.select("product_id", "conversion_rate", "popularity_score")),
        on="product_id",
        how="left",
    )
    gold_interactions = gold_interactions.fillna({"conversion_rate": 0.0, "popularity_score": 0.0})

    gold_interactions.write.mode("overwrite").partitionBy("category").parquet(gold_inter_path)
    print(f"  Gold interactions sauvegardées : {gold_inter_path}")

    if own_spark:
        spark.stop()

    print("\nGold terminé.\n")


if __name__ == "__main__":
    run_gold()
