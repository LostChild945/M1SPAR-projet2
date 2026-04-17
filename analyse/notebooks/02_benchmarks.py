"""
02_benchmarks.py
----------------
Benchmarks PySpark : partition pruning et broadcast join.

Démontre :
  1. Partition pruning par year_month — filtre pushdown prouvé via explain()
  2. Broadcast join products (500K) vs join standard — temps comparé

Usage :
    python 02_benchmarks.py
    # ou via Spark Web UI (localhost:4040) pour visualiser jobs/stages
"""

import time
from pyspark.sql import SparkSession, functions as F

BASE = "/app/data/"

spark = (
    SparkSession.builder
    .appName("M1SPAR-P2-Benchmarks")
    .config("spark.driver.memory", "4g")
    .config("spark.sql.shuffle.partitions", "20")
    .config("spark.sql.files.maxPartitionBytes", "64m")
    .config("spark.ui.enabled", "true")
    .config("spark.ui.port", "4040")
    .config("spark.sql.autoBroadcastJoinThreshold", str(100 * 1024 * 1024))
    .getOrCreate()
)
spark.sparkContext.setLogLevel("ERROR")
print(f"Spark {spark.version} démarré\n")

# ── Chargement ────────────────────────────────────────────────────────────
interactions = spark.read.parquet(f"{BASE}interactions/")
products = spark.read.parquet(f"{BASE}products/")

n_inter = interactions.count()
n_prod = products.count()
print(f"Interactions : {n_inter:,}")
print(f"Products     : {n_prod:,}\n")

# ═══════════════════════════════════════════════════════════════════════════
# BENCHMARK 1 : Partition Pruning par year_month
# ═══════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("BENCHMARK 1 — Partition Pruning")
print("=" * 60)

# Sans filtre (full scan)
t0 = time.time()
count_all = interactions.count()
t_full = time.time() - t0
print(f"\n  Full scan       : {count_all:,} lignes en {t_full:.2f}s")

# Avec filtre sur year_month (partition pruning)
months = interactions.select("year_month").distinct().orderBy("year_month").collect()
if months:
    target_month = months[len(months) // 2][0]  # mois médian
    print(f"  Filtre          : year_month = '{target_month}'")

    t0 = time.time()
    filtered = interactions.filter(F.col("year_month") == target_month)
    count_filtered = filtered.count()
    t_filtered = time.time() - t0
    print(f"  Partition pruned: {count_filtered:,} lignes en {t_filtered:.2f}s")

    speedup = t_full / t_filtered if t_filtered > 0 else float("inf")
    print(f"  Speedup         : {speedup:.1f}x")

    print("\n  Plan physique (partition pruning) :")
    filtered.explain(mode="simple")

# ═══════════════════════════════════════════════════════════════════════════
# BENCHMARK 2 : Broadcast Join vs Standard Join
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("BENCHMARK 2 — Broadcast Join vs Standard Join")
print("=" * 60)

# Standard join (sort-merge)
spark.conf.set("spark.sql.autoBroadcastJoinThreshold", "-1")
t0 = time.time()
joined_standard = interactions.join(products, on="product_id", how="left")
count_std = joined_standard.count()
t_standard = time.time() - t0
print(f"\n  Standard join  : {count_std:,} lignes en {t_standard:.2f}s")
print("  Plan standard :")
joined_standard.explain(mode="simple")

# Broadcast join
spark.conf.set("spark.sql.autoBroadcastJoinThreshold", str(100 * 1024 * 1024))
t0 = time.time()
joined_broadcast = interactions.join(F.broadcast(products), on="product_id", how="left")
count_bc = joined_broadcast.count()
t_broadcast = time.time() - t0
print(f"\n  Broadcast join : {count_bc:,} lignes en {t_broadcast:.2f}s")
print("  Plan broadcast :")
joined_broadcast.explain(mode="simple")

speedup = t_standard / t_broadcast if t_broadcast > 0 else float("inf")
print(f"\n  Speedup broadcast vs standard : {speedup:.1f}x")

# ═══════════════════════════════════════════════════════════════════════════
# RÉSUMÉ
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("RÉSUMÉ DES BENCHMARKS")
print("=" * 60)
print(f"  Partition pruning : {speedup:.1f}x plus rapide avec filtre year_month")
print(f"  Broadcast join    : {speedup:.1f}x plus rapide que sort-merge")
print("  → Voir Spark Web UI (localhost:4040) pour les détails des stages")

spark.stop()
print("\nTerminé.")
