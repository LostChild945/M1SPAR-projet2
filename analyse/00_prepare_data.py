"""
00_prepare_data.py
------------------
Convertit les fichiers Arrow téléchargés (dataset/raw/) en 3 tables Parquet
prêtes pour PySpark :

  data/interactions/   user_id, product_id, rating, timestamp,
                       interaction_type, converted, category, year_month
  data/products/       product_id, category, title, price, P1-P16
  data/users/          user_id, segment, total_purchases, avg_order_value, U1-U16

Stratégie mémoire : chaque catégorie est écrite sur disque immédiatement
après chargement — jamais plus d'une catégorie en RAM à la fois.

Traite uniquement les catégories déjà téléchargées.
Re-exécutable : les catégories déjà traitées sont skippées.

Usage :
    python 00_prepare_data.py
"""

import gc
import os
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from datasets import load_from_disk

RAW_REVIEWS = Path(os.environ.get("RAW_REVIEWS", "../dataset/raw/2023/reviews"))
RAW_META    = Path(os.environ.get("RAW_META",    "../dataset/raw/2023/metadata"))

OUT_DIR = Path("./data")

(OUT_DIR / "interactions").mkdir(parents=True, exist_ok=True)
(OUT_DIR / "products").mkdir(parents=True, exist_ok=True)
(OUT_DIR / "users").mkdir(parents=True, exist_ok=True)

KEEP_INTERACTIONS = ["user_id", "product_id", "rating", "timestamp",
                     "interaction_type", "converted", "category",
                     "year_month", "helpful_vote"]

KEEP_PRODUCTS = ["product_id", "category", "title", "price",
                 "average_rating", "rating_number", "store"]


def available_categories(base: Path) -> list[str]:
    if not base.exists():
        return []
    return sorted(d.name for d in base.iterdir() if d.is_dir())


def already_done(cat: str, table: str) -> bool:
    """Vérifie si une catégorie a déjà été convertie."""
    path = OUT_DIR / table / f"category={cat}"
    return path.exists()


def process_interactions_category(cat: str) -> int:
    """
    Charge une catégorie, la transforme et l'écrit sur disque.
    Retourne le nombre de lignes écrites (0 si skip/erreur).
    """
    if already_done(cat, "interactions"):
        print(f"    {cat} … [SKIP]")
        return 0

    path = RAW_REVIEWS / cat
    if not path.exists():
        return 0

    try:
        ds = load_from_disk(str(path))
        split = ds["full"] if "full" in ds else next(iter(ds.values()))
        df = split.to_pandas()
        del ds, split
        gc.collect()

        df = df.rename(columns={"asin": "product_id"})
        df["category"] = cat

        verified = df.get("verified_purchase", pd.Series(False, index=df.index))
        df["interaction_type"] = verified.map({True: "purchase", False: "review"})
        df["converted"] = (df["interaction_type"] == "purchase").astype(int)

        ts = pd.to_datetime(df["timestamp"], unit="ms", errors="coerce")
        df["year_month"] = ts.dt.to_period("M").astype(str)

        cols = [c for c in KEEP_INTERACTIONS if c in df.columns]
        df = df[cols]

        # Écriture Parquet partitionné par category (une partition = un dossier)
        table = pa.Table.from_pandas(df, preserve_index=False)
        pq.write_to_dataset(
            table,
            root_path=str(OUT_DIR / "interactions"),
            partition_cols=["category"],
        )
        n = len(df)
        print(f"    {cat} … {n:,} lignes")

        del df, table
        gc.collect()
        return n

    except Exception as e:
        print(f"    {cat} … ERREUR : {e}")
        return 0


def process_products_category(cat: str) -> int:
    """Charge les métadonnées d'une catégorie et les écrit sur disque."""
    out_path = OUT_DIR / "products" / f"{cat}.parquet"
    if out_path.exists():
        print(f"    {cat} … [SKIP]")
        return 0

    path = RAW_META / cat
    if not path.exists():
        return 0

    try:
        ds = load_from_disk(str(path))
        if hasattr(ds, "to_pandas"):
            df = ds.to_pandas()
        else:
            split = ds["full"] if "full" in ds else next(iter(ds.values()))
            df = split.to_pandas()
        del ds
        gc.collect()

        col_id = "parent_asin" if "parent_asin" in df.columns else "asin"
        df = df.rename(columns={col_id: "product_id"})
        df["category"] = cat

        cols = [c for c in KEEP_PRODUCTS if c in df.columns]
        df = df[cols].drop_duplicates("product_id")

        for i in range(1, 17):
            df[f"P{i}"] = 0.0

        df.to_parquet(out_path, index=False)
        n = len(df)
        print(f"    {cat} … {n:,} produits")

        del df
        gc.collect()
        return n

    except Exception as e:
        print(f"    {cat} … ERREUR : {e}")
        return 0


def build_users_from_parquet() -> int:
    """
    Construit la table users en lisant les interactions déjà écrites sur disque
    catégorie par catégorie — jamais tout en RAM.
    """
    out_path = OUT_DIR / "users" / "users.parquet"
    interactions_dir = OUT_DIR / "interactions"

    if not interactions_dir.exists():
        print("    Aucune interaction trouvée.")
        return 0

    print("    Agrégation par catégorie …")
    agg_parts = []

    for cat_dir in sorted(interactions_dir.iterdir()):
        if not cat_dir.is_dir():
            continue
        parquet_files = list(cat_dir.glob("*.parquet"))
        if not parquet_files:
            continue

        df = pd.read_parquet(cat_dir, columns=["user_id", "product_id", "converted", "rating"])
        part = (
            df.groupby("user_id")
            .agg(
                total_interactions=("product_id", "count"),
                total_purchases=("converted", "sum"),
                avg_rating=("rating", "mean"),
            )
            .reset_index()
        )
        agg_parts.append(part)
        del df, part
        gc.collect()

    if not agg_parts:
        return 0

    print("    Fusion des agrégations …")
    combined = pd.concat(agg_parts, ignore_index=True)
    del agg_parts
    gc.collect()

    users = (
        combined.groupby("user_id")
        .agg(
            total_interactions=("total_interactions", "sum"),
            total_purchases=("total_purchases", "sum"),
            avg_rating=("avg_rating", "mean"),
        )
        .reset_index()
    )
    del combined
    gc.collect()

    p90 = users["total_purchases"].quantile(0.90)
    p50 = users["total_purchases"].quantile(0.50)

    def segment(x):
        if x >= p90:   return "power_user"
        if x >= p50:   return "regular_user"
        return "casual_user"

    users["segment"]         = users["total_purchases"].apply(segment)
    users["avg_order_value"] = 0.0
    for i in range(1, 17):
        users[f"U{i}"] = 0.0

    users.to_parquet(out_path, index=False)
    n = len(users)
    print(f"    {n:,} utilisateurs sauvegardés")
    return n


if __name__ == "__main__":
    all_reviews = available_categories(RAW_REVIEWS)
    all_meta    = available_categories(RAW_META)

    # Intersection reviews ∩ metadata — seules les catégories disponibles des deux côtés
    common_cats  = sorted(set(all_reviews) & set(all_meta))[:4]
    cats_reviews = common_cats
    cats_meta    = common_cats

    print(f"Reviews disponibles  : {all_reviews}")
    print(f"Metadata disponibles : {all_meta}")
    print(f"Catégories communes  : {common_cats}\n")

    # ── 1. Interactions ──────────────────────────────────────────────────
    print("[1/3] Construction table interactions (une catégorie à la fois) …")
    total_inter = 0
    for cat in cats_reviews:
        total_inter += process_interactions_category(cat)
    print(f"  → {total_inter:,} nouvelles interactions écrites\n")

    # ── 2. Products ──────────────────────────────────────────────────────
    print("[2/3] Construction table products …")
    total_prod = 0
    for cat in cats_meta:
        total_prod += process_products_category(cat)
    print(f"  → {total_prod:,} nouveaux produits écrits\n")

    # ── 3. Users ─────────────────────────────────────────────────────────
    print("[3/3] Construction table users (agrégation depuis disque) …")
    build_users_from_parquet()

    print("\nPréparation terminée.")
    print("Lancez maintenant : jupyter notebook notebooks/01_eda_reco.ipynb")
