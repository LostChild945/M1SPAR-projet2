"""
03_cold_start_bridge.py
------------------------
Stratégie production : combine ALS (warm) + bridge model (cold).

                    ┌──────────────────────────────────────┐
                    │      Embeddings finaux (P1-P16)      │
                    └──────────────────────────────────────┘
                                    │
               ┌────────────────────┼────────────────────┐
               ▼                    ▼                    ▼
        Warm Product          Cold Product           Cold User
      (≥ N interactions)    (< N interactions)    (jamais vu)
      P1-P16 = ALS facts    P1-P16 = Bridge(ST)   U1-U16 = mean(P1-P16
                                                       des produits vus)

Bridge model = MLP qui apprend à projeter
  ST_embedding (384 dims) → ALS_factor (16 dims)
Entraîné sur les produits warm : on dispose des deux représentations.

Output :
  data/products/products_with_embeddings.parquet   (P1-P16 + is_cold)
  data/users/users_with_embeddings.parquet          (U1-U16 + is_cold)
  data/bridge_model/bridge_model.pkl

Usage :
    python 03_cold_start_bridge.py
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error

DATA_DIR         = Path("./data")
MIN_INTERACTIONS = 5    # seuil warm vs cold
ALS_RANK         = 16

# =============================================================================
# FILTRE DE CATÉGORIES — optionnel
# -----------------------------------------------------------------------------
# Doit être identique au filtre utilisé dans 01 et 02, ou None pour tout.
# =============================================================================

TEST_5 = ["All_Beauty", "Amazon_Fashion", "Appliances", "Arts_Crafts_and_Sewing", "Automotive"]

FILTER_CATEGORIES = TEST_5   # ← mettre None pour toutes les catégories

# =============================================================================

(DATA_DIR / "bridge_model").mkdir(parents=True, exist_ok=True)


def load(path) -> pd.DataFrame:
    return pd.read_parquet(path)


if __name__ == "__main__":
    # ── Chargement ────────────────────────────────────────────────────────
    print("Chargement des données …")
    products     = load(DATA_DIR / "products")
    users        = load(DATA_DIR / "users")
    interactions = load(DATA_DIR / "interactions")
    als_items    = load(DATA_DIR / "als_item_factors")
    als_users    = load(DATA_DIR / "als_user_factors")
    content_emb  = load(DATA_DIR / "content_embeddings" / "content_embeddings.parquet")

    if FILTER_CATEGORIES is not None:
        print(f"  → Filtre actif : {FILTER_CATEGORIES}")
        products     = products[products["category"].isin(FILTER_CATEGORIES)]
        interactions = interactions[interactions["category"].isin(FILTER_CATEGORIES)]
        # Garder seulement les produits/users présents dans les catégories filtrées
        valid_products = set(products["product_id"])
        valid_users    = set(interactions["user_id"])
        als_items      = als_items[als_items["product_id"].isin(valid_products)]
        als_users      = als_users[als_users["user_id"].isin(valid_users)]
        content_emb    = content_emb[content_emb["product_id"].isin(valid_products)]
        users          = users[users["user_id"].isin(valid_users)]

    p_cols   = [f"P{i}" for i in range(1, ALS_RANK + 1)]
    u_cols   = [f"U{i}" for i in range(1, ALS_RANK + 1)]
    emb_cols = [c for c in content_emb.columns if c.startswith("emb_")]

    # ── Classifier warm / cold ────────────────────────────────────────────
    print("\nClassification warm / cold …")
    prod_counts = interactions.groupby("product_id").size()
    user_counts = interactions.groupby("user_id").size()

    warm_products = set(prod_counts[prod_counts >= MIN_INTERACTIONS].index)
    warm_users    = set(user_counts[user_counts >= MIN_INTERACTIONS].index)

    all_product_ids = set(products["product_id"])
    all_user_ids    = set(users["user_id"])
    cold_products   = all_product_ids - warm_products
    cold_users      = all_user_ids    - warm_users

    print(f"  Produits  warm : {len(warm_products):>8,}  ({len(warm_products)/len(all_product_ids)*100:.1f}%)")
    print(f"  Produits  cold : {len(cold_products):>8,}  ({len(cold_products)/len(all_product_ids)*100:.1f}%)")
    print(f"  Users     warm : {len(warm_users):>8,}  ({len(warm_users)/len(all_user_ids)*100:.1f}%)")
    print(f"  Users     cold : {len(cold_users):>8,}  ({len(cold_users)/len(all_user_ids)*100:.1f}%)")

    # ── Bridge model : ST(384) → ALS(16) ─────────────────────────────────
    print("\nEntraînement du bridge model …")
    # Données d'entraînement = produits warm avec ALS factor ET content embedding
    warm_als = als_items[als_items["product_id"].isin(warm_products)]
    train_df = warm_als.merge(content_emb, on="product_id", how="inner")

    X_train = train_df[emb_cols].values.astype(np.float32)
    Y_train = train_df[p_cols].values.astype(np.float32)
    print(f"  Exemples d'entraînement : {len(X_train):,}")

    bridge = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPRegressor(
            hidden_layer_sizes=(256, 64),
            activation="relu",
            max_iter=300,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=15,
            random_state=42,
            verbose=True,
        )),
    ])
    bridge.fit(X_train, Y_train)

    Y_pred = bridge.predict(X_train)
    rmse = np.sqrt(mean_squared_error(Y_train, Y_pred))
    print(f"  RMSE (train) : {rmse:.6f}")

    with open(DATA_DIR / "bridge_model" / "bridge_model.pkl", "wb") as f:
        pickle.dump(bridge, f)
    print("  Bridge model sauvegardé.")

    # ── Embeddings finaux produits ─────────────────────────────────────────
    print("\nConstruction P1-P16 finaux pour tous les produits …")

    # 1. Base = produits avec facteurs ALS (warm)
    products_final = products.merge(
        als_items[["product_id"] + p_cols],
        on="product_id", how="left", suffixes=("_old", "")
    )
    # Supprimer les vieilles colonnes placeholder
    for col in p_cols:
        if f"{col}_old" in products_final.columns:
            products_final.drop(columns=[f"{col}_old"], inplace=True)

    # 2. Cold products → prédire via bridge
    cold_mask = products_final[p_cols[0]].isna()
    cold_ids  = products_final.loc[cold_mask, "product_id"].values

    if len(cold_ids) > 0:
        cold_content = content_emb[content_emb["product_id"].isin(cold_ids)]
        if not cold_content.empty:
            X_cold = cold_content[emb_cols].values.astype(np.float32)
            Y_cold = bridge.predict(X_cold).astype(np.float32)
            cold_df = pd.DataFrame(Y_cold, columns=p_cols)
            cold_df["product_id"] = cold_content["product_id"].values
            products_final = products_final.merge(
                cold_df, on="product_id", how="left", suffixes=("", "_bridge")
            )
            for col in p_cols:
                bridge_col = f"{col}_bridge"
                if bridge_col in products_final.columns:
                    products_final[col] = products_final[col].fillna(products_final[bridge_col])
                    products_final.drop(columns=[bridge_col], inplace=True)
        print(f"  → {len(cold_ids):,} produits cold traités via bridge")

    # 3. Produits sans aucun embedding (ni ALS ni content) → vecteur zéro
    for col in p_cols:
        products_final[col] = products_final[col].fillna(0.0)

    products_final["is_cold"] = products_final["product_id"].isin(cold_products)
    products_final.to_parquet(
        DATA_DIR / "products" / "products_with_embeddings.parquet", index=False
    )
    print(f"  → {len(products_final):,} produits sauvegardés")

    # ── Embeddings finaux utilisateurs ────────────────────────────────────
    print("\nConstruction U1-U16 finaux pour tous les utilisateurs …")

    # 1. Warm users → facteurs ALS
    users_final = users.merge(
        als_users[["user_id"] + u_cols],
        on="user_id", how="left", suffixes=("_old", "")
    )
    for col in u_cols:
        if f"{col}_old" in users_final.columns:
            users_final.drop(columns=[f"{col}_old"], inplace=True)

    # 2. Cold users → moyenne des P1-P16 des produits avec lesquels ils ont interagi
    cold_user_mask = users_final[u_cols[0]].isna()
    cold_user_ids  = users_final.loc[cold_user_mask, "user_id"].values

    if len(cold_user_ids) > 0:
        # Récupérer les interactions des cold users
        cold_inter = interactions[interactions["user_id"].isin(cold_user_ids)]
        cold_inter = cold_inter.merge(
            products_final[["product_id"] + p_cols],
            on="product_id", how="left"
        )
        # Moyenne des embeddings produits par user
        rename_map = {f"P{i}": f"U{i}" for i in range(1, ALS_RANK + 1)}
        cold_user_emb = (
            cold_inter.groupby("user_id")[p_cols]
            .mean()
            .reset_index()
            .rename(columns=rename_map)
        )
        users_final = users_final.merge(
            cold_user_emb, on="user_id", how="left", suffixes=("", "_avg")
        )
        for col in u_cols:
            avg_col = f"{col}_avg"
            if avg_col in users_final.columns:
                users_final[col] = users_final[col].fillna(users_final[avg_col])
                users_final.drop(columns=[avg_col], inplace=True)
        print(f"  → {len(cold_user_ids):,} cold users traités par moyenne des items")

    # 3. Users sans aucune info → vecteur zéro
    for col in u_cols:
        users_final[col] = users_final[col].fillna(0.0)

    users_final["is_cold"] = users_final["user_id"].isin(cold_users)
    users_final.to_parquet(
        DATA_DIR / "users" / "users_with_embeddings.parquet", index=False
    )
    print(f"  → {len(users_final):,} utilisateurs sauvegardés")

    # ── Résumé final ──────────────────────────────────────────────────────
    n_prod  = len(products_final)
    n_users = len(users_final)
    warm_p  = (~products_final["is_cold"]).sum()
    warm_u  = (~users_final["is_cold"]).sum()

    print("\n" + "="*50)
    print("RÉSUMÉ DES EMBEDDINGS PRODUITS")
    print("="*50)
    print(f"  Warm (ALS)      : {warm_p:>8,}  ({warm_p/n_prod*100:.1f}%)")
    print(f"  Cold (bridge)   : {n_prod-warm_p:>8,}  ({(n_prod-warm_p)/n_prod*100:.1f}%)")
    print(f"  Total           : {n_prod:>8,}")
    print()
    print("RÉSUMÉ DES EMBEDDINGS UTILISATEURS")
    print("="*50)
    print(f"  Warm (ALS)      : {warm_u:>8,}  ({warm_u/n_users*100:.1f}%)")
    print(f"  Cold (avg items): {n_users-warm_u:>8,}  ({(n_users-warm_u)/n_users*100:.1f}%)")
    print(f"  Total           : {n_users:>8,}")
    print()
    print("Pipeline terminé. Les embeddings sont prêts pour le modèle de reco.")
    print(f"  → data/products/products_with_embeddings.parquet")
    print(f"  → data/users/users_with_embeddings.parquet")
    print(f"  → data/bridge_model/bridge_model.pkl")
