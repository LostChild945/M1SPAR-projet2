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
    python 03_cold_start_bridge.py --skip-train   # réutilise le bridge model existant
"""

import gc
import pickle
import argparse
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

FILTER_CATEGORIES = ["All_Beauty", "Amazon_Fashion"]   # 2 catégories pour les tests rapides

# =============================================================================

(DATA_DIR / "bridge_model").mkdir(parents=True, exist_ok=True)

p_cols = [f"P{i}" for i in range(1, ALS_RANK + 1)]
u_cols = [f"U{i}" for i in range(1, ALS_RANK + 1)]


def load(path) -> pd.DataFrame:
    return pd.read_parquet(path)


def filter_by_category(df, col="category"):
    if FILTER_CATEGORIES is not None:
        return df[df[col].isin(FILTER_CATEGORIES)]
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-train", action="store_true",
                        help="Skip training, load bridge model from disk")
    args = parser.parse_args()

    # =====================================================================
    # PHASE 1 : Classification warm / cold
    #   On ne charge que interactions pour calculer les seuils.
    # =====================================================================
    print("Phase 1 — Classification warm / cold …")
    print("  → interactions …")
    interactions_df = filter_by_category(load(DATA_DIR / "interactions"))

    prod_counts = interactions_df.groupby("product_id").size()
    user_counts = interactions_df.groupby("user_id").size()

    warm_products = set(prod_counts[prod_counts >= MIN_INTERACTIONS].index)
    warm_users    = set(user_counts[user_counts >= MIN_INTERACTIONS].index)
    valid_products = set(interactions_df["product_id"])
    valid_users    = set(interactions_df["user_id"])

    del prod_counts, user_counts, interactions_df
    gc.collect()

    print(f"  warm products : {len(warm_products):,}")
    print(f"  warm users    : {len(warm_users):,}")

    # =====================================================================
    # PHASE 2 : Bridge model (train ou load)
    # =====================================================================
    bridge_path = DATA_DIR / "bridge_model" / "bridge_model.pkl"

    if args.skip_train and bridge_path.exists():
        print("\nPhase 2 — Chargement du bridge model existant …")
        with open(bridge_path, "rb") as f:
            bridge = pickle.load(f)
        # On a besoin de emb_cols — les lire depuis content_embeddings
        content_cols = pd.read_parquet(
            DATA_DIR / "content_embeddings" / "content_embeddings.parquet",
            columns=["product_id"]  # lecture minimale pour les noms de colonnes
        )
        # Récupérer les vrais noms de colonnes
        all_cols = pd.read_parquet(
            DATA_DIR / "content_embeddings" / "content_embeddings.parquet"
        ).columns.tolist()
        emb_cols = [c for c in all_cols if c.startswith("emb_")]
        del content_cols, all_cols
        print("  Bridge model chargé depuis le disque.")
    else:
        print("\nPhase 2 — Entraînement du bridge model …")
        print("  → als_item_factors …")
        als_items_df = load(DATA_DIR / "als_item_factors")
        if FILTER_CATEGORIES is not None:
            als_items_df = als_items_df[als_items_df["product_id"].isin(valid_products)]

        print("  → content_embeddings …")
        content_emb = load(DATA_DIR / "content_embeddings" / "content_embeddings.parquet")
        if FILTER_CATEGORIES is not None:
            content_emb = content_emb[content_emb["product_id"].isin(valid_products)]
        emb_cols = [c for c in content_emb.columns if c.startswith("emb_")]

        warm_als = als_items_df[als_items_df["product_id"].isin(warm_products)]
        train_df = warm_als.merge(content_emb, on="product_id", how="inner")
        del warm_als, als_items_df, content_emb
        gc.collect()

        X_train = train_df[emb_cols].values.astype(np.float32)
        Y_train = train_df[p_cols].values.astype(np.float32)
        print(f"  Exemples d'entraînement : {len(X_train):,}")
        del train_df

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

        rmse = np.sqrt(mean_squared_error(Y_train, bridge.predict(X_train)))
        print(f"  RMSE (train) : {rmse:.6f}")
        del X_train, Y_train
        gc.collect()

        with open(bridge_path, "wb") as f:
            pickle.dump(bridge, f)
        print("  Bridge model sauvegardé.")

    # =====================================================================
    # PHASE 3 : Embeddings finaux produits (P1-P16)
    #   Charge : products, als_item_factors, content_embeddings
    #   Libère tout avant la phase 4.
    # =====================================================================
    print("\nPhase 3 — Construction P1-P16 finaux pour tous les produits …")

    print("  → products …")
    products_df = filter_by_category(load(DATA_DIR / "products"))
    all_product_ids = set(products_df["product_id"])
    cold_products   = all_product_ids - warm_products

    print("  → als_item_factors …")
    als_items_df = load(DATA_DIR / "als_item_factors")
    if FILTER_CATEGORIES is not None:
        als_items_df = als_items_df[als_items_df["product_id"].isin(valid_products)]

    # Merge ALS factors
    products_final = products_df.merge(
        als_items_df[["product_id"] + p_cols],
        on="product_id", how="left", suffixes=("_old", "")
    )
    del products_df, als_items_df
    gc.collect()

    for col in p_cols:
        if f"{col}_old" in products_final.columns:
            products_final.drop(columns=[f"{col}_old"], inplace=True)

    # Cold products → bridge
    cold_mask = products_final[p_cols[0]].isna()
    cold_ids  = products_final.loc[cold_mask, "product_id"].values

    if len(cold_ids) > 0:
        print(f"  → content_embeddings pour {len(cold_ids):,} cold products …")

        # Charger content_embeddings filtré sur les cold ids seulement
        cold_ids_set = set(cold_ids)
        content_emb = load(DATA_DIR / "content_embeddings" / "content_embeddings.parquet")
        if FILTER_CATEGORIES is not None:
            content_emb = content_emb[content_emb["product_id"].isin(valid_products)]
        content_emb = content_emb[content_emb["product_id"].isin(cold_ids_set)]
        gc.collect()

        # Prédire par batches de 200k et stocker dans un dict {product_id → P1..P16}
        BATCH = 200_000
        n_batches = (len(content_emb) + BATCH - 1) // BATCH
        predicted_ids = []
        predicted_vals = []
        for i in range(n_batches):
            batch = content_emb.iloc[i * BATCH : (i + 1) * BATCH]
            X_cold = batch[emb_cols].values.astype(np.float32)
            Y_cold = bridge.predict(X_cold).astype(np.float32)
            predicted_ids.append(batch["product_id"].values)
            predicted_vals.append(Y_cold)
            del X_cold, Y_cold, batch
            gc.collect()
            print(f"    batch {i+1}/{n_batches}")

        del content_emb
        gc.collect()

        # Construire le DataFrame des prédictions et merger
        cold_pred = pd.DataFrame(
            np.concatenate(predicted_vals),
            columns=[f"{c}_bridge" for c in p_cols],
        )
        cold_pred["product_id"] = np.concatenate(predicted_ids)
        del predicted_ids, predicted_vals
        gc.collect()

        products_final = products_final.merge(cold_pred, on="product_id", how="left")
        del cold_pred
        for col in p_cols:
            bridge_col = f"{col}_bridge"
            if bridge_col in products_final.columns:
                products_final[col] = products_final[col].fillna(products_final[bridge_col])
                products_final.drop(columns=[bridge_col], inplace=True)
        gc.collect()
        print(f"  → {len(cold_ids):,} produits cold traités via bridge")

    del bridge
    gc.collect()

    # Vecteur zéro pour les produits sans embedding
    for col in p_cols:
        products_final[col] = products_final[col].fillna(0.0)

    products_final["is_cold"] = products_final["product_id"].isin(cold_products)
    products_final.to_parquet(
        DATA_DIR / "products" / "products_with_embeddings.parquet", index=False
    )
    n_prod = len(products_final)
    warm_p = (~products_final["is_cold"]).sum()
    print(f"  → {n_prod:,} produits sauvegardés")

    # Garder seulement le lookup P1-P16 pour la phase 4
    product_emb_lookup = products_final[["product_id"] + p_cols].copy()
    del products_final
    gc.collect()

    # =====================================================================
    # PHASE 4 : Embeddings finaux utilisateurs (U1-U16)
    #   Charge : users, als_user_factors, interactions
    # =====================================================================
    print("\nPhase 4 — Construction U1-U16 finaux pour tous les utilisateurs …")

    print("  → users …")
    users_df = load(DATA_DIR / "users")
    if FILTER_CATEGORIES is not None:
        users_df = users_df[users_df["user_id"].isin(valid_users)]
    all_user_ids = set(users_df["user_id"])
    cold_users   = all_user_ids - warm_users

    print("  → als_user_factors …")
    als_users_df = load(DATA_DIR / "als_user_factors")
    if FILTER_CATEGORIES is not None:
        als_users_df = als_users_df[als_users_df["user_id"].isin(valid_users)]

    users_final = users_df.merge(
        als_users_df[["user_id"] + u_cols],
        on="user_id", how="left", suffixes=("_old", "")
    )
    del users_df, als_users_df
    gc.collect()

    for col in u_cols:
        if f"{col}_old" in users_final.columns:
            users_final.drop(columns=[f"{col}_old"], inplace=True)

    # Cold users → moyenne des P1-P16 des produits vus
    cold_user_mask = users_final[u_cols[0]].isna()
    cold_user_ids  = users_final.loc[cold_user_mask, "user_id"].values

    if len(cold_user_ids) > 0:
        print(f"  → interactions pour {len(cold_user_ids):,} cold users …")
        interactions_df = filter_by_category(load(DATA_DIR / "interactions"))
        cold_inter = interactions_df[interactions_df["user_id"].isin(cold_user_ids)]
        del interactions_df
        gc.collect()

        cold_inter = cold_inter.merge(product_emb_lookup, on="product_id", how="left")
        rename_map = {f"P{i}": f"U{i}" for i in range(1, ALS_RANK + 1)}
        cold_user_emb = (
            cold_inter.groupby("user_id")[p_cols]
            .mean()
            .reset_index()
            .rename(columns=rename_map)
        )
        del cold_inter
        gc.collect()

        users_final = users_final.merge(
            cold_user_emb, on="user_id", how="left", suffixes=("", "_avg")
        )
        del cold_user_emb
        for col in u_cols:
            avg_col = f"{col}_avg"
            if avg_col in users_final.columns:
                users_final[col] = users_final[col].fillna(users_final[avg_col])
                users_final.drop(columns=[avg_col], inplace=True)
        print(f"  → {len(cold_user_ids):,} cold users traités par moyenne des items")

    del product_emb_lookup
    gc.collect()

    for col in u_cols:
        users_final[col] = users_final[col].fillna(0.0)

    users_final["is_cold"] = users_final["user_id"].isin(cold_users)
    users_final.to_parquet(
        DATA_DIR / "users" / "users_with_embeddings.parquet", index=False
    )
    n_users = len(users_final)
    warm_u  = (~users_final["is_cold"]).sum()
    print(f"  → {n_users:,} utilisateurs sauvegardés")

    # =====================================================================
    # Résumé final
    # =====================================================================
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
