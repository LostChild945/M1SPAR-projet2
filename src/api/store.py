"""
Chargement des données au démarrage.
Tout est stocké dans l'instance globale `state` utilisée par les autres modules.
"""
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(os.environ.get("DATA_DIR", "../../analyse/data"))

P_COLS = [f"P{i}" for i in range(1, 17)]
U_COLS = [f"U{i}" for i in range(1, 17)]


class AppState:
    # Produits
    product_ids: list[str] = []
    product_index: dict[str, int] = {}       # product_id → ligne dans la matrice
    product_categories: list[str] = []
    product_is_cold: np.ndarray = None
    product_matrix: np.ndarray = None        # (n_products, 16) brut  — pour dot product
    product_matrix_norm: np.ndarray = None   # (n_products, 16) normalisé — pour cosine

    # Utilisateurs
    user_index: dict[str, int] = {}          # user_id → ligne dans la matrice
    user_matrix: np.ndarray = None           # (n_users, 16)
    user_segments: dict[str, str] = {}

    # Interactions (pour exclude_purchased + popularité)
    purchased: dict[str, set] = {}
    popular_product_ids: list[str] = []

    def load(self) -> None:
        print(f"DATA_DIR = {DATA_DIR.resolve()}")

        # ── Produits ──────────────────────────────────────────────────────────
        products_path = DATA_DIR / "products" / "products_with_embeddings.parquet"
        if not products_path.exists():
            raise FileNotFoundError(
                f"Embeddings produits introuvables : {products_path}\n"
                "Lance d'abord le pipeline : 00 → 01 → 02 → 03"
            )

        print("Chargement des produits …")
        products = pd.read_parquet(products_path)
        self.product_ids = products["product_id"].tolist()
        self.product_index = {pid: i for i, pid in enumerate(self.product_ids)}
        self.product_categories = products["category"].fillna("").tolist()
        self.product_is_cold = (
            products["is_cold"].values
            if "is_cold" in products.columns
            else np.zeros(len(self.product_ids), dtype=bool)
        )

        mat = products[P_COLS].fillna(0.0).values.astype(np.float32)
        self.product_matrix = mat
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.product_matrix_norm = mat / norms
        print(f"  → {len(self.product_ids):,} produits chargés")

        # ── Utilisateurs ──────────────────────────────────────────────────────
        users_path = DATA_DIR / "users" / "users_with_embeddings.parquet"
        if not users_path.exists():
            raise FileNotFoundError(f"Embeddings utilisateurs introuvables : {users_path}")

        print("Chargement des utilisateurs …")
        users = pd.read_parquet(users_path)
        user_ids = users["user_id"].tolist()
        self.user_index = {uid: i for i, uid in enumerate(user_ids)}
        self.user_matrix = users[U_COLS].fillna(0.0).values.astype(np.float32)
        if "segment" in users.columns:
            self.user_segments = dict(zip(users["user_id"], users["segment"]))
        print(f"  → {len(self.user_index):,} utilisateurs chargés")

        # ── Interactions ──────────────────────────────────────────────────────
        inter_path = DATA_DIR / "interactions"
        if inter_path.exists():
            print("Chargement des interactions (historique + popularité) …")
            df = pd.read_parquet(inter_path, columns=["user_id", "product_id"])
            self.purchased = df.groupby("user_id")["product_id"].apply(set).to_dict()
            top_products = df["product_id"].value_counts().index.tolist()
            self.popular_product_ids = [
                pid for pid in top_products if pid in self.product_index
            ][:500]
            del df
            print(f"  → {len(self.purchased):,} users avec historique d'achat")

        print("Données prêtes.\n")


state = AppState()
