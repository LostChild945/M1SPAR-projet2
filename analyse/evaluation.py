"""
evaluation.py
-------------
Évaluation des modèles de recommandation :

  - NDCG@10 pour ALS
  - NDCG@10 pour baseline popularité
  - Lift ALS vs popularité

Split temporel : 80% train / 20% test par user (dernières interactions).
Échantillon : 1000 users avec ≥ 10 interactions.

Usage :
    from evaluation import evaluate_als, evaluate_popularity, compute_lift
"""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
import implicit

SAMPLE_USERS = 1000
MIN_INTERACTIONS_EVAL = 10


def ndcg_at_k(predicted: list[str], relevant: set[str], k: int = 10) -> float:
    """Calcule le NDCG@k pour une liste de prédictions."""
    dcg = 0.0
    for i, item in enumerate(predicted[:k]):
        if item in relevant:
            dcg += 1.0 / np.log2(i + 2)  # position 1-indexed

    # IDCG = DCG parfait (tous les relevants en haut)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant), k)))
    if idcg == 0:
        return 0.0
    return dcg / idcg


def split_train_test(interactions: pd.DataFrame, test_ratio: float = 0.2):
    """
    Split temporel par user : les dernières `test_ratio` interactions
    de chaque user vont dans le test set.
    """
    interactions = interactions.sort_values(["user_id", "timestamp"])

    def split_user(group):
        n = len(group)
        split_idx = int(n * (1 - test_ratio))
        group = group.copy()
        group["split"] = ["train"] * split_idx + ["test"] * (n - split_idx)
        return group

    interactions = interactions.groupby("user_id", group_keys=False).apply(split_user)
    train = interactions[interactions["split"] == "train"].drop(columns=["split"])
    test = interactions[interactions["split"] == "test"].drop(columns=["split"])
    return train, test


def evaluate_als(
    interactions: pd.DataFrame,
    als_rank: int = 16,
    als_iter: int = 15,
    als_reg: float = 0.1,
    sample_users: int = SAMPLE_USERS,
    k: int = 10,
) -> tuple[float, implicit.als.AlternatingLeastSquares]:
    """
    Entraîne ALS sur le train set et évalue NDCG@k sur le test set.

    Returns:
        (ndcg_mean, model)
    """
    # Sélectionner les users avec assez d'interactions
    user_counts = interactions["user_id"].value_counts()
    eligible = user_counts[user_counts >= MIN_INTERACTIONS_EVAL].index
    if len(eligible) > sample_users:
        eligible = np.random.choice(eligible, size=sample_users, replace=False)
    subset = interactions[interactions["user_id"].isin(eligible)]

    train, test = split_train_test(subset)

    # Indexation
    all_users = subset["user_id"].unique()
    all_items = subset["product_id"].unique()
    user_to_idx = {u: i for i, u in enumerate(all_users)}
    item_to_idx = {p: i for i, p in enumerate(all_items)}
    idx_to_item = {i: p for p, i in item_to_idx.items()}

    # Matrice sparse train
    train_mapped = train[train["product_id"].isin(item_to_idx)]
    signal = (
        train_mapped.assign(
            user_idx=train_mapped["user_id"].map(user_to_idx).astype(np.int32),
            product_idx=train_mapped["product_id"].map(item_to_idx).astype(np.int32),
        )
        .groupby(["user_idx", "product_idx"])
        .size()
        .reset_index(name="count")
    )

    user_item = csr_matrix(
        (signal["count"].values.astype(np.float32),
         (signal["user_idx"].values, signal["product_idx"].values)),
        shape=(len(all_users), len(all_items)),
    )

    # Train ALS
    model = implicit.als.AlternatingLeastSquares(
        factors=als_rank,
        iterations=als_iter,
        regularization=als_reg,
        use_gpu=False,
    )
    model.fit(user_item)

    # Évaluer NDCG@k
    test_by_user = test.groupby("user_id")["product_id"].apply(set).to_dict()
    ndcg_scores = []

    for user_id, relevant in test_by_user.items():
        if user_id not in user_to_idx:
            continue
        user_idx = user_to_idx[user_id]
        ids, scores = model.recommend(user_idx, user_item[user_idx], N=k, filter_already_liked_items=True)
        predicted = [idx_to_item.get(i, "") for i in ids]
        ndcg = ndcg_at_k(predicted, relevant, k)
        ndcg_scores.append(ndcg)

    ndcg_mean = float(np.mean(ndcg_scores)) if ndcg_scores else 0.0
    return ndcg_mean, model


def evaluate_popularity(
    interactions: pd.DataFrame,
    sample_users: int = SAMPLE_USERS,
    k: int = 10,
) -> float:
    """
    Baseline popularité : recommande les k produits les plus populaires.
    Retourne NDCG@k moyen.
    """
    user_counts = interactions["user_id"].value_counts()
    eligible = user_counts[user_counts >= MIN_INTERACTIONS_EVAL].index
    if len(eligible) > sample_users:
        eligible = np.random.choice(eligible, size=sample_users, replace=False)
    subset = interactions[interactions["user_id"].isin(eligible)]

    train, test = split_train_test(subset)

    # Top-k global
    top_k = train["product_id"].value_counts().head(k).index.tolist()

    # Évaluer
    test_by_user = test.groupby("user_id")["product_id"].apply(set).to_dict()
    ndcg_scores = []

    for user_id, relevant in test_by_user.items():
        ndcg = ndcg_at_k(top_k, relevant, k)
        ndcg_scores.append(ndcg)

    return float(np.mean(ndcg_scores)) if ndcg_scores else 0.0


def compute_lift(ndcg_als: float, ndcg_pop: float) -> float:
    """Calcule le lift en % de ALS vs popularité."""
    if ndcg_pop == 0:
        return float("inf") if ndcg_als > 0 else 0.0
    return (ndcg_als - ndcg_pop) / ndcg_pop * 100
