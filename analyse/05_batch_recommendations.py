"""
05_batch_recommendations.py
---------------------------
Génère les recommandations batch top-10 pour tous les users actifs.

Utilise les embeddings finaux (products_with_embeddings + users_with_embeddings)
pour calculer les scores via dot product.

Output :
  data/delta/gold/recommendations/   (Parquet partitionné)

Usage :
    python 05_batch_recommendations.py
"""

import os
from datetime import datetime

import mlflow
import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path(os.environ.get("DATA_DIR", "./data"))

TOP_K = 10


if __name__ == "__main__":
    mlflow.set_tracking_uri(f"file:{DATA_DIR / 'mlruns'}")
    mlflow.set_experiment("batch_recommendations")

    print("Chargement des embeddings …")
    products = pd.read_parquet(DATA_DIR / "products" / "products_with_embeddings.parquet")
    users = pd.read_parquet(DATA_DIR / "users" / "users_with_embeddings.parquet")

    p_cols = [f"P{i}" for i in range(1, 17)]
    u_cols = [f"U{i}" for i in range(1, 17)]

    product_ids = products["product_id"].values
    product_matrix = products[p_cols].values.astype(np.float32)
    user_ids = users["user_id"].values
    user_matrix = users[u_cols].values.astype(np.float32)

    print(f"  {len(user_ids):,} users × {len(product_ids):,} produits")

    # ── Batch scoring ────────────────────────────────────────────────────────
    print(f"\nGénération batch top-{TOP_K} …")
    batch_size = 1000
    all_recs = []

    with mlflow.start_run(run_name=f"batch_{datetime.now().strftime('%Y%m%d_%H%M')}"):
        mlflow.log_param("top_k", TOP_K)
        mlflow.log_param("n_users", len(user_ids))
        mlflow.log_param("n_products", len(product_ids))

        for start in range(0, len(user_ids), batch_size):
            end = min(start + batch_size, len(user_ids))
            batch_users = user_matrix[start:end]
            batch_user_ids = user_ids[start:end]

            # scores = batch_users (batch, 16) @ product_matrix.T (16, n_products)
            scores = batch_users @ product_matrix.T

            for i, uid in enumerate(batch_user_ids):
                top_idx = np.argpartition(scores[i], -TOP_K)[-TOP_K:]
                top_idx = top_idx[np.argsort(scores[i][top_idx])[::-1]]

                for rank, idx in enumerate(top_idx, 1):
                    all_recs.append({
                        "user_id": uid,
                        "product_id": product_ids[idx],
                        "score": float(scores[i][idx]),
                        "rank": rank,
                    })

            if (end % 10000) == 0 or end == len(user_ids):
                print(f"  {end:,}/{len(user_ids):,} users traités")

        recs_df = pd.DataFrame(all_recs)
        recs_df["generated_at"] = datetime.now().isoformat()

        # ── Écriture ─────────────────────────────────────────────────────
        out_path = DATA_DIR / "delta" / "gold" / "recommendations"
        out_path.mkdir(parents=True, exist_ok=True)
        recs_df.to_parquet(out_path / "recommendations.parquet", index=False)

        mlflow.log_metric("n_recommendations", len(recs_df))
        mlflow.log_metric("avg_score", float(recs_df["score"].mean()))

        print(f"\n  {len(recs_df):,} recommandations générées")
        print(f"  Score moyen : {recs_df['score'].mean():.4f}")
        print(f"  Sauvegardé : {out_path}")

    print("\nBatch terminé.")
