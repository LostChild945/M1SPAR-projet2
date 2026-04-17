"""
04_grid_search_als.py
---------------------
Grid search sur les hyperparamètres ALS avec MLflow tracking.

Grid : rank × regParam × maxIter = 3 × 3 × 2 = 18 combinaisons.
Évaluation : NDCG@10 sur échantillon de 1000 users.
Baseline : popularité pour calcul du lift.

Output :
  - 18 runs MLflow dans l'expérience "als_grid_search"
  - Meilleur modèle tagué "Production"

Usage :
    python 04_grid_search_als.py
"""

import os
import pickle
import tempfile
import numpy as np
import pandas as pd
import mlflow
from pathlib import Path
from itertools import product as cartesian

from evaluation import evaluate_als, evaluate_popularity, compute_lift

DATA_DIR = Path(os.environ.get("DATA_DIR", "./data"))

# ── Grid ────────────────────────────────────────────────────────────────────
RANKS      = [8, 16, 32]
REG_PARAMS = [0.01, 0.1, 0.5]
MAX_ITERS  = [10, 20]

# ── Filtre catégories ───────────────────────────────────────────────────────
TEST_5 = ["All_Beauty", "Amazon_Fashion", "Appliances", "Arts_Crafts_and_Sewing", "Automotive"]
FILTER_CATEGORIES = ["All_Beauty", "Amazon_Fashion"]  # 2 catégories pour les tests rapides


if __name__ == "__main__":
    mlflow.set_tracking_uri(f"file:{DATA_DIR / 'mlruns'}")
    mlflow.set_experiment("als_grid_search")

    print("Chargement des interactions …")
    interactions = pd.read_parquet(DATA_DIR / "interactions")
    if FILTER_CATEGORIES is not None:
        interactions = interactions[interactions["category"].isin(FILTER_CATEGORIES)]
    print(f"  → {len(interactions):,} interactions\n")

    # ── Baseline popularité ─────────────────────────────────────────────────
    print("Calcul baseline popularité …")
    np.random.seed(42)
    ndcg_pop = evaluate_popularity(interactions, k=10)
    print(f"  NDCG@10 popularité : {ndcg_pop:.4f}\n")

    # ── Grid search ─────────────────────────────────────────────────────────
    grid = list(cartesian(RANKS, REG_PARAMS, MAX_ITERS))
    print(f"Grid search : {len(grid)} combinaisons\n")

    best_ndcg = -1.0
    best_run_id = None
    best_model = None

    for i, (rank, reg, iters) in enumerate(grid, 1):
        print(f"[{i}/{len(grid)}] rank={rank}, reg={reg}, iter={iters}")
        np.random.seed(42)

        with mlflow.start_run(run_name=f"als_r{rank}_reg{reg}_i{iters}"):
            mlflow.log_param("rank", rank)
            mlflow.log_param("regularization", reg)
            mlflow.log_param("iterations", iters)
            mlflow.log_param("min_interactions_eval", 10)
            mlflow.log_param("sample_users", 1000)

            ndcg_als, model = evaluate_als(
                interactions,
                als_rank=rank,
                als_iter=iters,
                als_reg=reg,
            )

            lift = compute_lift(ndcg_als, ndcg_pop)

            mlflow.log_metric("ndcg_at_10", ndcg_als)
            mlflow.log_metric("ndcg_popularity_baseline", ndcg_pop)
            mlflow.log_metric("lift_vs_popularity", lift)
            mlflow.log_metric("n_user_factors", model.user_factors.shape[0])
            mlflow.log_metric("n_item_factors", model.item_factors.shape[0])

            print(f"  NDCG@10={ndcg_als:.4f}  lift={lift:+.1f}%")

            # Sauvegarder le modèle comme artifact
            with tempfile.TemporaryDirectory() as tmp:
                model_path = os.path.join(tmp, "als_model.pkl")
                with open(model_path, "wb") as f:
                    pickle.dump(model, f)
                mlflow.log_artifact(model_path, artifact_path="model")

            if ndcg_als > best_ndcg:
                best_ndcg = ndcg_als
                best_run_id = mlflow.active_run().info.run_id
                best_model = model

    # ── Enregistrer le meilleur modèle dans le Model Registry ──────────────
    if best_run_id:
        with mlflow.start_run(run_id=best_run_id):
            mlflow.set_tag("model_stage", "Production")

        # Sauver le best model dans un format MLflow-compatible
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = os.path.join(tmp, "als_model")
            os.makedirs(model_dir)
            # Pickle du modèle
            with open(os.path.join(model_dir, "model.pkl"), "wb") as f:
                pickle.dump(best_model, f)
            # MLmodel descriptor minimal
            with open(os.path.join(model_dir, "MLmodel"), "w") as f:
                f.write("artifact_path: als_model\n")
                f.write("flavors:\n")
                f.write("  python_function:\n")
                f.write("    loader_module: mlflow.pyfunc\n")

            # Logger comme artifact dans le best run
            with mlflow.start_run(run_id=best_run_id):
                mlflow.log_artifacts(model_dir, artifact_path="als_model")

        # Enregistrer dans le registry
        model_uri = f"runs:/{best_run_id}/als_model"
        mlflow.register_model(model_uri, "als-recommender")
        print(f"\nMeilleur run : {best_run_id} (NDCG@10={best_ndcg:.4f})")
        print(f"  → Modèle enregistré dans le registry : als-recommender")

    print("\nGrid search terminé. Consultez MLflow UI : mlflow ui --backend-store-uri file:./data/mlruns")
