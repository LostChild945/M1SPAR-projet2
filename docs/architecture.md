# Architecture technique — P2 Système de recommandation Amazon

## Schéma architecture end-to-end

```
┌─────────────────────────────────────────────────────────────────────────┐
│              ARCHITECTURE P2 — RECOMMANDATION AMAZON                    │
│              Batch + Cache Redis (pas de streaming)                     │
└─────────────────────────────────────────────────────────────────────────┘

[Sources]           [ETL Batch]           [Features]        [Serving]
interactions ──▶  PySpark ETL        ──▶  Redis Cache  ──▶  FastAPI
products          Parquet                  (Top-100 recs     /recommend
users             interactions/            précalculés)      /similar
(Amazon 2023)     products/                Parquet           /feedback
                  users/                   embeddings
                       │                       │                 │
                       ▼                       ▼                 ▼
                   MLflow                Modèles :           Streamlit
                   Tracking             - ALS MLlib         Dashboard
                   Registry             - ST Embed.         CTR /
                   (artefacts)          - Bridge MLP        segments /
                                        - Hybride           saisonnalité
                                             │
                                             ▼
                                        A/B Testing
                                        (scipy.stats)
                                        control vs treatment

Légende :
  ██ Implémenté    : PySpark ETL, Parquet, ALS MLlib, ST Embeddings, Bridge MLP
  ░░ À implémenter : MLflow, Redis, FastAPI, Streamlit, A/B Testing
```

---

## État d'implémentation

### Implémenté (`analyse/`)

| Script | Rôle | Output |
|---|---|---|
| `00_prepare_data.py` | Arrow → 3 tables Parquet | `data/interactions/`, `data/products/`, `data/users/` |
| `01_compute_content_embeddings.py` | Sentence Transformers (all-MiniLM-L6-v2, 384 dims) | `data/content_embeddings/content_embeddings.parquet` |
| `02_train_als.py` | ALS MLlib (rank=16, iter=15, reg=0.1, implicit) | `data/als_item_factors/`, `data/als_user_factors/` |
| `03_cold_start_bridge.py` | Bridge MLP sklearn (384→16) + fusion warm/cold | `data/products/products_with_embeddings.parquet`, `data/users/users_with_embeddings.parquet` |
| `eda_reco.py` | EDA 5 questions + 3 visualisations (Docker) | `outputs/viz1_*.png`, `viz2_*.png`, `viz3_*.png` |

### Dépendances réelles (`analyse/requirements.txt`)

```
pyspark>=3.5.0
pandas>=2.0.0
pyarrow>=14.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
datasets==2.21.0
sentence-transformers>=2.7.0
scikit-learn>=1.4.0
jupyter>=1.0.0
```

### À implémenter (cible P2)

- **MLflow** — tracking des runs ALS (rank, reg, RMSE, n_users, n_items)
- **Redis** — cache des Top-100 recs précalculées (TTL 25h)
- **FastAPI** — endpoints `/recommend`, `/similar`, `/feedback`, `/ab_results`
- **Streamlit** — dashboard CTR / segmentation / saisonnalité
- **A/B Testing** — `scipy.stats.proportions_ztest` sur CTR control vs treatment

---

## Justification des choix techniques

| Composant | Choix retenu | Alternative rejetée | Justification |
|---|---|---|---|
| Processing | **Spark ALS (MLlib)** | Surprise, LightFM | Scalabilité native 100M+ interactions ; distribué sur cluster |
| Cold start produit | **ST → Bridge MLP** | Content-based pur | ST encode la sémantique (384 dims) ; MLP projette vers l'espace ALS (16 dims) |
| Cold start user | **Moyenne P1-P16 des items vus** | KNN user-based | Approximation sans réentraînement ; converge dès 1 interaction |
| Hybridation | **Weighted combination ALS + Bridge** | Cascade | Tunable par poids `α` ; équilibre personnalisation / popularité |
| Stockage | **Parquet partitionné par category** | Delta Lake, Hive | Lecture filtrée rapide avec PySpark ; pas de dépendance externe |
| Cache serving | **Redis** (cible) | Memcached | Latence <5 ms pour Top-100 précalculés ; TTL natif |
| Feature store | **Parquet + Redis** (cible) | Feast, Tecton | Feast = overhead opérationnel injustifié pour ce scope |
| Tracking | **MLflow** (cible) | W&B, Comet | Open-source, intégration PySpark native, UI locale sans compte cloud |
| API | **FastAPI** (cible) | Flask | Async, OpenAPI auto-générée, validation pydantic |
| A/B Testing | **scipy.stats** (cible) | Optimizely | Suffisant pour test de significativité ; aucune dépendance externe |
| Dashboard | **Streamlit** (cible) | Dash | Prototypage rapide, Python pur |
| Embeddings texte | **all-MiniLM-L6-v2** | sentence-t5, OpenAI | 80 MB, CPU-only, pas de GPU requis, qualité suffisante pour le bridge |

---

## Architecture des modèles (implémentée)

### Stratégie warm / cold

```
Seuil : MIN_INTERACTIONS = 5  (02_train_als.py, 03_cold_start_bridge.py)

                    ┌──────────────────────────────────────┐
                    │      Embeddings finaux (P1-P16)      │
                    └──────────────────────────────────────┘
                                    │
               ┌────────────────────┼────────────────────┐
               ▼                    ▼                    ▼
        Warm Product          Cold Product           Cold User
      (≥ 5 interactions)    (< 5 interactions)    (jamais vu)
      P1-P16 = ALS facts    P1-P16 = Bridge(ST)   U1-U16 = mean(P1-P16
                                                       des items vus)
```

### Hyperparamètres ALS (`02_train_als.py`)

```python
ALS_RANK         = 16      # dimensions P1-P16 / U1-U16
ALS_MAX_ITER     = 15
ALS_REG_PARAM    = 0.1
ALS_IMPLICIT     = True    # signal = nb d'interactions par paire
```

### Architecture Bridge MLP (`03_cold_start_bridge.py`)

```
ST(384) → StandardScaler → Dense(256, ReLU) → Dense(64, ReLU) → ALS(16)

Pipeline sklearn :
  - StandardScaler  : normalise les embeddings ST en entrée
  - MLPRegressor    : hidden_layer_sizes=(256, 64), max_iter=300,
                      early_stopping=True, validation_fraction=0.1
```

---

## Décision Batch vs Streaming

```
DÉCISION : BATCH avec cache Redis pour P2
```

| Critère | Streaming | Batch + Cache (choix P2) |
|---|---|---|
| Latence reco | <10 ms (inutile) | <50 ms via Redis ✓ |
| Coût infra | Élevé (Kafka, Flink) | Faible (cron nuit) ✓ |
| Complexité ops | Haute | Faible ✓ |
| Fraîcheur données | Temps réel | J-1 (acceptable e-commerce) ✓ |
| Cohérence A/B | Difficile | Simple (snapshot quotidien) ✓ |

### Pipeline batch quotidien (cible)

```
02h00  ALS re-training (interactions J-1)
       → MLflow log (rank, reg, RMSE, n_users, n_items)
       → artefacts : als_item_factors/, als_user_factors/

03h00  Bridge model re-fit (produits warm uniquement)
       → data/bridge_model/bridge_model.pkl

03h30  Précalcul Top-100 recs par user actif (30j)
       → Redis : SET reco:{user_id} → JSON [product_id, score, source]
       → TTL : 25h

À la requête /recommend/{user_id} :
  a. HIT Redis  → réponse en <5 ms
  b. MISS Redis → ALS inference on-the-fly → <50 ms
  c. Cold start → Bridge MLP → <50 ms
```

---

## Plan A/B Testing (cible)

```python
# Hypothèse : modèle hybride (ALS + Bridge) améliore le CTR vs ALS seul

AB_CONFIG = {
    "experiment_name": "hybrid_vs_als_only",
    "control":   {"model": "ALS_only",   "traffic_pct": 50},
    "treatment": {"model": "ALS_Hybrid", "traffic_pct": 50},
    "metric":    "CTR",
    "duration":  "7 jours minimum",
    "min_users": 10_000,
    "alpha":     0.05,
    "power":     0.80,
}

from scipy import stats

def evaluate_ab_test(control_clicks, control_views,
                     treatment_clicks, treatment_views):
    ctr_c = control_clicks / control_views
    ctr_t = treatment_clicks / treatment_views
    _, p_value = stats.proportions_ztest(
        [treatment_clicks, control_clicks],
        [treatment_views,  control_views]
    )
    lift = (ctr_t - ctr_c) / ctr_c * 100
    return {
        "ctr_control":   round(ctr_c, 4),
        "ctr_treatment": round(ctr_t, 4),
        "lift_pct":      round(lift, 2),
        "p_value":       round(p_value, 4),
        "significant":   p_value < 0.05,
    }
```

### Métriques par segment

| Segment | Métrique principale | Seuil succès |
|---|---|---|
| power_user | CTR + achats/semaine | +3% lift minimum |
| regular_user | CTR | +5% lift minimum |
| casual_user | Taux de retour 7j | +2% lift minimum |

> Ne pas agréger les métriques cross-segment — un lift global peut masquer une
> dégradation sur les power_users (top 10% en revenus).

---

## Estimation ressources cluster

```
DÉVELOPPEMENT (5 catégories test, local) :
  Machine    : poste dev, driver Spark 4 GB (eda_reco.py)
  Docker     : mem_limit 8g (docker-compose.yml)
  Coût       : 0 $

ENTRAÎNEMENT ALS (23 catégories, ~14 GB Parquet, ~100M interactions) :
  Driver     : 28 GB RAM minimum
  Workers    : 4× 14 GB RAM
  Justification :
    - 14 GB Parquet → ~6 GB en mémoire Spark compressé
    - ALS rank=16, 15 itérations → pic mémoire ~3× dataset
    - shuffle.partitions=100 → ~600 MB/partition sur 4 workers

SERVING (cible demo) :
  FastAPI    : 2 vCPUs, 4 GB RAM
  Redis      : 13 GB, 100K ops/s
  Streamlit  : même machine que FastAPI
```

---

## Checklist livrable

- [x] Pipeline ETL Parquet (`00_prepare_data.py`)
- [x] Embeddings contenu ST (`01_compute_content_embeddings.py`)
- [x] Entraînement ALS + facteurs latents (`02_train_als.py`)
- [x] Bridge model cold start + fusion (`03_cold_start_bridge.py`)
- [x] EDA 5 questions + visualisations (`eda_reco.py`, Docker)
- [x] `docs/architecture.md` — ce fichier
- [x] `src/api/openapi.yaml` — spécification OpenAPI
- [ ] MLflow tracking des runs ALS
- [ ] Redis cache Top-100 recs
- [ ] FastAPI `/recommend`, `/similar`, `/feedback`, `/ab_results`
- [ ] Streamlit dashboard
- [ ] A/B Testing `scipy.stats`
