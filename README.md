# M1SPAR-projet2 — Système de recommandation Amazon

## Présentation du projet

Ce projet a pour objectif de construire un **système de recommandation** basé sur les données Amazon Reviews (2022 et 2023).
Il couvre l'intégralité de la chaîne : téléchargement des données brutes, exploration, construction des embeddings, et entraînement du modèle.

---

## Structure du projet

```
M1SPAR-projet2/
├── dataset/                        # Données brutes et script de téléchargement
│   ├── download_datasets.py        # Téléchargement via HuggingFace
│   ├── requirements.txt
│   └── raw/                        # Fichiers Arrow téléchargés (généré au run)
│       ├── 2023/
│       │   ├── reviews/            # raw_review_<Catégorie>
│       │   └── metadata/           # raw_meta_<Catégorie>
│       └── 2022/
│           ├── reviews/
│           └── metadata/
│
└── analyse/                        # Pipeline d'analyse et d'embeddings
    ├── requirements.txt
    ├── 00_prepare_data.py          # Arrow → 3 tables Parquet
    ├── 01_compute_content_embeddings.py  # Sentence Transformers (384 dims)
    ├── 02_train_als.py             # Entraînement ALS → facteurs latents
    ├── 03_cold_start_bridge.py     # Bridge model cold start + fusion finale
    ├── notebooks/
    │   └── 01_eda_reco.ipynb       # EDA complète (5 questions + visualisations)
    ├── data/                       # Tables Parquet (généré au run)
    │   ├── interactions/
    │   ├── products/
    │   └── users/
    └── outputs/                    # Visualisations générées (généré au run)
```

---

## Prérequis

- Python 3.10+
- Java 11+ (requis par PySpark)
- Environnement virtuel activé

```bash
# Créer et activer l'environnement
python -m venv venv
source venv/Scripts/activate   # Git Bash (Windows)
# ou
venv\Scripts\activate.bat      # CMD

# Installer les dépendances dataset
pip install -r dataset/requirements.txt

# Installer les dépendances analyse
pip install -r analyse/requirements.txt
```

---

## Étape 1 — Téléchargement des données

### Pourquoi ce dataset ?

Le dataset **Amazon Reviews 2023** (McAuley Lab, UCSD) est la référence académique pour les systèmes de recommandation.
Il contient **571 millions de reviews** réparties sur **33 catégories** de produits, couvrant les données jusqu'en 2023.

Il fournit deux types de fichiers par catégorie :
- **Reviews** : interactions utilisateur-produit (note, texte, achat vérifié, date)
- **Métadonnées** : informations produit (titre, prix, description, catégorie)

### Pourquoi HuggingFace ?

La bibliothèque `datasets` de HuggingFace permet de télécharger et de sauvegarder les données au format **Apache Arrow** (`.arrow`).
Ce format est colonnaire, non compressé, et optimisé pour une lecture très rapide en mémoire — idéal pour traiter des millions de lignes.

### Format Arrow vs autres formats

| Format | Lisible | Vitesse lecture | Taille disque |
|--------|---------|-----------------|---------------|
| `.jsonl` | Oui | Lent | Grand |
| `.csv` | Oui | Moyen | Moyen |
| `.parquet` | Non | Rapide | Petit (compressé) |
| `.arrow` | Non | **Très rapide** | Moyen |

### Compatibilité

La bibliothèque `datasets` version 3.0+ a supprimé le support des scripts de chargement custom.
Le dataset Amazon Reviews 2023 utilise encore ce mécanisme — il faut donc utiliser la **version 2.21.0**.

```bash
python dataset/download_datasets.py
```

Le script :
- Télécharge les **32 catégories** du dataset 2023 (reviews + métadonnées)
- Télécharge les **31 catégories** du dataset 2022
- Sauvegarde chaque catégorie sur disque (reprise automatique en cas d'interruption)

> **Note :** Le téléchargement complet représente plusieurs centaines de Go.
> Le pipeline fonctionne avec les catégories partiellement téléchargées.

---

## Étape 2 — Préparation des données (`00_prepare_data.py`)

### Pourquoi 3 tables ?

Les algorithmes de recommandation (ALS, etc.) raisonnent sur 3 entités distinctes :

| Table | Contenu | Clé primaire |
|---|---|---|
| `interactions` | Qui a acheté / noté quoi, quand | `(user_id, product_id)` |
| `products` | Informations sur les produits | `product_id` |
| `users` | Profil agrégé des utilisateurs | `user_id` |

Cette structure est le schéma standard des systèmes de recommandation (collaborative filtering).

### Dérivation des colonnes

Comme Amazon Reviews ne fournit que des reviews (pas de logs de navigation), certaines colonnes sont dérivées :

- **`interaction_type`** : `purchase` si `verified_purchase = True`, sinon `review`
- **`converted`** : `1` si achat vérifié, `0` sinon (signal binaire pour ALS implicite)
- **`year_month`** : extrait du timestamp en millisecondes (pour l'analyse de saisonnalité)
- **`segment`** : `power_user` (top 10%), `regular_user` (p50-p90), `casual_user` (bottom 50%)

### Format de sortie : Parquet

Les 3 tables sont sauvegardées en **Parquet** (partitionné par `category` pour `interactions`).
Parquet est le format natif de PySpark et est compressé automatiquement — il réduit l'espace disque et accélère les lectures filtrées.

```bash
cd analyse
python 00_prepare_data.py
```

---

## Étape 3 — Exploration du dataset (`notebooks/01_eda_reco.ipynb`)

Le notebook répond à 5 questions fondamentales qui justifient les choix d'architecture.

### Q1 · Sparsité de la matrice user-item

**Pourquoi c'est important :**
La sparsité mesure la proportion de paires (user, produit) non observées.
Une matrice très creuse signifie que chaque utilisateur n'a interagi qu'avec une infime partie du catalogue.

**Résultat attendu :** > 99.9% de la matrice est vide.

**Impact architectural :**
- ALS (Alternating Least Squares) est conçu pour les matrices creuses
- Les embeddings (P1-P16) sont indispensables pour couvrir les paires non observées

### Q2 · Distribution des types d'interactions

**Pourquoi c'est important :**
Dans Amazon Reviews, toutes les interactions sont des reviews.
La distinction `verified_purchase` (achat réel) vs simple review est cruciale :
- Achat vérifié = **signal fort** (explicit feedback)
- Review non vérifiée = **signal faible** (implicit feedback)

**Impact architectural :**
- Le champ `helpful_vote` peut pondérer la qualité des reviews dans la fonction de perte

### Q3 · Biais de notation (J-shaped distribution)

**Pourquoi c'est important :**
Amazon présente un biais positif massif (survivorship bias) : les utilisateurs notent
surtout les produits qu'ils aiment. La distribution des notes est en forme de J.

**Résultat attendu :** ~55% de notes à 5 étoiles.

**Impact architectural :**
- Corriger par position bias dans la fonction de perte ALS
- Privilégier le feedback implicite (achat) sur le rating brut

### Q4 · Distribution longue traîne (Loi de Zipf)

**Pourquoi c'est important :**
La majorité des produits reçoivent très peu d'avis.
Cela crée un problème de **cold start** : le modèle ne peut pas recommander
des produits qu'il n'a pas vus pendant l'entraînement.

**Résultat attendu :** ~60% des produits ont moins de 5 interactions.

**Impact architectural :**
- Les embeddings de contenu (Sentence Transformers) sont obligatoires pour les produits rares
- Le bridge model permet de projeter ces produits dans l'espace ALS

### Q5 · Segmentation utilisateurs & Saisonnalité

**Pourquoi c'est important :**
Les power users (≈10% des users) génèrent une part disproportionnée des achats.
Dégrader leurs recommandations pour optimiser le CTR moyen serait une erreur métier critique.

**Impact architectural :**
- Évaluer les modèles séparément par segment
- Ne pas fusionner les métriques power_user / casual_user

```bash
jupyter notebook notebooks/01_eda_reco.ipynb
```

Les visualisations sont sauvegardées dans `outputs/` :
- `viz1_sparsity_heatmap.png` — heatmap de la sparsité
- `viz2_rating_distribution.png` — distribution des notes (J-shaped)
- `viz3_category_share.png` — part des catégories par volume

---

## Étape 4 — Embeddings pour la production avec cold start

### Le problème du cold start

Un système de recommandation ne peut pas recommander un **nouveau produit** (jamais vu à l'entraînement)
ni personnaliser pour un **nouvel utilisateur** (aucun historique).
C'est le problème du cold start — inévitable en production.

### Solution : combinaison ALS + Sentence Transformers + Bridge model

```
                    ┌──────────────────────────────────────┐
                    │      Embeddings finaux (P1-P16)      │
                    └──────────────────────────────────────┘
                                    │
               ┌────────────────────┼────────────────────┐
               ▼                    ▼                    ▼
        Warm Product          Cold Product           Cold User
      (≥ 5 interactions)    (< 5 interactions)    (jamais vu)
      P1-P16 = ALS facts    P1-P16 = Bridge(ST)   U1-U16 = mean(P)
```

| Cas | Méthode | Pourquoi |
|---|---|---|
| Produit warm | Facteurs latents ALS | Encodent les comportements réels d'achat |
| Produit cold | Bridge model(ST) | Projette la sémantique du titre vers l'espace ALS |
| User warm | Facteurs latents ALS | Encodent les préférences réelles |
| User cold | Moyenne des items vus | Approximation rapide sans historique |

---

### Script 01 — Sentence Transformers (`01_compute_content_embeddings.py`)

**Pourquoi Sentence Transformers ?**
Le modèle `all-MiniLM-L6-v2` encode le texte (titre + description) en vecteurs denses de 384 dimensions.
Ces vecteurs capturent la **sémantique** : deux produits similaires auront des embeddings proches,
même s'ils n'ont aucun utilisateur en commun.

**Normalisation L2 :** les embeddings sont normalisés (norme = 1) pour que la similarité cosinus
soit équivalente au produit scalaire — plus efficace pour la recherche de voisins.

```bash
python 01_compute_content_embeddings.py
# Output : data/content_embeddings/content_embeddings.parquet (colonnes emb_0…emb_383)
```

---

### Script 02 — Entraînement ALS (`02_train_als.py`)

**Pourquoi ALS ?**
ALS (Alternating Least Squares) est l'algorithme de factorisation matricielle standard
pour les matrices d'interactions creuses. Il décompose la matrice user-item en deux matrices de rang faible :
- `U` (n_users × rank) — représentation latente des utilisateurs
- `V` (n_products × rank) — représentation latente des produits

Le produit `U × Vᵀ` reconstitue les interactions manquantes = prédictions de recommandation.

**Mode implicit feedback (`implicitPrefs=True`) :**
Plutôt que d'utiliser les ratings bruts (biaisés vers les 5 étoiles — cf. Q3),
on utilise le nombre d'interactions par paire comme signal de confiance.
Plus un utilisateur a interagi avec un produit, plus le signal est fort.

**Hyperparamètres :**
- `rank=16` : dimension des embeddings (P1-P16 / U1-U16)
- `regParam=0.1` : régularisation L2 pour éviter le sur-apprentissage
- `maxIter=15` : itérations d'optimisation alternée

```bash
python 02_train_als.py
# Output : data/als_item_factors/ + data/als_user_factors/
```

---

### Script 03 — Bridge model cold start (`03_cold_start_bridge.py`)

**Pourquoi un bridge model ?**
ALS produit des embeddings excellents pour les produits warm, mais ne couvre pas les produits cold.
Sentence Transformers couvre tous les produits mais n'encode pas les comportements d'achat.

Le **bridge model** (MLP) apprend à projeter l'espace sémantique (384 dims) vers l'espace comportemental ALS (16 dims).
Il est entraîné sur les produits warm (où on dispose des deux représentations) et appliqué aux produits cold.

**Architecture du MLP :**
```
ST(384) → Dense(256, ReLU) → Dense(64, ReLU) → ALS(16)
```

**Stratégie cold user :**
Pour un utilisateur sans historique, on calcule la **moyenne des embeddings P1-P16** des produits
avec lesquels il a interagi (même une seule interaction suffit).
C'est une approximation efficace qui ne nécessite pas de réentraîner ALS.

```bash
python 03_cold_start_bridge.py
# Output : data/products/products_with_embeddings.parquet
#          data/users/users_with_embeddings.parquet
#          data/bridge_model/bridge_model.pkl
```

---

## Ordre d'exécution complet

```bash
# 0. Environnement
source venv/Scripts/activate
pip install -r dataset/requirements.txt
pip install -r analyse/requirements.txt

# 1. Télécharger les données (long — peut tourner en parallèle)
python dataset/download_datasets.py

# 2. Construire les 3 tables Parquet
cd analyse
python 00_prepare_data.py

# 3. Embeddings de contenu (Sentence Transformers)
python 01_compute_content_embeddings.py

# 4. Entraîner ALS
python 02_train_als.py

# 5. Bridge model cold start + fusion finale
python 03_cold_start_bridge.py

# 6. Explorer le dataset
jupyter notebook notebooks/01_eda_reco.ipynb
```

---

## Décisions techniques clés

| Décision | Raison |
|---|---|
| `datasets==2.21.0` | La v3+ a supprimé `trust_remote_code` — incompatible avec le script HuggingFace du dataset |
| Format Arrow pour le stockage brut | Lecture très rapide, format natif HuggingFace |
| Format Parquet pour l'analyse | Compressé, partitionnable, natif PySpark |
| ALS en mode implicit | Les ratings Amazon sont biaisés (55% de 5 étoiles) — le signal d'achat est plus fiable |
| `rank=16` pour ALS | Compromis expressivité / coût mémoire / cohérence avec P1-P16 |
| `all-MiniLM-L6-v2` | Modèle léger (rapide), qualité suffisante pour le bridge, pas de GPU requis |
| MLP comme bridge | Apprend une projection non-linéaire ST → ALS ; plus expressif qu'une régression linéaire |
| Seuil warm/cold à 5 interactions | En dessous, les facteurs ALS sont instables (peu de signal) |
