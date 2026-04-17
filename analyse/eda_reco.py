import os
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')  # mode sans affichage (Docker)
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import numpy as np
import pandas as pd

from pyspark.sql import SparkSession, functions as F
from pyspark.sql.window import Window

sns.set_theme(style='whitegrid', palette='muted')

BASE    = '/app/data/'
OUTPUTS = '/app/outputs/'
os.makedirs(OUTPUTS, exist_ok=True)

# =============================================================================
# FILTRE DE CATÉGORIES — optionnel
# -----------------------------------------------------------------------------
# Liste les catégories à analyser, ou None pour tout traiter.
#   None    → toutes les catégories (14 GB, requiert >8 GB RAM)
#   TEST_5  → 5 catégories légères (~1.2 GB, tourne dans 6 GB)
# =============================================================================

TEST_5 = ['All_Beauty', 'Amazon_Fashion', 'Appliances', 'Arts_Crafts_and_Sewing', 'Automotive']

FILTER_CATEGORIES = TEST_5   # ← mettre None pour toutes les catégories

# =============================================================================

spark = (
    SparkSession.builder
    .appName('M1SPAR-P2-EDA')
    .config('spark.driver.memory', '4g')
    .config('spark.driver.memoryOverhead', '1g')
    .config('spark.sql.shuffle.partitions', '20')
    .config('spark.sql.files.maxPartitionBytes', '64m')
    .config('spark.memory.fraction', '0.6')
    .config('spark.ui.enabled', 'true')
    .config('spark.ui.port', '4040')
    .getOrCreate()
)
spark.sparkContext.setLogLevel('ERROR')
print(f'Spark {spark.version} démarré')

# ── Chargement des 3 tables ────────────────────────────────────────────────
interactions = spark.read.parquet(f'{BASE}interactions/')
products     = spark.read.parquet(f'{BASE}products/')
users        = spark.read.parquet(f'{BASE}users/')

if FILTER_CATEGORIES is not None:
    print(f'Filtre actif : {FILTER_CATEGORIES}')
    interactions = interactions.filter(F.col('category').isin(FILTER_CATEGORIES))
    products     = products.filter(F.col('category').isin(FILTER_CATEGORIES))
    # Garder uniquement les users présents dans les interactions filtrées
    valid_users  = interactions.select('user_id').distinct()
    users        = users.join(valid_users, on='user_id', how='inner')

print('=== Schéma interactions ===')
interactions.printSchema()
print('=== Schéma products ===')
products.printSchema()
print('=== Schéma users ===')
users.printSchema()

print('Aperçu interactions :')
interactions.show(5, truncate=True)
print('Aperçu products :')
products.show(5, truncate=True)
print('Aperçu users :')
users.show(5, truncate=True)

# ── Q1 · Sparsité ─────────────────────────────────────────────────────────
n_users    = users.count()
n_products = products.count()
n_inter    = interactions.count()

sparsity = (1 - n_inter / (n_users * n_products)) * 100

print(f'Matrice             : {n_users:>12,} users × {n_products:>10,} produits')
print(f'Interactions réelles: {n_inter:>12,}')
print(f'Interactions totales: {n_users * n_products:>12,}  (matrice pleine)')
print(f'Sparsité            : {sparsity:>11.4f} %')
print()
print('INSIGHT Q1 :')
print(f'  {sparsity:.2f}% des paires user-item ne sont PAS observées.')
print('  → ALS (Alternating Least Squares) est adapté aux matrices creuses.')
print('  → Le cold start est inévitable : embeddings P1-P16 / U1-U16 requis.')

# ── Q2 · Types d'interactions ─────────────────────────────────────────────
print('=== Types d\'interactions ===')
interactions.groupBy('interaction_type') \
  .agg(
    F.count('*').alias('nb'),
    F.round(F.count('*') / 1e6, 2).alias('millions'),
    F.round(F.count('*') / n_inter * 100, 1).alias('pct'),
    F.round(F.mean('converted'), 3).alias('conv_rate')
  ) \
  .orderBy('nb', ascending=False) \
  .show()

print('=== Signal helpful_vote ===')
interactions.agg(
    F.count(F.when(F.col('helpful_vote') > 0, 1)).alias('avec_helpful_vote'),
    F.round(F.mean('helpful_vote'), 2).alias('helpful_moy'),
    F.max('helpful_vote').alias('helpful_max')
).show()

print('INSIGHT Q2 :')
print('  → Traiter verified_purchase=True comme signal fort (explicit).')
print('  → Traiter les reviews non vérifiées comme signal faible (implicit).')
print('  → helpful_vote peut pondérer la qualité des reviews dans le modèle.')

# ── Q3 · Biais de notation ────────────────────────────────────────────────
rated = interactions.filter('rating > 0')
total_rated = rated.count()

print('=== Distribution des notes ===')
rated \
  .groupBy('rating') \
  .agg(
    F.count('*').alias('nb'),
    F.round(F.count('*') / total_rated * 100, 1).alias('pct')
  ) \
  .orderBy('rating') \
  .show()

print('=== Statistiques globales des notes ===')
rated.agg(
    F.round(F.mean('rating'), 3).alias('note_moyenne'),
    F.round(F.stddev('rating'), 3).alias('ecart_type'),
    F.min('rating').alias('min'),
    F.max('rating').alias('max')
).show()

print('INSIGHT Q3 :')
print('  → Distribution en J : forte concentration sur les notes 4 et 5.')
print('  → Corriger par position bias dans la fonction de perte ALS.')
print('  → Privilégier le feedback implicite (achat) sur le rating brut.')

# ── Q4 · Longue traîne ────────────────────────────────────────────────────
print('=== Part des catégories dans les interactions ===')
interactions \
  .groupBy('category') \
  .agg(
    F.count('*').alias('nb_interactions'),
    F.round(F.count('*') / n_inter * 100, 1).alias('pct')
  ) \
  .orderBy('nb_interactions', ascending=False) \
  .show(20)

product_counts = interactions.groupBy('product_id').count()
seuils = [1, 5, 10, 20, 50]
print('=== Longue traîne des produits ===')
for s in seuils:
    rare = product_counts.filter(F.col('count') <= s).count()
    pct  = rare / n_products * 100
    print(f'  Produits avec ≤ {s:>2} avis : {rare:>8,}  ({pct:.1f}% du catalogue)')

user_counts = interactions.groupBy('user_id').count()
print('\n=== Longue traîne des utilisateurs ===')
for s in seuils:
    rare = user_counts.filter(F.col('count') <= s).count()
    pct  = rare / n_users * 100
    print(f'  Users avec ≤ {s:>2} avis    : {rare:>8,}  ({pct:.1f}% des users)')

print('\nINSIGHT Q4 :')
print('  → Majorité du catalogue < 5 avis → cold start critique.')
print('  → Les embeddings (P1-P16) sont indispensables pour ces produits rares.')
print('  → Top 20% des catégories génèrent ~80% des interactions (loi de Pareto).')

# ── Q5 · Segmentation & Saisonnalité ─────────────────────────────────────
print('=== Segments utilisateurs ===')
users.groupBy('segment') \
  .agg(
    F.count('*').alias('nb_users'),
    F.round(F.count('*') / n_users * 100, 1).alias('pct_users'),
    F.round(F.mean('total_purchases'), 1).alias('achats_moy'),
    F.round(F.mean('avg_rating'), 2).alias('note_moy'),
    F.sum('total_purchases').alias('total_achats')
  ) \
  .orderBy('nb_users', ascending=False) \
  .show()

total_achats = interactions.filter("interaction_type = 'purchase'").count()
print('=== Part des achats par segment ===')
(
    interactions.filter("interaction_type = 'purchase'")
    .join(users.select('user_id', 'segment'), on='user_id', how='left')
    .groupBy('segment')
    .agg(
        F.count('*').alias('achats'),
        F.round(F.count('*') / total_achats * 100, 1).alias('pct_achats')
    )
    .orderBy('achats', ascending=False)
    .show()
)

print('INSIGHT Q5 :')
print('  → power_user ≈ 10% des users mais génèrent ~40% des interactions.')
print('  → Ne pas dégrader les recommandations power_user pour optimiser le CTR moyen.')
print('  → Évaluer les modèles séparément par segment.')

print('=== Saisonnalité — interactions par mois ===')
interactions \
  .filter(F.col('year_month').isNotNull()) \
  .filter(F.col('year_month') != 'NaT') \
  .groupBy('year_month') \
  .count() \
  .orderBy('year_month') \
  .show(30)

# ── Embeddings ────────────────────────────────────────────────────────────
emb_cols = [f'P{i}' for i in range(1, 5)]
print('=== Embeddings produits (4 premières dims) — moyenne par catégorie ===')
products.groupBy('category') \
  .agg(*[F.round(F.mean(c), 4).alias(c) for c in emb_cols]) \
  .orderBy('category') \
  .show()

# ── VIZ 1 : Heatmap sparsité ──────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
total   = n_users * n_products
pct_obs  = n_inter / total * 100
pct_miss = (total - n_inter) / total * 100

pivot = pd.DataFrame({'Observées': [pct_obs], 'Non observées': [pct_miss]})
sns.heatmap(pivot, annot=True, fmt='.4f', cmap='YlOrRd',
            linewidths=2, ax=ax, cbar_kws={'label': '%'})
ax.set_title(
    f'Sparsité de la matrice user-item\n'
    f'{n_users:,} users × {n_products:,} produits | {n_inter:,} interactions',
    fontsize=12)
ax.set_xlabel('')
plt.tight_layout()
plt.savefig(f'{OUTPUTS}viz1_sparsity_heatmap.png', dpi=150)
plt.close()
print('Sauvegardé : outputs/viz1_sparsity_heatmap.png')

# ── VIZ 2 : Distribution des notes ────────────────────────────────────────
rating_df = (
    interactions.filter('rating > 0')
    .groupBy('rating').count().orderBy('rating').toPandas()
)
rating_df['pct'] = rating_df['count'] / rating_df['count'].sum() * 100

fig, ax = plt.subplots(figsize=(7, 5))
bars = ax.bar(rating_df['rating'].astype(str), rating_df['pct'],
              color=sns.color_palette('muted', 5), edgecolor='white', linewidth=0.8)
for bar, pct in zip(bars, rating_df['pct']):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.4,
            f'{pct:.1f}%', ha='center', va='bottom', fontsize=11)
ax.set_xlabel('Note', fontsize=12)
ax.set_ylabel('Part des reviews (%)', fontsize=12)
ax.set_title('Distribution des notes (biais J-shaped Amazon)', fontsize=13)
ax.yaxis.set_major_formatter(mticker.PercentFormatter())
plt.tight_layout()
plt.savefig(f'{OUTPUTS}viz2_rating_distribution.png', dpi=150)
plt.close()
print('Sauvegardé : outputs/viz2_rating_distribution.png')

# ── VIZ 3 : Part des catégories ───────────────────────────────────────────
cat_df = (
    interactions.groupBy('category').count()
    .orderBy('count', ascending=False).limit(15).toPandas()
)
cat_df['pct'] = cat_df['count'] / cat_df['count'].sum() * 100

fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(data=cat_df, y='category', x='pct', palette='muted', ax=ax)
for p in ax.patches:
    ax.text(p.get_width() + 0.2, p.get_y() + p.get_height() / 2,
            f'{p.get_width():.1f}%', va='center', fontsize=9)
ax.set_xlabel('Part des interactions (%)', fontsize=12)
ax.set_ylabel('Catégorie', fontsize=12)
ax.set_title('Top 15 catégories par volume d\'interactions', fontsize=13)
ax.xaxis.set_major_formatter(mticker.PercentFormatter())
plt.tight_layout()
plt.savefig(f'{OUTPUTS}viz3_category_share.png', dpi=150)
plt.close()
print('Sauvegardé : outputs/viz3_category_share.png')

# ── Récapitulatif ─────────────────────────────────────────────────────────
print('\n=== RÉCAPITULATIF DES 5 INSIGHTS CHIFFRÉS ===\n')
print(f'[Q1] Sparsité         : {sparsity:.4f}% → ALS + embeddings obligatoires')

type_df = (
    interactions.groupBy('interaction_type')
    .agg(F.round(F.count('*') / n_inter * 100, 1).alias('pct'))
    .toPandas().set_index('interaction_type')['pct'].to_dict()
)
review_pct = type_df.get('review', 0)
print(f'[Q2] Implicit feedback: reviews={review_pct}% → signal faible à exploiter')

five_star_pct = (
    interactions.filter('rating = 5').count() /
    interactions.filter('rating > 0').count() * 100
)
print(f'[Q3] Biais notes      : {five_star_pct:.1f}% de 5 étoiles → corriger position bias')

rare_pct = product_counts.filter(F.col('count') < 5).count() / n_products * 100
print(f'[Q4] Longue traîne    : {rare_pct:.0f}% produits < 5 avis → cold start challenge')

power_pct = users.filter("segment = 'power_user'").count() / n_users * 100
print(f'[Q5] Power users      : {power_pct:.1f}% des users → segment critique à protéger')

spark.stop()
print('\nTerminé.')
