"""
bronze_ingestion.py
-------------------
Couche Bronze : ingestion brute des fichiers Arrow vers Parquet.

Délègue le traitement à analyse/00_prepare_data.py qui convertit
catégorie par catégorie (une seule en RAM à la fois).

Output :
  data/interactions/   (partitionné par category)
  data/products/       (un .parquet par catégorie)
  data/users/          (users.parquet agrégé)

Usage :
    python -m src.etl.bronze_ingestion
    # ou directement : python analyse/00_prepare_data.py
"""

import subprocess
import sys
from pathlib import Path

ANALYSE_DIR = Path(__file__).resolve().parents[2] / "analyse"


def run_bronze():
    """Lance le script de préparation des données brutes (couche Bronze)."""
    script = ANALYSE_DIR / "00_prepare_data.py"
    if not script.exists():
        raise FileNotFoundError(f"Script introuvable : {script}")

    print("=" * 60)
    print("BRONZE — Ingestion Arrow → Parquet")
    print("=" * 60)
    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=str(ANALYSE_DIR),
    )
    if result.returncode != 0:
        raise RuntimeError(f"Bronze ingestion échouée (code {result.returncode})")
    print("\nBronze terminé.\n")


if __name__ == "__main__":
    run_bronze()
