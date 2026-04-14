"""
Téléchargement des datasets Amazon Reviews 2023 et 2022 complets via HuggingFace.
Source: https://amazon-reviews-2023.github.io/data_loading/huggingface.html
"""

import os
import datasets
from datasets import load_dataset

datasets.logging.set_verbosity_error()

# Répertoire de sauvegarde
SAVE_DIR = os.path.join(os.path.dirname(__file__), "raw")
os.makedirs(SAVE_DIR, exist_ok=True)

# =============================================================================
# CONFIGURATION — modifier uniquement cette variable
# -----------------------------------------------------------------------------
# Nombre de catégories à télécharger (les N premières par ordre alphabétique).
#   None  → toutes les catégories disponibles
#   6     → les 6 premières
#   9     → les 9 premières (jusqu'à CDs_and_Vinyl)
#   12    → les 12 premières
# =============================================================================

N_CATEGORIES = 9

# =============================================================================

# -------------------------------------------------------------------------
# Toutes les catégories disponibles pour Amazon Reviews 2023
# -------------------------------------------------------------------------
CATEGORIES_2023 = [
    "All_Beauty",
    "Amazon_Fashion",
    "Appliances",
    "Arts_Crafts_and_Sewing",
    "Automotive",
    "Baby_Products",
    "Beauty_and_Personal_Care",
    "Books",
    "CDs_and_Vinyl",
    "Cell_Phones_and_Accessories",
    "Clothing_Shoes_and_Jewelry",
    "Digital_Music",
    "Electronics",
    "Gift_Cards",
    "Grocery_and_Gourmet_Food",
    "Health_and_Household",
    "Home_and_Kitchen",
    "Industrial_and_Scientific",
    "Kindle_Store",
    "Luxury_Beauty",
    "Movies_and_TV",
    "Musical_Instruments",
    "Office_Products",
    "Patio_Lawn_and_Garden",
    "Pet_Supplies",
    "Software",
    "Sports_and_Outdoors",
    "Subscription_Boxes",
    "Tools_and_Home_Improvement",
    "Toys_and_Games",
    "Video_Games",
    "Unknown",
]

# -------------------------------------------------------------------------
# Toutes les catégories disponibles pour Amazon Reviews 2022
# -------------------------------------------------------------------------
CATEGORIES_2022 = [
    "All_Beauty",
    "Amazon_Fashion",
    "Appliances",
    "Arts_Crafts_and_Sewing",
    "Automotive",
    "Baby_Products",
    "Beauty_and_Personal_Care",
    "Books",
    "CDs_and_Vinyl",
    "Cell_Phones_and_Accessories",
    "Clothing_Shoes_and_Jewelry",
    "Digital_Music",
    "Electronics",
    "Gift_Cards",
    "Grocery_and_Gourmet_Food",
    "Health_and_Household",
    "Home_and_Kitchen",
    "Industrial_and_Scientific",
    "Kindle_Store",
    "Luxury_Beauty",
    "Movies_and_TV",
    "Musical_Instruments",
    "Office_Products",
    "Patio_Lawn_and_Garden",
    "Pet_Supplies",
    "Software",
    "Sports_and_Outdoors",
    "Tools_and_Home_Improvement",
    "Toys_and_Games",
    "Video_Games",
    "Unknown",
]


def download_2023_reviews(categories: list[str]) -> None:
    """Télécharge les reviews (raw_review_*) pour chaque catégorie 2023."""
    print("\n=== Amazon Reviews 2023 — Reviews ===")
    for category in categories:
        config_name = f"raw_review_{category}"
        save_path = os.path.join(SAVE_DIR, "2023", "reviews", category)
        if os.path.exists(save_path):
            print(f"  [SKIP] {category} (déjà téléchargé)")
            continue
        print(f"  Téléchargement reviews : {category} …", end=" ", flush=True)
        try:
            ds = load_dataset(
                "McAuley-Lab/Amazon-Reviews-2023",
                config_name,
                trust_remote_code=True,
            )
            ds.save_to_disk(save_path)
            print("OK")
        except Exception as exc:
            print(f"ERREUR — {exc}")


def download_2023_metadata(categories: list[str]) -> None:
    """Télécharge les métadonnées (raw_meta_*) pour chaque catégorie 2023."""
    print("\n=== Amazon Reviews 2023 — Métadonnées ===")
    for category in categories:
        config_name = f"raw_meta_{category}"
        save_path = os.path.join(SAVE_DIR, "2023", "metadata", category)
        if os.path.exists(save_path):
            print(f"  [SKIP] {category} (déjà téléchargé)")
            continue
        print(f"  Téléchargement metadata : {category} …", end=" ", flush=True)
        try:
            ds = load_dataset(
                "McAuley-Lab/Amazon-Reviews-2023",
                config_name,
                split="full",
            )
            ds.save_to_disk(save_path)
            print("OK")
        except Exception as exc:
            print(f"ERREUR — {exc}")


def download_2022_reviews(categories: list[str]) -> None:
    """Télécharge les reviews pour chaque catégorie 2022."""
    print("\n=== Amazon Reviews 2022 — Reviews ===")
    for category in categories:
        config_name = f"raw_review_{category}"
        save_path = os.path.join(SAVE_DIR, "2022", "reviews", category)
        if os.path.exists(save_path):
            print(f"  [SKIP] {category} (déjà téléchargé)")
            continue
        print(f"  Téléchargement reviews : {category} …", end=" ", flush=True)
        try:
            ds = load_dataset(
                "McAuley-Lab/Amazon-Reviews-2022",
                config_name,
                trust_remote_code=True,
            )
            ds.save_to_disk(save_path)
            print("OK")
        except Exception as exc:
            print(f"ERREUR — {exc}")


def download_2022_metadata(categories: list[str]) -> None:
    """Télécharge les métadonnées pour chaque catégorie 2022."""
    print("\n=== Amazon Reviews 2022 — Métadonnées ===")
    for category in categories:
        config_name = f"raw_meta_{category}"
        save_path = os.path.join(SAVE_DIR, "2022", "metadata", category)
        if os.path.exists(save_path):
            print(f"  [SKIP] {category} (déjà téléchargé)")
            continue
        print(f"  Téléchargement metadata : {category} …", end=" ", flush=True)
        try:
            ds = load_dataset(
                "McAuley-Lab/Amazon-Reviews-2022",
                config_name,
                split="full",
            )
            ds.save_to_disk(save_path)
            print("OK")
        except Exception as exc:
            print(f"ERREUR — {exc}")


if __name__ == "__main__":
    cats_2023 = CATEGORIES_2023[:N_CATEGORIES] if N_CATEGORIES else CATEGORIES_2023
    cats_2022 = CATEGORIES_2022[:N_CATEGORIES] if N_CATEGORIES else CATEGORIES_2022

    print(f"N_CATEGORIES         : {N_CATEGORIES if N_CATEGORIES else 'toutes'}")
    print(f"Catégories 2023      : {cats_2023}")
    print(f"Dossier de sauvegarde : {os.path.abspath(SAVE_DIR)}\n")

    # --- 2023 ---
    download_2023_reviews(cats_2023)
    download_2023_metadata(cats_2023)

    print("\nTéléchargement terminé.")
