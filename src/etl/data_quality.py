"""
data_quality.py
---------------
Validation des données avec Great Expectations (API programmatique).

Checks :
  - rating ∈ [0, 5]
  - conversion_rate ∈ [0.17, 0.23] (17-23% global)
  - user_id et product_id non null
  - nb de lignes > 0

Usage :
    python -m src.etl.data_quality
"""

import os
import sys

import pandas as pd
import great_expectations as gx

BASE = os.environ.get("DATA_DIR", "./analyse/data")


def validate_interactions(df: pd.DataFrame) -> dict:
    """Valide la table interactions avec Great Expectations."""
    context = gx.get_context()
    ds = context.sources.pandas_default
    asset = ds.add_dataframe_asset(name="interactions")
    batch_request = asset.build_batch_request(dataframe=df)

    expectations = [
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="rating", min_value=0, max_value=5,
        ),
        gx.expectations.ExpectColumnValuesToNotBeNull(column="user_id"),
        gx.expectations.ExpectColumnValuesToNotBeNull(column="product_id"),
        gx.expectations.ExpectTableRowCountToBeGreaterThan(value=0),
    ]

    suite = context.add_expectation_suite(expectation_suite_name="interactions_suite")
    for exp in expectations:
        suite.add_expectation(exp)

    validator = context.get_validator(
        batch_request=batch_request,
        expectation_suite_name="interactions_suite",
    )
    results = validator.validate()

    return results


def validate_gold_products(df: pd.DataFrame) -> dict:
    """Valide la table Gold products."""
    context = gx.get_context()
    ds = context.sources.pandas_default
    asset = ds.add_dataframe_asset(name="gold_products")
    batch_request = asset.build_batch_request(dataframe=df)

    mean_conv = df["conversion_rate"].mean() if "conversion_rate" in df.columns else 0.0

    expectations = [
        gx.expectations.ExpectColumnValuesToNotBeNull(column="product_id"),
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="conversion_rate", min_value=0.0, max_value=1.0,
        ),
        gx.expectations.ExpectTableRowCountToBeGreaterThan(value=0),
    ]

    suite = context.add_expectation_suite(expectation_suite_name="gold_products_suite")
    for exp in expectations:
        suite.add_expectation(exp)

    validator = context.get_validator(
        batch_request=batch_request,
        expectation_suite_name="gold_products_suite",
    )
    results = validator.validate()

    # Check global conversion rate separately
    print(f"  Conversion rate moyenne : {mean_conv:.3f}")
    if 0.17 <= mean_conv <= 0.23:
        print("  ✓ Conversion rate dans la plage attendue [17%, 23%]")
    else:
        print(f"  ✗ Conversion rate hors plage : {mean_conv:.3f} (attendu [0.17, 0.23])")

    return results


def run_quality_checks():
    """Lance toutes les validations."""
    print("=" * 60)
    print("DATA QUALITY — Great Expectations")
    print("=" * 60)

    # Interactions Bronze
    print("\n[1/2] Validation interactions …")
    inter_path = os.path.join(BASE, "interactions")
    if os.path.exists(inter_path):
        df_inter = pd.read_parquet(inter_path)
        res = validate_interactions(df_inter)
        passed = sum(1 for r in res.results if r.success)
        total = len(res.results)
        print(f"  Résultat : {passed}/{total} checks passés")
    else:
        print("  ⚠ Dossier interactions introuvable, skip")

    # Gold products
    print("\n[2/2] Validation Gold products …")
    gold_path = os.path.join(BASE, "gold", "products")
    if os.path.exists(gold_path):
        df_gold = pd.read_parquet(gold_path)
        res = validate_gold_products(df_gold)
        passed = sum(1 for r in res.results if r.success)
        total = len(res.results)
        print(f"  Résultat : {passed}/{total} checks passés")
    else:
        print("  ⚠ Dossier Gold products introuvable, skip")

    print("\nValidation terminée.")


if __name__ == "__main__":
    run_quality_checks()
