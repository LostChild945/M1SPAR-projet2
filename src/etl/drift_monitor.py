"""
drift_monitor.py
----------------
Détection de drift sur le taux de conversion par mois.

Méthode : test de Kolmogorov-Smirnov (KS) entre chaque paire de mois
consécutifs sur la colonne `converted`.

Output :
  data/drift_report.parquet   (month_a, month_b, ks_statistic, p_value, drift_detected)

Usage :
    python -m src.etl.drift_monitor
"""

import os

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

BASE = os.environ.get("DATA_DIR", "./analyse/data")


def compute_drift(base=BASE) -> pd.DataFrame:
    """Calcule le test KS sur `converted` entre mois consécutifs."""
    inter_path = os.path.join(base, "interactions")

    print("=" * 60)
    print("DRIFT MONITOR — Test KS par mois")
    print("=" * 60)

    df = pd.read_parquet(inter_path, columns=["year_month", "converted"])
    df = df.dropna(subset=["year_month"])
    df = df[df["year_month"] != "NaT"]

    months = sorted(df["year_month"].unique())
    print(f"  Mois disponibles : {len(months)}")

    results = []
    for i in range(len(months) - 1):
        m_a, m_b = months[i], months[i + 1]
        a = df.loc[df["year_month"] == m_a, "converted"].values
        b = df.loc[df["year_month"] == m_b, "converted"].values

        if len(a) < 10 or len(b) < 10:
            continue

        stat, p_val = ks_2samp(a, b)
        results.append({
            "month_a": m_a,
            "month_b": m_b,
            "n_a": len(a),
            "n_b": len(b),
            "conv_rate_a": float(np.mean(a)),
            "conv_rate_b": float(np.mean(b)),
            "ks_statistic": float(stat),
            "p_value": float(p_val),
            "drift_detected": p_val < 0.05,
        })

    report = pd.DataFrame(results)

    if not report.empty:
        out_path = os.path.join(base, "drift_report.parquet")
        report.to_parquet(out_path, index=False)
        print(f"\n  Rapport sauvegardé : {out_path}")

        n_drift = report["drift_detected"].sum()
        print(f"  Paires de mois : {len(report)}")
        print(f"  Drift détecté  : {n_drift} paire(s)")

        print("\n  Détails :")
        for _, row in report.iterrows():
            flag = "⚠ DRIFT" if row["drift_detected"] else "  OK"
            print(
                f"    {row['month_a']} → {row['month_b']}  "
                f"KS={row['ks_statistic']:.4f}  p={row['p_value']:.4f}  "
                f"conv={row['conv_rate_a']:.3f}→{row['conv_rate_b']:.3f}  {flag}"
            )
    else:
        print("  Pas assez de données pour le test KS.")

    print("\nDrift monitoring terminé.")
    return report


if __name__ == "__main__":
    compute_drift()
