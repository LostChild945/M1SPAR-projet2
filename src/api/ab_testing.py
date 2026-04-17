"""
A/B testing :
  - Attribution déterministe control/treatment via hash MD5 de l'user_id (50/50)
  - Calcul CTR + test Z bilatéral (scipy) pour détecter un lift significatif
"""
import hashlib
import os
from datetime import date

from scipy.stats import norm

import db
from schemas import ABResultsResponse

# Date de début de l'expérience (configurable via env var)
_start_str = os.environ.get("EXPERIMENT_START", "2026-04-14")
EXPERIMENT_START = date.fromisoformat(_start_str)


def get_variant(user_id: str) -> str:
    """Retourne 'control' ou 'treatment' de façon déterministe."""
    digest = int(hashlib.md5(user_id.encode()).hexdigest(), 16)
    return "control" if digest % 2 == 0 else "treatment"


def get_ab_results(experiment_id: str, segment: str | None) -> ABResultsResponse:
    stats = db.get_ab_stats(segment)

    n_c = stats["n_control"]
    n_t = stats["n_treatment"]
    clicks_c = stats["clicks_control"]
    clicks_t = stats["clicks_treatment"]

    ctr_c = clicks_c / n_c if n_c > 0 else 0.0
    ctr_t = clicks_t / n_t if n_t > 0 else 0.0
    lift = (ctr_t - ctr_c) / ctr_c * 100 if ctr_c > 0 else 0.0

    # Test Z bilatéral sur proportions (implémentation manuelle via scipy.stats.norm)
    p_value = 1.0
    if n_c > 0 and n_t > 0 and (clicks_c + clicks_t) > 0:
        p_pool = (clicks_c + clicks_t) / (n_c + n_t)
        se = (p_pool * (1 - p_pool) * (1 / n_c + 1 / n_t)) ** 0.5
        if se > 0:
            z = (ctr_t - ctr_c) / se
            p_value = float(2 * norm.sf(abs(z)))

    n_days = max((date.today() - EXPERIMENT_START).days + 1, 1)

    return ABResultsResponse(
        experiment_id=experiment_id,
        start_date=str(EXPERIMENT_START),
        n_days=n_days,
        control_ctr=round(ctr_c, 4),
        treatment_ctr=round(ctr_t, 4),
        lift_pct=round(lift, 2),
        p_value=round(p_value, 4),
        significant=p_value < 0.05,
        n_users_control=n_c,
        n_users_treatment=n_t,
    )
