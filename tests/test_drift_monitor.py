"""Tests pour le drift monitor."""

import os
import tempfile

import numpy as np
import pandas as pd
import pytest

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "etl"))


def test_ks_detects_drift():
    """Le test KS détecte un changement significatif de distribution."""
    from scipy.stats import ks_2samp

    rng = np.random.default_rng(42)
    # Mois A : conversion ~20%
    a = rng.binomial(1, 0.20, size=5000)
    # Mois B : conversion ~40% (drift)
    b = rng.binomial(1, 0.40, size=5000)

    stat, p_value = ks_2samp(a, b)
    assert p_value < 0.05
    assert stat > 0.1


def test_ks_no_drift():
    """Le test KS ne détecte pas de drift quand les distributions sont stables."""
    from scipy.stats import ks_2samp

    rng = np.random.default_rng(42)
    a = rng.binomial(1, 0.20, size=5000)
    b = rng.binomial(1, 0.20, size=5000)

    stat, p_value = ks_2samp(a, b)
    assert p_value > 0.05
