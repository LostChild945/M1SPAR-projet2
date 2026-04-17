"""Tests pour le module cache."""

import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "api"))


def test_cache_graceful_degradation_on_get():
    """cache.get retourne None si Redis est indisponible."""
    import cache
    cache._client = None

    result = cache.get("some_key")
    assert result is None


def test_cache_graceful_degradation_on_set():
    """cache.set ne lève pas d'exception si Redis est indisponible."""
    import cache
    cache._client = None

    cache.set("some_key", {"data": "test"})
