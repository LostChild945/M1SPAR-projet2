"""
Cache Redis optionnel.
Si Redis est indisponible au démarrage ou en cours d'exécution,
toutes les opérations dégradent silencieusement (pas de cache).
"""
import json
import os
from typing import Any

import redis as redis_lib

_REDIS_HOST = os.environ.get("REDIS_HOST", "localhost")
_REDIS_PORT = int(os.environ.get("REDIS_PORT", 6379))
_client = None          # None = pas encore testé, False = indisponible


def _get() -> redis_lib.Redis | None:
    global _client
    if _client is None:
        try:
            r = redis_lib.Redis(
                host=_REDIS_HOST,
                port=_REDIS_PORT,
                decode_responses=True,
                socket_connect_timeout=1,
            )
            r.ping()
            _client = r
            print(f"Redis connecté ({_REDIS_HOST}:{_REDIS_PORT})")
        except Exception as e:
            _client = False
            print(f"Redis indisponible ({e}) — cache désactivé")
    return _client if _client else None


def get(key: str) -> Any | None:
    r = _get()
    if not r:
        return None
    try:
        val = r.get(key)
        return json.loads(val) if val else None
    except Exception:
        return None


def set(key: str, value: Any, ttl: int = 300) -> None:
    r = _get()
    if not r:
        return
    try:
        r.setex(key, ttl, json.dumps(value))
    except Exception:
        pass
