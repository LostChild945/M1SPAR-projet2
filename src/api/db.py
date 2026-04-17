"""
Persistance SQLite pour le feedback utilisateur et les impressions A/B.
Utilise des connexions thread-local pour la compatibilité avec uvicorn.
"""
import os
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(os.environ.get("DB_PATH", "/app/api_data/feedback.db"))
_local = threading.local()

_DDL = """
CREATE TABLE IF NOT EXISTS feedback (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id          TEXT NOT NULL,
    product_id       TEXT NOT NULL,
    interaction_type TEXT NOT NULL,
    ab_variant       TEXT NOT NULL,
    ts               TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS impressions (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id    TEXT NOT NULL,
    ab_variant TEXT NOT NULL,
    ts         TEXT NOT NULL
);
"""


def _conn() -> sqlite3.Connection:
    if not hasattr(_local, "conn"):
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.executescript(_DDL)
        conn.commit()
        _local.conn = conn
    return _local.conn


def insert_feedback(
    user_id: str,
    product_id: str,
    interaction_type: str,
    ab_variant: str,
    ts: datetime,
) -> None:
    _conn().execute(
        "INSERT INTO feedback (user_id, product_id, interaction_type, ab_variant, ts) "
        "VALUES (?, ?, ?, ?, ?)",
        (user_id, product_id, interaction_type, ab_variant, ts.isoformat()),
    )
    _conn().commit()


def insert_impression(user_id: str, ab_variant: str) -> None:
    _conn().execute(
        "INSERT INTO impressions (user_id, ab_variant, ts) VALUES (?, ?, ?)",
        (user_id, ab_variant, datetime.now(timezone.utc).isoformat()),
    )
    _conn().commit()


def get_ab_stats(segment: str | None = None) -> dict:
    """
    Retourne les compteurs bruts pour le calcul du CTR par variant.
    `segment` est réservé pour une extension future (filtrage par segment user).
    """
    conn = _conn()

    imp_rows = conn.execute(
        "SELECT ab_variant, COUNT(DISTINCT user_id) AS n FROM impressions GROUP BY ab_variant"
    ).fetchall()
    impressions = {r["ab_variant"]: r["n"] for r in imp_rows}

    fb_rows = conn.execute(
        "SELECT ab_variant, COUNT(*) AS n FROM feedback "
        "WHERE interaction_type IN ('click', 'purchase') GROUP BY ab_variant"
    ).fetchall()
    interactions = {r["ab_variant"]: r["n"] for r in fb_rows}

    return {
        "n_control":          impressions.get("control", 0),
        "n_treatment":        impressions.get("treatment", 0),
        "clicks_control":     interactions.get("control", 0),
        "clicks_treatment":   interactions.get("treatment", 0),
    }
