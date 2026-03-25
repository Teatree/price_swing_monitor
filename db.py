"""
PostgreSQL storage for Price Swing Monitor.
Stores every qualifying market with pre-end and current prices.
Falls back to in-memory-only if DATABASE_URL is not set.
"""

import os
import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

_conn = None


def _get_conn():
    """Lazy-init a single persistent connection."""
    global _conn
    if _conn is not None:
        try:
            _conn.cursor().execute("SELECT 1")
            return _conn
        except Exception:
            _conn = None

    url = os.environ.get("DATABASE_URL")
    if not url:
        return None

    import psycopg2
    _conn = psycopg2.connect(url, connect_timeout=5)
    _conn.autocommit = True
    return _conn


def init_db():
    """Create the table if it doesn't exist. Safe to call on every startup."""
    conn = _get_conn()
    if not conn:
        logger.info("DATABASE_URL not set — using in-memory log only")
        return False

    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS price_swings (
                id              SERIAL PRIMARY KEY,
                recorded_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                sport           TEXT NOT NULL,
                event_title     TEXT,
                market_question TEXT NOT NULL,
                condition_id    TEXT NOT NULL,
                polymarket_url  TEXT,
                volume          REAL,
                end_date        TIMESTAMPTZ,
                market_closed   BOOLEAN,
                outcome_1_name  TEXT,
                outcome_1_pre   REAL,
                outcome_1_cur   REAL,
                outcome_2_name  TEXT,
                outcome_2_pre   REAL,
                outcome_2_cur   REAL,
                outcome_3_name  TEXT,
                outcome_3_pre   REAL,
                outcome_3_cur   REAL
            );
        """)
        # Index for common queries
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_ps_sport      ON price_swings (sport);
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_ps_recorded    ON price_swings (recorded_at);
        """)
        cur.execute("""
            CREATE UNIQUE INDEX IF NOT EXISTS idx_ps_condition
                ON price_swings (condition_id);
        """)
    logger.info("PostgreSQL connected — price_swings table ready")
    return True


def insert_swing(result: dict) -> bool:
    """Insert a qualifying market. Returns True on success, False if skipped/error."""
    conn = _get_conn()
    if not conn:
        return False

    outcomes = result.get("outcomes", [])

    def _o(idx, field):
        if idx < len(outcomes):
            return outcomes[idx].get(field)
        return None

    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO price_swings (
                    sport, event_title, market_question, condition_id,
                    polymarket_url, volume, end_date, market_closed,
                    outcome_1_name, outcome_1_pre, outcome_1_cur,
                    outcome_2_name, outcome_2_pre, outcome_2_cur,
                    outcome_3_name, outcome_3_pre, outcome_3_cur
                ) VALUES (
                    %s, %s, %s, %s,
                    %s, %s, %s, %s,
                    %s, %s, %s,
                    %s, %s, %s,
                    %s, %s, %s
                )
                ON CONFLICT (condition_id) DO UPDATE SET
                    outcome_1_cur = EXCLUDED.outcome_1_cur,
                    outcome_2_cur = EXCLUDED.outcome_2_cur,
                    outcome_3_cur = EXCLUDED.outcome_3_cur,
                    recorded_at   = NOW()
            """, (
                result.get("sport"),
                result.get("event_title"),
                result.get("market_question"),
                result.get("condition_id"),
                result.get("polymarket_url"),
                result.get("volume"),
                result.get("end_date"),
                result.get("closed"),
                _o(0, "name"), _o(0, "pre_end_price"), _o(0, "current_price"),
                _o(1, "name"), _o(1, "pre_end_price"), _o(1, "current_price"),
                _o(2, "name"), _o(2, "pre_end_price"), _o(2, "current_price"),
            ))
        return True
    except Exception as e:
        logger.error("DB insert error: %s", e)
        return False


def get_all_swings(limit: int = 200, sport: str | None = None) -> list[dict]:
    """Fetch logged swings, newest first."""
    conn = _get_conn()
    if not conn:
        return []

    where = ""
    params: list = []
    if sport:
        where = "WHERE sport = %s"
        params.append(sport)

    try:
        with conn.cursor() as cur:
            cur.execute(f"""
                SELECT recorded_at, sport, event_title, market_question,
                       condition_id, polymarket_url, volume, end_date,
                       market_closed,
                       outcome_1_name, outcome_1_pre, outcome_1_cur,
                       outcome_2_name, outcome_2_pre, outcome_2_cur,
                       outcome_3_name, outcome_3_pre, outcome_3_cur
                FROM price_swings
                {where}
                ORDER BY recorded_at DESC
                LIMIT %s
            """, params + [limit])
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, row)) for row in cur.fetchall()]
    except Exception as e:
        logger.error("DB query error: %s", e)
        return []


def get_count() -> int:
    conn = _get_conn()
    if not conn:
        return 0
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM price_swings")
            return cur.fetchone()[0]
    except Exception:
        return 0
