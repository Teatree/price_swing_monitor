"""
Price Swing Monitor — Polymarket Sports Markets
Finds markets where a favored outcome's price crashed after the market ended.
"""

import os
import json
import time
import logging
from datetime import datetime, timedelta, timezone
from threading import Lock

import requests
from flask import Flask, render_template, jsonify, request as req

from db import init_db, insert_swing, get_all_swings, get_count

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Init DB on startup (no-op if DATABASE_URL not set)
with app.app_context():
    _db_available = init_db()

GAMMA = "https://gamma-api.polymarket.com"
CLOB = "https://clob.polymarket.com"
DELAY = 0.2  # seconds between CLOB API calls

# ---------------------------------------------------------------------------
# Sport → Polymarket series IDs (from /sports endpoint)
# ---------------------------------------------------------------------------
SPORT_SERIES = {
    "CS2":      [10310],
    "Dota2":    [10309],
    "Valorant": [10369],
    "LoL":      [10311],
    "Soccer":   [10188, 10193, 10204, 10189, 10203, 10194, 10195],
    "NFL":      [10187],
    "NBA":      [10345],
    "NHL":      [10346],
}

# ---------------------------------------------------------------------------
# In-memory log of qualified markets
# ---------------------------------------------------------------------------
market_log: list[dict] = []
seen_conditions: set[str] = set()
_log_lock = Lock()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse(val):
    """Parse a JSON-string field from Gamma API."""
    if isinstance(val, str):
        try:
            return json.loads(val)
        except (json.JSONDecodeError, TypeError):
            return val
    return val


def _iso_to_dt(s: str | None):
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        return None

# ---------------------------------------------------------------------------
# Gamma API
# ---------------------------------------------------------------------------

def fetch_events(series_id: int, *, closed: bool, cutoff_dt=None,
                  limit: int = 50) -> list[dict]:
    """Paginated fetch of events for one series_id.
    Stops early if all events are older than cutoff_dt."""
    out = []
    offset = 0
    while offset < 500:
        params = {
            "series_id": str(series_id),
            "closed": str(closed).lower(),
            "order": "endDate",
            "ascending": "false",
            "limit": limit,
            "offset": offset,
        }
        if not closed:
            params["active"] = "true"
        try:
            r = requests.get(f"{GAMMA}/events", params=params, timeout=15)
            r.raise_for_status()
            batch = r.json()
        except Exception as e:
            logger.warning("Gamma error series=%s: %s", series_id, e)
            break
        if not batch:
            break
        out.extend(batch)

        # Early exit: if last event in batch is older than cutoff, stop paging
        if cutoff_dt and batch:
            last_end = _iso_to_dt(batch[-1].get("endDate"))
            if last_end and last_end < cutoff_dt:
                break

        if len(batch) < limit:
            break
        offset += len(batch)
        time.sleep(DELAY)
    return out

# ---------------------------------------------------------------------------
# CLOB API helpers
# ---------------------------------------------------------------------------

def clob_price(token_id: str) -> float | None:
    """Current best-ask price via REST."""
    try:
        r = requests.get(f"{CLOB}/price",
                         params={"token_id": token_id, "side": "BUY"}, timeout=5)
        return float(r.json().get("price", 0))
    except Exception:
        return None


def clob_book_depth(token_id: str, levels: int = 5) -> float:
    """Total shares across top N ask levels."""
    try:
        r = requests.get(f"{CLOB}/book",
                         params={"token_id": token_id}, timeout=5)
        asks = r.json().get("asks", [])
        return sum(float(a.get("size", 0)) for a in asks[:levels])
    except Exception:
        return 0.0


def clob_pre_end_price(token_id: str, end_unix: int) -> float | None:
    """
    Price just before market end, from CLOB /prices-history.
    Uses explicit startTs/endTs (interval=max is buggy for resolved markets).
    """
    try:
        r = requests.get(f"{CLOB}/prices-history", params={
            "market": token_id,
            "startTs": end_unix - 86400,   # 24 h before end
            "endTs": end_unix,
            "fidelity": 10,                # 10-min data points
        }, timeout=10)
        if r.status_code != 200:
            return None
        history = r.json().get("history", [])
        if not history:
            return None
        return float(history[-1].get("p", 0))
    except Exception as e:
        logger.debug("prices-history error %s: %s", token_id, e)
        return None

# ---------------------------------------------------------------------------
# Core scan
# ---------------------------------------------------------------------------

def scan_markets(
    sports: list[str],
    hours: float = 3.0,
    min_volume: float = 10_000,
    min_shares: float = 0,
    start_price_min: float = 0.68,
    current_price_max: float = 0.40,
) -> list[dict]:
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(hours=hours)
    results: list[dict] = []
    seen_cids: set[str] = set()

    for sport in sports:
        series_ids = SPORT_SERIES.get(sport, [])
        for sid in series_ids:
            # Fetch both closed and still-active-but-ended events
            for closed in (True, False):
                events = fetch_events(sid, closed=closed, cutoff_dt=cutoff)
                logger.info("  %s series=%s closed=%s → %d events",
                            sport, sid, closed, len(events))
                total_markets = 0
                for event in events:
                    for market in event.get("markets") or []:
                        total_markets += 1
                        cid = market.get("conditionId", "")
                        if cid in seen_cids:
                            continue
                        seen_cids.add(cid)
                        r = _process_market(market, event, sport,
                                            now, cutoff,
                                            min_volume, min_shares,
                                            start_price_min, current_price_max)
                        if r:
                            results.append(r)
                logger.info("    → %d total markets in those events", total_markets)
    logger.info("Scan complete: %d qualifying markets", len(results))
    return results


def _process_market(market, event, sport, now, cutoff,
                    min_volume, min_shares, start_min, cur_max):
    """Return a result dict if this market qualifies, else None."""

    # --- end-date filter ---
    end_dt = _iso_to_dt(market.get("endDate"))
    if not end_dt or end_dt > now or end_dt < cutoff:
        return None

    # --- volume filter (cheap, no API call) ---
    try:
        vol = float(market.get("volumeNum") or 0)
    except (TypeError, ValueError):
        vol = 0
    if vol < min_volume:
        return None

    # --- parse outcomes / tokens ---
    outcomes = _parse(market.get("outcomes"))
    tokens   = _parse(market.get("clobTokenIds"))
    if not outcomes or not tokens or len(outcomes) != len(tokens):
        return None

    # --- skip player-prop style Yes/No (individual player markets) ---
    if set(outcomes) == {"Yes", "No"}:
        return None

    # --- Moneyline / team-win filter ---
    # Only keep markets where two (or three) teams compete to win.
    # These have "vs" in the question or are labelled "Match Winner".
    # Exclude O/U, Spread, Handicap, Game/Map/Set sub-markets, 1H, props.
    q = (market.get("question") or "").lower()
    EXCLUDE_KW = ["o/u", "over", "under", "spread", "handicap", "total",
                  "game 1", "game 2", "game 3", "game 4", "game 5",
                  "game 6", "game 7", "map 1", "map 2", "map 3",
                  "map 4", "map 5", "set 1", "set 2", "set 3",
                  "1h ", "rushing", "receiving", "touchdown",
                  "passing", "assists", "rebounds", "points scored"]
    if any(kw in q for kw in EXCLUDE_KW):
        return None
    if "vs" not in q and "match winner" not in q:
        return None

    # --- current prices (from Gamma outcomePrices first, CLOB fallback) ---
    gamma_prices = _parse(market.get("outcomePrices"))
    current: list[float | None] = []
    if gamma_prices and len(gamma_prices) == len(outcomes):
        current = [float(p) for p in gamma_prices]
    else:
        for tok in tokens:
            current.append(clob_price(tok))
            time.sleep(DELAY)

    # --- pre-end prices from CLOB history ---
    end_unix = int(end_dt.timestamp())
    pre_end: list[float | None] = []

    # Optimisation: for 2-outcome markets, fetch only first token
    if len(tokens) == 2:
        p0 = clob_pre_end_price(tokens[0], end_unix)
        time.sleep(DELAY)
        pre_end = [p0, round(1 - p0, 4) if p0 is not None else None]
    else:
        for tok in tokens:
            pre_end.append(clob_pre_end_price(tok, end_unix))
            time.sleep(DELAY)

    # Fallback: if history unavailable but market is resolved (closed),
    # we know the current gamma prices ARE the resolved prices (0/1).
    # Invert them to estimate pre-end: the loser was likely the favorite.
    if all(p is None for p in pre_end) and gamma_prices:
        gp = [float(p) for p in gamma_prices]
        if market.get("closed"):
            # Resolved: current gamma = resolution (1/0).
            # Can't reliably recover pre-end price → skip.
            pass
        else:
            # Active but ended: gamma prices are still live trading prices.
            # We don't have history, so pre-end is unknown → skip.
            pass

    # --- build outcome list ---
    outcome_data = []
    for i, (name, tok) in enumerate(zip(outcomes, tokens)):
        cur = current[i] if i < len(current) else None
        pre = pre_end[i] if i < len(pre_end) else None
        pct = None
        if pre and pre > 0 and cur is not None:
            pct = round(((cur - pre) / pre) * 100, 1)
        outcome_data.append({
            "name": name,
            "token_id": tok,
            "current_price": cur,
            "pre_end_price": pre,
            "pct_change": pct,
        })

    # --- price-swing filters ---
    # Highest pre-end price must exceed start_min
    pre_vals = [o["pre_end_price"] for o in outcome_data if o["pre_end_price"] is not None]
    if not pre_vals:
        return None  # can't evaluate without history
    max_pre = max(pre_vals)
    if max_pre < start_min:
        return None

    # The previously-favored outcome's current price must be below cur_max
    fav_idx = next(i for i, o in enumerate(outcome_data) if o["pre_end_price"] == max_pre)
    fav_cur = outcome_data[fav_idx]["current_price"]
    if fav_cur is None or fav_cur >= cur_max:
        return None

    # --- depth filter (only for qualifying markets) ---
    if min_shares > 0:
        for o in outcome_data:
            d = clob_book_depth(o["token_id"])
            o["shares_available"] = d
            time.sleep(DELAY)
            if d < min_shares:
                return None

    # --- build result ---
    event_slug = event.get("slug", "")
    result = {
        "event_title": event.get("title", ""),
        "market_question": market.get("question", ""),
        "condition_id": market.get("conditionId", ""),
        "sport": sport,
        "end_date": market.get("endDate"),
        "end_dt_unix": end_unix,
        "polymarket_url": f"https://polymarket.com/event/{event_slug}" if event_slug else "",
        "volume": vol,
        "outcomes": outcome_data,
        "closed": bool(market.get("closed")),
    }

    _log_if_new(result)
    return result

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _log_if_new(result: dict):
    cid = result.get("condition_id", "")
    if not cid:
        return

    # Write to Postgres (upserts — safe to call repeatedly)
    insert_swing(result)

    # Also keep in-memory copy
    with _log_lock:
        if cid in seen_conditions:
            return
        seen_conditions.add(cid)
        entry = {
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
            "market_name": result["market_question"],
            "sport": result["sport"],
            "volume": result["volume"],
        }
        for i, o in enumerate(result["outcomes"]):
            label = o["name"]
            entry[f"outcome_{i+1}_name"] = label
            entry[f"outcome_{i+1}_pre"] = o["pre_end_price"]
            entry[f"outcome_{i+1}_cur"] = o["current_price"]
        market_log.append(entry)


# ---------------------------------------------------------------------------
# Flask routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/scan")
def api_scan():
    sports = [s for s in req.args.get("sports", "CS2,Dota2,Valorant,LoL,Soccer,NFL,NBA,NHL").split(",") if s]
    hours           = float(req.args.get("hours", 3))
    min_volume      = float(req.args.get("min_volume", 10000))
    min_shares      = float(req.args.get("min_shares", 0))
    start_price_min = float(req.args.get("start_price_min", 68)) / 100
    current_price_max = float(req.args.get("current_price_max", 40)) / 100

    t0 = time.time()
    results = scan_markets(sports, hours, min_volume, min_shares,
                           start_price_min, current_price_max)
    elapsed = round(time.time() - t0, 1)

    # Sort: biggest swing first (most negative pct_change)
    results.sort(key=lambda r: min(
        (o["pct_change"] for o in r["outcomes"] if o["pct_change"] is not None),
        default=0,
    ))

    return jsonify({
        "results": results,
        "count": len(results),
        "scanned_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_s": elapsed,
    })


@app.route("/api/debug")
def api_debug():
    """Show raw recently-ended moneyline markets before price filters apply."""
    sports = [s for s in req.args.get("sports", "CS2,Dota2,Valorant,LoL,Soccer,NFL,NBA,NHL").split(",") if s]
    hours = float(req.args.get("hours", 3))

    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(hours=hours)
    found = []

    for sport in sports:
        for sid in SPORT_SERIES.get(sport, []):
            for closed in (True, False):
                events = fetch_events(sid, closed=closed, cutoff_dt=cutoff)
                for event in events:
                    for mkt in event.get("markets") or []:
                        end_dt = _iso_to_dt(mkt.get("endDate"))
                        if not end_dt or end_dt > now or end_dt < cutoff:
                            continue
                        q = (mkt.get("question") or "")
                        outcomes = _parse(mkt.get("outcomes"))
                        gamma_prices = _parse(mkt.get("outcomePrices"))
                        vol = float(mkt.get("volumeNum") or 0)
                        found.append({
                            "sport": sport,
                            "question": q,
                            "outcomes": outcomes,
                            "gamma_prices": gamma_prices,
                            "volume": vol,
                            "closed": mkt.get("closed"),
                            "end_date": mkt.get("endDate"),
                            "is_moneyline": "vs" in q.lower() or "match winner" in q.lower(),
                        })

    return jsonify({"count": len(found), "markets": found})


@app.route("/api/log")
def api_log():
    # Prefer Postgres, fall back to in-memory
    if _db_available:
        sport = req.args.get("sport")
        limit = int(req.args.get("limit", 200))
        rows = get_all_swings(limit=limit, sport=sport)
        # Serialize datetimes
        for r in rows:
            for k, v in r.items():
                if isinstance(v, datetime):
                    r[k] = v.strftime("%Y-%m-%d %H:%M:%S UTC")
        return jsonify({"entries": rows, "total": get_count(), "source": "postgres"})
    with _log_lock:
        return jsonify({"entries": list(market_log), "total": len(market_log), "source": "memory"})


@app.route("/api/db-test")
def api_db_test():
    """Quick check that Postgres is connected and working."""
    if not _db_available:
        return jsonify({"ok": False, "error": "DATABASE_URL not set"}), 503
    try:
        count = get_count()
        return jsonify({"ok": True, "row_count": count})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
