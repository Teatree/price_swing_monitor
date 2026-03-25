"""
Price Swing Monitor — Polymarket Sports Markets
Shows LIVE sports moneyline markets and tracks price swings during the match.

Live detection (Option C — "breakout from stability"):
  1. Fetch the last 8h of 5-minute price history for the first token
  2. Walk backwards to find the last "stable" plateau: a stretch of >=6
     consecutive points (≥30 min) where all prices are within ±2c of each other
  3. The stable price = median of that plateau = pre-match line
  4. If the current price has moved >=5c from the stable price, the match is LIVE
  5. Pre-match markets (price still near the plateau) are skipped
"""

import os
import json
import time
import logging
from datetime import datetime, timedelta, timezone
from threading import Lock
from statistics import median

import requests
from flask import Flask, render_template, jsonify, request as req

from db import init_db, insert_swing, get_all_swings, get_count

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

with app.app_context():
    _db_available = init_db()

GAMMA = "https://gamma-api.polymarket.com"
CLOB = "https://clob.polymarket.com"
DELAY = 0.2

# Live detection parameters
STABLE_BAND_CENTS = 2      # max ±2c variation to count as "stable"
STABLE_MIN_POINTS = 6      # minimum 6 data points (~30 min at 5-min fidelity)
BREAKOUT_CENTS = 5          # current price must be >=5c away from stable price

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
# In-memory log
# ---------------------------------------------------------------------------
market_log: list[dict] = []
seen_conditions: set[str] = set()
_log_lock = Lock()
MAX_MEMORY_LOG = 500  # cap in-memory log to prevent unbounded growth

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse(val):
    if isinstance(val, str):
        try:
            return json.loads(val)
        except (json.JSONDecodeError, TypeError):
            return val
    return val


def _parse_dt(s: str | None) -> datetime | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        pass
    try:
        return datetime.strptime(s, "%Y-%m-%d %H:%M:%S%z")
    except Exception:
        pass
    try:
        return datetime.strptime(s + "+00:00", "%Y-%m-%d %H:%M:%S%z")
    except Exception:
        return None

# ---------------------------------------------------------------------------
# Gamma API
# ---------------------------------------------------------------------------

def fetch_active_events(series_id: int, limit: int = 50) -> list[dict]:
    out = []
    offset = 0
    while offset < 500:
        params = {
            "series_id": str(series_id),
            "active": "true",
            "closed": "false",
            "order": "endDate",
            "ascending": "false",
            "limit": limit,
            "offset": offset,
        }
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
        if len(batch) < limit:
            break
        offset += len(batch)
        time.sleep(DELAY)
    return out

# ---------------------------------------------------------------------------
# CLOB API helpers
# ---------------------------------------------------------------------------

def clob_price(token_id: str) -> float | None:
    try:
        r = requests.get(f"{CLOB}/price",
                         params={"token_id": token_id, "side": "BUY"}, timeout=5)
        return float(r.json().get("price", 0))
    except Exception:
        return None


def clob_book_depth(token_id: str, levels: int = 5) -> float:
    try:
        r = requests.get(f"{CLOB}/book",
                         params={"token_id": token_id}, timeout=5)
        asks = r.json().get("asks", [])
        return sum(float(a.get("size", 0)) for a in asks[:levels])
    except Exception:
        return 0.0


def clob_price_history(token_id: str, now_unix: int) -> list[float]:
    """Fetch last 8 hours of price data at 5-minute intervals.
    Returns list of prices (floats), newest last."""
    try:
        r = requests.get(f"{CLOB}/prices-history", params={
            "market": token_id,
            "startTs": now_unix - 28800,   # 8h ago
            "endTs":   now_unix,
            "fidelity": 5,                 # 5-min data points
        }, timeout=10)
        if r.status_code != 200:
            return []
        history = r.json().get("history", [])
        return [float(pt.get("p", 0)) for pt in history]
    except Exception as e:
        logger.debug("prices-history error %s: %s", token_id, e)
        return []


def find_stable_price(prices: list[float]) -> float | None:
    """Walk backwards through price history to find the last stable plateau.

    A plateau is STABLE_MIN_POINTS (6) consecutive prices where
    max - min <= STABLE_BAND_CENTS/100. Returns the median of that plateau,
    or None if no stable region is found.
    """
    if len(prices) < STABLE_MIN_POINTS:
        return None

    band = STABLE_BAND_CENTS / 100

    # Walk backwards from the end of the history
    # We want the LAST stable region before the current volatility
    # Start from the end and try to find where volatility begins,
    # then look for stability before that.
    for end in range(len(prices) - 1, STABLE_MIN_POINTS - 2, -1):
        # Check if [end - STABLE_MIN_POINTS + 1 .. end] is stable
        start = end - STABLE_MIN_POINTS + 1
        window = prices[start:end + 1]
        if max(window) - min(window) <= band:
            # Found a stable region — now extend it backwards as far as possible
            while start > 0 and max(prices[start - 1:end + 1]) - min(prices[start - 1:end + 1]) <= band:
                start -= 1
            plateau = prices[start:end + 1]
            return round(median(plateau), 4)

    return None


def detect_live(prices: list[float], current_price: float):
    """Determine if a market is live using the breakout-from-stability method.

    Returns (is_live: bool, stable_price: float|None).
    - stable_price is the pre-match line (median of last stable plateau)
    - is_live is True if current price has broken out >=5c from stable
    """
    stable = find_stable_price(prices)
    if stable is None:
        return False, None

    movement = abs(current_price - stable)
    is_live = movement >= (BREAKOUT_CENTS / 100)
    return is_live, stable

# ---------------------------------------------------------------------------
# Moneyline detection
# ---------------------------------------------------------------------------

EXCLUDE_KW = [
    "o/u", "over", "under", "spread", "handicap", "total",
    "game 1", "game 2", "game 3", "game 4", "game 5",
    "game 6", "game 7", "map 1", "map 2", "map 3",
    "map 4", "map 5", "set 1", "set 2", "set 3",
    "1h ", "rushing", "receiving", "touchdown",
    "passing", "assists", "rebounds", "points scored",
    "odd/even", "odd or even",
]


def _is_moneyline(question: str, outcomes: list) -> bool:
    if set(outcomes) == {"Yes", "No"}:
        return False
    q = question.lower()
    if any(kw in q for kw in EXCLUDE_KW):
        return False
    return "vs" in q or "match winner" in q

# ---------------------------------------------------------------------------
# Core scan
# ---------------------------------------------------------------------------

def scan_markets(
    sports: list[str],
    min_volume: float = 10_000,
    min_shares: float = 0,
    start_price_min: float = 0.68,
    current_price_max: float = 0.40,
) -> list[dict]:
    """Find live moneyline markets using breakout-from-stability detection."""
    now = datetime.now(timezone.utc)
    now_unix = int(now.timestamp())
    results: list[dict] = []
    seen_cids: set[str] = set()

    for sport in sports:
        series_ids = SPORT_SERIES.get(sport, [])
        for sid in series_ids:
            events = fetch_active_events(sid)
            logger.info("  %s series=%s → %d active events", sport, sid, len(events))
            mkt_count = 0
            live_count = 0
            for event in events:
                for market in event.get("markets") or []:
                    mkt_count += 1
                    cid = market.get("conditionId", "")
                    if cid in seen_cids:
                        continue
                    seen_cids.add(cid)
                    r = _process_market(market, event, sport, now, now_unix,
                                        min_volume, min_shares,
                                        start_price_min, current_price_max)
                    if r:
                        live_count += 1
                        results.append(r)
            logger.info("    → %d markets checked, %d live", mkt_count, live_count)
    logger.info("Scan complete: %d live markets", len(results))
    return results


def _process_market(market, event, sport, now, now_unix,
                    min_volume, min_shares, start_min, cur_max):
    """Return a result dict for live moneyline markets.
    Uses breakout-from-stability to detect if match is in progress."""

    # --- Hard filters ---

    if market.get("closed"):
        return None

    try:
        vol = float(market.get("volumeNum") or 0)
    except (TypeError, ValueError):
        vol = 0
    if vol < min_volume:
        return None

    outcomes = _parse(market.get("outcomes"))
    tokens   = _parse(market.get("clobTokenIds"))
    if not outcomes or not tokens or len(outcomes) != len(tokens):
        return None

    question = market.get("question") or ""
    if not _is_moneyline(question, outcomes):
        return None

    # --- Fetch current best-ask prices from CLOB ---

    current: list[float | None] = []
    for tok in tokens:
        current.append(clob_price(tok))
        time.sleep(DELAY)

    # Skip if any outcome >= 99c (match effectively decided)
    if any(c is not None and c >= 0.99 for c in current):
        return None

    # --- Live detection via price history on first token ---
    cur0 = current[0] if current else None
    if cur0 is None:
        return None

    history = clob_price_history(tokens[0], now_unix)
    time.sleep(DELAY)

    is_live, stable_price = detect_live(history, cur0)
    if not is_live:
        return None

    # --- Build pre-match prices from stable price ---
    # stable_price is for token[0]. For 2-outcome markets, token[1] = 1 - token[0]
    pre_match: list[float | None] = []
    if stable_price is not None:
        pre_match.append(stable_price)
        if len(tokens) == 2:
            pre_match.append(round(1 - stable_price, 4))
        else:
            # For 3+ outcomes, fetch history for each additional token
            for tok in tokens[1:]:
                h = clob_price_history(tok, now_unix)
                time.sleep(DELAY)
                s = find_stable_price(h)
                pre_match.append(s)
    else:
        pre_match = [None] * len(tokens)

    # --- Build outcome list ---
    outcome_data = []
    for i, (name, tok) in enumerate(zip(outcomes, tokens)):
        cur = current[i] if i < len(current) else None
        pre = pre_match[i] if i < len(pre_match) else None
        change = None
        if pre is not None and cur is not None:
            change = round((cur - pre) * 100, 1)
        outcome_data.append({
            "name": name,
            "token_id": tok,
            "current_price": cur,
            "pre_match_price": pre,
            "change_cents": change,
        })

    # --- Soft filters (swing criteria — flag but don't discard) ---
    qualified = True
    disqualify_reason = None

    pre_vals = [o["pre_match_price"] for o in outcome_data if o["pre_match_price"] is not None]
    if not pre_vals:
        qualified = False
        disqualify_reason = "No pre-match price available"
    else:
        max_pre = max(pre_vals)
        fav_idx = next(i for i, o in enumerate(outcome_data) if o["pre_match_price"] == max_pre)
        fav_cur = outcome_data[fav_idx]["current_price"]

        if max_pre < start_min:
            qualified = False
            disqualify_reason = f"Pre-match fav {max_pre*100:.0f}\u00a2 < {start_min*100:.0f}\u00a2 threshold"
        elif fav_cur is None:
            qualified = False
            disqualify_reason = "Current price unavailable"
        elif fav_cur >= cur_max:
            qualified = False
            disqualify_reason = f"Fav now {fav_cur*100:.0f}\u00a2 \u2265 {cur_max*100:.0f}\u00a2 threshold"

    # Depth filter only for qualified
    if qualified and min_shares > 0:
        for o in outcome_data:
            d = clob_book_depth(o["token_id"])
            o["shares_available"] = d
            time.sleep(DELAY)
            if d < min_shares:
                qualified = False
                disqualify_reason = f"Insufficient depth ({d:.0f} < {min_shares:.0f} shares)"
                break

    # --- Build result ---
    event_slug = event.get("slug", "")

    result = {
        "event_title": event.get("title", ""),
        "market_question": question,
        "condition_id": market.get("conditionId", ""),
        "sport": sport,
        "game_start": market.get("gameStartTime"),
        "polymarket_url": f"https://polymarket.com/event/{event_slug}" if event_slug else "",
        "volume": vol,
        "outcomes": outcome_data,
        "qualified": qualified,
        "disqualify_reason": disqualify_reason,
    }

    if qualified:
        _log_if_new(result)
    return result

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _log_if_new(result: dict):
    cid = result.get("condition_id", "")
    if not cid:
        return

    insert_swing(result)

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
            entry[f"outcome_{i+1}_name"] = o["name"]
            entry[f"outcome_{i+1}_pre"] = o["pre_match_price"]
            entry[f"outcome_{i+1}_cur"] = o["current_price"]
        market_log.append(entry)
        # Cap in-memory log
        if len(market_log) > MAX_MEMORY_LOG:
            market_log[:] = market_log[-MAX_MEMORY_LOG:]
        # Cap seen set (old entries don't matter — DB handles dedup)
        if len(seen_conditions) > 2000:
            seen_conditions.clear()

# ---------------------------------------------------------------------------
# Flask routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/scan")
def api_scan():
    sports = [s for s in req.args.get("sports", "CS2,Dota2,Valorant,LoL,Soccer,NFL,NBA,NHL").split(",") if s]
    min_volume      = float(req.args.get("min_volume", 10000))
    min_shares      = float(req.args.get("min_shares", 0))
    start_price_min = float(req.args.get("start_price_min", 68)) / 100
    current_price_max = float(req.args.get("current_price_max", 40)) / 100

    t0 = time.time()
    results = scan_markets(sports, min_volume, min_shares,
                           start_price_min, current_price_max)
    elapsed = round(time.time() - t0, 1)

    qualified = [r for r in results if r.get("qualified")]
    dimmed    = [r for r in results if not r.get("qualified")]

    # Qualified: biggest absolute swing first
    qualified.sort(key=lambda r: min(
        (o["change_cents"] for o in r["outcomes"] if o["change_cents"] is not None),
        default=0,
    ))
    # Dimmed: by volume (most liquid first)
    dimmed.sort(key=lambda r: r.get("volume", 0), reverse=True)

    return jsonify({
        "results": qualified + dimmed,
        "count_qualified": len(qualified),
        "count_total": len(results),
        "scanned_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_s": elapsed,
    })


@app.route("/api/log")
def api_log():
    if _db_available:
        sport = req.args.get("sport")
        limit = int(req.args.get("limit", 200))
        rows = get_all_swings(limit=limit, sport=sport)
        for r in rows:
            for k, v in r.items():
                if isinstance(v, datetime):
                    r[k] = v.strftime("%Y-%m-%d %H:%M:%S UTC")
        return jsonify({"entries": rows, "total": get_count(), "source": "postgres"})
    with _log_lock:
        return jsonify({"entries": list(market_log), "total": len(market_log), "source": "memory"})


@app.route("/api/db-test")
def api_db_test():
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
