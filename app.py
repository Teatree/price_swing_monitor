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
from threading import Lock, Thread, Event
from statistics import median

import requests
from flask import Flask, render_template, jsonify, request as req

from db import init_db, insert_swing, get_all_swings, get_count, get_unresolved, mark_resolved

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

with app.app_context():
    _db_available = init_db()

# ---------------------------------------------------------------------------
# Server-side monitor state (survives browser tab close)
# ---------------------------------------------------------------------------
_monitor_lock = Lock()
_monitor_stop = Event()       # set() to signal the bg thread to stop
_monitor_thread: Thread | None = None
_monitor_config: dict = {}    # params the monitor was started with
_monitor_started_at: str = ""
_last_results: dict = {}      # cached results from most recent scan
_last_scan_at: str = ""

GAMMA = "https://gamma-api.polymarket.com"
CLOB = "https://clob.polymarket.com"
DELAY = 0.15

# Live detection parameters
STABLE_BAND_CENTS = 3      # max ±3c variation to count as "stable"
STABLE_MIN_POINTS = 4      # minimum 4 data points (~20 min at 5-min fidelity)
BREAKOUT_CENTS = 5          # current price must be >=5c away from stable price
RESOLUTION_CHECK_INTERVAL = 120  # check unresolved markets every 2 minutes
RESOLUTION_THRESHOLD = 0.95      # outcome price >= this = winner

# ---------------------------------------------------------------------------
# Sport → Polymarket identifiers
# Each sport has either series_ids (queried individually) or a tag_id
# (single query catches ALL leagues under that tag).
# tag_id 100350 = all soccer, 64 = all esports
# ---------------------------------------------------------------------------
SPORT_CONFIG = {
    "CS2":      {"series": [10310]},
    "Dota2":    {"series": [10309]},
    "Valorant": {"series": [10369]},
    "LoL":      {"series": [10311]},
    "Soccer":   {"series": [
        10188,  # EPL
        10193,  # La Liga
        10194,  # Bundesliga
        10195,  # Ligue 1
        10203,  # Serie A
        10204,  # UCL
        10209,  # UEL
        10189,  # MLS
        10238,  # FIFA Friendlies
        10243,  # UEFA Qualifiers
        10246,  # CONCACAF
        10286,  # Eredivisie
        10289,  # Libertadores
        10290,  # Liga MX
        10292,  # Turkish League
        10330,  # Primeira Liga
        10359,  # Brasileirao
        10361,  # Saudi Pro League
        10444,  # K-League
        10230,  # EFL Championship
        10285,  # Argentina Primera
        10287,  # Italian Cup
        10317,  # DFB Pokal
        10360,  # J-League
        10438,  # A-League
    ], "tags": [
        102866, # La Liga 2
    ]},
    "NFL":      {"series": [10187]},
    "NBA":      {"series": [10345]},
    "NHL":      {"series": [10346]},
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

def fetch_active_events(series_id: int = None, tag_id: int = None,
                        limit: int = 100, max_events: int = 500) -> list[dict]:
    """Fetch active non-closed events by series_id or tag_id."""
    out = []
    offset = 0
    while offset < max_events:
        params = {
            "active": "true",
            "closed": "false",
            "order": "startDate",
            "ascending": "false",
            "limit": limit,
            "offset": offset,
        }
        if series_id:
            params["series_id"] = str(series_id)
        if tag_id:
            params["tag_id"] = str(tag_id)
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

def clob_best_ask(token_id: str) -> tuple[float | None, float]:
    """Get best ask price from the order book (most reliable price source).

    Returns (price, total_depth_shares).
    Filters out stale asks >= 95c (known Polymarket /book bug).
    Falls back to /price endpoint if /book fails.
    """
    try:
        r = requests.get(f"{CLOB}/book",
                         params={"token_id": token_id}, timeout=5)
        data = r.json()
        asks = data.get("asks", [])
        # Sort by price ascending, filter out stale >= 95c
        valid = []
        for a in asks:
            p = float(a.get("price", 0))
            s = float(a.get("size", 0))
            if 0 < p < 0.95 and s > 0:
                valid.append((p, s))
        valid.sort(key=lambda x: x[0])
        if valid:
            best_price = valid[0][0]
            total_depth = sum(s for _, s in valid[:5])
            return best_price, total_depth
    except Exception as e:
        logger.debug("book error %s: %s", token_id, e)

    # Fallback: /price endpoint
    try:
        r = requests.get(f"{CLOB}/price",
                         params={"token_id": token_id, "side": "BUY"}, timeout=5)
        p = float(r.json().get("price", 0))
        if p > 0:
            return p, 0.0
    except Exception:
        pass

    return None, 0.0


def clob_price_history(token_id: str, now_unix: int) -> list[dict]:
    """Fetch last 8 hours of price data at 5-minute intervals.
    Returns list of {t: unix_timestamp, p: price} dicts, newest last."""
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
        return [{"t": int(pt.get("t", 0)), "p": float(pt.get("p", 0))} for pt in history]
    except Exception as e:
        logger.debug("prices-history error %s: %s", token_id, e)
        return []


def _find_all_plateaus(prices: list[float], band: float, min_points: int):
    """Scan forward through prices and collect all non-overlapping plateaus.
    Returns list of (start_idx, end_idx, median_price, length)."""
    plateaus = []
    i = 0
    while i <= len(prices) - min_points:
        window = prices[i:i + min_points]
        if max(window) - min(window) <= band:
            # Found start of a plateau — extend it forward
            end = i + min_points - 1
            while end + 1 < len(prices) and max(prices[i:end + 2]) - min(prices[i:end + 2]) <= band:
                end += 1
            plateau = prices[i:end + 1]
            plateaus.append((i, end, round(median(plateau), 4), len(plateau)))
            i = end + 1  # skip past this plateau
        else:
            i += 1
    return plateaus


def find_stable_price_and_breakout(history: list[dict]) -> tuple[float | None, int | None]:
    """Find the pre-match price using multi-plateau scoring.

    1. Find ALL stable plateaus in the 8h price history
    2. Filter to candidates where current price broke out >=5c
    3. Score each candidate (length, position, post-plateau volatility)
    4. Return the highest-scoring plateau as the pre-match line

    Scoring:
      +2 per point of length (longer = more likely pre-match)
      +5 if earliest candidate (first plateau is usually pre-match)
      +3 if followed by a sharp move >=5c within 2 data points
      +2 if plateau is in the first half of history
    """
    if len(history) < STABLE_MIN_POINTS:
        return None, None

    prices = [pt["p"] for pt in history]
    current = prices[-1]
    band = STABLE_BAND_CENTS / 100
    breakout_threshold = BREAKOUT_CENTS / 100

    # Step 1: find all plateaus
    all_plateaus = _find_all_plateaus(prices, band, STABLE_MIN_POINTS)
    if not all_plateaus:
        return None, None

    # Step 2: filter to candidates (current price broke out from plateau)
    candidates = []
    for start, end, med, length in all_plateaus:
        if abs(current - med) >= breakout_threshold:
            candidates.append((start, end, med, length))

    if not candidates:
        return None, None

    # Step 3: score each candidate
    midpoint = len(prices) // 2
    best_score = -1
    best = None

    for idx, (start, end, med, length) in enumerate(candidates):
        score = length * 2

        # Earliest candidate bonus
        if idx == 0:
            score += 5

        # First-half-of-history bonus
        if start < midpoint:
            score += 2

        # Post-plateau volatility: sharp move within 2 points after plateau
        for j in range(end + 1, min(end + 3, len(prices))):
            if abs(prices[j] - med) >= breakout_threshold:
                score += 3
                break

        if score > best_score or (score == best_score and best and start < best[0]):
            best_score = score
            best = (start, end, med, length)

    if not best:
        return None, None

    # Step 4: return winner
    stable_price = best[2]
    breakout_idx = best[1] + 1
    breakout_ts = history[breakout_idx]["t"] if breakout_idx < len(history) else None
    return stable_price, breakout_ts


def detect_live(history: list[dict], current_price: float):
    """Determine if a market is live using multi-plateau scoring.

    Returns (is_live, stable_price, breakout_ts).
    """
    stable, breakout_ts = find_stable_price_and_breakout(history)
    if stable is None:
        return False, None, None
    return True, stable, breakout_ts

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
    """Detect moneyline for esports/NBA/NFL/NHL (team names as outcomes)."""
    if set(outcomes) == {"Yes", "No"}:
        return False
    q = question.lower()
    if any(kw in q for kw in EXCLUDE_KW):
        return False
    return "vs" in q or "match winner" in q


def _is_soccer_moneyline(market: dict) -> str | None:
    """Detect soccer-style moneyline (Yes/No per outcome).
    Returns 'team_win', 'draw', or None."""
    outcomes = _parse(market.get("outcomes"))
    if set(outcomes) != {"Yes", "No"}:
        return None
    smt = (market.get("sportsMarketType") or "").lower()
    if smt != "moneyline":
        return None
    q = (market.get("question") or "").lower()
    if "draw" in q:
        return "draw"
    if "win" in q:
        return "team_win"
    return None

# ---------------------------------------------------------------------------
# Core scan
# ---------------------------------------------------------------------------

def scan_markets(
    sports: list[str],
    min_volume: float = 10_000,
    min_shares: float = 0,
    start_price_min: float = 0.68,
    swing_min: float = 0.20,
) -> list[dict]:
    """Find live moneyline markets using breakout-from-stability detection."""
    now = datetime.now(timezone.utc)
    now_unix = int(now.timestamp())
    results: list[dict] = []
    seen_cids: set[str] = set()

    for sport in sports:
        cfg = SPORT_CONFIG.get(sport, {})
        # Collect all events — either by tag (one call) or by series IDs
        all_events = []
        for sid in cfg.get("series", []):
            evts = fetch_active_events(series_id=sid)
            all_events.extend(evts)
        for tid in cfg.get("tags", []):
            evts = fetch_active_events(tag_id=tid)
            all_events.extend(evts)
        n_src = len(cfg.get("series", [])) + len(cfg.get("tags", []))
        logger.info("  %s → %d sources, %d active events", sport, n_src, len(all_events))

        mkt_count = 0
        live_count = 0
        seen_event_slugs: set[str] = set()
        for event in all_events:
            event_slug = event.get("slug", "")

            # --- Soccer: group 3 Yes/No markets per match ---
            if sport == "Soccer":
                if event_slug in seen_event_slugs:
                    continue
                seen_event_slugs.add(event_slug)
                mkt_count += 1
                r = _process_soccer_event(event, sport, now, now_unix,
                                          min_volume, min_shares,
                                          start_price_min, swing_min)
                if r:
                    live_count += 1
                    results.append(r)
                continue

            # --- Esports / NBA / NFL / NHL: single market per match ---
            for market in event.get("markets") or []:
                mkt_count += 1
                cid = market.get("conditionId", "")
                if cid in seen_cids:
                    continue
                seen_cids.add(cid)
                r = _process_market(market, event, sport, now, now_unix,
                                    min_volume, min_shares,
                                    start_price_min, swing_min)
                if r:
                    live_count += 1
                    results.append(r)
        logger.info("    → %d markets checked, %d live", mkt_count, live_count)
    logger.info("Scan complete: %d live markets", len(results))
    return results


def _process_market(market, event, sport, now, now_unix,
                    min_volume, min_shares, start_min, swing_min):
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

    # --- Quick pre-filter using Gamma prices (FREE, no API call) ---
    gamma_prices = _parse(market.get("outcomePrices"))
    if gamma_prices and len(gamma_prices) == len(outcomes):
        gp = [float(p) for p in gamma_prices]
        # If any outcome >= 99c, match is decided → skip
        if any(p >= 0.99 for p in gp):
            return None

    # --- Fetch price history for live detection (1 CLOB call) ---
    history = clob_price_history(tokens[0], now_unix)
    time.sleep(DELAY)

    if not history:
        return None

    # Use the latest point from history as current price for token 0
    cur0 = float(history[-1]["p"])

    is_live, stable_price, breakout_ts = detect_live(history, cur0)
    if not is_live:
        return None

    # --- Now fetch real best-ask prices from order book (only for live markets) ---
    current: list[float | None] = []
    depths: list[float] = []
    for tok in tokens:
        price, depth = clob_best_ask(tok)
        current.append(price)
        depths.append(depth)
        time.sleep(DELAY)

    # Re-check with real prices: skip if any outcome >= 99c
    if any(c is not None and c >= 0.99 for c in current):
        return None

    # --- Duration since breakout (how long the match has been live) ---
    live_minutes = None
    if breakout_ts:
        live_minutes = int((now_unix - breakout_ts) / 60)

    # --- Build pre-match prices from stable price ---
    pre_match: list[float | None] = []
    if stable_price is not None:
        pre_match.append(stable_price)
        if len(tokens) == 2:
            pre_match.append(round(1 - stable_price, 4))
        else:
            for tok in tokens[1:]:
                h = clob_price_history(tok, now_unix)
                time.sleep(DELAY)
                s, _ = find_stable_price_and_breakout(h)
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
            "shares_available": depths[i] if i < len(depths) else 0,
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
        else:
            fav_drop = max_pre - fav_cur  # how much the fav has fallen
            if fav_drop < swing_min:
                qualified = False
                disqualify_reason = f"Fav drop {fav_drop*100:.0f}\u00a2 < {swing_min*100:.0f}\u00a2 swing threshold"

    # Depth filter (depth already fetched from book)
    if qualified and min_shares > 0:
        for o in outcome_data:
            if o["shares_available"] < min_shares:
                qualified = False
                disqualify_reason = f"Insufficient depth ({o['shares_available']:.0f} < {min_shares:.0f} shares)"
                break

    # --- Build result ---
    event_slug = event.get("slug", "")

    result = {
        "event_title": event.get("title", ""),
        "market_question": question,
        "condition_id": market.get("conditionId", ""),
        "sport": sport,
        "game_start": market.get("gameStartTime"),
        "live_minutes": live_minutes,
        "polymarket_url": f"https://polymarket.com/event/{event_slug}" if event_slug else "",
        "volume": vol,
        "outcomes": outcome_data,
        "qualified": qualified,
        "disqualify_reason": disqualify_reason,
    }

    if qualified:
        _log_if_new(result)
    return result


def _process_soccer_event(event, sport, now, now_unix,
                          min_volume, min_shares, start_min, swing_min):
    """Process a soccer event with 3 separate Yes/No moneyline markets.
    Groups them into a single result with 2 or 3 outcomes (Team A, Team B, Draw)."""
    markets = event.get("markets") or []
    title = event.get("title", "")
    event_slug = event.get("slug", "")

    # Find the team-win and draw markets
    team_markets = []  # (team_name, market_dict)
    draw_market = None
    total_vol = 0

    for m in markets:
        if m.get("closed"):
            continue
        mtype = _is_soccer_moneyline(m)
        if not mtype:
            continue
        try:
            vol = float(m.get("volumeNum") or 0)
        except (TypeError, ValueError):
            vol = 0
        total_vol += vol

        q = m.get("question", "")
        if mtype == "draw":
            draw_market = m
        elif mtype == "team_win":
            # Extract team name from "Will TEAM win on DATE?"
            name = q.replace("Will ", "").split(" win ")[0].strip()
            if not name:
                name = q
            team_markets.append((name, m))

    if len(team_markets) < 2:
        return None
    if total_vol < min_volume:
        return None

    # Build outcome list: each team-win market's Yes token = that team's price
    # Draw market's Yes token = draw probability
    outcome_entries = []  # (name, token_id, market_for_resolution)
    for team_name, mkt in team_markets:
        tokens = _parse(mkt.get("clobTokenIds"))
        if not tokens:
            continue
        # tokens[0] = Yes token (team wins)
        outcome_entries.append((team_name, tokens[0], mkt))
    if draw_market:
        dtokens = _parse(draw_market.get("clobTokenIds"))
        if dtokens:
            outcome_entries.append(("Draw", dtokens[0], draw_market))

    if len(outcome_entries) < 2:
        return None

    # Use the highest-volume team market for live detection
    main_token = outcome_entries[0][1]

    # --- Fetch price history for live detection ---
    history = clob_price_history(main_token, now_unix)
    time.sleep(DELAY)
    if not history:
        return None

    cur0 = float(history[-1]["p"])
    is_live, stable_price, breakout_ts = detect_live(history, cur0)
    if not is_live:
        return None

    # --- Fetch current prices from order book ---
    current_prices = []
    depths = []
    for _, tok, _ in outcome_entries:
        price, depth = clob_best_ask(tok)
        current_prices.append(price)
        depths.append(depth)
        time.sleep(DELAY)

    # Skip if any outcome >= 99c
    if any(c is not None and c >= 0.99 for c in current_prices):
        return None

    # --- Pre-match prices ---
    pre_match_prices = []
    if stable_price is not None:
        pre_match_prices.append(stable_price)
        # Fetch history for other outcomes
        for _, tok, _ in outcome_entries[1:]:
            h = clob_price_history(tok, now_unix)
            time.sleep(DELAY)
            s, _ = find_stable_price_and_breakout(h)
            pre_match_prices.append(s)
    else:
        pre_match_prices = [None] * len(outcome_entries)

    live_minutes = int((now_unix - breakout_ts) / 60) if breakout_ts else None

    # --- Build outcome data ---
    outcome_data = []
    for i, (name, tok, _) in enumerate(outcome_entries):
        cur = current_prices[i] if i < len(current_prices) else None
        pre = pre_match_prices[i] if i < len(pre_match_prices) else None
        change = round((cur - pre) * 100, 1) if pre is not None and cur is not None else None
        outcome_data.append({
            "name": name,
            "token_id": tok,
            "current_price": cur,
            "pre_match_price": pre,
            "change_cents": change,
            "shares_available": depths[i] if i < len(depths) else 0,
        })

    # --- Soft filters ---
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
        else:
            fav_drop = max_pre - fav_cur
            if fav_drop < swing_min:
                qualified = False
                disqualify_reason = f"Fav drop {fav_drop*100:.0f}\u00a2 < {swing_min*100:.0f}\u00a2 swing threshold"

    if qualified and min_shares > 0:
        for o in outcome_data:
            if o["shares_available"] < min_shares:
                qualified = False
                disqualify_reason = f"Insufficient depth ({o['shares_available']:.0f} < {min_shares:.0f} shares)"
                break

    # Use first team market's condition_id for tracking
    condition_id = outcome_entries[0][2].get("conditionId", "")

    result = {
        "event_title": title,
        "market_question": title,
        "condition_id": condition_id,
        "sport": sport,
        "game_start": None,
        "live_minutes": live_minutes,
        "polymarket_url": f"https://polymarket.com/event/{event_slug}" if event_slug else "",
        "volume": total_vol,
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
# Resolution checker (called from monitor loop, not a separate thread)
# ---------------------------------------------------------------------------
RESOLVE_BATCH_SIZE = 5  # check at most 5 unresolved markets per scan cycle

def check_resolutions(send_tg: bool = False):
    """Check a batch of unresolved swing markets. Called after each monitor scan.
    Only checks RESOLVE_BATCH_SIZE markets per call to limit memory/API usage."""
    try:
        unresolved = get_unresolved()
        if not unresolved:
            return
        # Only check a few per cycle — they'll all get checked over time
        batch = unresolved[:RESOLVE_BATCH_SIZE]
        logger.info("Resolution check: %d/%d unresolved (batch=%d)",
                     len(batch), len(unresolved), RESOLVE_BATCH_SIZE)

        for mkt in batch:
            cid = mkt["condition_id"]
            prices = _check_resolution_prices(mkt)
            if not prices:
                continue

            # Attach pre-match prices from the DB row
            db_pres = {}
            for i in range(1, 4):
                n = mkt.get(f"outcome_{i}_name")
                p = mkt.get(f"outcome_{i}_pre")
                if n:
                    db_pres[n] = p
            prices = [(name, cur, db_pres.get(name)) for name, cur, _ in prices]

            # Check if any outcome hit the threshold
            winner = None
            for name, cur_price, pre_price in prices:
                if cur_price >= RESOLUTION_THRESHOLD:
                    winner = (name, cur_price, pre_price)
                    break

            if winner:
                w_name, w_final, w_pre = winner
                losers = [(n, cp, pp) for n, cp, pp in prices if n != w_name]
                if losers:
                    loser = max(losers, key=lambda x: x[2] or 0)
                    l_name, l_final, l_pre = loser
                else:
                    l_name, l_final, l_pre = None, None, None

                all_pres = [(n, pp) for n, _, pp in prices if pp is not None]
                fav_name = max(all_pres, key=lambda x: x[1])[0] if all_pres else None
                fav_won = (w_name == fav_name) if fav_name else None

                mark_resolved(cid, w_name, w_pre, w_final,
                              l_name, l_pre, l_final, fav_won)
                logger.info("Resolved: %s → winner=%s (fav_won=%s)",
                            mkt.get("market_question", cid), w_name, fav_won)

                # Send resolution notification to Telegram
                if send_tg:
                    try:
                        tg_send_resolution(mkt, w_name, w_pre,
                                           l_name, l_pre, fav_won)
                    except Exception as e:
                        logger.error("TG resolution send error: %s", e)

            time.sleep(DELAY)

    except Exception as e:
        logger.error("Resolution check error: %s", e)


def _check_resolution_prices(db_row: dict):
    """Check if a market is resolved by looking up the event via slug.
    Returns list of (outcome_name, current_price, None) or None."""
    try:
        # Extract slug from polymarket_url
        url = db_row.get("polymarket_url") or ""
        slug = url.split("/event/")[-1] if "/event/" in url else ""
        condition_id = db_row.get("condition_id", "")
        if not slug:
            return None

        r = requests.get(f"{GAMMA}/events",
                         params={"slug": slug}, timeout=10)
        if r.status_code != 200:
            return None
        events = r.json()
        if not events:
            return None
        event = events[0]

        # Find the moneyline market matching our condition_id
        for mkt in event.get("markets") or []:
            if mkt.get("conditionId") != condition_id:
                continue

            outcomes = _parse(mkt.get("outcomes"))
            gamma_prices = _parse(mkt.get("outcomePrices"))
            if not outcomes or not gamma_prices:
                continue

            gp = [float(p) for p in gamma_prices]
            # Resolved if market is closed OR any outcome >= threshold
            if mkt.get("closed") or any(p >= RESOLUTION_THRESHOLD for p in gp):
                return [(name, gp[i], None) for i, name in enumerate(outcomes)]

            return None  # found our market but it's not resolved yet

        return None

    except Exception as e:
        logger.debug("Resolution price check error %s: %s", condition_id, e)
        return None


# ---------------------------------------------------------------------------
# Background monitor loop
# ---------------------------------------------------------------------------

def _monitor_loop(config: dict):
    """Runs in a background thread. Scans repeatedly until _monitor_stop is set."""
    global _last_results, _last_scan_at

    sports = config.get("sports", list(SPORT_CONFIG.keys()))
    min_volume = config.get("min_volume", 10000)
    min_shares = config.get("min_shares", 0)
    start_price_min = config.get("start_price_min", 0.68)
    swing_min = config.get("swing_min", 0.40)
    interval = max(15, config.get("interval", 60))
    send_tg = config.get("telegram", False)

    logger.info("Monitor started: interval=%ds sports=%s tg=%s", interval, sports, send_tg)

    while not _monitor_stop.is_set():
        try:
            t0 = time.time()
            results = scan_markets(sports, min_volume, min_shares,
                                   start_price_min, swing_min)
            elapsed = round(time.time() - t0, 1)

            qualified = [r for r in results if r.get("qualified")]
            dimmed = [r for r in results if not r.get("qualified")]

            qualified.sort(key=lambda r: min(
                (o["change_cents"] for o in r["outcomes"] if o["change_cents"] is not None),
                default=0,
            ))
            dimmed.sort(key=lambda r: r.get("volume", 0), reverse=True)

            scan_data = {
                "results": qualified + dimmed,
                "count_qualified": len(qualified),
                "count_total": len(results),
                "scanned_at": datetime.now(timezone.utc).isoformat(),
                "elapsed_s": elapsed,
            }

            with _monitor_lock:
                _last_results = scan_data
                _last_scan_at = scan_data["scanned_at"]

            # Telegram: edit-in-place or new message when market set changes
            if send_tg:
                try:
                    tg_handle_swing_update(qualified)
                except Exception as e:
                    logger.error("Monitor TG error: %s", e)

            logger.info("Monitor scan done: %d qualified, %d total, %.1fs",
                        len(qualified), len(results), elapsed)

            # Check a few unresolved markets for resolution (lightweight)
            if _db_available:
                check_resolutions(send_tg=send_tg)

        except Exception as e:
            import traceback
            logger.error("Monitor scan error: %s\n%s", e, traceback.format_exc())

        # Self-ping to prevent Render free tier from spinning down
        try:
            requests.get(f"{SITE_URL}api/monitor/status", timeout=5)
        except Exception:
            pass

        # Sleep in small increments so we can stop quickly
        for _ in range(interval):
            if _monitor_stop.is_set():
                break
            time.sleep(1)

    logger.info("Monitor stopped")


# ---------------------------------------------------------------------------
# Flask routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/scan")
def api_scan():
    """One-shot scan (used by Scan Now button when monitor is off)."""
    sports = [s for s in req.args.get("sports", "CS2,Dota2,Valorant,LoL,Soccer,NFL,NBA,NHL").split(",") if s]
    min_volume      = float(req.args.get("min_volume", 10000))
    min_shares      = float(req.args.get("min_shares", 0))
    start_price_min = float(req.args.get("start_price_min", 68)) / 100
    swing_min = float(req.args.get("swing_min", 20)) / 100

    t0 = time.time()
    results = scan_markets(sports, min_volume, min_shares,
                           start_price_min, swing_min)
    elapsed = round(time.time() - t0, 1)

    qualified = [r for r in results if r.get("qualified")]
    dimmed    = [r for r in results if not r.get("qualified")]

    qualified.sort(key=lambda r: min(
        (o["change_cents"] for o in r["outcomes"] if o["change_cents"] is not None),
        default=0,
    ))
    dimmed.sort(key=lambda r: r.get("volume", 0), reverse=True)

    return jsonify({
        "results": qualified + dimmed,
        "count_qualified": len(qualified),
        "count_total": len(results),
        "scanned_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_s": elapsed,
    })


@app.route("/api/monitor/start", methods=["POST"])
def api_monitor_start():
    """Start server-side background monitoring."""
    global _monitor_thread, _monitor_config, _monitor_started_at

    with _monitor_lock:
        if _monitor_thread and _monitor_thread.is_alive():
            return jsonify({"ok": True, "status": "already_running"})

    data = req.get_json(silent=True) or {}
    config = {
        "sports": [s for s in data.get("sports", "CS2,Dota2,Valorant,LoL,Soccer,NFL,NBA,NHL").split(",") if s],
        "min_volume": float(data.get("min_volume", 10000)),
        "min_shares": float(data.get("min_shares", 0)),
        "start_price_min": float(data.get("start_price_min", 68)) / 100,
        "swing_min": float(data.get("swing_min", 20)) / 100,
        "interval": int(data.get("interval", 60)),
        "telegram": bool(data.get("telegram", False)),
    }

    _monitor_stop.clear()
    _monitor_config = config
    _monitor_started_at = datetime.now(timezone.utc).isoformat()
    _monitor_thread = Thread(target=_monitor_loop, args=(config,), daemon=True)
    _monitor_thread.start()

    return jsonify({"ok": True, "status": "started", "config": config})


@app.route("/api/monitor/stop", methods=["POST"])
def api_monitor_stop():
    """Stop server-side background monitoring."""
    global _monitor_thread
    _monitor_stop.set()
    if _monitor_thread:
        _monitor_thread.join(timeout=5)
        _monitor_thread = None
    return jsonify({"ok": True, "status": "stopped"})


@app.route("/api/monitor/status")
def api_monitor_status():
    """Check if server-side monitor is running + get latest cached results."""
    running = _monitor_thread is not None and _monitor_thread.is_alive()
    with _monitor_lock:
        return jsonify({
            "running": running,
            "config": _monitor_config if running else None,
            "started_at": _monitor_started_at if running else None,
            "last_scan_at": _last_scan_at,
            "results": _last_results if _last_results else None,
        })


@app.route("/api/monitor/reset", methods=["POST", "GET"])
def api_monitor_reset():
    """Force-stop monitoring. GET allowed so you can just hit the URL in a browser."""
    global _monitor_thread, _last_results, _last_scan_at
    _monitor_stop.set()
    if _monitor_thread:
        _monitor_thread.join(timeout=5)
        _monitor_thread = None
    with _monitor_lock:
        _last_results = {}
        _last_scan_at = ""
    return jsonify({"ok": True, "status": "reset"})


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


TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")
SITE_URL = os.environ.get("SITE_URL", "https://price-swing-monitor.onrender.com/")

# Tracked message state for edit-in-place
_tg_swing_msg_id: int | None = None     # message_id of the current swing alert
_tg_swing_market_ids: set[str] = set()  # condition_ids in the current message


def _tg_fmt_num(n) -> str:
    if n is None:
        return "?"
    if n >= 1e6:
        return f"{n/1e6:.1f}M"
    if n >= 1e3:
        return f"{n/1e3:.1f}K"
    return str(round(n))


def _find_fav_idx(outcomes: list[dict]) -> int:
    """Find the index of the pre-match favourite (highest pre_match_price)."""
    best_idx, best_pre = -1, -1
    for i, o in enumerate(outcomes):
        pre = o.get("pre_match_price")
        if pre is not None and pre > best_pre:
            best_pre = pre
            best_idx = i
    return best_idx


def _format_telegram_message(markets: list[dict]) -> str:
    """Format swing markets for Telegram (HTML mode)."""
    from html import escape as h
    n = len(markets)
    lines = [f"\U0001F6A8 <b>Price Swing Alert</b> \u2014 {n} market{'s' if n != 1 else ''}", ""]

    for r in markets:
        sport = h(r.get("sport", ""))
        question = h(r.get("market_question", r.get("event_title", "")))
        url = r.get("polymarket_url", "")
        vol = r.get("volume", 0)
        live_m = r.get("live_minutes")
        live_str = ""
        if live_m is not None:
            if live_m < 60:
                live_str = f" | \u23f1 {live_m}m"
            else:
                live_str = f" | \u23f1 {live_m // 60}h{live_m % 60}m"

        lines.append(f"<b>{sport}</b> | Vol ${_tg_fmt_num(vol)}{live_str}")
        if url:
            lines.append(f'<a href="{url}">{question}</a>')
        else:
            lines.append(question)

        outcomes = r.get("outcomes", [])
        fav_idx = _find_fav_idx(outcomes)

        for i, o in enumerate(outcomes):
            name = h(o.get("name", "?"))
            pre = o.get("pre_match_price")
            cur = o.get("current_price")
            change = o.get("change_cents")
            shares = o.get("shares_available")

            fav_mark = "\u2B50 " if i == fav_idx else "  "
            pre_s = f"{pre*100:.0f}" if pre is not None else "-"
            cur_s = f"{cur*100:.0f}" if cur is not None else "-"
            chg_s = ""
            if change is not None:
                sign = "+" if change > 0 else ""
                chg_s = f" ({sign}{change:.0f}c)"
            shares_s = f" [{_tg_fmt_num(shares)} sh]" if shares else ""
            lines.append(f"{fav_mark}{name}: {pre_s}\u00a2 \u2192 {cur_s}\u00a2{chg_s}{shares_s}")
        lines.append("")

    ts = datetime.now(timezone.utc).strftime("%H:%M UTC")
    lines.append(f'\U0001F552 {ts} | <a href="{SITE_URL}">Open Monitor</a>')
    return "\n".join(lines)


def _format_resolution_message(mkt: dict, winner_name: str, winner_pre: float,
                                loser_name: str, loser_pre: float,
                                fav_won: bool) -> str:
    """Format a resolved market notification."""
    from html import escape as h
    question = h(mkt.get("market_question", ""))
    sport = h(mkt.get("sport", ""))

    # Determine who was the fav
    fav_emoji = "\u2B50"
    if fav_won:
        verdict = "\u2705 Favourite won"
        w_label = f"{fav_emoji} {h(winner_name)}"
        l_label = f"  {h(loser_name)}"
    else:
        verdict = "\u274C Favourite lost (upset!)"
        w_label = f"  {h(winner_name)}"
        l_label = f"{fav_emoji} {h(loser_name)}"

    w_pre_s = f"{winner_pre*100:.0f}" if winner_pre is not None else "?"
    l_pre_s = f"{loser_pre*100:.0f}" if loser_pre is not None else "?"

    lines = [
        f"\U0001F3C1 <b>Match Resolved</b> | {sport}",
        f"<b>{question}</b>",
        "",
        f"\U0001F3C6 Winner: {w_label} (was {w_pre_s}\u00a2 pre-match)",
        f"\U0001F4A4 Loser: {l_label} (was {l_pre_s}\u00a2 pre-match)",
        "",
        verdict,
    ]
    return "\n".join(lines)


def tg_send(text: str) -> dict:
    """Send a new Telegram message. Returns the API response."""
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
            json={
                "chat_id": TELEGRAM_CHAT_ID,
                "text": text,
                "parse_mode": "HTML",
                "disable_web_page_preview": True,
            },
            timeout=10,
        )
        return r.json()
    except Exception as e:
        logger.error("Telegram send error: %s", e)
        return {"ok": False, "error": str(e)}


def tg_edit(message_id: int, text: str) -> dict:
    """Edit an existing Telegram message in-place."""
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/editMessageText",
            json={
                "chat_id": TELEGRAM_CHAT_ID,
                "message_id": message_id,
                "text": text,
                "parse_mode": "HTML",
                "disable_web_page_preview": True,
            },
            timeout=10,
        )
        return r.json()
    except Exception as e:
        logger.error("Telegram edit error: %s", e)
        return {"ok": False, "error": str(e)}


def tg_handle_swing_update(qualified: list[dict]):
    """Smart Telegram updates: new message when market set changes, edit otherwise."""
    global _tg_swing_msg_id, _tg_swing_market_ids

    if not qualified:
        # No swing markets — clear tracking (keep old message in chat)
        _tg_swing_msg_id = None
        _tg_swing_market_ids = set()
        return

    current_ids = {r.get("condition_id", "") for r in qualified}
    text = _format_telegram_message(qualified)

    if _tg_swing_msg_id and current_ids == _tg_swing_market_ids:
        # Same set of markets — just edit prices in place
        result = tg_edit(_tg_swing_msg_id, text)
        if not result.get("ok"):
            # Edit failed (message too old, deleted, etc.) — send new
            result = tg_send(text)
            if result.get("ok"):
                _tg_swing_msg_id = result["result"]["message_id"]
    else:
        # Market set changed — send a new message, keep old one in chat
        result = tg_send(text)
        if result.get("ok"):
            _tg_swing_msg_id = result["result"]["message_id"]
            _tg_swing_market_ids = current_ids


def tg_send_resolution(mkt: dict, winner_name: str, winner_pre: float,
                       loser_name: str, loser_pre: float, fav_won: bool):
    """Send a resolution notification (always a new message)."""
    text = _format_resolution_message(mkt, winner_name, winner_pre,
                                       loser_name, loser_pre, fav_won)
    tg_send(text)


@app.route("/api/telegram-send", methods=["POST"])
def api_telegram_send():
    """Send swing markets to Telegram. Expects JSON body with {markets: [...]}."""
    data = req.get_json(silent=True) or {}
    markets = data.get("markets", [])
    if not markets:
        return jsonify({"ok": False, "error": "No markets provided"}), 400

    text = _format_telegram_message(markets)
    result = tg_send(text)
    return jsonify(result)


@app.route("/api/telegram-test", methods=["POST"])
def api_telegram_test():
    """Send a test message with fake market data."""
    fake_markets = [
        {
            "sport": "CS2",
            "market_question": "CS2: Team Alpha vs Team Beta (BO3)",
            "polymarket_url": "https://polymarket.com/event/test-market",
            "volume": 85000,
            "live_minutes": 47,
            "outcomes": [
                {"name": "Team Alpha", "pre_match_price": 0.72, "current_price": 0.35,
                 "change_cents": -37.0, "shares_available": 4250},
                {"name": "Team Beta", "pre_match_price": 0.28, "current_price": 0.65,
                 "change_cents": 37.0, "shares_available": 1800},
            ],
        },
        {
            "sport": "NBA",
            "market_question": "Lakers vs Celtics",
            "polymarket_url": "https://polymarket.com/event/test-nba",
            "volume": 250000,
            "live_minutes": 92,
            "outcomes": [
                {"name": "Lakers", "pre_match_price": 0.68, "current_price": 0.30,
                 "change_cents": -38.0, "shares_available": 12500},
                {"name": "Celtics", "pre_match_price": 0.32, "current_price": 0.70,
                 "change_cents": 38.0, "shares_available": 8300},
            ],
        },
    ]
    text = _format_telegram_message(fake_markets)
    result = tg_send(text)
    return jsonify(result)


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
