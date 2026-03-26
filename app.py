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

from db import init_db, insert_swing, get_all_swings, get_count

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


def find_stable_price_and_breakout(history: list[dict]) -> tuple[float | None, int | None]:
    """Walk backwards through price history to find the last stable plateau.

    A plateau is STABLE_MIN_POINTS (6) consecutive prices where
    max - min <= STABLE_BAND_CENTS/100.

    Returns (stable_price, breakout_timestamp):
    - stable_price: median of the plateau (pre-match line)
    - breakout_timestamp: unix timestamp of the first point AFTER the plateau
      (i.e., when the match started / prices began moving)
    """
    if len(history) < STABLE_MIN_POINTS:
        return None, None

    prices = [pt["p"] for pt in history]
    band = STABLE_BAND_CENTS / 100

    for end in range(len(prices) - 1, STABLE_MIN_POINTS - 2, -1):
        start = end - STABLE_MIN_POINTS + 1
        window = prices[start:end + 1]
        if max(window) - min(window) <= band:
            # Extend plateau backwards
            while start > 0 and max(prices[start - 1:end + 1]) - min(prices[start - 1:end + 1]) <= band:
                start -= 1
            plateau = prices[start:end + 1]
            stable_price = round(median(plateau), 4)
            # Breakout = first point after the plateau
            breakout_idx = end + 1
            breakout_ts = history[breakout_idx]["t"] if breakout_idx < len(history) else None
            return stable_price, breakout_ts

    return None, None


def detect_live(history: list[dict], current_price: float):
    """Determine if a market is live using the breakout-from-stability method.

    Returns (is_live, stable_price, breakout_ts).
    - stable_price is the pre-match line (median of last stable plateau)
    - breakout_ts is the unix timestamp when prices started moving
    - is_live is True if current price has broken out >=5c from stable
    """
    stable, breakout_ts = find_stable_price_and_breakout(history)
    if stable is None:
        return False, None, None

    movement = abs(current_price - stable)
    is_live = movement >= (BREAKOUT_CENTS / 100)
    return is_live, stable, breakout_ts

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

    # --- Now fetch real best-ask prices from CLOB (only for live markets) ---
    current: list[float | None] = []
    for tok in tokens:
        current.append(clob_price(tok))
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
# Background monitor loop
# ---------------------------------------------------------------------------

def _monitor_loop(config: dict):
    """Runs in a background thread. Scans repeatedly until _monitor_stop is set."""
    global _last_results, _last_scan_at

    sports = config.get("sports", list(SPORT_SERIES.keys()))
    min_volume = config.get("min_volume", 10000)
    min_shares = config.get("min_shares", 0)
    start_price_min = config.get("start_price_min", 0.68)
    current_price_max = config.get("current_price_max", 0.40)
    interval = max(15, config.get("interval", 60))
    send_tg = config.get("telegram", False)

    logger.info("Monitor started: interval=%ds sports=%s tg=%s", interval, sports, send_tg)

    while not _monitor_stop.is_set():
        try:
            t0 = time.time()
            results = scan_markets(sports, min_volume, min_shares,
                                   start_price_min, current_price_max)
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

            # Telegram: send if enabled and there are qualified markets
            if send_tg and qualified:
                try:
                    text = _format_telegram_message(qualified)
                    send_telegram(text)
                except Exception as e:
                    logger.error("Monitor TG send error: %s", e)

            logger.info("Monitor scan done: %d qualified, %d total, %.1fs",
                        len(qualified), len(results), elapsed)
        except Exception as e:
            logger.error("Monitor scan error: %s", e)

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
    current_price_max = float(req.args.get("current_price_max", 40)) / 100

    t0 = time.time()
    results = scan_markets(sports, min_volume, min_shares,
                           start_price_min, current_price_max)
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
        "current_price_max": float(data.get("current_price_max", 40)) / 100,
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


def _format_telegram_message(markets: list[dict]) -> str:
    """Format a list of swing markets into a Telegram message (HTML mode)."""
    from html import escape as h
    lines = ["\U0001F6A8 <b>Price Swing Alert</b>", ""]

    for r in markets:
        sport = h(r.get("sport", ""))
        question = h(r.get("market_question", r.get("event_title", "")))
        url = r.get("polymarket_url", "")
        vol = r.get("volume", 0)
        live_m = r.get("live_minutes")
        live_str = ""
        if live_m is not None:
            if live_m < 60:
                live_str = f" | Live {live_m}m"
            else:
                live_str = f" | Live {live_m // 60}h {live_m % 60}m"

        vol_str = _tg_fmt_num(vol)
        lines.append(f"<b>{sport}</b> | Vol ${vol_str}{live_str}")
        if url:
            lines.append(f'<a href="{url}">{question}</a>')
        else:
            lines.append(question)

        for o in r.get("outcomes", []):
            name = h(o.get("name", "?"))
            pre = o.get("pre_match_price")
            cur = o.get("current_price")
            change = o.get("change_cents")
            pre_s = f"{pre*100:.1f}" if pre is not None else "-"
            cur_s = f"{cur*100:.1f}" if cur is not None else "-"
            chg_s = ""
            if change is not None:
                sign = "+" if change > 0 else ""
                chg_s = f" ({sign}{change:.1f}c)"
            lines.append(f"  {name}: {pre_s}c \u2192 {cur_s}c{chg_s}")
        lines.append("")

    lines.append(f'<a href="{SITE_URL}">Open Monitor</a>')
    return "\n".join(lines)


def _tg_fmt_num(n) -> str:
    if n is None:
        return "?"
    if n >= 1e6:
        return f"{n/1e6:.1f}M"
    if n >= 1e3:
        return f"{n/1e3:.1f}K"
    return str(round(n))


def send_telegram(text: str) -> dict:
    """Send a message via Telegram Bot API. Returns the API response."""
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


@app.route("/api/telegram-send", methods=["POST"])
def api_telegram_send():
    """Send swing markets to Telegram. Expects JSON body with {markets: [...]}."""
    data = req.get_json(silent=True) or {}
    markets = data.get("markets", [])
    if not markets:
        return jsonify({"ok": False, "error": "No markets provided"}), 400

    text = _format_telegram_message(markets)
    result = send_telegram(text)
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
                {"name": "Team Alpha", "pre_match_price": 0.72, "current_price": 0.35, "change_cents": -37.0},
                {"name": "Team Beta", "pre_match_price": 0.28, "current_price": 0.65, "change_cents": 37.0},
            ],
        },
        {
            "sport": "NBA",
            "market_question": "Lakers vs Celtics",
            "polymarket_url": "https://polymarket.com/event/test-nba",
            "volume": 250000,
            "live_minutes": 92,
            "outcomes": [
                {"name": "Lakers", "pre_match_price": 0.68, "current_price": 0.30, "change_cents": -38.0},
                {"name": "Celtics", "pre_match_price": 0.32, "current_price": 0.70, "change_cents": 38.0},
            ],
        },
    ]
    text = _format_telegram_message(fake_markets)
    result = send_telegram(text)
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
