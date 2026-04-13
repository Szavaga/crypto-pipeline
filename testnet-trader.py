"""
Binance Testnet Autotrader
Executes real trades on Binance Spot Testnet (virtual money, real API mechanics)
based on ML model signals from predict-today.py.

Adds stop-loss (4%) and take-profit (8%) via OCO orders — the key upgrade
over paper-trader.py which relies only on daily signal exits.

Usage: python testnet-trader.py
       (or via run-all.py automatically after predict-today.py)

Prerequisites:
  1. pip install python-dotenv
  2. Create .env with TESTNET_API_KEY and TESTNET_SECRET_KEY
     Get keys at https://testnet.binance.vision (log in with GitHub)
"""

import pandas as pd
import json
import os
import sys
import math
import hmac
import hashlib
import time
import urllib.parse
import requests
from datetime import datetime, timezone

# Load .env file if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed; fall back to environment variables

# Telegram notifications (optional — graceful degradation if module missing)
try:
    import telegram_notify as _tg_mod
    _TG_OK = True
except ImportError:
    _TG_OK = False

def _tg(msg: str):
    if _TG_OK:
        _tg_mod.send_message(msg)

# ── Constants ──────────────────────────────────────────────────────────────────

TESTNET_BASE   = "https://testnet.binance.vision"
BINANCE_BASE   = "https://api.binance.com"   # real Binance for market data (candles/ATR)
DATA_DIR       = "data"
POSITIONS_PATH = os.path.join(DATA_DIR, "testnet_positions.json")
LEDGER_PATH    = os.path.join(DATA_DIR, "testnet_ledger.csv")
SIGNAL_LOG     = os.path.join(DATA_DIR, "signal_log.csv")

COINS        = ["ETH", "BTC", "SOL"]
COIN_SYMBOLS = {"ETH": "ETHUSDT", "BTC": "BTCUSDT", "SOL": "SOLUSDT"}

INITIAL_CAPITAL = 200.0   # USDT allocated per coin on testnet
COMMISSION      = 0.001   # 0.1% Binance fee (per side)
SLIPPAGE_PCT    = 0.0005  # 0.05% market-order slippage (testnet book is thin vs mainnet)

SL_PCT         = 0.04     # 4% stop-loss
TP_PCT         = 0.08     # 8% take-profit (2:1 R/R)
SL_LIMIT_OFFSET = 0.002   # 0.2% below stop trigger to ensure fill
CONF_THRESHOLD = {"BTC": 55.0, "ETH": 57.0, "SOL": 65.0}  # per-coin minimum prob_up % to enter
MAX_KELLY_PCT  = 25.0     # cap Kelly position size at 25% of balance

# Cache for exchange info (avoid repeated API calls)
_lot_size_cache: dict = {}

# Server time offset (ms) — set by test_connectivity() on startup
_server_time_offset: int = 0


# ── Authentication ─────────────────────────────────────────────────────────────

def load_keys() -> tuple:
    """Load API keys from environment (.env file or system env)."""
    api_key = os.environ.get("TESTNET_API_KEY", "").strip()
    secret  = os.environ.get("TESTNET_SECRET_KEY", "").strip()
    if not api_key or not secret:
        raise RuntimeError(
            "Missing TESTNET_API_KEY or TESTNET_SECRET_KEY.\n"
            "Create a .env file with these keys.\n"
            "Get testnet keys at: https://testnet.binance.vision"
        )
    return api_key, secret


def _sign(params: dict, secret: str) -> dict:
    """Add timestamp and HMAC-SHA256 signature to a request params dict."""
    params["timestamp"]  = int(time.time() * 1000) + _server_time_offset
    params["recvWindow"] = 5000
    query     = urllib.parse.urlencode(params)
    signature = hmac.new(secret.encode(), query.encode(), hashlib.sha256).hexdigest()
    params["signature"] = signature
    return params


def signed_request(method: str, endpoint: str, params: dict,
                   api_key: str, secret: str) -> dict:
    """Execute a signed Binance API request. Raises RuntimeError on failure."""
    signed_params = _sign(params.copy(), secret)
    url     = TESTNET_BASE + endpoint
    headers = {"X-MBX-APIKEY": api_key}

    if method == "GET":
        resp = requests.get(url, params=signed_params, headers=headers, timeout=10)
    elif method == "POST":
        resp = requests.post(url, params=signed_params, headers=headers, timeout=10)
    elif method == "DELETE":
        resp = requests.delete(url, params=signed_params, headers=headers, timeout=10)
    else:
        raise ValueError(f"Unsupported method: {method}")

    if resp.status_code != 200:
        raise RuntimeError(
            f"Binance API error {resp.status_code}: {resp.text}"
        )
    return resp.json()


# ── Connectivity check ─────────────────────────────────────────────────────────

def test_connectivity(api_key: str, secret: str) -> bool:
    """
    Verify testnet connectivity, sync server time, and validate API keys.
    Sets global _server_time_offset to compensate for Windows clock drift.
    """
    global _server_time_offset

    # 1. Ping
    try:
        resp = requests.get(TESTNET_BASE + "/api/v3/ping", timeout=10)
        if resp.status_code != 200:
            print(f"  ✗  Testnet ping failed: {resp.status_code}")
            return False
    except Exception as e:
        print(f"  ✗  Cannot reach testnet ({e})")
        return False

    # 2. Server time sync (fix Windows clock drift, error -1021)
    try:
        resp = requests.get(TESTNET_BASE + "/api/v3/time", timeout=10)
        server_ms = resp.json()["serverTime"]
        local_ms  = int(time.time() * 1000)
        _server_time_offset = server_ms - local_ms
        if abs(_server_time_offset) > 1000:
            print(f"  ⚠  Clock offset {_server_time_offset}ms — adjusting timestamps")
    except Exception:
        pass  # Non-fatal; proceed with 0 offset

    # 3. Validate keys by fetching account
    try:
        signed_request("GET", "/api/v3/account", {}, api_key, secret)
        print("  ✓  Testnet API keys valid")
        return True
    except RuntimeError as e:
        print(f"  ✗  API key validation failed: {e}")
        return False


# ── Exchange info & lot sizing ─────────────────────────────────────────────────

def get_lot_size(symbol: str) -> dict:
    """
    Fetch LOT_SIZE and PRICE_FILTER constraints for a symbol.
    Returns dict with step_size, min_qty, min_notional, tick_size.
    Cached in module-level dict.
    """
    if symbol in _lot_size_cache:
        return _lot_size_cache[symbol]

    resp = requests.get(
        TESTNET_BASE + "/api/v3/exchangeInfo",
        params={"symbol": symbol},
        timeout=10
    )
    if resp.status_code != 200:
        # Fallback defaults for ETHUSDT
        return {"step_size": 0.0001, "min_qty": 0.0001,
                "min_notional": 10.0,  "tick_size": 0.01}

    filters = resp.json()["symbols"][0]["filters"]
    result  = {"step_size": 0.0001, "min_qty": 0.0001,
               "min_notional": 10.0, "tick_size": 0.01}

    for f in filters:
        if f["filterType"] == "LOT_SIZE":
            result["step_size"] = float(f["stepSize"])
            result["min_qty"]   = float(f["minQty"])
        elif f["filterType"] == "PRICE_FILTER":
            result["tick_size"] = float(f["tickSize"])
        elif f["filterType"] == "MIN_NOTIONAL":
            result["min_notional"] = float(f.get("minNotional", 10.0))
        elif f["filterType"] == "NOTIONAL":
            result["min_notional"] = float(f.get("minNotional", 10.0))

    _lot_size_cache[symbol] = result
    return result


def snap_to_lot_size(raw_qty: float, step_size: float, min_qty: float) -> float:
    """
    Snap quantity DOWN to nearest step_size. Always floor, never round up
    (rounding up could exceed available balance).
    Returns 0.0 if result is below min_qty.
    """
    # Determine decimal precision from step_size string representation
    step_str  = f"{step_size:.10f}".rstrip("0")
    decimals  = len(step_str.split(".")[1]) if "." in step_str else 0
    steps     = math.floor(raw_qty / step_size)
    quantity  = round(steps * step_size, decimals)
    return quantity if quantity >= min_qty else 0.0


def round_price(price: float, tick_size: float) -> float:
    """Round price to tick_size precision."""
    tick_str = f"{tick_size:.10f}".rstrip("0")
    decimals = len(tick_str.split(".")[1]) if "." in tick_str else 2
    steps    = round(price / tick_size)
    return round(steps * tick_size, decimals)


# ── Price & balance ────────────────────────────────────────────────────────────

def get_current_price(symbol: str) -> float:
    """Fetch live price from Binance testnet."""
    try:
        resp = requests.get(
            TESTNET_BASE + "/api/v3/ticker/price",
            params={"symbol": symbol},
            timeout=10
        )
        return float(resp.json()["price"])
    except Exception:
        return 0.0


def get_4h_confirmation(symbol: str) -> bool:
    """
    Check if the most recent 4H candle confirms a bullish bias.
    Requires 4H RSI14 > 50 AND 4H MACD histogram > 0.
    Uses real Binance (not testnet) for reliable candle data.
    Returns True if confirmed, False if no confirmation (skip entry).
    """
    try:
        resp = requests.get(
            BINANCE_BASE + "/api/v3/klines",
            params={"symbol": symbol, "interval": "4h", "limit": 30},
            timeout=10
        )
        candles = resp.json()
        closes  = [float(c[4]) for c in candles]
        highs   = [float(c[2]) for c in candles]
        lows    = [float(c[3]) for c in candles]

        # RSI14
        deltas = [closes[i] - closes[i-1] for i in range(1, len(closes))]
        gains  = [d if d > 0 else 0 for d in deltas]
        losses = [-d if d < 0 else 0 for d in deltas]
        avg_gain = sum(gains[-14:]) / 14
        avg_loss = sum(losses[-14:]) / 14
        rsi = 100 - (100 / (1 + avg_gain / avg_loss)) if avg_loss > 0 else 100

        # MACD (12, 26, 9) — simple EMA approximation
        def ema(data, period):
            k = 2 / (period + 1)
            e = data[0]
            for v in data[1:]:
                e = v * k + e * (1 - k)
            return e

        macd_line   = ema(closes, 12) - ema(closes, 26)
        signal_line = ema(closes[-9:], 9) if len(closes) >= 9 else macd_line
        macd_hist   = macd_line - signal_line

        confirmed = rsi > 50 and macd_hist > 0
        print(f"  4H check: RSI={rsi:.1f}  MACD_hist={macd_hist:.4f}  → {'✓ confirmed' if confirmed else '✗ no confirm'}")
        return confirmed
    except Exception as e:
        print(f"  ⚠  4H confirmation failed ({e}) — proceeding anyway")
        return True  # fail open: don't block trades on API errors


def get_atr14(symbol: str) -> float:
    """
    Fetch last 20 daily candles from real Binance and compute ATR14.
    Returns ATR as an absolute price value (e.g. 2500.0 for BTC).
    Returns 0.0 on failure (caller falls back to fixed SL/TP).
    """
    try:
        resp = requests.get(
            BINANCE_BASE + "/api/v3/klines",
            params={"symbol": symbol, "interval": "1d", "limit": 20},
            timeout=10
        )
        candles = resp.json()
        trs = []
        for i in range(1, len(candles)):
            high  = float(candles[i][2])
            low   = float(candles[i][3])
            prev_close = float(candles[i-1][4])
            tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
            trs.append(tr)
        return sum(trs[-14:]) / 14 if len(trs) >= 14 else 0.0
    except Exception as e:
        print(f"  ⚠  ATR14 fetch failed ({e}) — using fixed SL/TP")
        return 0.0


def get_account_balance(api_key: str, secret: str) -> dict:
    """Fetch free balances from testnet account. Returns {asset: free_amount}."""
    data = signed_request("GET", "/api/v3/account", {}, api_key, secret)
    return {b["asset"]: float(b["free"]) for b in data.get("balances", [])
            if float(b["free"]) > 0}


# ── State management ───────────────────────────────────────────────────────────

def load_positions() -> dict:
    """
    Load testnet positions from JSON. Initializes fresh state if file missing.
    Key difference from paper_positions.json: adds entry_order_id,
    oco_order_list_id, sl_price, tp_price for real order tracking.
    Any coins in COINS but missing from the saved file get initialized fresh.
    """
    default = lambda: {
        "in_position":       False,
        "entry_price":       0.0,
        "quantity":          0.0,
        "value_usdt":        0.0,
        "entry_date":        "",
        "entry_order_id":    "",
        "oco_order_list_id": -1,
        "sl_price":          0.0,
        "tp_price":          0.0,
        "confidence":        0.0,
        "kelly_pct":         0.0,
        "balance":           INITIAL_CAPITAL,
    }

    if os.path.exists(POSITIONS_PATH):
        with open(POSITIONS_PATH, "r") as f:
            positions = json.load(f)
        # Add any newly configured coins that aren't saved yet
        for coin in COINS:
            if coin not in positions:
                print(f"  ℹ  {coin} not in positions file — initializing with ${INITIAL_CAPITAL}")
                positions[coin] = default()
        return positions

    return {coin: default() for coin in COINS}


def save_positions(positions: dict):
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(POSITIONS_PATH, "w") as f:
        json.dump(positions, f, indent=2)


def load_ledger() -> pd.DataFrame:
    if os.path.exists(LEDGER_PATH):
        return pd.read_csv(LEDGER_PATH, parse_dates=["date"])
    return pd.DataFrame(columns=[
        "date", "coin", "action", "price", "quantity", "value_usdt",
        "commission", "pnl", "pnl_pct", "balance_after",
        "signal_confidence", "kelly_pct", "reason",
        "order_id", "oco_order_list_id",
    ])


def save_ledger(df: pd.DataFrame):
    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_csv(LEDGER_PATH, index=False)


# ── Signal loader ──────────────────────────────────────────────────────────────

def get_latest_signals(today: str) -> dict:
    """
    Read signal_log.csv, return most recent signal per coin for today only.
    Stale signal guard: skip if signal date != today (missed run protection).
    """
    if not os.path.exists(SIGNAL_LOG):
        return {}

    df = pd.read_csv(SIGNAL_LOG)
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")

    latest = {}
    for coin in COINS:
        rows = df[df["coin"] == coin].sort_values("date", ascending=False)
        if not rows.empty:
            row = rows.iloc[0].to_dict()
            if row["date"] == today:
                latest[coin] = row
    return latest


# ── Order placement ────────────────────────────────────────────────────────────

def place_market_buy(symbol: str, quantity: float,
                     api_key: str, secret: str) -> dict:
    """
    Place a MARKET BUY order. Returns {order_id, avg_price, executed_qty}.
    avg_price is computed from fills for accuracy.
    """
    params = {
        "symbol":   symbol,
        "side":     "BUY",
        "type":     "MARKET",
        "quantity": quantity,
    }
    resp = signed_request("POST", "/api/v3/order", params, api_key, secret)

    # Compute weighted average fill price
    fills = resp.get("fills", [])
    if fills:
        total_qty  = sum(float(f["qty"]) for f in fills)
        total_cost = sum(float(f["price"]) * float(f["qty"]) for f in fills)
        avg_price  = total_cost / total_qty if total_qty > 0 else 0.0
    else:
        avg_price = float(resp.get("price", 0))

    return {
        "order_id":     str(resp["orderId"]),
        "avg_price":    avg_price,
        "executed_qty": float(resp["executedQty"]),
    }


def place_oco_sell(symbol: str, quantity: float, tp_price: float,
                   sl_price: float, tick_size: float,
                   api_key: str, secret: str) -> dict:
    """
    Place an OCO (One-Cancels-Other) SELL order with:
      - LIMIT_MAKER leg at tp_price (take-profit)
      - STOP_LOSS_LIMIT leg at sl_price trigger, sl_limit_price fill

    sl_limit_price is set 0.2% below sl_price to ensure the limit fills
    in fast-moving markets.

    Returns {order_list_id, tp_price, sl_price} on success.
    Raises RuntimeError on failure.
    """
    sl_limit_price = round_price(sl_price * (1 - SL_LIMIT_OFFSET), tick_size)
    tp_rounded     = round_price(tp_price, tick_size)
    sl_rounded     = round_price(sl_price, tick_size)

    params = {
        "symbol":          symbol,
        "side":            "SELL",
        "quantity":        quantity,
        "aboveType":       "LIMIT_MAKER",
        "abovePrice":      tp_rounded,
        "belowType":       "STOP_LOSS_LIMIT",
        "belowStopPrice":  sl_rounded,
        "belowPrice":      sl_limit_price,
    }
    resp = signed_request("POST", "/api/v3/orderList/oco", params, api_key, secret)

    return {
        "order_list_id": int(resp["orderListId"]),
        "tp_price":      tp_rounded,
        "sl_price":      sl_rounded,
    }


def cancel_oco(symbol: str, order_list_id: int,
               api_key: str, secret: str) -> bool:
    """Cancel an active OCO order list. Returns True on success."""
    try:
        params = {"symbol": symbol, "orderListId": order_list_id}
        signed_request("DELETE", "/api/v3/orderList", params, api_key, secret)
        return True
    except RuntimeError as e:
        print(f"  ⚠  Could not cancel OCO {order_list_id}: {e}")
        return False


def place_market_sell(symbol: str, quantity: float,
                      api_key: str, secret: str) -> dict:
    """Place a MARKET SELL order. Returns {order_id, avg_price, executed_qty}."""
    params = {
        "symbol":   symbol,
        "side":     "SELL",
        "type":     "MARKET",
        "quantity": quantity,
    }
    resp = signed_request("POST", "/api/v3/order", params, api_key, secret)

    fills = resp.get("fills", [])
    if fills:
        total_qty  = sum(float(f["qty"]) for f in fills)
        total_cost = sum(float(f["price"]) * float(f["qty"]) for f in fills)
        avg_price  = total_cost / total_qty if total_qty > 0 else 0.0
    else:
        avg_price = float(resp.get("price", 0))

    return {
        "order_id":     str(resp["orderId"]),
        "avg_price":    avg_price,
        "executed_qty": float(resp["executedQty"]),
    }


# ── OCO fill detection ─────────────────────────────────────────────────────────

def check_oco_fills(positions: dict, api_key: str, secret: str) -> list:
    """
    For each coin in_position with an active OCO, check if it has fired.
    Binance executes OCO in real-time; we just read the result on each daily run.

    Returns list of fill event dicts:
      {coin, reason ("TP_HIT"|"SL_HIT"), exit_price, exit_qty, order_id, oco_order_list_id}
    """
    fill_events = []

    for coin in COINS:
        pos = positions.get(coin, {})
        if not pos.get("in_position") or pos.get("oco_order_list_id", -1) == -1:
            continue

        order_list_id = pos["oco_order_list_id"]
        symbol        = COIN_SYMBOLS[coin]

        try:
            params = {"symbol": symbol, "orderListId": order_list_id}
            resp   = signed_request("GET", "/api/v3/orderList", params,
                                    api_key, secret)
        except RuntimeError as e:
            print(f"  ⚠  Could not check OCO {order_list_id} for {coin}: {e}")
            continue

        list_status = resp.get("listOrderStatus", "")

        if list_status == "ALL_DONE":
            # Find the FILLED order in the pair
            filled_order = None
            for order in resp.get("orders", []):
                if order.get("status") == "FILLED":
                    filled_order = order
                    break

            if not filled_order:
                # Both might show as FILLED in rare cases; use the last one
                orders = resp.get("orders", [])
                filled_order = orders[-1] if orders else None

            if not filled_order:
                print(f"  ⚠  OCO {order_list_id} ALL_DONE but no FILLED order found")
                continue

            # Determine exit price from cummulativeQuoteQty / executedQty
            exec_qty   = float(filled_order.get("executedQty", 0) or 0)
            cum_quote  = float(filled_order.get("cummulativeQuoteQty", 0) or 0)
            exit_price = (cum_quote / exec_qty) if exec_qty > 0 else float(filled_order.get("price", 0))

            # Determine if TP or SL hit: compare against stored tp_price with 2% tolerance
            tp_threshold = pos["tp_price"] * 0.98
            reason = "TP_HIT" if exit_price >= tp_threshold else "SL_HIT"

            fill_events.append({
                "coin":              coin,
                "reason":            reason,
                "exit_price":        exit_price,
                "exit_qty":          exec_qty if exec_qty > 0 else pos["quantity"],
                "order_id":          str(filled_order.get("orderId", "")),
                "oco_order_list_id": order_list_id,
            })
            tag = "TP HIT" if reason == "TP_HIT" else "SL HIT"
            print(f"  {'✓' if reason == 'TP_HIT' else '✗'}  {coin} OCO {order_list_id} → {tag} @ ${exit_price:,.2f}")

        elif list_status == "EXECUTING":
            print(f"  —  {coin} OCO {order_list_id} still open (EXECUTING)")
        else:
            print(f"  ?  {coin} OCO {order_list_id} status: {list_status}")

    return fill_events


# ── Trade logic ────────────────────────────────────────────────────────────────

def process_fills(positions: dict, ledger: pd.DataFrame,
                  fills: list, today: str) -> tuple:
    """
    Record realized P&L for each OCO fill event (TP_HIT or SL_HIT).
    Updates positions and appends rows to ledger.
    """
    new_rows = []

    for event in fills:
        coin        = event["coin"]
        exit_price  = event["exit_price"]
        exit_qty    = event["exit_qty"]
        reason      = event["reason"]
        pos         = positions[coin]

        exit_value  = exit_qty * exit_price
        commission  = exit_value * COMMISSION
        net_exit    = exit_value - commission
        pnl         = net_exit - pos["value_usdt"]
        pnl_pct     = pnl / pos["value_usdt"] * 100 if pos["value_usdt"] > 0 else 0

        pos["balance"]          += net_exit
        pos["in_position"]       = False
        pos["entry_price"]       = 0.0
        pos["quantity"]          = 0.0
        pos["value_usdt"]        = 0.0
        pos["entry_order_id"]    = ""
        pos["oco_order_list_id"] = -1
        pos["sl_price"]          = 0.0
        pos["tp_price"]          = 0.0

        action = "SELL_TP" if reason == "TP_HIT" else "SELL_SL"
        new_rows.append({
            "date":               today,
            "coin":               coin,
            "action":             action,
            "price":              round(exit_price, 4),
            "quantity":           round(exit_qty, 6),
            "value_usdt":         round(net_exit, 2),
            "commission":         round(commission, 4),
            "pnl":                round(pnl, 4),
            "pnl_pct":            round(pnl_pct, 2),
            "balance_after":      round(pos["balance"], 2),
            "signal_confidence":  pos["confidence"],
            "kelly_pct":          pos["kelly_pct"],
            "reason":             reason,
            "order_id":           event["order_id"],
            "oco_order_list_id":  event["oco_order_list_id"],
        })
        emoji = "✓" if pnl >= 0 else "✗"
        print(f"  {emoji}  {coin} {action}  exit=${exit_price:,.2f}  "
              f"P&L=${pnl:+.2f} ({pnl_pct:+.2f}%)  "
              f"balance=${pos['balance']:.2f}")
        tg_emoji = "🎯" if reason == "TP_HIT" else "💥"
        tg_label = "TP HIT" if reason == "TP_HIT" else "SL HIT"
        _tg(
            f"{tg_emoji} <b>TESTNET {tg_label} — {coin}</b>\n"
            f"Fill: <b>${exit_price:,.2f}</b>\n"
            f"PnL: <b>${pnl:+.2f} ({pnl_pct:+.1f}%)</b>\n"
            f"Balance: ${pos['balance']:.2f}"
        )

    if new_rows:
        ledger = pd.concat([ledger, pd.DataFrame(new_rows)], ignore_index=True)

    return positions, ledger


def execute_exits_by_signal(positions: dict, signals: dict,
                             ledger: pd.DataFrame, today: str,
                             api_key: str, secret: str) -> tuple:
    """
    If the ML model signals STAY OUT and we are in position, exit immediately.
    Cancel the active OCO first, then place a MARKET SELL.
    This overrides the SL/TP and closes the trade based on the model's view.
    """
    new_rows = []

    for coin in COINS:
        pos    = positions.get(coin, {})
        signal = signals.get(coin, {})

        if not pos.get("in_position"):
            continue

        sig_text = str(signal.get("signal", ""))
        prob_up  = float(signal.get("prob_up", 50))

        # Exit if signal is bearish (STAY OUT) or confidence too low
        is_bearish = "OUT" in sig_text or (
            "SKIP" not in sig_text and "BUY" not in sig_text and sig_text
        )
        if not is_bearish:
            continue

        symbol = COIN_SYMBOLS[coin]

        # Cancel active OCO before placing market sell
        oco_id = pos.get("oco_order_list_id", -1)
        if oco_id != -1:
            cancelled = cancel_oco(symbol, oco_id, api_key, secret)
            if cancelled:
                print(f"  ✓  {coin} OCO {oco_id} cancelled")
            # Even if cancel fails, attempt market sell (OCO may have already fired)

        # Place market sell
        try:
            fill = place_market_sell(symbol, pos["quantity"], api_key, secret)
        except RuntimeError as e:
            print(f"  ✗  {coin} market sell failed: {e}")
            # Position may be stuck — leave in_position=True for manual review
            continue

        exit_price = fill["avg_price"] * (1 - SLIPPAGE_PCT)  # worst-case fill on sell
        exit_qty   = fill["executed_qty"]
        exit_value = exit_qty * exit_price
        commission = exit_value * COMMISSION
        net_exit   = exit_value - commission
        pnl        = net_exit - pos["value_usdt"]
        pnl_pct    = pnl / pos["value_usdt"] * 100 if pos["value_usdt"] > 0 else 0

        pos["balance"]          += net_exit
        pos["in_position"]       = False
        pos["entry_price"]       = 0.0
        pos["quantity"]          = 0.0
        pos["value_usdt"]        = 0.0
        pos["entry_order_id"]    = ""
        pos["oco_order_list_id"] = -1
        pos["sl_price"]          = 0.0
        pos["tp_price"]          = 0.0

        new_rows.append({
            "date":               today,
            "coin":               coin,
            "action":             "SELL_SIGNAL",
            "price":              round(exit_price, 4),
            "quantity":           round(exit_qty, 6),
            "value_usdt":         round(net_exit, 2),
            "commission":         round(commission, 4),
            "pnl":                round(pnl, 4),
            "pnl_pct":            round(pnl_pct, 2),
            "balance_after":      round(pos["balance"], 2),
            "signal_confidence":  round(prob_up, 1),
            "kelly_pct":          0.0,
            "reason":             f"SELL_SIGNAL — {sig_text}",
            "order_id":           fill["order_id"],
            "oco_order_list_id":  oco_id,
        })
        emoji = "✓" if pnl >= 0 else "✗"
        print(f"  {emoji}  {coin} SIGNAL EXIT  @ ${exit_price:,.2f}  "
              f"P&L=${pnl:+.2f} ({pnl_pct:+.2f}%)  "
              f"balance=${pos['balance']:.2f}")
        _tg(
            f"🔴 <b>TESTNET SELL (signal exit) — {coin}</b>\n"
            f"Exit: <b>${exit_price:,.2f}</b>  qty={exit_qty}\n"
            f"PnL: <b>${pnl:+.2f} ({pnl_pct:+.1f}%)</b>\n"
            f"Balance: ${pos['balance']:.2f}"
        )

    if new_rows:
        ledger = pd.concat([ledger, pd.DataFrame(new_rows)], ignore_index=True)

    return positions, ledger


def execute_entries(positions: dict, signals: dict, ledger: pd.DataFrame,
                    today: str, api_key: str, secret: str,
                    lot_info_by_coin: dict) -> tuple:
    """
    For each coin with a BUY signal and no open position:
    1. Calculate Kelly-sized position
    2. Snap quantity to lot size
    3. Place MARKET BUY
    4. Place OCO SELL with SL and TP
    If OCO placement fails after a buy, the position is left open without
    stops (logged as WARNING) — the signal-based exit in the next run serves
    as a fallback.
    """
    new_rows = []

    for coin in COINS:
        pos    = positions.get(coin, {})
        signal = signals.get(coin, {})

        if pos.get("in_position"):
            print(f"  —  {coin} already in position — skipping entry")
            continue

        if not signal:
            print(f"  —  {coin} no signal today — skipping")
            continue

        sig_text  = str(signal.get("signal", ""))
        prob_up   = float(signal.get("prob_up", 50))
        kelly_pct = float(signal.get("kelly_pct", 0))

        if "BUY" not in sig_text or prob_up < CONF_THRESHOLD.get(coin, 55.0):
            status = "STAY OUT" if "OUT" in sig_text else "SKIP"
            print(f"  —  {coin} {status} (prob_up={prob_up:.1f}%)  balance=${pos['balance']:.2f}")
            continue

        # 4H entry confirmation — skip if 4H momentum is not bullish
        if not get_4h_confirmation(COIN_SYMBOLS[coin]):
            print(f"  —  {coin} BUY signal but 4H not confirmed — skipping entry")
            continue

        lot_info = lot_info_by_coin.get(coin, {})
        balance = pos["balance"]
        if balance < lot_info.get("min_notional", 10.0):
            print(f"  ⚠  {coin} balance ${balance:.2f} too low — skipping")
            continue

        # Position sizing
        use_kelly    = min(kelly_pct, MAX_KELLY_PCT) if kelly_pct > 0 else 10.0
        invest_amt   = balance * use_kelly / 100
        current_price = get_current_price(COIN_SYMBOLS[coin])
        if current_price == 0:
            print(f"  ⚠  {coin} could not fetch price — skipping")
            continue

        raw_qty  = invest_amt / current_price
        step_size = lot_info["step_size"]
        min_qty   = lot_info["min_qty"]
        quantity  = snap_to_lot_size(raw_qty, step_size, min_qty)

        if quantity == 0:
            print(f"  ⚠  {coin} quantity rounds to 0 (invest=${invest_amt:.2f}) — skipping")
            continue

        actual_cost = quantity * current_price
        if actual_cost < lot_info["min_notional"]:
            print(f"  ⚠  {coin} order value ${actual_cost:.2f} below min notional — skipping")
            continue

        symbol = COIN_SYMBOLS[coin]
        print(f"  →  {coin} BUY  qty={quantity}  ~${actual_cost:.2f}  "
              f"kelly={use_kelly:.1f}%  conf={prob_up:.1f}%")

        # Place market buy
        try:
            buy_fill = place_market_buy(symbol, quantity, api_key, secret)
        except RuntimeError as e:
            print(f"  ✗  {coin} market buy failed: {e}")
            continue

        entry_price  = buy_fill["avg_price"] * (1 + SLIPPAGE_PCT)  # worst-case fill vs quote
        executed_qty = buy_fill["executed_qty"]
        entry_cost   = executed_qty * entry_price
        commission   = entry_cost * COMMISSION
        net_cost     = entry_cost + commission  # total cash spent (includes slippage + fee)

        # Calculate OCO prices — ATR-based if available, fixed fallback
        tick_size = lot_info["tick_size"]
        atr = get_atr14(COIN_SYMBOLS[coin])
        if atr > 0:
            raw_sl_pct = atr / entry_price
            # Clamp: SL between 2% and 8% of entry price
            sl_pct = max(0.02, min(0.08, raw_sl_pct * 2))
            tp_pct = sl_pct * 2   # maintain 2:1 R/R
            print(f"  ATR14=${atr:,.2f}  → SL={sl_pct*100:.1f}%  TP={tp_pct*100:.1f}%")
        else:
            sl_pct = SL_PCT
            tp_pct = TP_PCT
        tp_price  = entry_price * (1 + tp_pct)
        sl_price  = entry_price * (1 - sl_pct)

        # Place OCO sell order (SL + TP)
        oco_order_list_id = -1
        try:
            oco = place_oco_sell(symbol, executed_qty, tp_price, sl_price,
                                  tick_size, api_key, secret)
            oco_order_list_id = oco["order_list_id"]
            tp_price = oco["tp_price"]
            sl_price = oco["sl_price"]
            print(f"  ✓  {coin} OCO placed  "
                  f"SL=${sl_price:,.2f}  TP=${tp_price:,.2f}  "
                  f"orderListId={oco_order_list_id}")
        except RuntimeError as e:
            print(f"  ⚠  {coin} OCO placement FAILED: {e}")
            print(f"      Position open without stops — will exit on next STAY OUT signal")

        # Update position state
        pos["in_position"]       = True
        pos["entry_price"]       = entry_price
        pos["quantity"]          = executed_qty
        pos["value_usdt"]        = net_cost
        pos["entry_date"]        = today
        pos["entry_order_id"]    = buy_fill["order_id"]
        pos["oco_order_list_id"] = oco_order_list_id
        pos["sl_price"]          = sl_price
        pos["tp_price"]          = tp_price
        pos["confidence"]        = prob_up
        pos["kelly_pct"]         = use_kelly
        pos["balance"]          -= net_cost

        new_rows.append({
            "date":               today,
            "coin":               coin,
            "action":             "BUY",
            "price":              round(entry_price, 4),
            "quantity":           round(executed_qty, 6),
            "value_usdt":         round(net_cost, 2),
            "commission":         round(commission, 4),
            "pnl":                0.0,
            "pnl_pct":            0.0,
            "balance_after":      round(pos["balance"], 2),
            "signal_confidence":  round(prob_up, 1),
            "kelly_pct":          round(use_kelly, 1),
            "reason":             str(signal.get("confidence", "")),
            "order_id":           buy_fill["order_id"],
            "oco_order_list_id":  oco_order_list_id,
        })
        print(f"  ✓  {coin} ENTERED  @ ${entry_price:,.2f}  "
              f"qty={executed_qty}  cost=${net_cost:.2f}  "
              f"balance=${pos['balance']:.2f}")
        _tg(
            f"🟢 <b>TESTNET BUY — {coin}</b>\n"
            f"Entry: <b>${entry_price:,.2f}</b>  qty={executed_qty}\n"
            f"Cost: ${net_cost:.2f}  Kelly: {use_kelly:.1f}%\n"
            f"SL: ${sl_price:,.2f}  TP: ${tp_price:,.2f}\n"
            f"Balance left: ${pos['balance']:.2f}"
        )

    if new_rows:
        ledger = pd.concat([ledger, pd.DataFrame(new_rows)], ignore_index=True)

    return positions, ledger


# ── Summary ────────────────────────────────────────────────────────────────────

def print_summary(positions: dict, today: str):
    print(f"\n{'─'*52}")
    print("  Testnet Portfolio")
    print(f"{'─'*52}")
    total = 0.0
    for coin in COINS:
        pos = positions.get(coin, {})
        bal = pos.get("balance", 0)
        if pos.get("in_position"):
            price    = get_current_price(COIN_SYMBOLS[coin])
            curr_val = pos["quantity"] * price
            unreal   = curr_val - pos["value_usdt"]
            unreal_p = unreal / pos["value_usdt"] * 100 if pos["value_usdt"] > 0 else 0
            total   += bal + curr_val
            sl_str   = f"SL=${pos['sl_price']:,.2f}" if pos["sl_price"] > 0 else "no SL"
            tp_str   = f"TP=${pos['tp_price']:,.2f}" if pos["tp_price"] > 0 else "no TP"
            print(f"  {coin:<4} IN POSITION  "
                  f"entry=${pos['entry_price']:,.2f}  now=${price:,.2f}  "
                  f"unreal={unreal_p:+.2f}%  {sl_str}  {tp_str}  "
                  f"cash=${bal:.2f}")
        else:
            total += bal
            print(f"  {coin:<4} FLAT         balance=${bal:.2f}")

    initial = INITIAL_CAPITAL * len(COINS)
    ret_pct  = (total / initial - 1) * 100 if initial > 0 else 0
    print(f"\n  Total: ${total:,.2f}  ({ret_pct:+.2f}% from ${initial:.0f})")


def save_summary(positions: dict, ledger: pd.DataFrame, today: str):
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    total_bal = sum(p.get("balance", 0) for p in positions.values())
    summary = {
        "updated":  now,
        "today":    today,
        "initial":  INITIAL_CAPITAL * len(COINS),
        "balance":  total_bal,
        "coins":    {
            coin: {
                "in_position":  positions[coin].get("in_position", False),
                "balance":      positions[coin].get("balance", 0),
                "entry_price":  positions[coin].get("entry_price", 0),
                "sl_price":     positions[coin].get("sl_price", 0),
                "tp_price":     positions[coin].get("tp_price", 0),
                "quantity":     positions[coin].get("quantity", 0),
                "value_usdt":   positions[coin].get("value_usdt", 0),
                "kelly_pct":    positions[coin].get("kelly_pct", 0),
                "confidence":   positions[coin].get("confidence", 0),
                "entry_date":   positions[coin].get("entry_date", ""),
            }
            for coin in COINS
        },
    }
    path = os.path.join(DATA_DIR, "testnet_summary.json")
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    return path


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    now   = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    print(f"\n{'='*52}")
    print("  Binance Testnet Autotrader")
    print(f"  {now}")
    print(f"{'='*52}\n")

    # Load API keys
    try:
        api_key, secret = load_keys()
    except RuntimeError as e:
        print(f"  ✗  {e}")
        sys.exit(1)

    # Connectivity check (also syncs clock)
    print("  Checking testnet connectivity...")
    if not test_connectivity(api_key, secret):
        print("  ✗  Cannot connect to testnet — aborting")
        sys.exit(1)   # Exit with error so GitHub Actions shows a FAILED step

    # Load state
    positions = load_positions()
    ledger    = load_ledger()

    # Load signals (today only)
    signals = get_latest_signals(today)
    if not signals:
        print(f"  ⚠  No signals for {today} — run predict-today.py first")
        _tg(f"⚠️ Testnet: no signals for {today} — entries skipped")
        # Still check fills / manage open positions
    else:
        print(f"  Signals loaded for: {list(signals.keys())}")

    # Fetch exchange info
    print("\n  Fetching exchange info...")
    lot_info_by_coin = {}
    for coin in COINS:
        symbol = COIN_SYMBOLS[coin]
        info   = get_lot_size(symbol)
        lot_info_by_coin[coin] = info
        print(f"  {symbol}: stepSize={info['step_size']}  "
              f"minNotional=${info['min_notional']}")

    print("\n  Checking OCO fills (SL/TP)...")
    fills = check_oco_fills(positions, api_key, secret)
    if fills:
        positions, ledger = process_fills(positions, ledger, fills, today)
    else:
        print("  —  No OCO fills detected")

    if signals:
        print("\n  Checking signal exits...")
        positions, ledger = execute_exits_by_signal(
            positions, signals, ledger, today, api_key, secret
        )

        print("\n  Checking signal entries...")
        positions, ledger = execute_entries(
            positions, signals, ledger, today, api_key, secret, lot_info_by_coin
        )

    # Save state (CSV/JSON)
    save_positions(positions)
    save_ledger(ledger)
    summary_path = save_summary(positions, ledger, today)

    # Save state (DB)
    try:
        import db
        db.init_schema()
        for coin, pos in positions.items():
            db.try_write(db.upsert_position, "testnet", coin, pos)
        # Write any new ledger rows to DB
        new_rows = ledger[ledger["date"] == today].to_dict("records")
        for row in new_rows:
            db.try_write(db.insert_trade, "testnet", row)
        if new_rows:
            print(f"  ✓  Trades → DB ({len(new_rows)} rows)")
        print(f"  ✓  Positions → DB")
    except Exception as e:
        print(f"  ⚠  DB skipped: {e}")

    # Print summary
    print_summary(positions, today)

    # Telegram end-of-run summary
    total_equity = sum(
        p.get("balance", 0) + p.get("value_usdt", 0) for p in positions.values()
    )
    coin_lines = []
    for coin in COINS:
        p = positions.get(coin, {})
        status = "IN POSITION" if p.get("in_position") else "flat"
        coin_lines.append(f"  {coin}: {status}  bal=${p.get('balance', 0):.2f}")
    _tg(
        f"📊 <b>Testnet Run Complete — {today}</b>\n"
        + "\n".join(coin_lines)
        + f"\nTotal equity: <b>${total_equity:.2f}</b>"
    )

    print(f"\n  ✓  Positions → {POSITIONS_PATH}")
    print(f"  ✓  Ledger    → {LEDGER_PATH}")
    print(f"  ✓  Summary   → {summary_path}")
    print(f"{'='*52}\n")


if __name__ == "__main__":
    main()
