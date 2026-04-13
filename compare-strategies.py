"""
Strategy Comparison + Paper Trading — runs 4 simple strategies alongside the ensemble.
Each strategy starts with $200 and opens/closes positions every 8-hour cycle.

Strategies:
  ensemble  — the full XGBoost ensemble from predict-today.py
  fund      — Funding Rate Contrarian (rule-based)
  rsi       — RSI Mean Reversion (rule-based)
  ml3       — Simple3 XGBoost (d_rsi14 + funding_rate + d_vol_ratio)
  ml5       — Simple5 XGBoost (above + fear_greed + d_macd_hist)

Position sizing: 20% of available capital per BUY signal.
Positions are held for one cycle (open now, close next run).
Commission: 0.1% entry + 0.1% exit = 0.2% round-trip.

Files written:
  data/simple_paper_state.json  — current capital + open positions per strategy
  data/simple_paper_ledger.csv  — full trade log
  data/strategy_comparison.json — signal comparison snapshot

Usage: python compare-strategies.py
Run after: predict-today.py && testnet-trader.py
"""

import csv
import importlib.util
import json
import os
import pickle
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests

DATA_DIR  = "data"
MODEL_DIR = "models"

COINS = {"BTC": "BTCUSDT", "ETH": "ETHUSDT", "SOL": "SOLUSDT",
         "AVAX": "AVAXUSDT", "LINK": "LINKUSDT"}

BINANCE_URL  = "https://api.binance.com/api/v3/klines"
FAPI_URL     = "https://fapi.binance.com/fapi/v1/fundingRate"
YFINANCE_MAP = {"BTCUSDT": "BTC-USD", "ETHUSDT": "ETH-USD", "SOLUSDT": "SOL-USD",
                "AVAXUSDT": "AVAX-USD", "LINKUSDT": "LINK-USD"}

CONF_THRESH     = 0.55
INITIAL_CAPITAL = 200.0
POSITION_SIZE   = 0.20    # 20% of cash per BUY signal
COMMISSION      = 0.001   # 0.1% per leg (0.2% round-trip)

STRATEGIES  = ["ensemble", "fund", "rsi", "ml3", "ml5"]
STATE_FILE  = os.path.join(DATA_DIR, "simple_paper_state.json")
LEDGER_FILE = os.path.join(DATA_DIR, "simple_paper_ledger.csv")


# ── Data helpers ──────────────────────────────────────────────────────────────

def _fetch_yf(symbol: str, limit: int) -> pd.DataFrame:
    import yfinance as yf
    yf_sym = YFINANCE_MAP.get(symbol, symbol)
    df = yf.Ticker(yf_sym).history(period="60d", interval="1d", auto_adjust=True)
    df = df.rename(columns={"Open": "open", "High": "high", "Low": "low",
                             "Close": "close", "Volume": "volume"})
    return df[["open", "high", "low", "close", "volume"]].tail(limit).reset_index(drop=True)


def fetch_daily(symbol: str, limit: int = 50) -> pd.DataFrame:
    try:
        resp = requests.get(BINANCE_URL,
            params={"symbol": symbol, "interval": "1d", "limit": limit},
            timeout=15)
        if resp.status_code == 451 or resp.status_code != 200:
            raise RuntimeError("geo-blocked or error")
        raw = resp.json()
        df  = pd.DataFrame(raw, columns=[
            "ts", "open", "high", "low", "close", "volume",
            "ct", "qv", "trades", "tb", "tq", "ignore"])
        for c in ["open", "high", "low", "close", "volume"]:
            df[c] = df[c].astype(float)
        return df[["open", "high", "low", "close", "volume"]]
    except Exception:
        try:
            return _fetch_yf(symbol, limit)
        except Exception:
            return pd.DataFrame()


def get_live_features(symbol: str) -> tuple:
    """Compute d_rsi14, d_macd_hist, d_vol_ratio. Returns (features_dict, current_price)."""
    df = fetch_daily(symbol, 50)
    if df.empty or len(df) < 20:
        return {}, 0.0

    price = float(df["close"].iloc[-1])

    try:
        import pandas_ta as ta
        d_rsi14     = float(ta.rsi(df["close"], length=14).iloc[-1])
        macd_df     = ta.macd(df["close"], fast=12, slow=26, signal=9)
        mh_col      = [c for c in macd_df.columns if c.startswith("MACDh_")][0]
        d_macd_hist = float(macd_df[mh_col].iloc[-1])
    except Exception:
        delta       = df["close"].diff()
        gain        = delta.clip(lower=0).rolling(14).mean()
        loss        = (-delta.clip(upper=0)).rolling(14).mean()
        rs          = gain / loss.replace(0, float("nan"))
        rsi_s       = 100 - 100 / (1 + rs)
        d_rsi14     = float(rsi_s.iloc[-1]) if not np.isnan(rsi_s.iloc[-1]) else 50.0
        ema12       = df["close"].ewm(span=12, adjust=False).mean()
        ema26       = df["close"].ewm(span=26, adjust=False).mean()
        macd_line   = ema12 - ema26
        sig_line    = macd_line.ewm(span=9, adjust=False).mean()
        d_macd_hist = float((macd_line - sig_line).iloc[-1])

    vol_ma20    = df["volume"].rolling(20).mean()
    d_vol_ratio = float(df["volume"].iloc[-1] / vol_ma20.iloc[-1]) \
                  if vol_ma20.iloc[-1] > 0 else 1.0

    if np.isnan(d_rsi14):      d_rsi14     = 50.0
    if np.isnan(d_macd_hist):  d_macd_hist = 0.0
    if np.isnan(d_vol_ratio):  d_vol_ratio = 1.0

    return {
        "d_rsi14":     round(d_rsi14, 2),
        "d_macd_hist": round(d_macd_hist, 6),
        "d_vol_ratio": round(d_vol_ratio, 3),
    }, price


def get_funding_rate(symbol: str) -> float:
    try:
        r    = requests.get(FAPI_URL, params={"symbol": symbol, "limit": 1}, timeout=10)
        data = r.json()
        return float(data[-1]["fundingRate"]) * 100
    except Exception:
        return 0.0


def get_fear_greed() -> int:
    try:
        r = requests.get("https://api.alternative.me/fng/?limit=1&format=json", timeout=10)
        return int(r.json()["data"][0]["value"])
    except Exception:
        return 50


# ── Signal generators ─────────────────────────────────────────────────────────

def strategy_funding_contrarian(funding_rate: float) -> str:
    if funding_rate < -0.01:
        return "BUY"
    elif funding_rate > 0.05:
        return "STAY OUT"
    return "SKIP"


def strategy_rsi_reversion(d_rsi14: float, d_vol_ratio: float) -> str:
    if d_rsi14 < 35 and d_vol_ratio > 1.2:
        return "BUY"
    elif d_rsi14 > 70:
        return "STAY OUT"
    return "SKIP"


def strategy_simple_ml(coin: str, suffix: str, features: dict) -> str:
    model_path = os.path.join(MODEL_DIR, f"{coin}_{suffix}.pkl")
    if not os.path.exists(model_path):
        return "N/A"
    try:
        with open(model_path, "rb") as fh:
            art = pickle.load(fh)
        vals = np.array([features.get(f, 0.0) for f in art["features"]], dtype=float)
        vals = np.nan_to_num(vals, nan=0.0).reshape(1, -1)
        Xs   = art["scaler"].transform(vals)
        p    = float(art["model"].predict_proba(Xs)[0][1])
        if p >= CONF_THRESH:
            return "BUY"
        elif (1 - p) >= CONF_THRESH:
            return "STAY OUT"
        return "SKIP"
    except Exception as e:
        print(f"      ⚠  {coin} {suffix}: {e}")
        return "ERR"


def load_ensemble_signals() -> dict:
    log_path = os.path.join(DATA_DIR, "signal_log.csv")
    if not os.path.exists(log_path):
        print("  ⚠  signal_log.csv not found — run predict-today.py first")
        return {}
    df = pd.read_csv(log_path)
    if df.empty:
        return {}
    latest = df.groupby("coin").last().reset_index()
    result = {}
    for _, row in latest.iterrows():
        sig = str(row.get("signal", "SKIP"))
        result[row["coin"]] = "BUY" if "BUY" in sig else "STAY OUT" if "OUT" in sig else "SKIP"
    return result


# ── Paper trading ─────────────────────────────────────────────────────────────

def load_state() -> dict:
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as fh:
            return json.load(fh)
    return {s: {"capital": INITIAL_CAPITAL, "positions": {}} for s in STRATEGIES}


def save_state(state: dict):
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(STATE_FILE, "w") as fh:
        json.dump(state, fh, indent=2)


def close_all_positions(state: dict, prices: dict, ts: str) -> list:
    """Close every open position at current prices. Returns closed trade records."""
    records = []
    for strat in STRATEGIES:
        for coin, pos in list(state[strat]["positions"].items()):
            price_now   = prices.get(coin, 0.0)
            if price_now <= 0:
                continue
            entry_price = pos["entry_price"]
            size_usd    = pos["size_usd"]
            proceeds    = size_usd * (price_now / entry_price)
            exit_fee    = proceeds * COMMISSION
            net_return  = proceeds - exit_fee
            pnl         = net_return - size_usd           # net P&L vs original stake

            state[strat]["capital"] += net_return
            records.append({
                "ts":           ts,
                "strategy":     strat,
                "coin":         coin,
                "action":       "CLOSE",
                "entry_price":  round(entry_price, 4),
                "exit_price":   round(price_now, 4),
                "size_usd":     round(size_usd, 4),
                "ret_pct":      round((price_now / entry_price - 1) * 100, 3),
                "pnl_usd":      round(pnl, 4),
                "capital":      round(state[strat]["capital"], 4),
                "win":          pnl > 0,
            })
        state[strat]["positions"] = {}
    return records


def open_new_positions(state: dict, signals_by_coin: dict, prices: dict, ts: str) -> list:
    """Open positions for every BUY signal. signals_by_coin: {coin: {strat: signal}}."""
    records = []
    for strat in STRATEGIES:
        for coin, sig_map in signals_by_coin.items():
            if sig_map.get(strat) != "BUY":
                continue
            price_now = prices.get(coin, 0.0)
            if price_now <= 0:
                continue
            cash     = state[strat]["capital"]
            size_usd = cash * POSITION_SIZE
            entry_fee = size_usd * COMMISSION
            state[strat]["capital"] -= (size_usd + entry_fee)
            state[strat]["positions"][coin] = {
                "entry_price": round(price_now, 4),
                "size_usd":    round(size_usd, 4),
                "entry_ts":    ts,
            }
            records.append({
                "ts":          ts,
                "strategy":    strat,
                "coin":        coin,
                "action":      "OPEN",
                "entry_price": round(price_now, 4),
                "exit_price":  None,
                "size_usd":    round(size_usd, 4),
                "ret_pct":     None,
                "pnl_usd":     round(-entry_fee, 4),
                "capital":     round(state[strat]["capital"], 4),
                "win":         None,
            })
    return records


def append_ledger(records: list):
    if not records:
        return
    os.makedirs(DATA_DIR, exist_ok=True)
    fieldnames = ["ts", "strategy", "coin", "action", "entry_price", "exit_price",
                  "size_usd", "ret_pct", "pnl_usd", "capital", "win"]
    write_header = not os.path.exists(LEDGER_FILE)
    with open(LEDGER_FILE, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        if write_header:
            w.writeheader()
        w.writerows(records)


def total_portfolio_value(state: dict, prices: dict) -> dict:
    """Capital + mark-to-market value of open positions."""
    totals = {}
    for strat in STRATEGIES:
        val = state[strat]["capital"]
        for coin, pos in state[strat]["positions"].items():
            price_now = prices.get(coin, pos["entry_price"])
            val += pos["size_usd"] * (price_now / pos["entry_price"])
        totals[strat] = round(val, 2)
    return totals


# ── Telegram ──────────────────────────────────────────────────────────────────

def send_telegram(text: str):
    try:
        spec = importlib.util.spec_from_file_location("tg", "telegram_notify.py")
        tg   = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(tg)
        if tg.TELEGRAM_TOKEN:
            tg.send_message(text)
        else:
            print("  ⚠  Telegram not configured")
    except Exception as e:
        print(f"  ⚠  Telegram failed: {e}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    now      = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    ts       = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")

    print(f"\n{'='*70}")
    print(f"  Strategy Comparison + Paper Trading — {now}")
    print(f"{'='*70}\n")

    state            = load_state()
    ensemble_signals = load_ensemble_signals()
    fear_greed       = get_fear_greed()
    print(f"  Fear & Greed: {fear_greed}\n")

    # ── Fetch features + prices for all coins ─────────────────────────────────
    rows             = []     # signal comparison rows
    prices           = {}     # {coin: current_price}
    signals_by_coin  = {}     # {coin: {strat: signal}}

    for ticker, symbol in COINS.items():
        print(f"  ── {ticker}", end=" ", flush=True)

        feats, price = get_live_features(symbol)
        funding_rate = get_funding_rate(symbol)
        feats["funding_rate"] = funding_rate
        feats["fear_greed"]   = fear_greed

        d_rsi14     = feats.get("d_rsi14", 50.0)
        d_vol_ratio = feats.get("d_vol_ratio", 1.0)

        ens_sig  = ensemble_signals.get(ticker, "N/A")
        fund_sig = strategy_funding_contrarian(funding_rate)
        rsi_sig  = strategy_rsi_reversion(d_rsi14, d_vol_ratio)
        ml3_sig  = strategy_simple_ml(ticker, "simple3", feats)
        ml5_sig  = strategy_simple_ml(ticker, "simple5", feats)

        prices[ticker]          = price
        signals_by_coin[ticker] = {
            "ensemble": ens_sig,
            "fund":     fund_sig,
            "rsi":      rsi_sig,
            "ml3":      ml3_sig,
            "ml5":      ml5_sig,
        }

        rows.append({
            "coin":         ticker,
            "price":        price,
            "ensemble":     ens_sig,
            "fund":         fund_sig,
            "rsi":          rsi_sig,
            "ml3":          ml3_sig,
            "ml5":          ml5_sig,
            "funding_rate": round(funding_rate, 4),
            "d_rsi14":      round(d_rsi14, 1),
            "d_vol_ratio":  round(d_vol_ratio, 2),
        })

        n_buy = sum(1 for s in signals_by_coin[ticker].values() if s == "BUY")
        print(f"  ${price:>10,.2f}  RSI={d_rsi14:.1f}  fund={funding_rate:+.5f}%  → {n_buy}/5 BUY")
        time.sleep(0.3)

    # ── Paper trading: close then open ────────────────────────────────────────
    closed = close_all_positions(state, prices, ts)
    opened = open_new_positions(state, signals_by_coin, prices, ts)
    append_ledger(closed + opened)

    # Portfolio value = cash + open positions MTM
    portfolio = total_portfolio_value(state, prices)
    save_state(state)

    # ── Signal comparison table ───────────────────────────────────────────────
    def agree(sig, ens):
        if sig == ens and sig in ("BUY", "STAY OUT"):
            return sig + " ✓"
        return sig

    print(f"\n  {'─'*68}")
    print(f"  {'Coin':<6}  {'Ensemble':>10}  {'FundRate':>10}  "
          f"{'RSI-MR':>10}  {'Simple3':>9}  {'Simple5':>9}")
    print(f"  {'─'*68}")
    for r in rows:
        ens = r["ensemble"]
        print(f"  {r['coin']:<6}  {ens:>10}  {agree(r['fund'],ens):>10}  "
              f"{agree(r['rsi'],ens):>10}  {agree(r['ml3'],ens):>9}  {agree(r['ml5'],ens):>9}")
    print(f"  {'─'*68}")

    # ── Paper portfolio summary ───────────────────────────────────────────────
    strat_labels = {
        "ensemble": "Ensemble",
        "fund":     "FundRate",
        "rsi":      "RSI-MR",
        "ml3":      "Simple3",
        "ml5":      "Simple5",
    }
    print(f"\n  Paper Portfolio (started ${INITIAL_CAPITAL:.0f} each):")
    print(f"  {'─'*44}")
    print(f"  {'Strategy':<12}  {'Value':>8}  {'P&L':>8}  {'Return':>8}")
    print(f"  {'─'*44}")
    for strat in STRATEGIES:
        val = portfolio[strat]
        pnl = val - INITIAL_CAPITAL
        ret = (val / INITIAL_CAPITAL - 1) * 100
        bar = "+" if pnl >= 0 else ""
        print(f"  {strat_labels[strat]:<12}  ${val:>7.2f}  {bar}{pnl:>+7.2f}  {ret:>+7.1f}%")
    print(f"  {'─'*44}")

    # Consensus
    print(f"\n  Consensus:")
    for r in rows:
        strats = list(signals_by_coin[r["coin"]].values())
        n_buy  = sum(1 for s in strats if s == "BUY")
        n_out  = sum(1 for s in strats if s == "STAY OUT")
        if   n_buy >= 4: verdict = f"STRONG BUY  ({n_buy}/5 agree)"
        elif n_buy == 3: verdict = f"BUY         ({n_buy}/5 agree)"
        elif n_buy == 2: verdict = f"Weak BUY    ({n_buy}/5 agree)"
        elif n_out >= 3: verdict = f"STAY OUT    ({n_out}/5 agree)"
        else:            verdict = "No consensus — skip"
        print(f"    {r['coin']}: {verdict:<28} "
              f"RSI={r['d_rsi14']:.1f}  fund={r['funding_rate']:+.4f}%")

    print(f"\n  {'='*68}\n")

    # ── Save comparison JSON ──────────────────────────────────────────────────
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(os.path.join(DATA_DIR, "strategy_comparison.json"), "w") as fh:
        json.dump({
            "date":       date_str,
            "fear_greed": fear_greed,
            "portfolio":  portfolio,
            "strategies": rows,
        }, fh, indent=2)
    print(f"  ✓  Saved → data/strategy_comparison.json")
    print(f"  ✓  Closed {len(closed)} / Opened {len(opened)} paper trades")
    print(f"  ✓  Ledger → {LEDGER_FILE}")

    # ── Telegram ──────────────────────────────────────────────────────────────
    def abbr(s: str) -> str:
        return {"BUY": "BUY", "STAY OUT": "OUT", "SKIP": "---",
                "N/A": "N/A", "ERR": "ERR"}.get(s, s[:3])

    lines = [f"📊 <b>Strategy Comparison — {date_str}</b>\n"]

    # Signal table
    hdr = f"{'':6} {'Ens':>5} {'Fund':>5} {'RSI':>5} {'ML3':>5} {'ML5':>5}"
    lines.append(f"<code>{hdr}</code>")
    for r in rows:
        ens  = r["ensemble"]
        icon = "🟢" if ens == "BUY" else "🔴" if ens == "STAY OUT" else "⚫"
        row  = (f"{r['coin']:<6} {abbr(ens):>5} {abbr(r['fund']):>5} "
                f"{abbr(r['rsi']):>5} {abbr(r['ml3']):>5} {abbr(r['ml5']):>5}")
        lines.append(f"{icon} <code>{row}</code>")

    lines.append("")

    # Portfolio summary
    lines.append("<b>Paper Portfolio ($200 start each):</b>")
    for strat in STRATEGIES:
        val = portfolio[strat]
        pnl = val - INITIAL_CAPITAL
        ret = (val / INITIAL_CAPITAL - 1) * 100
        emoji = "📈" if pnl > 0 else "📉" if pnl < 0 else "➡️"
        lines.append(f"{emoji} <code>{strat_labels[strat]:<12} ${val:>7.2f}  {ret:>+6.1f}%</code>")

    lines.append("")

    # Consensus
    consensus_found = False
    for r in rows:
        n_buy = sum(1 for s in signals_by_coin[r["coin"]].values() if s == "BUY")
        if n_buy >= 3:
            lines.append(f"✅ {r['coin']}: {n_buy}/5 agree → BUY")
            consensus_found = True
    if not consensus_found:
        lines.append("⚫ No strong consensus today")

    lines.append(f"\nF&amp;G: {fear_greed}")
    lines.append("<i>⚠ Not financial advice.</i>")

    send_telegram("\n".join(lines))
    print(f"  ✓  Telegram sent\n")


if __name__ == "__main__":
    main()