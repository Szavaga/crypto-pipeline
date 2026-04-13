"""
Strategy Comparison — runs 4 simple strategies alongside the ensemble.
Shows which strategies agree and sends a single Telegram comparison message.

Strategies:
  1. Funding Rate Contrarian  (rule-based)
  2. RSI Mean Reversion       (rule-based)
  3. Simple3 XGBoost          (d_rsi14 + funding_rate + d_vol_ratio)
  4. Simple5 XGBoost          (above + fear_greed + d_macd_hist)

Usage: python compare-strategies.py
Run after: predict-today.py && testnet-trader.py
"""

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

CONF_THRESH = 0.55


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


def get_live_features(symbol: str) -> dict:
    """Compute d_rsi14, d_macd_hist, d_vol_ratio from recent daily candles."""
    df = fetch_daily(symbol, 50)
    if df.empty or len(df) < 20:
        return {}

    try:
        import pandas_ta as ta
        d_rsi14     = float(ta.rsi(df["close"], length=14).iloc[-1])
        macd_df     = ta.macd(df["close"], fast=12, slow=26, signal=9)
        mh_col      = [c for c in macd_df.columns if c.startswith("MACDh_")][0]
        d_macd_hist = float(macd_df[mh_col].iloc[-1])
    except Exception:
        # Fallback: manual EMA-based computation
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

    # Replace NaN with safe defaults
    if np.isnan(d_rsi14):      d_rsi14     = 50.0
    if np.isnan(d_macd_hist):  d_macd_hist = 0.0
    if np.isnan(d_vol_ratio):  d_vol_ratio = 1.0

    return {
        "d_rsi14":     round(d_rsi14, 2),
        "d_macd_hist": round(d_macd_hist, 6),
        "d_vol_ratio": round(d_vol_ratio, 3),
    }


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


# ── Strategies ────────────────────────────────────────────────────────────────

def strategy_funding_contrarian(funding_rate: float) -> str:
    """BUY on extreme shorts (squeeze likely), STAY OUT on overleveraged longs."""
    if funding_rate < -0.01:
        return "BUY"
    elif funding_rate > 0.05:
        return "STAY OUT"
    return "SKIP"


def strategy_rsi_reversion(d_rsi14: float, d_vol_ratio: float) -> str:
    """Oversold + volume confirmation = BUY. Overbought = STAY OUT."""
    if d_rsi14 < 35 and d_vol_ratio > 1.2:
        return "BUY"
    elif d_rsi14 > 70:
        return "STAY OUT"
    return "SKIP"


def strategy_simple_ml(coin: str, suffix: str, features: dict) -> str:
    """Run a simple XGBoost model. Returns BUY / STAY OUT / SKIP / N/A."""
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
        print(f"      ⚠  {coin} {suffix} error: {e}")
        return "ERR"


# ── Ensemble signal loader ────────────────────────────────────────────────────

def load_ensemble_signals() -> dict:
    """Load the most recent ensemble signal per coin from signal_log.csv."""
    log_path = os.path.join(DATA_DIR, "signal_log.csv")
    if not os.path.exists(log_path):
        print("  ⚠  signal_log.csv not found — run predict-today.py first")
        return {}
    df = pd.read_csv(log_path)
    if df.empty:
        return {}
    # Last row per coin
    latest = df.groupby("coin").last().reset_index()
    result = {}
    for _, row in latest.iterrows():
        sig = str(row.get("signal", "SKIP"))
        if "BUY" in sig:
            result[row["coin"]] = "BUY"
        elif "OUT" in sig:
            result[row["coin"]] = "STAY OUT"
        else:
            result[row["coin"]] = "SKIP"
    return result


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

    print(f"\n{'='*70}")
    print(f"  Strategy Comparison — {now}")
    print(f"{'='*70}\n")

    ensemble_signals = load_ensemble_signals()
    fear_greed       = get_fear_greed()
    print(f"  Fear & Greed: {fear_greed}\n")

    rows = []

    for ticker, symbol in COINS.items():
        print(f"  ── {ticker}", end=" ", flush=True)

        feats        = get_live_features(symbol)
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

        rows.append({
            "coin":         ticker,
            "ensemble":     ens_sig,
            "fund":         fund_sig,
            "rsi":          rsi_sig,
            "ml3":          ml3_sig,
            "ml5":          ml5_sig,
            "funding_rate": round(funding_rate, 4),
            "d_rsi14":      round(d_rsi14, 1),
            "d_vol_ratio":  round(d_vol_ratio, 2),
        })

        strategies = [ens_sig, fund_sig, rsi_sig, ml3_sig, ml5_sig]
        n_buy = sum(1 for s in strategies if s == "BUY")
        print(f"  RSI={d_rsi14:.1f}  fund={funding_rate:+.4f}%  vol={d_vol_ratio:.2f}  "
              f"→ {n_buy}/5 BUY")
        time.sleep(0.3)

    # ── Console comparison table ──────────────────────────────────────────────

    def agree(sig, ens):
        """Mark ✓ when strategy matches ensemble on a directional signal."""
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

    # ── Consensus ─────────────────────────────────────────────────────────────
    print(f"\n  Consensus:")
    for r in rows:
        strats = [r["ensemble"], r["fund"], r["rsi"], r["ml3"], r["ml5"]]
        n_buy  = sum(1 for s in strats if s == "BUY")
        n_out  = sum(1 for s in strats if s == "STAY OUT")
        if n_buy >= 4:
            verdict = f"STRONG BUY  ({n_buy}/5 agree)"
        elif n_buy == 3:
            verdict = f"BUY         ({n_buy}/5 agree)"
        elif n_buy == 2:
            verdict = f"Weak BUY    ({n_buy}/5 agree)"
        elif n_out >= 3:
            verdict = f"STAY OUT    ({n_out}/5 agree)"
        else:
            verdict = "No consensus — skip"
        print(f"    {r['coin']}: {verdict:<28} "
              f"(RSI={r['d_rsi14']:.1f}  fund={r['funding_rate']:+.4f}%  vol={r['d_vol_ratio']:.2f})")

    print(f"\n  {'='*68}\n")

    # ── Save JSON ─────────────────────────────────────────────────────────────
    os.makedirs(DATA_DIR, exist_ok=True)
    out_json = os.path.join(DATA_DIR, "strategy_comparison.json")
    with open(out_json, "w") as fh:
        json.dump({"date": date_str, "fear_greed": fear_greed, "strategies": rows}, fh, indent=2)
    print(f"  ✓  Saved → {out_json}")

    # ── Telegram ──────────────────────────────────────────────────────────────
    def abbr(s: str) -> str:
        return {"BUY": "BUY", "STAY OUT": "OUT", "SKIP": "---",
                "N/A": "N/A", "ERR": "ERR"}.get(s, s[:3])

    lines = [f"📊 <b>Strategy Comparison — {date_str}</b>\n"]
    header = f"{'':6} {'Ens':>5} {'Fund':>6} {'RSI':>5} {'ML3':>4} {'ML5':>4}"
    lines.append(f"<code>{header}</code>")

    for r in rows:
        ens = r["ensemble"]
        icon = "🟢" if ens == "BUY" else "🔴" if ens == "STAY OUT" else "⚫"
        row_str = (f"{r['coin']:<6} {abbr(ens):>5} {abbr(r['fund']):>6} "
                   f"{abbr(r['rsi']):>5} {abbr(r['ml3']):>4} {abbr(r['ml5']):>4}")
        lines.append(f"{icon} <code>{row_str}</code>")

    lines.append("")
    consensus_found = False
    for r in rows:
        strats = [r["ensemble"], r["fund"], r["rsi"], r["ml3"], r["ml5"]]
        n_buy  = sum(1 for s in strats if s == "BUY")
        if n_buy >= 3:
            lines.append(f"✅ {r['coin']}: {n_buy}/5 strategies agree → BUY")
            consensus_found = True

    if not consensus_found:
        lines.append("⚫ No strong consensus signals today")

    lines.append(f"\nF&amp;G: {fear_greed}")
    lines.append("<i>⚠ Not financial advice.</i>")

    send_telegram("\n".join(lines))
    print(f"  ✓  Telegram sent")


if __name__ == "__main__":
    main()