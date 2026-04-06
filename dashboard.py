"""
Dashboard Data Generator
Lekéri az összes adatot és elmenti egy data/dashboard_data.json fájlba.
A dashboard.html ebből olvassa be az adatokat fetch()-el.

Usage: python dashboard.py
Majd nyisd meg: dashboard.html a böngészőben
"""

import requests
import pandas as pd
import json
import math
import os
from datetime import datetime, timezone

DATA_DIR    = "data"
LOG_PATH    = os.path.join(DATA_DIR, "signal_log.csv")
JSON_PATH   = os.path.join(DATA_DIR, "dashboard_data.json")
BINANCE_URL = "https://api.binance.com/api/v3/klines"

COINS = {"BTC": "BTCUSDT", "ETH": "ETHUSDT", "SOL": "SOLUSDT"}


def clean(obj):
    """Rekurzívan tisztítja az adatot JSON-kompatibilisre."""
    if isinstance(obj, dict):
        return {str(k): clean(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [clean(v) for v in obj]
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return 0
        return round(obj, 6)
    if isinstance(obj, str):
        return (obj.replace("\u2014", "-")
                   .replace("\u2013", "-")
                   .replace("\u2019", "'"))
    if hasattr(obj, 'item'):  # numpy scalar
        return obj.item()
    return obj


def fetch_prices(symbol, days=120):
    try:
        r = requests.get(BINANCE_URL,
            params={"symbol": symbol, "interval": "1d", "limit": days},
            timeout=15)
        return [{"t": int(c[0]), "o": float(c[1]), "h": float(c[2]),
                 "l": float(c[3]), "c": float(c[4]), "v": float(c[5])}
                for c in r.json()]
    except Exception as e:
        print(f"  ⚠  Prices error ({symbol}): {e}")
        return []


def fetch_ticker(symbol):
    try:
        r = requests.get("https://api.binance.com/api/v3/ticker/24hr",
            params={"symbol": symbol}, timeout=10)
        d = r.json()
        return {
            "price":  float(d["lastPrice"]),
            "change": float(d["priceChangePercent"]),
            "high":   float(d["highPrice"]),
            "low":    float(d["lowPrice"]),
            "volume": float(d["quoteVolume"]),
        }
    except Exception as e:
        print(f"  ⚠  Ticker error ({symbol}): {e}")
        return {}


def fetch_fear_greed(limit=60):
    try:
        r = requests.get(
            f"https://api.alternative.me/fng/?limit={limit}&format=json",
            timeout=10)
        return [{"t": int(d["timestamp"]) * 1000,
                 "v": int(d["value"]),
                 "l": str(d["value_classification"])}
                for d in r.json().get("data", [])]
    except Exception as e:
        print(f"  ⚠  Fear & Greed error: {e}")
        return []


def fetch_global():
    try:
        r = requests.get("https://api.coingecko.com/api/v3/global", timeout=10)
        d = r.json()["data"]
        return {
            "btc_dom":    round(float(d["market_cap_percentage"].get("btc", 0)), 1),
            "eth_dom":    round(float(d["market_cap_percentage"].get("eth", 0)), 1),
            "total_mcap": float(d["total_market_cap"].get("usd", 0)),
            "mcap_change": round(float(d.get("market_cap_change_percentage_24h_usd", 0)), 2),
        }
    except Exception as e:
        print(f"  ⚠  Global data error: {e}")
        return {}


def fetch_funding(symbol="ETHUSDT"):
    try:
        r = requests.get("https://fapi.binance.com/fapi/v1/fundingRate",
            params={"symbol": symbol, "limit": 3}, timeout=10)
        rates = [float(d["fundingRate"]) * 100 for d in r.json()]
        return round(sum(rates) / len(rates), 6) if rates else 0.0
    except Exception as e:
        print(f"  ⚠  Funding error ({symbol}): {e}")
        return 0.0


def load_signals():
    result = {"BTC": [], "ETH": [], "SOL": []}
    if not os.path.exists(LOG_PATH):
        return result
    try:
        df = pd.read_csv(LOG_PATH, on_bad_lines="skip")
        required = ["date", "coin", "price", "signal", "prob_up", "prob_down"]
        for col in required:
            if col not in df.columns:
                print(f"  ⚠  Missing column in log: {col}")
                return result

        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
        df = df.dropna(subset=["date", "coin"])

        # Safe defaults for optional columns
        defaults = {"kelly_pct": 0, "fear_greed": 0, "btc_dom": 0,
                    "confidence": "", "fg_label": "", "funding": 0, "signal": ""}
        for col, default in defaults.items():
            if col not in df.columns:
                df[col] = default

        # Convert numeric columns
        for col in ["price", "prob_up", "prob_down", "kelly_pct",
                    "fear_greed", "btc_dom", "funding"]:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

        # Convert string columns — remove problematic characters
        for col in ["confidence", "fg_label", "signal"]:
            df[col] = df[col].astype(str).str.replace("\u2014", "-", regex=False)
            df[col] = df[col].str.replace("\u2013", "-", regex=False)

        for coin in ["BTC", "ETH", "SOL"]:
            rows = df[df["coin"] == coin].sort_values("date", ascending=False)
            result[coin] = rows.to_dict("records")

    except Exception as e:
        print(f"  ⚠  Could not load signal log: {e}")

    return result


def main():
    print(f"\n{'='*50}")
    print(f"  Dashboard Data Generator")
    print(f"{'='*50}\n")

    # Collect all data
    print("  Fetching prices...")
    prices  = {}
    tickers = {}
    for ticker, symbol in COINS.items():
        prices[ticker]  = fetch_prices(symbol, 120)
        tickers[ticker] = fetch_ticker(symbol)
        t = tickers[ticker]
        print(f"  ✓  {ticker}: {len(prices[ticker])} candles  "
              f"${t.get('price', 0):,.2f} ({t.get('change', 0):+.2f}%)")

    print("\n  Fetching sentiment...")
    fg   = fetch_fear_greed(60)
    glbl = fetch_global()
    eth_fund = fetch_funding("ETHUSDT")
    btc_fund = fetch_funding("BTCUSDT")
    if fg:
        print(f"  ✓  Fear & Greed: {fg[0]['v']} ({fg[0]['l']})")
    if glbl:
        print(f"  ✓  BTC dom: {glbl.get('btc_dom')}%  "
              f"ETH dom: {glbl.get('eth_dom')}%")

    print("\n  Loading signals...")
    signals = load_signals()
    total   = sum(len(v) for v in signals.values())
    latest  = {coin: rows[0] if rows else {} for coin, rows in signals.items()}

    # Stats per coin
    stats = {}
    for coin, rows in signals.items():
        total_c = len(rows)
        stats[coin] = {
            "total":    total_c,
            "buys":     sum(1 for r in rows if "BUY" in str(r.get("signal", ""))),
            "stays":    sum(1 for r in rows if "OUT" in str(r.get("signal", ""))),
            "skips":    sum(1 for r in rows if "SKIP" in str(r.get("signal", ""))),
            "avg_conf": round(sum(float(r.get("prob_up", 50)) for r in rows) / total_c, 1) if total_c else 0,
        }

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    # Build data object
    data = {
        "updated":  now,
        "prices":   prices,
        "tickers":  tickers,
        "signals":  signals,
        "latest":   latest,
        "stats":    stats,
        "fg":       fg,
        "glbl":     glbl,
        "eth_fund": eth_fund,
        "btc_fund": btc_fund,
    }

    # Clean and save
    data = clean(data)
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=True, indent=2)

    print(f"  ✓  {total} signals loaded")
    print(f"\n  ✓  Data saved → {JSON_PATH}")
    print(f"  Open in browser:")
    print(f"  http://localhost:8080")
    print(f"  Start the server with: python server.py")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()