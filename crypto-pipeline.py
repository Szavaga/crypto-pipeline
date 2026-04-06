"""
Crypto Data Pipeline — BTC, ETH, SOL
Fetches OHLCV data from Binance's public API (no account or API key needed).
Binance returns up to 1000 candles per request — we page through to get
multiple years of daily data.

Requirements:
    pip install requests pandas

Usage:
    python crypto-pipeline.py
"""

import requests
import pandas as pd
import time
import os
from datetime import datetime, timezone

# ── Config ────────────────────────────────────────────────────────────────────

COINS = {
    "BTC": "BTCUSDT",
    "ETH": "ETHUSDT",
    "SOL": "SOLUSDT",
}

DAYS       = 365 * 3   # 3 years of history
INTERVAL   = "1d"      # daily candles
OUTPUT_DIR = "data"

BINANCE_URL = "https://api.binance.com/api/v3/klines"

# ── Fetch ─────────────────────────────────────────────────────────────────────

def fetch_ohlcv(symbol: str, days: int = 365) -> pd.DataFrame:
    """
    Fetch daily OHLCV candles from Binance for a single symbol.
    Pages through in chunks of 1000 candles to get full history.
    """
    end_ms   = int(datetime.now(timezone.utc).timestamp() * 1000)
    start_ms = end_ms - (days * 24 * 60 * 60 * 1000)

    all_candles = []
    chunk_start = start_ms

    print(f"  Fetching {symbol}...", end="", flush=True)

    while chunk_start < end_ms:
        params = {
            "symbol":    symbol,
            "interval":  INTERVAL,
            "startTime": chunk_start,
            "endTime":   end_ms,
            "limit":     1000,  # Binance max per request
        }

        resp = requests.get(BINANCE_URL, params=params, timeout=15)

        if resp.status_code != 200:
            raise RuntimeError(
                f"Binance returned {resp.status_code}: {resp.text}"
            )

        candles = resp.json()
        if not candles:
            break

        all_candles.extend(candles)

        # Move start forward past the last candle we got
        chunk_start = candles[-1][0] + 1

        # If we got fewer than 1000 we've reached the end
        if len(candles) < 1000:
            break

        time.sleep(0.1)  # be polite to Binance rate limits

    print(f" {len(all_candles)} candles")

    if not all_candles:
        raise RuntimeError(f"No data returned for {symbol}")

    # Binance kline format:
    # [open_time, open, high, low, close, volume, close_time, ...]
    df = pd.DataFrame(all_candles, columns=[
        "timestamp_ms", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades",
        "taker_buy_base", "taker_buy_quote", "ignore"
    ])

    df["date"]   = pd.to_datetime(df["timestamp_ms"], unit="ms").dt.date
    df["open"]   = df["open"].astype(float)
    df["high"]   = df["high"].astype(float)
    df["low"]    = df["low"].astype(float)
    df["close"]  = df["close"].astype(float)
    df["volume"] = df["volume"].astype(float)

    df = df[["date", "open", "high", "low", "close", "volume"]]
    df = df.sort_values("date").reset_index(drop=True)

    return df


def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["daily_return"] = df["close"].pct_change() * 100
    df["price_range"]  = df["high"] - df["low"]
    df["close_ma7"]    = df["close"].rolling(window=7).mean()
    df["close_ma30"]   = df["close"].rolling(window=30).mean()
    return df


def print_summary(ticker: str, df: pd.DataFrame):
    print(f"\n  {'─'*45}")
    print(f"  {ticker} — {len(df)} rows  ({df['date'].iloc[0]} → {df['date'].iloc[-1]})")
    print(f"  Close: min=${df['close'].min():>12,.2f}   max=${df['close'].max():>12,.2f}")
    print(f"  Latest: ${df['close'].iloc[-1]:>12,.2f}  ({df['date'].iloc[-1]})")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"\n{'='*50}")
    print(f"  Crypto Data Pipeline (Binance)")
    print(f"  Fetching {DAYS} days of OHLCV data")
    print(f"  Coins: {', '.join(COINS.keys())}")
    print(f"{'='*50}\n")

    all_data = {}

    for ticker, symbol in COINS.items():
        try:
            df = fetch_ohlcv(symbol, days=DAYS)
            df = add_basic_features(df)
            all_data[ticker] = df

            out_path = os.path.join(OUTPUT_DIR, f"{ticker}_ohlcv.csv")
            df.to_csv(out_path, index=False)
            print(f"  ✓  Saved → {out_path}")
            print_summary(ticker, df)

        except Exception as e:
            print(f"\n  ✗  Error fetching {ticker}: {e}")

        time.sleep(0.5)

    if all_data:
        combined = pd.concat(
            [df.assign(coin=ticker) for ticker, df in all_data.items()],
            ignore_index=True
        )
        combined.to_csv(os.path.join(OUTPUT_DIR, "all_coins_combined.csv"), index=False)
        print(f"\n  ✓  Combined → data/all_coins_combined.csv")

    print(f"\n{'='*50}")
    print(f"  Done! Run feature-engineering.py next")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()