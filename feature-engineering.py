"""
Feature Engineering v3 — Multi-Timeframe (1H + 4H + 1D)
Fetches three timeframes from Binance and combines their features
into a single row per day. The model sees momentum at all scales.

Strategy:
  - 1D features: macro trend, long-term RSI, EMA50
  - 4H features: medium trend, MACD crossovers, BB position
  - 1H features: short-term momentum, entry timing, volume spikes

Requirements: pip install pandas pandas-ta requests
Usage: python feature-engineering.py
"""

import pandas as pd
import pandas_ta as ta
import requests
import time
import os
import math
from datetime import datetime, timezone

INPUT_DIR  = "data"
OUTPUT_DIR = "data"
COINS      = ["BTC", "ETH", "SOL"]

BINANCE_URL = "https://api.binance.com/api/v3/klines"
FAPI_URL    = "https://fapi.binance.com/fapi/v1/fundingRate"

TIMEFRAMES = {
    "1d": {"interval": "1d", "limit": 1100, "days": 1095},
    "4h": {"interval": "4h", "limit": 1100, "days": 183},  # ~6 months of 4H
    "1h": {"interval": "1h", "limit": 1100, "days": 45},   # ~45 days of 1H
}


# ── Data fetching ─────────────────────────────────────────────────────────────

def fetch_candles(symbol: str, interval: str, days: int) -> pd.DataFrame:
    """Fetch OHLCV candles from Binance for any timeframe."""
    end_ms   = int(datetime.now(timezone.utc).timestamp() * 1000)
    start_ms = end_ms - (days * 24 * 60 * 60 * 1000)

    all_candles = []
    chunk_start = start_ms

    while chunk_start < end_ms:
        params = {
            "symbol":    symbol,
            "interval":  interval,
            "startTime": chunk_start,
            "endTime":   end_ms,
            "limit":     1000,
        }
        resp = requests.get(BINANCE_URL, params=params, timeout=15)
        if resp.status_code != 200:
            raise RuntimeError(f"Binance {resp.status_code}: {resp.text[:100]}")

        candles = resp.json()
        if not candles:
            break

        all_candles.extend(candles)
        chunk_start = candles[-1][0] + 1
        if len(candles) < 1000:
            break
        time.sleep(0.1)

    if not all_candles:
        return pd.DataFrame()

    df = pd.DataFrame(all_candles, columns=[
        "ts", "open", "high", "low", "close", "volume",
        "close_time", "quote_vol", "trades", "tb_base", "tb_quote", "ignore"
    ])
    df["datetime"] = pd.to_datetime(df["ts"].astype(int), unit="ms")
    df["date"]     = df["datetime"].dt.date

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)

    return df[["datetime", "date", "open", "high", "low", "close", "volume"]].reset_index(drop=True)


# ── Feature computation ───────────────────────────────────────────────────────

def compute_features(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """
    Compute technical indicators and prefix all columns.
    prefix = "d" for daily, "h4" for 4H, "h1" for 1H
    """
    d = df.copy()

    # Trend
    d["ema9"]  = ta.ema(d["close"], length=9)
    d["ema21"] = ta.ema(d["close"], length=21)
    d["ema50"] = ta.ema(d["close"], length=50)
    d["ema_cross"] = (d["ema9"] > d["ema21"]).astype(int)  # 1=bullish cross
    d["price_vs_ema50"] = (d["close"] - d["ema50"]) / d["ema50"] * 100  # % above/below

    # Momentum
    d["rsi14"]   = ta.rsi(d["close"], length=14)
    d["rsi7"]    = ta.rsi(d["close"], length=7)
    d["roc5"]    = ta.roc(d["close"], length=5)
    d["roc10"]   = ta.roc(d["close"], length=10)

    # MACD
    macd = ta.macd(d["close"], fast=12, slow=26, signal=9)
    if macd is not None and not macd.empty:
        cols = {c: c for c in macd.columns}
        macd_col    = [c for c in macd.columns if c.startswith("MACD_")][0]
        signal_col  = [c for c in macd.columns if c.startswith("MACDs_")][0]
        hist_col    = [c for c in macd.columns if c.startswith("MACDh_")][0]
        d["macd_line"]   = macd[macd_col]
        d["macd_signal"] = macd[signal_col]
        d["macd_hist"]   = macd[hist_col]
        d["macd_cross"]  = ((d["macd_hist"] > 0) & (d["macd_hist"].shift(1) <= 0)).astype(int)

    # Bollinger Bands
    bb = ta.bbands(d["close"], length=20, std=2)
    if bb is not None and not bb.empty:
        bbu = [c for c in bb.columns if c.startswith("BBU")][0]
        bbm = [c for c in bb.columns if c.startswith("BBM")][0]
        bbl = [c for c in bb.columns if c.startswith("BBL")][0]
        d["bb_width"]    = (bb[bbu] - bb[bbl]) / bb[bbm] * 100
        d["bb_position"] = (d["close"] - bb[bbl]) / (bb[bbu] - bb[bbl])  # 0=lower 1=upper
        d["bb_squeeze"]  = (d["bb_width"] < d["bb_width"].rolling(20).mean()).astype(int)

    # Volatility
    d["atr14"]    = ta.atr(d["high"], d["low"], d["close"], length=14)
    d["atr_pct"]  = d["atr14"] / d["close"] * 100  # normalized ATR

    # Volume
    if d["volume"].notna().sum() > 20:
        d["vol_ratio"] = d["volume"] / d["volume"].rolling(20).mean()
        d["vol_spike"] = (d["vol_ratio"] > 2.0).astype(int)
        d["obv"]       = ta.obv(d["close"], d["volume"])
        d["obv_slope"] = d["obv"].diff(5)  # OBV trend over 5 bars

    # Price action
    d["body_pct"]     = abs(d["close"] - d["open"]) / d["open"] * 100
    d["upper_wick"]   = (d["high"] - d[["close","open"]].max(axis=1)) / d["open"] * 100
    d["lower_wick"]   = (d[["close","open"]].min(axis=1) - d["low"]) / d["open"] * 100
    d["is_green"]     = (d["close"] > d["open"]).astype(int)
    d["close_pos"]    = (d["close"] - d["low"]) / (d["high"] - d["low"]).replace(0, float("nan"))

    # Returns
    d["ret1"]  = d["close"].pct_change(1)  * 100
    d["ret3"]  = d["close"].pct_change(3)  * 100
    d["ret7"]  = d["close"].pct_change(7)  * 100

    # Keep only feature columns (drop OHLCV)
    skip = ["datetime", "date", "open", "high", "low", "close", "volume"]
    feat_cols = [c for c in d.columns if c not in skip]

    # Add prefix
    result = d[["datetime", "date"]].copy()
    for col in feat_cols:
        result[f"{prefix}_{col}"] = d[col]

    return result


def resample_to_daily(df_feat: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """
    For sub-daily timeframes (4H, 1H), aggregate features to daily level.
    We take the LAST value of each day (most recent bar = most current signal).
    Also add some aggregated stats (max RSI, min RSI, count of bullish bars).
    """
    df_feat["date"] = pd.to_datetime(df_feat["date"])

    # Last value per day = most recent intraday signal
    last = df_feat.groupby("date").last().reset_index()
    last = last.drop(columns=["datetime"], errors="ignore")

    # Add intraday aggregates for RSI and returns
    rsi_col = f"{prefix}_rsi14"
    ret_col = f"{prefix}_ret1"
    grn_col = f"{prefix}_is_green"

    if rsi_col in df_feat.columns:
        agg = df_feat.groupby("date").agg(
            **{f"{prefix}_rsi_max":    (rsi_col, "max"),
               f"{prefix}_rsi_min":    (rsi_col, "min"),
               f"{prefix}_green_pct":  (grn_col, "mean"),  # % of green bars today
               f"{prefix}_n_bars":     (ret_col, "count")}
        ).reset_index()
        last = last.merge(agg, on="date", how="left")

    return last


# ── External data ─────────────────────────────────────────────────────────────

def fetch_fear_greed(limit=1500) -> pd.DataFrame:
    print("  Fetching Fear & Greed...")
    try:
        resp = requests.get(
            f"https://api.alternative.me/fng/?limit={limit}&format=json",
            timeout=15)
        data = resp.json().get("data", [])
        df   = pd.DataFrame(data)
        df["date"]       = pd.to_datetime(df["timestamp"].astype(int), unit="s").dt.date
        df["fear_greed"] = df["value"].astype(int)
        df["fg_extreme_fear"]  = (df["fear_greed"] < 25).astype(int)
        df["fg_extreme_greed"] = (df["fear_greed"] > 75).astype(int)
        df["fg_7d_avg"]        = df["fear_greed"].rolling(7).mean()
        df["fg_momentum"]      = df["fear_greed"].diff(3)  # sentiment momentum
        print(f"  ✓  Fear & Greed: {len(df)} days")
        return df[["date","fear_greed","fg_extreme_fear","fg_extreme_greed","fg_7d_avg","fg_momentum"]]
    except Exception as e:
        print(f"  ⚠  Fear & Greed failed: {e}")
        return pd.DataFrame()


def fetch_funding(symbol="BTCUSDT", days=1095) -> pd.DataFrame:
    print(f"  Fetching funding rates ({symbol})...")
    try:
        end_ms      = int(datetime.now(timezone.utc).timestamp() * 1000)
        start_ms    = end_ms - (days * 24 * 60 * 60 * 1000)
        all_rates   = []
        chunk_start = start_ms

        while chunk_start < end_ms:
            params = {"symbol": symbol, "startTime": chunk_start,
                      "endTime": end_ms, "limit": 1000}
            resp = requests.get(FAPI_URL, params=params, timeout=15)
            if resp.status_code != 200:
                break
            data = resp.json()
            if not data:
                break
            all_rates.extend(data)
            chunk_start = data[-1]["fundingTime"] + 1
            if len(data) < 1000:
                break
            time.sleep(0.1)

        if not all_rates:
            return pd.DataFrame()

        df = pd.DataFrame(all_rates)
        df["date"]         = pd.to_datetime(df["fundingTime"].astype(int), unit="ms").dt.date
        df["funding_rate"] = df["fundingRate"].astype(float) * 100
        daily = df.groupby("date")["funding_rate"].mean().reset_index()
        daily["funding_extreme_pos"] = (daily["funding_rate"] >  0.05).astype(int)
        daily["funding_extreme_neg"] = (daily["funding_rate"] < -0.01).astype(int)
        daily["funding_7d_avg"]      = daily["funding_rate"].rolling(7).mean()
        daily["funding_momentum"]    = daily["funding_rate"].diff(3)
        print(f"  ✓  Funding: {len(daily)} days")
        return daily
    except Exception as e:
        print(f"  ⚠  Funding failed: {e}")
        return pd.DataFrame()


def fetch_btc_dominance() -> pd.DataFrame:
    print("  Fetching BTC dominance...")
    try:
        resp = requests.get(
            "https://api.coingecko.com/api/v3/global/market_cap_chart",
            params={"vs_currency": "usd", "days": 365}, timeout=15)
        data    = resp.json().get("market_cap_percentage", {}).get("btc", [])
        df      = pd.DataFrame(data, columns=["ts", "btc_dominance"])
        df["date"]          = pd.to_datetime(df["ts"].astype(int), unit="ms").dt.date
        df["btc_dom_change"] = df["btc_dominance"].diff()
        df["btc_dom_7d_avg"] = df["btc_dominance"].rolling(7).mean()
        print(f"  ✓  BTC dominance: {len(df)} days")
        return df[["date","btc_dominance","btc_dom_change","btc_dom_7d_avg"]]
    except Exception as e:
        print(f"  ⚠  BTC dominance failed: {e}")
        return pd.DataFrame()


def fetch_dxy() -> pd.DataFrame:
    """Fetch US Dollar Index (DXY) via yfinance — strong dollar = bearish crypto."""
    print("  Fetching DXY (US Dollar Index)...")
    try:
        import yfinance as yf
        df = yf.Ticker("DX-Y.NYB").history(period="3y", interval="1d", auto_adjust=True)
        if df.empty:
            raise RuntimeError("yfinance returned no DXY data")
        df = df.rename(columns={"Close": "dxy_close"})
        df.index = pd.to_datetime(df.index, utc=True).tz_localize(None)
        df["date"]          = df.index.date
        df["dxy_ret5"]      = df["dxy_close"].pct_change(5).round(6)
        df["dxy_above_ma20"] = (df["dxy_close"] > df["dxy_close"].rolling(20).mean()).astype(int)
        print(f"  ✓  DXY: {len(df)} days")
        return df[["date", "dxy_close", "dxy_ret5", "dxy_above_ma20"]].reset_index(drop=True)
    except Exception as e:
        print(f"  ⚠  DXY failed: {e}")
        return pd.DataFrame()


# ── Timeframe alignment ───────────────────────────────────────────────────────

def build_mtf_features(symbol: str, ticker: str) -> pd.DataFrame:
    """
    Fetch all three timeframes, compute features, align to daily.
    Returns one row per day with features from all timeframes.
    """
    print(f"\n  Processing {ticker}...")

    # 1D — use full history
    print(f"    Fetching 1D candles...")
    df_1d = fetch_candles(symbol, "1d", days=1095)
    time.sleep(0.3)

    # 4H — last 6 months
    print(f"    Fetching 4H candles...")
    df_4h = fetch_candles(symbol, "4h", days=183)
    time.sleep(0.3)

    # 1H — last 45 days
    print(f"    Fetching 1H candles...")
    df_1h = fetch_candles(symbol, "1h", days=45)
    time.sleep(0.3)

    print(f"    Computing features...")

    # Compute features per timeframe
    feat_1d = compute_features(df_1d, "d")
    feat_4h = compute_features(df_4h, "h4")
    feat_1h = compute_features(df_1h, "h1")

    # Aggregate sub-daily to daily
    daily_4h = resample_to_daily(feat_4h, "h4")
    daily_1h = resample_to_daily(feat_1h, "h1")

    # Base: daily features
    feat_1d["date"] = pd.to_datetime(feat_1d["date"])
    feat_1d = feat_1d.drop(columns=["datetime"], errors="ignore")

    # Merge all timeframes on date
    merged = feat_1d.merge(daily_4h, on="date", how="left")
    merged = merged.merge(daily_1h, on="date", how="left")

    # Add OHLCV back for target + backtest
    ohlcv = df_1d[["date", "open", "high", "low", "close", "volume"]].copy()
    ohlcv["date"] = pd.to_datetime(ohlcv["date"])
    merged = merged.merge(ohlcv, on="date", how="left")

    # Multi-timeframe agreement features
    # These are the most powerful MTF signals
    if "d_ema_cross" in merged.columns and "h4_ema_cross" in merged.columns:
        merged["mtf_trend_agree"] = (
            (merged["d_ema_cross"] == 1) &
            (merged["h4_ema_cross"] == 1)
        ).astype(int)

    if "d_rsi14" in merged.columns and "h4_rsi14" in merged.columns:
        merged["mtf_rsi_bullish"] = (
            (merged["d_rsi14"] > 50) &
            (merged["h4_rsi14"] > 50)
        ).astype(int)
        merged["mtf_rsi_diverge"] = merged["d_rsi14"] - merged["h4_rsi14"]

    if "h4_macd_hist" in merged.columns and "d_macd_hist" in merged.columns:
        merged["mtf_macd_agree"] = (
            (merged["d_macd_hist"] > 0) &
            (merged["h4_macd_hist"] > 0)
        ).astype(int)

    # MTF strength score (0-3): how many timeframes are bullish
    score_cols = []
    if "d_ema_cross"  in merged.columns: score_cols.append("d_ema_cross")
    if "h4_ema_cross" in merged.columns: score_cols.append("h4_ema_cross")
    if "h1_ema_cross" in merged.columns: score_cols.append("h1_ema_cross")
    if score_cols:
        merged["mtf_score"] = merged[score_cols].sum(axis=1)

    # Target: 3-day forward return > 0.5% (filters noise vs. simple next-day direction)
    merged = merged.sort_values("date").reset_index(drop=True)
    fwd_return = merged["close"].shift(-3) / merged["close"] - 1
    merged["target"] = (fwd_return > 0.005).astype(int)

    # Forward-fill 4H/1H features for days with missing data
    h4_cols = [c for c in merged.columns if c.startswith("h4_")]
    h1_cols = [c for c in merged.columns if c.startswith("h1_")]
    merged[h4_cols + h1_cols] = merged[h4_cols + h1_cols].ffill()

    return merged


def print_summary(ticker: str, df: pd.DataFrame):
    exclude = ["date","open","high","low","close","volume","target"]
    feat_cols = [c for c in df.columns if c not in exclude and df[c].dtype != object]
    d_feats  = [c for c in feat_cols if c.startswith("d_")]
    h4_feats = [c for c in feat_cols if c.startswith("h4_")]
    h1_feats = [c for c in feat_cols if c.startswith("h1_")]
    mtf_feats = [c for c in feat_cols if c.startswith("mtf_")]
    up_pct = df["target"].mean() * 100 if "target" in df.columns else 0

    print(f"\n  {ticker}")
    print(f"  Rows:            {len(df)}")
    print(f"  Daily features:  {len(d_feats)}")
    print(f"  4H features:     {len(h4_feats)}")
    print(f"  1H features:     {len(h1_feats)}")
    print(f"  MTF features:    {len(mtf_feats)}")
    print(f"  Total features:  {len(feat_cols)}")
    print(f"  Target balance:  {up_pct:.1f}% up / {100-up_pct:.1f}% down")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"\n{'='*50}")
    print(f"  Feature Engineering v3 — Multi-Timeframe")
    print(f"  Timeframes: 1H + 4H + 1D")
    print(f"{'='*50}\n")

    # External data (once, shared across all coins)
    print("  Fetching external data...")
    fear_greed = fetch_fear_greed()
    funding    = fetch_funding("BTCUSDT")
    btc_dom    = fetch_btc_dominance()
    dxy        = fetch_dxy()
    print()

    all_featured = []
    COIN_SYMBOLS = {"BTC": "BTCUSDT", "ETH": "ETHUSDT", "SOL": "SOLUSDT"}

    for ticker in COINS:
        symbol = COIN_SYMBOLS[ticker]
        try:
            df = build_mtf_features(symbol, ticker)

            # Merge external data
            df["date"] = pd.to_datetime(df["date"])

            if not fear_greed.empty:
                fg = fear_greed.copy()
                fg["date"] = pd.to_datetime(fg["date"])
                df = df.merge(fg, on="date", how="left")

            if not funding.empty:
                fd = funding.copy()
                fd["date"] = pd.to_datetime(fd["date"])
                df = df.merge(fd, on="date", how="left")

            if not btc_dom.empty:
                bd = btc_dom.copy()
                bd["date"] = pd.to_datetime(bd["date"])
                df = df.merge(bd, on="date", how="left")

            if not dxy.empty:
                dx = dxy.copy()
                dx["date"] = pd.to_datetime(dx["date"])
                df = df.merge(dx, on="date", how="left")

            # Forward-fill external data gaps
            ext_cols = [c for c in df.columns if any(
                c.startswith(p) for p in ["fear_","fg_","funding_","btc_dom","dxy_"]
            )]
            df[ext_cols] = df[ext_cols].ffill()

            # Market regime: 50d/200d EMA relationship (golden cross = bull, death cross = bear)
            ema50  = df["close"].ewm(span=50,  adjust=False).mean()
            ema200 = df["close"].ewm(span=200, adjust=False).mean()
            df["regime_bull"]     = (ema50 > ema200).astype(int)
            df["regime_strength"] = ((ema50 - ema200) / ema200).round(6)

            # Feature interactions — capture combined signals the model can't see alone
            if "d_rsi14" in df.columns and "d_vol_ratio" in df.columns:
                df["rsi_vol_bull"]  = df["d_rsi14"] * df["d_vol_ratio"]
            if "d_macd_hist" in df.columns and "d_bb_position" in df.columns:
                df["macd_bb_pos"]   = df["d_macd_hist"] * df["d_bb_position"]
            if "d_ema_cross" in df.columns and "d_rsi14" in df.columns:
                df["ema_rsi_align"] = df["d_ema_cross"] * (df["d_rsi14"] - 50)
            if "funding_rate" in df.columns and "d_rsi14" in df.columns:
                df["funding_rsi"]   = df["funding_rate"] * df["d_rsi14"]

            # Drop rows where core daily features are missing
            core_cols = ["d_rsi14", "d_ema50", "d_macd_line"]
            available = [c for c in core_cols if c in df.columns]
            if available:
                df = df.dropna(subset=available)

            # Drop last row (no future target)
            df = df.iloc[:-1].reset_index(drop=True)

            # Save
            out_path = os.path.join(OUTPUT_DIR, f"{ticker}_features.csv")
            df.to_csv(out_path, index=False)
            print(f"  ✓  Saved → {out_path}")
            print_summary(ticker, df)

            all_featured.append(df.assign(coin=ticker))

        except Exception as e:
            print(f"  ✗  Error processing {ticker}: {e}")
            import traceback; traceback.print_exc()

        time.sleep(1)

    if all_featured:
        combined = pd.concat(all_featured, ignore_index=True)
        combined.to_csv(os.path.join(OUTPUT_DIR, "all_coins_features.csv"), index=False)
        print(f"\n  ✓  Combined → data/all_coins_features.csv")

    print(f"\n{'='*50}")
    print(f"  Done! Run train-model.py next")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()