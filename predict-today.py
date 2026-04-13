"""
Daily Prediction v4 — Multi-Timeframe
Fetches live 1H + 4H + 1D candles, computes MTF features,
and generates signals with confidence + Kelly sizing.

Requirements: pip install requests pandas xgboost scikit-learn lightgbm pandas-ta
Usage: python predict-today.py
"""

import requests
import pandas as pd
import pandas_ta as ta
import numpy as np
import pickle

# Must be defined before pickle.load() — train-model.py pickles _CalibratedModel
# instances under __main__, so this script needs the same class in its namespace.
class _CalibratedModel:
    def __init__(self, model, calibrator):
        self.model      = model
        self.calibrator = calibrator

    def predict_proba(self, X):
        raw = self.model.predict_proba(X)[:, 1]
        cal = self.calibrator.predict(raw)
        return np.column_stack([1 - cal, cal])
import os
import time
import math
from datetime import datetime, timezone

DATA_DIR    = "data"
MODEL_DIR   = "models"
CONF_THRESH = 0.55  # default fallback
CONF_THRESHOLD = {"BTC": 0.55, "ETH": 0.57, "SOL": 0.65}  # per-coin thresholds

COINS = {"BTC": "BTCUSDT", "ETH": "ETHUSDT", "SOL": "SOLUSDT"}
# yfinance ticker map (used as fallback when Binance returns 451/geo-block)
YFINANCE_MAP = {"BTCUSDT": "BTC-USD", "ETHUSDT": "ETH-USD", "SOLUSDT": "SOL-USD"}
BINANCE_URL = "https://api.binance.com/api/v3/klines"
FAPI_URL    = "https://fapi.binance.com/fapi/v1/fundingRate"


# ── Fetch helpers ─────────────────────────────────────────────────────────────

def _fetch_yfinance(symbol: str, interval: str, limit: int) -> pd.DataFrame:
    """Fallback data source when Binance is geo-blocked (HTTP 451)."""
    import yfinance as yf

    yf_symbol = YFINANCE_MAP.get(symbol, symbol)

    # yfinance doesn't have 4h — fetch 1h and resample
    if interval == "4h":
        fetch_interval = "1h"
        resample = True
    else:
        fetch_interval = interval
        resample = False

    # period needed to cover `limit` candles
    period_map = {"1h": "30d", "1d": "365d"}
    period = period_map.get(fetch_interval, "60d")

    ticker = yf.Ticker(yf_symbol)
    df = ticker.history(period=period, interval=fetch_interval, auto_adjust=True)

    if df.empty:
        raise RuntimeError(f"yfinance returned no data for {yf_symbol}")

    df = df.rename(columns={"Open": "open", "High": "high",
                             "Low": "low", "Close": "close", "Volume": "volume"})
    df.index = pd.to_datetime(df.index, utc=True).tz_localize(None)
    df["datetime"] = df.index
    df["date"] = df["datetime"].dt.date

    if resample:
        df = df.resample("4h", on="datetime").agg(
            open=("open", "first"), high=("high", "max"),
            low=("low", "min"), close=("close", "last"),
            volume=("volume", "sum")
        ).dropna().reset_index()
        df["date"] = df["datetime"].dt.date

    df = df[["datetime", "date", "open", "high", "low", "close", "volume"]]
    return df.tail(limit).reset_index(drop=True)


def fetch_candles(symbol, interval, limit=150):
    try:
        resp = requests.get(BINANCE_URL,
            params={"symbol": symbol, "interval": interval, "limit": limit},
            timeout=15)
        if resp.status_code == 451:
            raise RuntimeError("geo-blocked")
        if resp.status_code != 200:
            raise RuntimeError(f"Binance {resp.status_code}")
        raw = resp.json()
        df  = pd.DataFrame(raw, columns=[
            "ts","open","high","low","close","volume",
            "ct","qv","trades","tb","tq","ignore"
        ])
        df["datetime"] = pd.to_datetime(df["ts"].astype(int), unit="ms")
        df["date"]     = df["datetime"].dt.date
        for c in ["open","high","low","close","volume"]:
            df[c] = df[c].astype(float)
        return df[["datetime","date","open","high","low","close","volume"]].reset_index(drop=True)
    except RuntimeError:
        return _fetch_yfinance(symbol, interval, limit)


def get_fear_greed():
    try:
        r    = requests.get("https://api.alternative.me/fng/?limit=8&format=json", timeout=10)
        data = r.json()["data"]
        vals = [int(d["value"]) for d in data]
        fg   = vals[0]
        return {
            "fear_greed": fg, "fg_extreme_fear": int(fg<25),
            "fg_extreme_greed": int(fg>75), "fg_7d_avg": sum(vals)/len(vals),
            "fg_momentum": fg - vals[3] if len(vals)>3 else 0,
            "fg_label": data[0]["value_classification"],
        }
    except: return {}


def get_btc_dominance():
    try:
        r   = requests.get("https://api.coingecko.com/api/v3/global", timeout=10)
        d   = r.json()["data"]
        btc = d["market_cap_percentage"]["btc"]
        return {"btc_dominance": btc, "btc_dom_change": 0, "btc_dom_7d_avg": btc}
    except: return {}


def get_funding(symbol="BTCUSDT"):
    try:
        r     = requests.get(FAPI_URL, params={"symbol": symbol, "limit": 8}, timeout=10)
        rates = [float(d["fundingRate"])*100 for d in r.json()]
        fr    = rates[-1]
        return {
            "funding_rate": fr, "funding_extreme_pos": int(fr>0.05),
            "funding_extreme_neg": int(fr<-0.01),
            "funding_7d_avg": sum(rates)/len(rates),
            "funding_momentum": fr - rates[0] if rates else 0,
        }
    except: return {}


def get_dxy():
    """Fetch latest DXY (US Dollar Index) value via yfinance."""
    try:
        import yfinance as yf
        df = yf.Ticker("DX-Y.NYB").history(period="30d", interval="1d", auto_adjust=True)
        if df.empty:
            return {}
        closes = df["Close"].dropna()
        dxy    = float(closes.iloc[-1])
        ma20   = float(closes.rolling(20).mean().iloc[-1]) if len(closes) >= 20 else dxy
        ret5   = float((closes.iloc[-1] / closes.iloc[-6] - 1)) if len(closes) >= 6 else 0.0
        return {
            "dxy_close":       round(dxy, 4),
            "dxy_ret5":        round(ret5, 6),
            "dxy_above_ma20":  int(dxy > ma20),
        }
    except: return {}


# ── Feature computation (matches feature-engineering.py) ─────────────────────

def compute_features(df, prefix):
    d = df.copy()

    d["ema9"]  = ta.ema(d["close"], length=9)
    d["ema21"] = ta.ema(d["close"], length=21)
    d["ema50"] = ta.ema(d["close"], length=50)
    d["ema_cross"]       = (d["ema9"] > d["ema21"]).astype(int)
    d["price_vs_ema50"]  = (d["close"] - d["ema50"]) / d["ema50"] * 100

    d["rsi14"] = ta.rsi(d["close"], length=14)
    d["rsi7"]  = ta.rsi(d["close"], length=7)
    d["roc5"]  = ta.roc(d["close"], length=5)
    d["roc10"] = ta.roc(d["close"], length=10)

    macd = ta.macd(d["close"], fast=12, slow=26, signal=9)
    if macd is not None and not macd.empty:
        mc = [c for c in macd.columns if c.startswith("MACD_")][0]
        ms = [c for c in macd.columns if c.startswith("MACDs_")][0]
        mh = [c for c in macd.columns if c.startswith("MACDh_")][0]
        d["macd_line"]   = macd[mc]
        d["macd_signal"] = macd[ms]
        d["macd_hist"]   = macd[mh]
        d["macd_cross"]  = ((d["macd_hist"]>0) & (d["macd_hist"].shift(1)<=0)).astype(int)

    bb = ta.bbands(d["close"], length=20, std=2)
    if bb is not None and not bb.empty:
        bbu = [c for c in bb.columns if c.startswith("BBU")][0]
        bbm = [c for c in bb.columns if c.startswith("BBM")][0]
        bbl = [c for c in bb.columns if c.startswith("BBL")][0]
        d["bb_width"]    = (bb[bbu]-bb[bbl]) / bb[bbm] * 100
        d["bb_position"] = (d["close"]-bb[bbl]) / (bb[bbu]-bb[bbl])
        d["bb_squeeze"]  = (d["bb_width"] < d["bb_width"].rolling(20).mean()).astype(int)

    d["atr14"]   = ta.atr(d["high"], d["low"], d["close"], length=14)
    d["atr_pct"] = d["atr14"] / d["close"] * 100

    if d["volume"].notna().sum() > 20:
        d["vol_ratio"] = d["volume"] / d["volume"].rolling(20).mean()
        d["vol_spike"] = (d["vol_ratio"] > 2.0).astype(int)
        d["obv"]       = ta.obv(d["close"], d["volume"])
        d["obv_slope"] = d["obv"].diff(5)

    d["body_pct"]   = abs(d["close"]-d["open"]) / d["open"] * 100
    d["upper_wick"] = (d["high"]-d[["close","open"]].max(axis=1)) / d["open"] * 100
    d["lower_wick"] = (d[["close","open"]].min(axis=1)-d["low"]) / d["open"] * 100
    d["is_green"]   = (d["close"]>d["open"]).astype(int)
    rng = d["high"]-d["low"]
    d["close_pos"]  = (d["close"]-d["low"]) / rng.replace(0, float("nan"))

    d["ret1"] = d["close"].pct_change(1)*100
    d["ret3"] = d["close"].pct_change(3)*100
    d["ret7"] = d["close"].pct_change(7)*100

    skip = ["datetime","date","open","high","low","close","volume"]
    result = d[["datetime","date"]].copy()
    for col in [c for c in d.columns if c not in skip]:
        result[f"{prefix}_{col}"] = d[col]
    return result


def get_latest_row(df_feat):
    """Return the most recent row as a flat dict."""
    return df_feat.iloc[-1].to_dict()


def build_live_features(symbol):
    """Fetch all timeframes and return a single feature dict for today."""

    # Fetch all three timeframes
    df_1d = fetch_candles(symbol, "1d", limit=150)
    time.sleep(0.2)
    df_4h = fetch_candles(symbol, "4h", limit=150)
    time.sleep(0.2)
    df_1h = fetch_candles(symbol, "1h", limit=150)

    # Compute features
    feat_1d = compute_features(df_1d, "d")
    feat_4h = compute_features(df_4h, "h4")
    feat_1h = compute_features(df_1h, "h1")

    # Get latest row from each timeframe
    row_1d = get_latest_row(feat_1d)
    row_4h = get_latest_row(feat_4h)
    row_1h = get_latest_row(feat_1h)

    # Combine into single dict
    combined = {}
    combined.update(row_1d)
    combined.update(row_4h)
    combined.update(row_1h)

    # Add MTF confluence features
    try:
        combined["mtf_trend_agree"] = int(
            combined.get("d_ema_cross",0)==1 and
            combined.get("h4_ema_cross",0)==1
        )
        combined["mtf_rsi_bullish"] = int(
            combined.get("d_rsi14",50)>50 and
            combined.get("h4_rsi14",50)>50
        )
        combined["mtf_rsi_diverge"] = (
            combined.get("d_rsi14",50) - combined.get("h4_rsi14",50)
        )
        combined["mtf_macd_agree"] = int(
            combined.get("d_macd_hist",0)>0 and
            combined.get("h4_macd_hist",0)>0
        )
        score = sum([
            combined.get("d_ema_cross",0),
            combined.get("h4_ema_cross",0),
            combined.get("h1_ema_cross",0),
        ])
        combined["mtf_score"] = score
    except:
        pass

    # Feature interactions (must match feature-engineering.py)
    try:
        combined["rsi_vol_bull"]  = combined.get("d_rsi14", 50) * combined.get("d_vol_ratio", 1)
        combined["macd_bb_pos"]   = combined.get("d_macd_hist", 0) * combined.get("d_bb_position", 0.5)
        combined["ema_rsi_align"] = combined.get("d_ema_cross", 0) * (combined.get("d_rsi14", 50) - 50)
        combined["funding_rsi"]   = combined.get("funding_rate", 0) * combined.get("d_rsi14", 50)
    except:
        pass

    # Current price and date from daily
    combined["close"] = df_1d["close"].iloc[-1]
    combined["date"]  = df_1d["date"].iloc[-1]

    return combined


def kelly_fraction(p, avg_win, avg_loss, max_f=0.25):
    if avg_loss == 0: return 0.0
    b = avg_win / avg_loss
    f = (p * b - (1-p)) / b
    return round(min(max(0.0, f/2), max_f)*100, 1)


def ensemble_proba(artifacts, row_df):
    probas = []
    for name, model in artifacts["models"].items():
        Xs = artifacts["scalers"][name].transform(row_df.values)
        probas.append(model.predict_proba(Xs)[0][1])
    return float(np.mean(probas))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    print(f"\n{'='*54}")
    print(f"  Daily Signal Generator v4 — Multi-Timeframe")
    print(f"  {now}")
    print(f"{'='*54}\n")

    print("  Fetching external signals...")
    fg      = get_fear_greed()
    btc_dom = get_btc_dominance()
    funding = get_funding("BTCUSDT")
    dxy     = get_dxy()

    if fg:
        print(f"  Fear & Greed:  {fg['fear_greed']} — {fg.get('fg_label','')}")
    if btc_dom:
        print(f"  BTC Dominance: {btc_dom['btc_dominance']:.1f}%")
    if funding:
        print(f"  Funding Rate:  {funding['funding_rate']:.4f}%")
    if dxy:
        print(f"  DXY:           {dxy['dxy_close']:.2f}  ({'above' if dxy['dxy_above_ma20'] else 'below'} MA20)")

    external = {**fg, **btc_dom, **funding, **dxy}
    external.pop("fg_label", None)

    print()
    signals = []

    for ticker, symbol in COINS.items():
        print(f"  ── {ticker} {'─'*38}")
        try:
            # Load ensemble
            ap = os.path.join(MODEL_DIR, f"{ticker}_ensemble.pkl")
            if not os.path.exists(ap):
                raise FileNotFoundError(f"{ap} not found — run train-model.py first")
            with open(ap, "rb") as f:
                artifacts = pickle.load(f)

            feature_cols = artifacts["features"]

            # Build live MTF features
            print(f"  Fetching 1H + 4H + 1D candles...")
            live = build_live_features(symbol)
            live.update(external)

            # Build feature row
            missing = [f for f in feature_cols if f not in live]
            if missing:
                print(f"  ⚠  Missing features: {missing[:5]}{'...' if len(missing)>5 else ''}")
                # Fill missing with 0
                for f in missing:
                    live[f] = 0

            row = pd.DataFrame([{f: live.get(f, 0) for f in feature_cols}])

            # Replace NaN with 0
            row = row.fillna(0)

            prob_up   = ensemble_proba(artifacts, row)
            prob_down = 1 - prob_up

            coin_thresh = CONF_THRESHOLD.get(ticker, CONF_THRESH)
            if prob_up >= coin_thresh:
                signal    = "BUY / HOLD"
                arrow     = "▲"
                kelly_pct = kelly_fraction(prob_up, artifacts["avg_win"], artifacts["avg_loss"])
            elif prob_down >= coin_thresh:
                signal    = "STAY OUT"
                arrow     = "▼"
                kelly_pct = 0.0
            else:
                signal    = "SKIP — low confidence"
                arrow     = "—"
                kelly_pct = 0.0

            if prob_up >= 0.65:     conf_label = "Strong"
            elif prob_up >= 0.55:   conf_label = "Moderate"
            elif prob_down >= 0.65: conf_label = "Strong (bearish)"
            elif prob_down >= 0.55: conf_label = "Moderate (bearish)"
            else:                   conf_label = "Too weak to act"

            # MTF context
            mtf_score = live.get("mtf_score", 0)
            mtf_agree = live.get("mtf_trend_agree", 0)
            mtf_str   = f"{int(mtf_score)}/3 TF bullish" + (" ✓ All agree" if mtf_agree else "")

            price = live["close"]
            print(f"  Price:       ${price:,.2f}")
            print(f"  Signal:      {arrow}  {signal}")
            print(f"  Confidence:  UP {prob_up*100:.1f}%  DOWN {prob_down*100:.1f}%  [{conf_label}]")
            print(f"  MTF:         {mtf_str}")
            if kelly_pct > 0:
                print(f"  Position:    {kelly_pct:.1f}% of capital  (Half-Kelly)")
            else:
                print(f"  Position:    0% — do not enter")
            print()

            signals.append({
                "date":       str(live["date"]),
                "coin":       ticker,
                "price":      price,
                "signal":     signal,
                "prob_up":    round(prob_up*100, 1),
                "prob_down":  round(prob_down*100, 1),
                "kelly_pct":  kelly_pct,
                "confidence": conf_label,
                "mtf_score":  int(mtf_score),
                "fear_greed": fg.get("fear_greed",""),
                "fg_label":   fg.get("fg_label",""),
                "funding":    funding.get("funding_rate",""),
                "btc_dom":    btc_dom.get("btc_dominance",""),
            })

        except Exception as e:
            print(f"  ✗  Error: {e}\n")
            import traceback; traceback.print_exc()

        time.sleep(0.5)

    # Summary
    print(f"{'='*54}")
    print(f"  Summary")
    print(f"{'='*54}")
    print(f"  {'Coin':<6} {'Price':>12}  {'Signal':<22} {'Up%':>6}  {'MTF':>6}  {'Position'}")
    print(f"  {'─'*62}")
    for s in signals:
        arrow = "▲" if "BUY" in s["signal"] else "▼" if "OUT" in s["signal"] else "—"
        pos   = f"{s['kelly_pct']:.1f}% capital" if s["kelly_pct"]>0 else "skip"
        print(f"  {s['coin']:<6} ${s['price']:>11,.2f}  "
              f"{arrow} {s['signal']:<20}  "
              f"{s['prob_up']:>5.1f}%  {s['mtf_score']:>3}/3  {pos}")

    print(f"\n  Context:")
    if fg:
        zone = "Contrarian BUY" if fg.get("fg_extreme_fear") else \
               "Contrarian SELL" if fg.get("fg_extreme_greed") else "Neutral"
        print(f"  F&G {fg.get('fear_greed','?')} ({fg.get('fg_label','?')}) — {zone}")
    if funding:
        fr = funding.get("funding_rate",0)
        print(f"  Funding {fr:.4f}% — " +
              ("overleveraged LONG" if fr>0.05 else "squeeze risk" if fr<-0.01 else "neutral"))

    print(f"\n  ⚠  Model signals only — not financial advice.")
    print(f"{'='*54}\n")

    # Log to CSV
    if signals:
        log_path = os.path.join(DATA_DIR, "signal_log.csv")
        log_df   = pd.DataFrame(signals)
        header   = not os.path.exists(log_path)
        log_df.to_csv(log_path, mode="a", header=header, index=False)
        print(f"  Logged → {log_path}\n")

    # Log to DB
    try:
        import db
        db.init_schema()
        for s in signals:
            db.try_write(db.upsert_signal, s)
        if signals:
            print(f"  ✓  Signals → DB ({len(signals)} rows)")
    except Exception as e:
        print(f"  ⚠  DB skipped: {e}")

    # Telegram
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("tg", "telegram_notify.py")
        tg   = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(tg)
        if tg.TELEGRAM_TOKEN:
            tg.notify_signals(signals)
        else:
            print("  ⚠  Telegram skipped — TELEGRAM_TOKEN not set")
    except Exception as e:
        print(f"  ⚠  Telegram failed: {e}")


if __name__ == "__main__":
    main()