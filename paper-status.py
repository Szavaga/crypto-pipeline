"""
Paper Trading Status — run this anytime to check live portfolio values.
Fetches current prices, shows unrealized P&L on open positions, and
lists recent closed trades from the ledger.

Usage: python paper-status.py
"""

import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests

DATA_DIR  = "data"
STATE_FILE  = os.path.join(DATA_DIR, "simple_paper_state.json")
LEDGER_FILE = os.path.join(DATA_DIR, "simple_paper_ledger.csv")

COINS = {"BTC": "BTCUSDT", "ETH": "ETHUSDT", "SOL": "SOLUSDT",
         "AVAX": "AVAXUSDT", "LINK": "LINKUSDT"}
YFINANCE_MAP = {"BTCUSDT": "BTC-USD", "ETHUSDT": "ETH-USD", "SOLUSDT": "SOL-USD",
                "AVAXUSDT": "AVAX-USD", "LINKUSDT": "LINK-USD"}

INITIAL_CAPITAL = 200.0
STRATEGIES      = ["ensemble", "fund", "rsi", "ml3", "ml5"]
STRAT_LABELS    = {
    "ensemble": "Ensemble",
    "fund":     "FundRate",
    "rsi":      "RSI-MR",
    "ml3":      "Simple3",
    "ml5":      "Simple5",
}


def get_prices() -> dict:
    """Fetch latest price for each coin via Binance ticker (or yfinance fallback)."""
    prices = {}
    for ticker, symbol in COINS.items():
        try:
            resp = requests.get(
                "https://api.binance.com/api/v3/ticker/price",
                params={"symbol": symbol}, timeout=10)
            if resp.status_code == 200:
                prices[ticker] = float(resp.json()["price"])
                continue
        except Exception:
            pass
        try:
            import yfinance as yf
            yf_sym = YFINANCE_MAP[symbol]
            hist   = yf.Ticker(yf_sym).history(period="2d", interval="1h", auto_adjust=True)
            if not hist.empty:
                prices[ticker] = float(hist["Close"].iloc[-1])
        except Exception:
            pass
    return prices


def main():
    if not os.path.exists(STATE_FILE):
        print("\n  No paper state found. Run compare-strategies.py first.\n")
        return

    with open(STATE_FILE) as fh:
        state = json.load(fh)

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    print(f"\n{'='*66}")
    print(f"  Paper Portfolio Status — {now}")
    print(f"{'='*66}\n")

    print("  Fetching live prices...", end=" ", flush=True)
    prices = get_prices()
    print("  ".join(f"{c}=${v:,.2f}" for c, v in prices.items()))
    print()

    # ── Portfolio summary ──────────────────────────────────────────────────────
    print(f"  {'─'*62}")
    print(f"  {'Strategy':<12}  {'Cash':>8}  {'Invested':>9}  "
          f"{'Unreal P&L':>11}  {'Total':>8}  {'Ret%':>7}")
    print(f"  {'─'*62}")

    for strat in STRATEGIES:
        cash     = state[strat]["capital"]
        invested = 0.0
        unreal   = 0.0

        for coin, pos in state[strat]["positions"].items():
            p_now     = prices.get(coin, pos["entry_price"])
            cur_val   = pos["size_usd"] * (p_now / pos["entry_price"])
            invested += pos["size_usd"]
            unreal   += cur_val - pos["size_usd"]

        total = cash + invested + unreal
        ret   = (total / INITIAL_CAPITAL - 1) * 100
        print(f"  {STRAT_LABELS[strat]:<12}  ${cash:>7.2f}  ${invested:>8.2f}  "
              f"  {unreal:>+10.2f}  ${total:>7.2f}  {ret:>+6.1f}%")

    print(f"  {'─'*62}")

    # ── Open positions detail ──────────────────────────────────────────────────
    any_open = any(state[s]["positions"] for s in STRATEGIES)
    if any_open:
        print(f"\n  Open positions:")
        print(f"  {'─'*58}")
        print(f"  {'Strategy':<12}  {'Coin':<5}  {'Entry':>10}  "
              f"{'Now':>10}  {'Unreal':>9}  {'Since'}")
        print(f"  {'─'*58}")
        for strat in STRATEGIES:
            for coin, pos in state[strat]["positions"].items():
                p_entry  = pos["entry_price"]
                p_now    = prices.get(coin, p_entry)
                size_usd = pos["size_usd"]
                unreal   = size_usd * (p_now / p_entry - 1)
                entry_ts = pos.get("entry_ts", "?")
                print(f"  {STRAT_LABELS[strat]:<12}  {coin:<5}  "
                      f"${p_entry:>9,.4f}  ${p_now:>9,.4f}  "
                      f"{unreal:>+8.2f}  {entry_ts}")
        print(f"  {'─'*58}")

    # ── Recent closed trades ───────────────────────────────────────────────────
    if os.path.exists(LEDGER_FILE):
        df = pd.read_csv(LEDGER_FILE)
        closed = df[df["action"] == "CLOSE"].tail(15)
        if not closed.empty:
            print(f"\n  Last {len(closed)} closed trades:")
            print(f"  {'─'*64}")
            print(f"  {'Timestamp':<17}  {'Strategy':<10}  {'Coin':<5}  "
                  f"{'Ret%':>7}  {'P&L':>8}  {'Cycles':>7}  {'W/L'}")
            print(f"  {'─'*64}")
            for _, t in closed.iterrows():
                icon = "W" if str(t.get("win", "")).lower() == "true" else "L"
                ret  = f"{float(t['ret_pct']):>+.2f}%" if t["ret_pct"] != "" else "  open"
                pnl  = f"${float(t['pnl_usd']):>+.3f}" if t["pnl_usd"] != "" else "     —"
                cyc  = str(t.get("hold_cycles", "?"))
                print(f"  {str(t['ts']):<17}  {str(t['strategy']):<10}  "
                      f"{str(t['coin']):<5}  {ret:>7}  {pnl:>8}  {cyc:>7}  {icon}")
            print(f"  {'─'*64}")

        # Per-strategy win/loss summary
        if len(df[df["action"] == "CLOSE"]) > 0:
            print(f"\n  Win/Loss summary (all time):")
            for strat in STRATEGIES:
                s_df   = df[(df["strategy"] == strat) & (df["action"] == "CLOSE")]
                n      = len(s_df)
                if n == 0:
                    continue
                wins   = (s_df["win"].astype(str).str.lower() == "true").sum()
                total_pnl = s_df["pnl_usd"].astype(float).sum()
                wr     = wins / n * 100
                print(f"  {STRAT_LABELS[strat]:<12}  {n:>3} trades  "
                      f"WR={wr:.0f}%  total P&L=${total_pnl:>+.2f}")

    print(f"\n{'='*66}\n")


if __name__ == "__main__":
    main()
