"""
Paper Trading Engine
Automatically executes paper trades based on model signals.
Tracks positions, P&L, and performance metrics.

Runs daily after predict-today.py — reads the signal log,
opens/closes positions, and updates the paper trading ledger.

Usage: python paper-trader.py
       (or via run_all.py automatically)
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timezone, timedelta
import requests

DATA_DIR   = "data"
LEDGER_PATH = os.path.join(DATA_DIR, "paper_ledger.csv")
POSITIONS_PATH = os.path.join(DATA_DIR, "paper_positions.json")
SIGNAL_LOG = os.path.join(DATA_DIR, "signal_log.csv")

INITIAL_CAPITAL = 1000.0   # starting paper money per coin
COMMISSION      = 0.001    # 0.1% Binance fee simulation
CONF_THRESHOLD  = 55.0     # minimum confidence to enter

COINS = ["BTC", "ETH", "SOL"]
BINANCE_URL = "https://api.binance.com/api/v3/ticker/price"


# ── Price fetcher ─────────────────────────────────────────────────────────────

def get_current_price(symbol: str) -> float:
    try:
        r = requests.get(BINANCE_URL, params={"symbol": symbol}, timeout=10)
        return float(r.json()["price"])
    except:
        return 0.0


# ── Ledger management ─────────────────────────────────────────────────────────

def load_ledger() -> pd.DataFrame:
    if os.path.exists(LEDGER_PATH):
        return pd.read_csv(LEDGER_PATH, parse_dates=["date"])
    return pd.DataFrame(columns=[
        "date", "coin", "action", "price", "quantity", "value",
        "commission", "pnl", "pnl_pct", "balance_after",
        "signal_confidence", "kelly_pct", "reason"
    ])


def save_ledger(df: pd.DataFrame):
    df.to_csv(LEDGER_PATH, index=False)


def load_positions() -> dict:
    """Load open positions. Format: {coin: {entry_price, quantity, value, date, confidence}}"""
    if os.path.exists(POSITIONS_PATH):
        with open(POSITIONS_PATH, "r") as f:
            return json.load(f)
    # Initialize with equal capital per coin
    return {
        coin: {
            "in_position": False,
            "entry_price": 0.0,
            "quantity":    0.0,
            "value":       0.0,
            "entry_date":  "",
            "confidence":  0.0,
            "kelly_pct":   0.0,
            "balance":     INITIAL_CAPITAL,  # available cash
        }
        for coin in COINS
    }


def save_positions(positions: dict):
    with open(POSITIONS_PATH, "w") as f:
        json.dump(positions, f, indent=2)


# ── Signal loader ─────────────────────────────────────────────────────────────

def get_latest_signals() -> dict:
    """Get the most recent signal for each coin from signal_log.csv"""
    if not os.path.exists(SIGNAL_LOG):
        return {}

    df = pd.read_csv(SIGNAL_LOG)
    df["date"] = pd.to_datetime(df["date"])
    latest = {}
    for coin in COINS:
        rows = df[df["coin"] == coin].sort_values("date", ascending=False)
        if not rows.empty:
            latest[coin] = rows.iloc[0].to_dict()
    return latest


# ── Trade execution ───────────────────────────────────────────────────────────

def execute_trades(positions: dict, signals: dict,
                   ledger: pd.DataFrame, today: str) -> tuple:
    """
    For each coin:
    - If signal = BUY and not in position → ENTER
    - If signal = STAY OUT and in position → EXIT
    - If signal = SKIP → hold current state
    Returns updated positions and ledger.
    """
    new_rows = []

    COIN_SYMBOLS = {"BTC": "BTCUSDT", "ETH": "ETHUSDT", "SOL": "SOLUSDT"}

    for coin in COINS:
        pos    = positions[coin]
        signal = signals.get(coin, {})

        if not signal:
            continue

        sig_text   = str(signal.get("signal", ""))
        prob_up    = float(signal.get("prob_up", 50))
        kelly_pct  = float(signal.get("kelly_pct", 0))
        confidence = str(signal.get("confidence", ""))
        sig_date   = str(signal.get("date", today))

        # Only act on today's signal
        if sig_date != today:
            continue

        # Get current price
        price = get_current_price(COIN_SYMBOLS[coin])
        if price == 0:
            print(f"  ⚠  Could not fetch price for {coin} — skipping")
            continue

        # ── ENTER position ────────────────────────────────────────────────────
        if ("BUY" in sig_text and prob_up >= CONF_THRESHOLD
                and not pos["in_position"] and pos["balance"] > 10):

            # Position size based on Kelly
            if kelly_pct > 0:
                invest_pct = kelly_pct / 100
            else:
                invest_pct = 0.10  # default 10% if Kelly not set

            invest_amount = pos["balance"] * invest_pct
            commission    = invest_amount * COMMISSION
            net_invest    = invest_amount - commission
            quantity      = net_invest / price

            pos["in_position"] = True
            pos["entry_price"] = price
            pos["quantity"]    = quantity
            pos["value"]       = net_invest
            pos["entry_date"]  = today
            pos["confidence"]  = prob_up
            pos["kelly_pct"]   = kelly_pct
            pos["balance"]    -= invest_amount

            new_rows.append({
                "date":               today,
                "coin":               coin,
                "action":             "BUY",
                "price":              round(price, 4),
                "quantity":           round(quantity, 6),
                "value":              round(net_invest, 2),
                "commission":         round(commission, 4),
                "pnl":                0.0,
                "pnl_pct":            0.0,
                "balance_after":      round(pos["balance"], 2),
                "signal_confidence":  round(prob_up, 1),
                "kelly_pct":          round(kelly_pct, 1),
                "reason":             confidence,
            })
            print(f"  ✓  {coin} ENTERED  @ ${price:,.2f}  "
                  f"qty={quantity:.4f}  invested=${net_invest:.2f}  "
                  f"conf={prob_up:.1f}%")

        # ── EXIT position ─────────────────────────────────────────────────────
        elif ("OUT" in sig_text or "SKIP" not in sig_text and "BUY" not in sig_text) \
                and pos["in_position"]:

            exit_value  = pos["quantity"] * price
            commission  = exit_value * COMMISSION
            net_exit    = exit_value - commission
            pnl         = net_exit - pos["value"]
            pnl_pct     = pnl / pos["value"] * 100

            pos["balance"]    += net_exit
            pos["in_position"] = False
            pos["entry_price"] = 0.0
            pos["quantity"]    = 0.0
            pos["value"]       = 0.0

            new_rows.append({
                "date":               today,
                "coin":               coin,
                "action":             "SELL",
                "price":              round(price, 4),
                "quantity":           round(pos["quantity"] if pos["quantity"] else exit_value/price, 6),
                "value":              round(net_exit, 2),
                "commission":         round(commission, 4),
                "pnl":                round(pnl, 4),
                "pnl_pct":            round(pnl_pct, 2),
                "balance_after":      round(pos["balance"], 2),
                "signal_confidence":  round(prob_up, 1),
                "kelly_pct":          0.0,
                "reason":             f"Exit signal — {sig_text}",
            })
            emoji = "✓" if pnl >= 0 else "✗"
            print(f"  {emoji}  {coin} EXITED   @ ${price:,.2f}  "
                  f"P&L=${pnl:+.2f} ({pnl_pct:+.2f}%)  "
                  f"balance=${pos['balance']:.2f}")

        # ── HOLD / SKIP ───────────────────────────────────────────────────────
        else:
            status = "holding" if pos["in_position"] else "flat"
            unrealized = ""
            if pos["in_position"] and price > 0:
                unreal_pnl = (price - pos["entry_price"]) / pos["entry_price"] * 100
                unrealized = f"  unrealized={unreal_pnl:+.2f}%"
            print(f"  —  {coin} {status.upper():<8} @ ${price:,.2f}{unrealized}")

    # Append new rows to ledger
    if new_rows:
        new_df  = pd.DataFrame(new_rows)
        ledger  = pd.concat([ledger, new_df], ignore_index=True)

    return positions, ledger


# ── Portfolio summary ─────────────────────────────────────────────────────────

def portfolio_summary(positions: dict) -> dict:
    """Calculate current portfolio value including open positions."""
    COIN_SYMBOLS = {"BTC": "BTCUSDT", "ETH": "ETHUSDT", "SOL": "SOLUSDT"}
    total_cash     = 0.0
    total_invested = 0.0
    total_unreal   = 0.0

    rows = []
    for coin in COINS:
        pos   = positions[coin]
        cash  = pos["balance"]
        total_cash += cash

        if pos["in_position"]:
            price     = get_current_price(COIN_SYMBOLS[coin])
            curr_val  = pos["quantity"] * price
            unreal    = curr_val - pos["value"]
            unreal_pct = unreal / pos["value"] * 100 if pos["value"] > 0 else 0
            total_invested += curr_val
            total_unreal   += unreal
            rows.append({
                "coin":        coin,
                "status":      "IN POSITION",
                "entry_price": pos["entry_price"],
                "current_price": price,
                "quantity":    pos["quantity"],
                "invested":    pos["value"],
                "current_val": curr_val,
                "unrealized":  unreal,
                "unreal_pct":  unreal_pct,
                "cash":        cash,
            })
        else:
            rows.append({
                "coin":        coin,
                "status":      "FLAT",
                "entry_price": 0,
                "current_price": 0,
                "quantity":    0,
                "invested":    0,
                "current_val": 0,
                "unrealized":  0,
                "unreal_pct":  0,
                "cash":        cash,
            })

    total_portfolio = total_cash + total_invested
    total_start     = INITIAL_CAPITAL * len(COINS)
    total_return    = (total_portfolio / total_start - 1) * 100

    return {
        "positions":        rows,
        "total_cash":       total_cash,
        "total_invested":   total_invested,
        "total_unrealized": total_unreal,
        "total_portfolio":  total_portfolio,
        "total_return":     total_return,
        "initial":          total_start,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    now   = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    print(f"\n{'='*52}")
    print(f"  Paper Trading Engine")
    print(f"  {now}")
    print(f"{'='*52}\n")

    # Load state
    ledger    = load_ledger()
    positions = load_positions()
    signals   = get_latest_signals()

    if not signals:
        print("  ⚠  No signals found — run predict-today.py first")
        return

    print(f"  Signals loaded for: {list(signals.keys())}")
    print(f"  Processing trades...\n")

    # Execute trades
    positions, ledger = execute_trades(positions, signals, ledger, today)

    # Save state (CSV/JSON)
    save_positions(positions)
    save_ledger(ledger)

    # Save state (DB)
    try:
        import db
        db.init_schema()
        for coin, pos in positions.items():
            db.try_write(db.upsert_position, "paper", coin, pos)
        new_rows = ledger[ledger["date"].astype(str).str.startswith(today)].to_dict("records")
        for row in new_rows:
            db.try_write(db.insert_trade, "paper", row)
        if new_rows:
            print(f"  ✓  Trades → DB ({len(new_rows)} rows)")
    except Exception as e:
        print(f"  ⚠  DB skipped: {e}")

    # Portfolio summary
    summary = portfolio_summary(positions)

    print(f"\n{'─'*52}")
    print(f"  Portfolio Summary")
    print(f"{'─'*52}")
    for row in summary["positions"]:
        if row["status"] == "IN POSITION":
            print(f"  {row['coin']:<4} IN POSITION  "
                  f"entry=${row['entry_price']:,.2f}  "
                  f"now=${row['current_price']:,.2f}  "
                  f"unreal={row['unreal_pct']:+.2f}%  "
                  f"cash=${row['cash']:.2f}")
        else:
            print(f"  {row['coin']:<4} FLAT         cash=${row['cash']:.2f}")

    print(f"\n  Total portfolio:  ${summary['total_portfolio']:,.2f}")
    print(f"  Total return:     {summary['total_return']:+.2f}%  "
          f"(started ${summary['initial']:,.0f})")
    print(f"  Cash:             ${summary['total_cash']:,.2f}")
    print(f"  Invested:         ${summary['total_invested']:,.2f}")
    if summary["total_unrealized"] != 0:
        print(f"  Unrealized P&L:   ${summary['total_unrealized']:+.2f}")

    # Save summary to JSON for dashboard
    summary_path = os.path.join(DATA_DIR, "paper_summary.json")
    with open(summary_path, "w") as f:
        json.dump({
            "updated":    now,
            "today":      today,
            "summary":    summary,
            "initial":    INITIAL_CAPITAL * len(COINS),
        }, f, indent=2, default=str)

    print(f"\n  ✓  Ledger → {LEDGER_PATH}")
    print(f"  ✓  Summary → {summary_path}")
    print(f"{'='*52}\n")


if __name__ == "__main__":
    main()