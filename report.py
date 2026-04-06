"""
Report Generator — Daily & Weekly Performance Report
Reads the paper trading ledger and generates a clean text + HTML report.
Run manually or automatically via run_all.py.

Usage:
  python report.py          → daily report
  python report.py --weekly → weekly report
"""

import pandas as pd
import numpy as np
import json
import os
import argparse
from datetime import datetime, timezone, timedelta

DATA_DIR    = "data"
LEDGER_PATH = os.path.join(DATA_DIR, "paper_ledger.csv")
SUMMARY_PATH = os.path.join(DATA_DIR, "paper_summary.json")
REPORT_HTML = "report.html"


def load_data():
    ledger   = pd.read_csv(LEDGER_PATH, parse_dates=["date"]) \
               if os.path.exists(LEDGER_PATH) else pd.DataFrame()
    summary  = json.load(open(SUMMARY_PATH)) \
               if os.path.exists(SUMMARY_PATH) else {}
    return ledger, summary


def calc_metrics(trades: pd.DataFrame) -> dict:
    """Calculate performance metrics from a set of completed trades."""
    sells = trades[trades["action"] == "SELL"]

    if sells.empty:
        return {
            "n_trades": 0, "wins": 0, "losses": 0, "win_rate": 0,
            "total_pnl": 0, "avg_pnl": 0, "avg_win": 0, "avg_loss": 0,
            "profit_factor": 0, "best_trade": 0, "worst_trade": 0,
            "total_commission": 0,
        }

    wins   = sells[sells["pnl"] > 0]
    losses = sells[sells["pnl"] < 0]

    gross_profit = wins["pnl"].sum()
    gross_loss   = abs(losses["pnl"].sum())
    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    return {
        "n_trades":        len(sells),
        "wins":            len(wins),
        "losses":          len(losses),
        "win_rate":        len(wins) / len(sells) * 100 if len(sells) > 0 else 0,
        "total_pnl":       sells["pnl"].sum(),
        "avg_pnl":         sells["pnl"].mean(),
        "avg_win":         wins["pnl"].mean() if len(wins) > 0 else 0,
        "avg_loss":        losses["pnl"].mean() if len(losses) > 0 else 0,
        "profit_factor":   pf,
        "best_trade":      sells["pnl"].max() if len(sells) > 0 else 0,
        "worst_trade":     sells["pnl"].min() if len(sells) > 0 else 0,
        "total_commission":trades["commission"].sum(),
    }


def print_report(ledger: pd.DataFrame, summary: dict, period: str, since: str):
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    print(f"\n{'='*56}")
    print(f"  Crypto Paper Trading — {period} Report")
    print(f"  {now}")
    print(f"{'='*56}")

    if ledger.empty:
        print("\n  No trades recorded yet.")
        print("  Run predict-today.py and paper-trader.py first.")
        print(f"{'='*56}\n")
        return

    # Filter to period
    ledger["date"] = pd.to_datetime(ledger["date"])
    period_ledger  = ledger[ledger["date"] >= since]

    # ── Portfolio Status ───────────────────────────────────────────────────────
    if summary:
        s = summary.get("summary", {})
        print(f"\n  Portfolio Status ({summary.get('updated','')})")
        print(f"  {'─'*48}")
        print(f"  Initial capital:    ${s.get('initial', 0):>10,.2f}")
        print(f"  Current value:      ${s.get('total_portfolio', 0):>10,.2f}")
        ret = s.get('total_return', 0)
        print(f"  Total return:       {ret:>+10.2f}%  "
              f"({'▲' if ret >= 0 else '▼'} ${abs(s.get('total_portfolio',0) - s.get('initial',0)):,.2f})")
        print(f"  Cash available:     ${s.get('total_cash', 0):>10,.2f}")
        if s.get('total_invested', 0) > 0:
            print(f"  Currently invested: ${s.get('total_invested', 0):>10,.2f}")
            print(f"  Unrealized P&L:    ${s.get('total_unrealized', 0):>+10.2f}")

        # Per-coin status
        print(f"\n  Per-Coin Status:")
        for pos in s.get("positions", []):
            if pos["status"] == "IN POSITION":
                print(f"    {pos['coin']:<4} IN POSITION  "
                      f"entry=${pos['entry_price']:,.2f}  "
                      f"current=${pos.get('current_price',0):,.2f}  "
                      f"unreal={pos.get('unreal_pct',0):+.2f}%")
            else:
                print(f"    {pos['coin']:<4} FLAT         "
                      f"cash=${pos['cash']:.2f}")

    # ── Period Performance ─────────────────────────────────────────────────────
    print(f"\n  {period} Performance (since {since})")
    print(f"  {'─'*48}")

    for coin in ["BTC", "ETH", "SOL"]:
        coin_trades = period_ledger[period_ledger["coin"] == coin]
        m = calc_metrics(coin_trades)

        if m["n_trades"] == 0:
            buys  = len(coin_trades[coin_trades["action"]=="BUY"])
            print(f"  {coin}  {buys} open position(s), no closed trades yet")
            continue

        pf_str = f"{m['profit_factor']:.2f}" if m["profit_factor"] != float("inf") else "∞"
        print(f"\n  {coin}")
        print(f"    Trades:         {m['n_trades']}  "
              f"({m['wins']}W / {m['losses']}L)  "
              f"Win rate: {m['win_rate']:.1f}%")
        print(f"    Total P&L:      ${m['total_pnl']:+.4f}")
        print(f"    Profit Factor:  {pf_str}")
        print(f"    Avg win:        ${m['avg_win']:+.4f}  "
              f"Avg loss: ${m['avg_loss']:+.4f}")
        print(f"    Best trade:     ${m['best_trade']:+.4f}  "
              f"Worst: ${m['worst_trade']:+.4f}")
        print(f"    Commissions:    ${m['total_commission']:.4f}")

    # ── All trades this period ─────────────────────────────────────────────────
    sells = period_ledger[period_ledger["action"] == "SELL"]
    if not sells.empty:
        print(f"\n  Closed Trades ({period}):")
        print(f"  {'─'*48}")
        print(f"  {'Date':<12} {'Coin':<5} {'Price':>10}  "
              f"{'P&L':>10}  {'P&L%':>7}  {'Conf':>6}")
        print(f"  {'─'*56}")
        for _, row in sells.sort_values("date", ascending=False).iterrows():
            emoji = "▲" if row["pnl"] >= 0 else "▼"
            print(f"  {str(row['date'])[:10]:<12} "
                  f"{row['coin']:<5} "
                  f"${row['price']:>9,.2f}  "
                  f"{emoji} ${row['pnl']:>+8.4f}  "
                  f"{row['pnl_pct']:>+6.2f}%  "
                  f"{row['signal_confidence']:>5.1f}%")

    # ── Overall metrics ────────────────────────────────────────────────────────
    all_metrics = calc_metrics(period_ledger)
    if all_metrics["n_trades"] > 0:
        pf_str = f"{all_metrics['profit_factor']:.2f}" \
                 if all_metrics["profit_factor"] != float("inf") else "∞"
        print(f"\n  Overall {period} Metrics:")
        print(f"  {'─'*48}")
        print(f"  Total trades:    {all_metrics['n_trades']}  "
              f"({all_metrics['wins']}W / {all_metrics['losses']}L)")
        print(f"  Win rate:        {all_metrics['win_rate']:.1f}%")
        print(f"  Profit Factor:   {pf_str}")
        print(f"  Total P&L:       ${all_metrics['total_pnl']:+.4f}")
        print(f"  Avg trade:       ${all_metrics['avg_pnl']:+.4f}")
        print(f"  Best / Worst:    ${all_metrics['best_trade']:+.4f} / "
              f"${all_metrics['worst_trade']:+.4f}")
        print(f"  Total fees paid: ${all_metrics['total_commission']:.4f}")

    print(f"\n{'='*56}\n")

    return all_metrics


def generate_html_report(ledger: pd.DataFrame, summary: dict,
                          period: str, since: str):
    """Generate a clean HTML report file."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    if ledger.empty:
        rows_html = "<tr><td colspan='8' style='text-align:center;color:#64748b;padding:20px'>No trades yet</td></tr>"
    else:
        ledger["date"] = pd.to_datetime(ledger["date"])
        period_sells   = ledger[(ledger["date"] >= since) & (ledger["action"] == "SELL")]
        rows_html = ""
        for _, row in period_sells.sort_values("date", ascending=False).iterrows():
            color  = "#10b981" if row["pnl"] >= 0 else "#ef4444"
            arrow  = "▲" if row["pnl"] >= 0 else "▼"
            rows_html += f"""<tr>
              <td>{str(row['date'])[:10]}</td>
              <td style='color:{("color:#f59e0b" if row["coin"]=="BTC" else "color:#627eea" if row["coin"]=="ETH" else "color:#10b981")};font-weight:700'>{row['coin']}</td>
              <td>${row['price']:,.2f}</td>
              <td style='color:{color}'>{arrow} ${row['pnl']:+.4f}</td>
              <td style='color:{color}'>{row['pnl_pct']:+.2f}%</td>
              <td>{row['signal_confidence']:.1f}%</td>
              <td>${row['kelly_pct']:.1f}%</td>
              <td style='color:#64748b;font-size:11px'>{row.get('reason','')[:30]}</td>
            </tr>"""

    s           = summary.get("summary", {})
    total_val   = s.get("total_portfolio", LEDGER_PATH)
    total_ret   = s.get("total_return", 0)
    ret_color   = "#10b981" if total_ret >= 0 else "#ef4444"

    # Coin cards
    cards_html = ""
    for pos in s.get("positions", []):
        coin  = pos["coin"]
        color = "#f59e0b" if coin=="BTC" else "#627eea" if coin=="ETH" else "#10b981"
        if pos["status"] == "IN POSITION":
            uc    = "#10b981" if pos.get("unreal_pct",0)>=0 else "#ef4444"
            cards_html += f"""<div class='card'>
              <div class='coin' style='color:{color}'>{coin}</div>
              <div class='badge buy'>IN POSITION</div>
              <div class='stat-row'><span>Entry</span><span>${pos['entry_price']:,.2f}</span></div>
              <div class='stat-row'><span>Current</span><span>${pos.get('current_price',0):,.2f}</span></div>
              <div class='stat-row'><span>Unrealized</span><span style='color:{uc}'>{pos.get('unreal_pct',0):+.2f}%</span></div>
              <div class='stat-row'><span>Cash</span><span>${pos['cash']:.2f}</span></div>
            </div>"""
        else:
            cards_html += f"""<div class='card'>
              <div class='coin' style='color:{color}'>{coin}</div>
              <div class='badge flat'>FLAT</div>
              <div class='stat-row'><span>Available cash</span><span>${pos['cash']:.2f}</span></div>
            </div>"""

    html = f"""<!DOCTYPE html>
<html lang='en'>
<head>
<meta charset='UTF-8'>
<meta name='viewport' content='width=device-width,initial-scale=1'>
<title>Paper Trading Report — {period}</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#080b12;color:#e2e8f0;padding:24px;font-size:14px}}
h1{{font-size:20px;font-weight:700;margin-bottom:4px}}
.sub{{color:#64748b;font-size:12px;margin-bottom:28px}}
.section{{margin-bottom:28px}}
.slabel{{font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:1px;color:#64748b;margin-bottom:12px}}
.cards{{display:flex;gap:16px;flex-wrap:wrap;margin-bottom:24px}}
.card{{background:#0f1320;border:1px solid #252d45;border-radius:12px;padding:18px;min-width:200px;flex:1}}
.coin{{font-size:20px;font-weight:800;margin-bottom:8px}}
.badge{{display:inline-block;padding:3px 10px;border-radius:20px;font-size:11px;font-weight:600;margin-bottom:12px}}
.badge.buy{{background:rgba(16,185,129,.15);color:#10b981;border:1px solid rgba(16,185,129,.3)}}
.badge.flat{{background:#1d2540;color:#64748b;border:1px solid #252d45}}
.stat-row{{display:flex;justify-content:space-between;padding:4px 0;border-bottom:1px solid rgba(37,45,69,.5);font-size:13px}}
.stat-row:last-child{{border-bottom:none}}
.metrics{{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:12px;margin-bottom:24px}}
.metric{{background:#0f1320;border:1px solid #252d45;border-radius:10px;padding:14px;text-align:center}}
.metric .num{{font-size:22px;font-weight:700;margin-bottom:4px}}
.metric .lbl{{font-size:10px;color:#64748b;text-transform:uppercase;letter-spacing:.5px}}
.twrap{{overflow-x:auto;border-radius:10px;border:1px solid #252d45}}
table{{width:100%;border-collapse:collapse;font-size:12px}}
thead tr{{background:#0f1320}}
th{{padding:10px 14px;text-align:left;color:#64748b;font-weight:500;font-size:11px;text-transform:uppercase;border-bottom:1px solid #252d45}}
td{{padding:10px 14px;border-bottom:1px solid rgba(37,45,69,.4)}}
tr:last-child td{{border-bottom:none}}
tr:hover td{{background:rgba(29,37,64,.4)}}
.portfolio-val{{font-size:36px;font-weight:800;color:{ret_color};margin:8px 0}}
.portfolio-ret{{font-size:18px;color:{ret_color}}}
footer{{margin-top:32px;text-align:center;font-size:11px;color:#475569}}
</style>
</head>
<body>

<h1>Paper Trading Report — {period}</h1>
<div class='sub'>Generated: {now}</div>

<div class='section'>
  <div class='slabel'>Portfolio Value</div>
  <div class='portfolio-val'>${s.get('total_portfolio', 0):,.2f}</div>
  <div class='portfolio-ret'>{total_ret:+.2f}% since start
    (${abs(s.get('total_portfolio',0) - s.get('initial',3000)):,.2f}
    {'profit' if total_ret >= 0 else 'loss'})</div>
</div>

<div class='section'>
  <div class='slabel'>Position Status</div>
  <div class='cards'>{cards_html}</div>
</div>

<div class='section'>
  <div class='slabel'>{period} Closed Trades</div>
  <div class='twrap'>
    <table>
      <thead><tr>
        <th>Date</th><th>Coin</th><th>Exit Price</th>
        <th>P&amp;L ($)</th><th>P&amp;L (%)</th>
        <th>Confidence</th><th>Kelly</th><th>Reason</th>
      </tr></thead>
      <tbody>{rows_html}</tbody>
    </table>
  </div>
</div>

<footer>
  Crypto Paper Trader &nbsp;·&nbsp; Not financial advice &nbsp;·&nbsp; {now}
</footer>
</body>
</html>"""

    with open(REPORT_HTML, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  ✓  HTML report → {REPORT_HTML}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weekly", action="store_true", help="Weekly report")
    parser.add_argument("--daily",  action="store_true", help="Daily report (default)")
    args = parser.parse_args()

    now = datetime.now(timezone.utc)

    if args.weekly:
        period = "Weekly"
        since  = (now - timedelta(days=7)).strftime("%Y-%m-%d")
    else:
        period = "Daily"
        since  = now.strftime("%Y-%m-%d")

    ledger, summary = load_data()

    if ledger.empty and not summary:
        print("\n  No data yet — run predict-today.py and paper-trader.py first.\n")
        return

    print_report(ledger, summary, period, since)
    generate_html_report(ledger, summary, period, since)


if __name__ == "__main__":
    main()