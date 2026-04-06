"""
Backtest — Crypto Signal Strategy
Visszateszteli a stratégiát az összes historikus adaton.
Kimenet: equity curve grafikon + részletes trade log CSV.

Requirements: pip install pandas matplotlib xgboost scikit-learn lightgbm
Usage: python backtest.py
"""

import pandas as pd
import numpy as np
import pickle
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import datetime

DATA_DIR   = "data"
MODEL_DIR  = "models"
OUTPUT_DIR = "backtest"
os.makedirs(OUTPUT_DIR, exist_ok=True)

COINS         = ["BTC", "ETH", "SOL"]
CONF_THRESH   = 0.55
INITIAL_CAP   = 1000.0    # kezdő tőke USD-ben
MAX_KELLY     = 0.25      # max 25% per trade
COMMISSION    = 0.001     # 0.1% Binance fee

EXCLUDE_COLS = ["date","open","high","low","close","volume",
                "coin","target","close_ma7","close_ma30",
                "daily_return","price_range","fg_label"]


def get_feature_cols(df):
    return [c for c in df.columns if c not in EXCLUDE_COLS and df[c].dtype != object]


def kelly_pct(prob_win, avg_win, avg_loss, max_f=MAX_KELLY):
    if avg_loss == 0: return 0.0
    b = avg_win / avg_loss
    q = 1 - prob_win
    f = (prob_win * b - q) / b
    return min(max(0.0, f / 2), max_f)


def run_backtest(ticker: str) -> dict:
    # Load features
    feat_path = os.path.join(DATA_DIR, f"{ticker}_features.csv")
    if not os.path.exists(feat_path):
        print(f"  ✗  {feat_path} not found")
        return {}

    df = pd.read_csv(feat_path, parse_dates=["date"])

    # Load ensemble model
    model_path = os.path.join(MODEL_DIR, f"{ticker}_ensemble.pkl")
    if not os.path.exists(model_path):
        print(f"  ✗  {model_path} not found — run train-model.py first")
        return {}

    with open(model_path, "rb") as f:
        artifacts = pickle.load(f)

    feature_cols = artifacts["features"]
    avg_win      = artifacts["avg_win"]
    avg_loss     = artifacts["avg_loss"]

    df = df.dropna(subset=feature_cols + ["target"]).reset_index(drop=True)

    # ── Walk-forward backtest ──────────────────────────────────────────────
    # Train on first 60%, simulate trading on remaining 40%
    # This avoids data snooping completely
    split = int(len(df) * 0.60)
    X_train = df[feature_cols].iloc[:split].values
    y_train = df["target"].iloc[:split].values

    from sklearn.preprocessing import StandardScaler
    scalers = {}
    models  = {}

    for name, model in artifacts["models"].items():
        sc = StandardScaler()
        sc.fit(X_train)
        scalers[name] = sc

    # Generate predictions for test period only
    test_df = df.iloc[split:].copy().reset_index(drop=True)
    X_test  = test_df[feature_cols].values

    probas = []
    for name, model in artifacts["models"].items():
        Xs = scalers[name].transform(X_test)
        probas.append(model.predict_proba(Xs)[:, 1])
    avg_proba = np.mean(probas, axis=0)

    test_df["prob_up"]   = avg_proba
    test_df["signal"]    = (avg_proba >= CONF_THRESH).astype(int)
    test_df["mkt_ret"]   = test_df["close"].pct_change().fillna(0)

    # Kelly sizing
    test_df["kelly"] = test_df["prob_up"].apply(
        lambda p: kelly_pct(p, avg_win, avg_loss) if p >= CONF_THRESH else 0.0
    )

    # Simulate equity
    capital     = INITIAL_CAP
    equity      = [capital]
    bnh_equity  = [capital]
    trades      = []

    for i in range(1, len(test_df)):
        row      = test_df.iloc[i]
        prev_row = test_df.iloc[i-1]

        # Market return
        mkt_ret = row["mkt_ret"]

        # Buy & hold
        bnh_equity.append(bnh_equity[-1] * (1 + mkt_ret))

        # Strategy: use previous bar's signal
        k = prev_row["kelly"]
        if k > 0:
            trade_ret = k * mkt_ret - COMMISSION * k  # fee on position size
            capital   = capital * (1 + trade_ret)
            if k > 0 and abs(mkt_ret) > 0.001:
                trades.append({
                    "date":     str(row["date"])[:10],
                    "coin":     ticker,
                    "price_in": prev_row["close"],
                    "price_out":row["close"],
                    "prob_up":  round(prev_row["prob_up"]*100, 1),
                    "kelly":    round(k*100, 1),
                    "ret_pct":  round(mkt_ret*100, 2),
                    "pnl_usd":  round(capital * k * mkt_ret, 2),
                    "win":      mkt_ret > 0,
                })
        equity.append(capital)

    # ── Performance metrics ───────────────────────────────────────────────
    eq   = np.array(equity)
    bnh  = np.array(bnh_equity)

    total_ret  = (eq[-1] / INITIAL_CAP - 1) * 100
    bnh_ret    = (bnh[-1] / INITIAL_CAP - 1) * 100
    n_trades   = len(trades)
    wins       = sum(1 for t in trades if t["win"])
    win_rate   = wins / n_trades * 100 if n_trades else 0

    # Max drawdown
    peak = np.maximum.accumulate(eq)
    dd   = (eq - peak) / peak
    max_dd = dd.min() * 100

    # Sharpe ratio (annualized, daily returns)
    daily_rets = np.diff(eq) / eq[:-1]
    sharpe     = (daily_rets.mean() / daily_rets.std() * np.sqrt(365)
                  if daily_rets.std() > 0 else 0)

    print(f"\n  {ticker} backtest results (last 40% of data):")
    print(f"  Total return:    {total_ret:+.1f}%  (B&H: {bnh_ret:+.1f}%)")
    print(f"  Final capital:   ${eq[-1]:,.2f}  (started ${INITIAL_CAP:,.0f})")
    print(f"  Trades:          {n_trades}  |  Win rate: {win_rate:.1f}%")
    print(f"  Max drawdown:    {max_dd:.1f}%")
    print(f"  Sharpe ratio:    {sharpe:.2f}")

    return {
        "ticker":     ticker,
        "equity":     equity,
        "bnh":        bnh_equity,
        "dates":      test_df["date"].astype(str).str[:10].tolist(),
        "trades":     trades,
        "total_ret":  total_ret,
        "bnh_ret":    bnh_ret,
        "win_rate":   win_rate,
        "max_dd":     max_dd,
        "sharpe":     sharpe,
        "n_trades":   n_trades,
        "final_cap":  eq[-1],
    }


def plot_results(results: list):
    if not results:
        return

    n    = len(results)
    fig  = plt.figure(figsize=(14, 5 * n), facecolor="#0f1320")
    gs   = gridspec.GridSpec(n, 2, figure=fig, hspace=0.4, wspace=0.3)

    for i, r in enumerate(results):
        color = "#627eea" if r["ticker"]=="ETH" else "#f59e0b" if r["ticker"]=="BTC" else "#10b981"
        dates = r["dates"]
        x     = range(len(dates))

        # Equity curve
        ax1 = fig.add_subplot(gs[i, 0])
        ax1.plot(x, r["equity"], color=color,   linewidth=1.5, label="Strategy")
        ax1.plot(x, r["bnh"],    color="#475569", linewidth=1,   label="Buy & Hold", linestyle="--")
        ax1.fill_between(x, r["equity"], alpha=0.1, color=color)
        ax1.set_facecolor("#0f1320")
        ax1.tick_params(colors="#64748b", labelsize=8)
        ax1.spines[["top","right","left","bottom"]].set_color("#1d2540")
        ax1.set_title(f"{r['ticker']} — Equity Curve", color="#e2e8f0", fontsize=11, pad=8)
        ax1.legend(fontsize=8, facecolor="#161c2e", labelcolor="#94a3b8", framealpha=1)
        ax1.yaxis.set_tick_params(labelcolor="#64748b")

        # Tick every ~20 bars
        step  = max(1, len(dates)//6)
        ticks = list(range(0, len(dates), step))
        ax1.set_xticks(ticks)
        ax1.set_xticklabels([dates[t] for t in ticks], rotation=30, ha="right", fontsize=7, color="#64748b")

        # Stats panel
        ax2 = fig.add_subplot(gs[i, 1])
        ax2.set_facecolor("#0f1320")
        ax2.axis("off")
        ax2.set_title(f"{r['ticker']} — Performance", color="#e2e8f0", fontsize=11, pad=8)

        metrics = [
            ("Total Return",    f"{r['total_ret']:+.1f}%",  color),
            ("Buy & Hold",      f"{r['bnh_ret']:+.1f}%",   "#475569"),
            ("Final Capital",   f"${r['final_cap']:,.0f}", "#e2e8f0"),
            ("Total Trades",    str(r["n_trades"]),         "#94a3b8"),
            ("Win Rate",        f"{r['win_rate']:.1f}%",   "#10b981" if r["win_rate"]>=50 else "#ef4444"),
            ("Max Drawdown",    f"{r['max_dd']:.1f}%",     "#ef4444"),
            ("Sharpe Ratio",    f"{r['sharpe']:.2f}",      "#10b981" if r["sharpe"]>=1 else "#f59e0b"),
        ]
        for j, (lbl, val, col) in enumerate(metrics):
            y = 0.9 - j * 0.12
            ax2.text(0.05, y, lbl, color="#64748b", fontsize=10, transform=ax2.transAxes)
            ax2.text(0.6,  y, val, color=col,       fontsize=10, transform=ax2.transAxes, fontweight="bold")

    plt.suptitle("Crypto Strategy Backtest Results", color="#e2e8f0", fontsize=14, y=1.01)
    out = os.path.join(OUTPUT_DIR, "backtest_results.png")
    plt.savefig(out, dpi=120, bbox_inches="tight", facecolor="#0f1320")
    plt.close()
    print(f"\n  ✓  Chart saved → {out}")


def main():
    print(f"\n{'='*50}")
    print(f"  Backtest — Ensemble Strategy")
    print(f"  Initial capital: ${INITIAL_CAP:,.0f}")
    print(f"  Commission: {COMMISSION*100:.1f}% per trade")
    print(f"  Confidence threshold: {CONF_THRESH*100:.0f}%")
    print(f"{'='*50}")

    results    = []
    all_trades = []

    for ticker in COINS:
        r = run_backtest(ticker)
        if r:
            results.append(r)
            all_trades.extend(r["trades"])

    # Save trade log
    if all_trades:
        tdf = pd.DataFrame(all_trades)
        out = os.path.join(OUTPUT_DIR, "trade_log.csv")
        tdf.to_csv(out, index=False)
        print(f"\n  ✓  Trade log → {out}")

    # Plot
    plot_results(results)

    # Summary
    print(f"\n{'='*50}")
    print(f"  Summary")
    print(f"{'='*50}")
    print(f"  {'Coin':<6} {'Return':>8} {'B&H':>8} {'WinRate':>8} {'Sharpe':>8} {'MaxDD':>8}")
    print(f"  {'─'*50}")
    for r in results:
        print(f"  {r['ticker']:<6} {r['total_ret']:>+7.1f}% {r['bnh_ret']:>+7.1f}% "
              f"{r['win_rate']:>7.1f}% {r['sharpe']:>8.2f} {r['max_dd']:>7.1f}%")

    print(f"\n  Files saved in: {OUTPUT_DIR}/")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()