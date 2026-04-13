# Crypto ML Trading Pipeline

An end-to-end automated trading system that collects crypto market data, engineers multi-timeframe features, trains an ML ensemble, and generates BUY/STAY OUT signals — executed live on Binance Testnet with dynamic stop-loss and take-profit orders.

---

## How it works

```
crypto-pipeline.py        → fetch 3 years of OHLCV data (5 coins)
feature-engineering.py    → build 80+ features across 1H + 4H + 1D timeframes
train-model.py            → train XGBoost + LightGBM + RandomForest ensemble
predict-today.py          → generate today's signals + Kelly position sizing
testnet-trader.py         → execute real orders on Binance Testnet (3×/day)
compare-strategies.py     → run 4 simpler strategies in parallel for comparison
paper-status.py           → check live paper portfolio value between cron runs
server.py                 → Flask dashboard served in the browser
```

The pipeline runs automatically 3 times per day (00:40, 08:40, 16:40 UTC) via cron on an Oracle Cloud server. Each run fetches live market data, generates signals, manages open testnet positions, and commits state to Git.

---

## Features

### Data sources
- 3 years of daily OHLCV candles — Binance public API
- 4H and 1H intraday candles for multi-timeframe confirmation
- Fear & Greed Index — crowd sentiment
- BTC perpetual futures funding rates — leverage and sentiment signal
- BTC dominance — market rotation signal
- DXY (US Dollar Index) — macro dollar strength via yfinance
- Geographic fallback: yfinance used automatically when Binance returns 451

### Feature engineering
- **1D** — macro trend: EMA crossovers, RSI, MACD, Bollinger Bands, ATR, OBV, volume ratio
- **4H** — medium momentum: same indicators on 4-hour timeframe
- **1H** — entry timing: intraday momentum and volume spikes
- **MTF confluence** — trend agreement score across all 3 timeframes
- **Market regime** — golden/death cross (50d vs 200d EMA), regime strength
- **Feature interactions** — RSI × volume, MACD × BB position, EMA × RSI, funding × RSI
- **Top 25 features** selected per coin via mutual information (SelectKBest)

### ML ensemble
- 3 models per coin: **XGBoost**, **LightGBM**, **RandomForest**
- Hyperparameter tuning via **Optuna** (100 trials, objective: Profit Factor)
- **Walk-forward cross-validation** — test period always in the future, no data leakage
- **Isotonic probability calibration** — output probabilities reflect real historical win rates
- Per-coin confidence thresholds: BTC 55%, ETH 57%, SOL 65%, AVAX 60%, LINK 58%
- **Target**: next-day return > 0% (binary classification)

### Trade execution
- **4H entry confirmation gate** — RSI > 50 AND MACD > 0 on 4H timeframe required before entry
- **ATR-based dynamic SL/TP** — stop-loss = 2×ATR (clamped 2–8%), take-profit = 2:1 R/R
- **Half-Kelly position sizing** — mathematically optimal bet fraction, capped at 25% per trade
- **OCO orders** (One-Cancels-Other) on Binance Testnet — real order mechanics, paper capital
- Telegram notifications on entry, TP hit, SL hit, and daily summary

### Strategy comparison
- 4 simpler strategies run in parallel with the ensemble, each with $200 paper capital:
  - **Funding Rate Contrarian** — rule-based: buy on extreme shorts, stay out on overleveraged longs
  - **RSI Mean Reversion** — rule-based: oversold + volume confirmation
  - **Simple3 XGBoost** — 3-feature model: RSI, funding rate, volume ratio
  - **Simple5 XGBoost** — 5-feature model: above + Fear&Greed, MACD
- Positions held until signal changes (not forced closed each cycle)
- `paper-status.py` — fetch live prices and show unrealized P&L on open positions at any time
- Goal: determine empirically whether ensemble complexity beats simpler baselines

### Dashboard
- Live market pulse: Fear & Greed gauge, BTC dominance, funding rate
- Testnet and paper trading performance cards
- Signal table with confidence, MTF score, and Kelly sizing
- Pipeline trigger with live log streaming (SSE)

### Infrastructure
- Oracle Cloud Ubuntu server (always-on, 1GB RAM)
- Cron job: 3 runs/day, auto-commits state to Git after each run
- Two-layer firewall: UFW + OCI Security List
- SSH key authentication, fail2ban, non-standard SSH port
- Models trained locally, deployed to server via SCP

---

## Project structure

```
crypto-pipeline/
├── crypto-pipeline.py       # raw data collection
├── feature-engineering.py   # multi-timeframe feature builder
├── train-model.py           # ensemble training + Optuna tuning
├── train-simple.py          # lightweight 3/5-feature models for comparison
├── predict-today.py         # live signal generator
├── testnet-trader.py        # Binance Testnet execution engine
├── compare-strategies.py    # 4-strategy parallel comparison + paper trading
├── paper-status.py          # live portfolio status (run anytime)
├── backtest.py              # walk-forward backtesting tool
├── server.py                # Flask web server + REST API
├── dashboard.html           # live dashboard UI
├── run-all.py               # run full pipeline locally in order
├── .env                     # API keys (never committed)
├── data/                    # generated CSVs and JSON state
└── models/                  # trained model pickles
```

---

## Setup

```bash
pip install requests pandas pandas-ta xgboost lightgbm scikit-learn optuna flask matplotlib yfinance python-dotenv
```

Create a `.env` file:
```
TESTNET_API_KEY=your_key_here
TESTNET_SECRET_KEY=your_secret_here
TELEGRAM_TOKEN=your_bot_token       # optional
TELEGRAM_CHAT_ID=your_chat_id      # optional
```

Get Testnet API keys at [testnet.binance.vision](https://testnet.binance.vision)

---

## Key metrics explained

| Metric | What it means |
|---|---|
| **Profit Factor** | Gross profit ÷ gross loss — >1.5 is meaningful, >2.0 is strong |
| **Win Rate** | % of closed trades that were profitable |
| **Sharpe Ratio** | Return per unit of risk (annualised) — >1.0 is the target |
| **Max Drawdown** | Largest peak-to-trough decline during the test period |
| **Confident accuracy** | Accuracy on days where model confidence ≥ per-coin threshold |

---

## Coins

BTC, ETH, SOL, AVAX, LINK — each coin has its own trained ensemble model, independent position management, and per-coin confidence threshold calibrated to its historical signal quality.
