# Crypto ML Trading Pipeline

An end-to-end automated trading system that collects crypto market data, engineers multi-timeframe features, trains an ML ensemble, and generates daily BUY/STAY OUT signals — executed on Binance Testnet with stop-loss and take-profit orders.

---

## How it works

```
crypto-pipeline.py        → fetch 3 years of OHLCV data (BTC, ETH, SOL)
feature-engineering.py    → build 80+ features across 1H + 4H + 1D timeframes
train-model.py            → train XGBoost + LightGBM + RandomForest ensemble
predict-today.py          → generate today's signal + Kelly position size
paper-trader.py           → simulate trades (no real money)
testnet-trader.py         → execute real orders on Binance Testnet
dashboard.py / server.py  → serve live dashboard in the browser
```

Run everything in order with:
```
python run-all.py
```

Or launch the dashboard and trigger runs from the browser:
```
python server.py
```
Then open `http://localhost:8080`

---

## Features

### Data
- 3 years of daily OHLCV candles from Binance public API (no key required)
- Fear & Greed Index — crowd sentiment signal
- BTC perpetual futures funding rates — leverage/sentiment signal
- BTC dominance — market rotation signal

### Feature Engineering (multi-timeframe)
- **1D** — macro trend: EMA crossovers, RSI, MACD, Bollinger Bands, ATR, OBV
- **4H** — medium momentum: same indicators aggregated to daily
- **1H** — entry timing: intraday momentum and volume spikes
- **MTF confluence** — signals that agree across all 3 timeframes
- **Market regime** — golden/death cross (50d vs 200d EMA)

### ML Model
- Ensemble of 3 models: XGBoost, LightGBM, RandomForest
- Hyperparameter tuning via **Optuna** (100 trials, objective: Profit Factor)
- **Walk-forward cross-validation** — test always in the future, never leaked
- **Top 20 features** selected by mutual information
- **Isotonic probability calibration** — probabilities reflect real win rates
- **Target**: 3-day forward return > 0.5% (filters noise vs next-day direction)

### Trading
- Signals at 55% confidence threshold — below that = no action
- **Half-Kelly position sizing** — mathematically optimal bet fraction, capped at 25%
- **OCO orders** (One-Cancels-Other): take-profit +8%, stop-loss -4%
- Paper trading for signal validation before real capital
- Binance Testnet execution with HMAC-signed API requests

### Dashboard
- Live market pulse: Fear & Greed gauge, BTC dominance, funding rate
- Paper trading and testnet performance cards
- Daily signals table with confidence and Kelly sizing
- Pipeline trigger with live log streaming (SSE)

---

## Project structure

```
crypto-pipeline/
├── crypto-pipeline.py       # raw data collection
├── feature-engineering.py   # multi-timeframe feature builder
├── train-model.py           # ensemble training + Optuna tuning
├── predict-today.py         # live signal generator
├── paper-trader.py          # paper trading engine
├── testnet-trader.py        # Binance Testnet autotrader
├── run-all.py               # runs the full pipeline in order
├── server.py                # Flask web server + API
├── dashboard.html           # live dashboard UI
├── report.py                # HTML performance report generator
├── backtest.py              # standalone backtesting tool
├── .env                     # API keys (never committed)
└── data/                    # generated CSVs and JSON state (not committed)
    models/                  # trained model pickles (not committed)
```

---

## Setup

```bash
pip install requests pandas pandas-ta xgboost lightgbm scikit-learn optuna flask matplotlib
```

Create a `.env` file in the project root:
```
TESTNET_API_KEY=your_key_here
TESTNET_SECRET_KEY=your_secret_here
```

Get Testnet API keys at [testnet.binance.vision](https://testnet.binance.vision)

---

## Key metrics

| Metric | What it means |
|---|---|
| **Profit Factor** | Gross profit ÷ gross loss. >2.0 = excellent |
| **Win Rate** | % of trades that were profitable |
| **Max Drawdown** | Largest peak-to-trough portfolio decline |
| **Confident accuracy** | Accuracy on days where model confidence ≥ 55% |

---

## Coins

BTC, ETH, SOL — each gets its own trained model and independent position management.
