"""
Train Simple Models — Two stripped-down XGBoost models for strategy comparison.
No Optuna, no cross-validation — fast training, deterministic results.

Feature sets:
  simple3: d_rsi14, funding_rate, d_vol_ratio
  simple5: d_rsi14, funding_rate, d_vol_ratio, fear_greed, d_macd_hist

Saves to models/{coin}_simple3.pkl and models/{coin}_simple5.pkl

Usage: python train-simple.py
"""

import pandas as pd
import numpy as np
import os
import pickle
import warnings
warnings.filterwarnings("ignore")

from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

INPUT_DIR  = "data"
OUTPUT_DIR = "models"
os.makedirs(OUTPUT_DIR, exist_ok=True)

COINS = ["BTC", "ETH", "SOL", "AVAX", "LINK"]

FEATURES_3 = ["d_rsi14", "funding_rate", "d_vol_ratio"]
FEATURES_5 = ["d_rsi14", "funding_rate", "d_vol_ratio", "fear_greed", "d_macd_hist"]


def train_simple(df: pd.DataFrame, features: list, label: str) -> dict | None:
    """Train a single XGBoost and return artifacts dict, or None on failure."""
    available = [f for f in features if f in df.columns]
    missing   = set(features) - set(available)
    if missing:
        print(f"      ⚠  Missing columns: {missing}")

    if len(available) == 0:
        print(f"      ✗  No usable features for {label}")
        return None

    df_clean = df.dropna(subset=available + ["target"]).reset_index(drop=True)
    if len(df_clean) < 100:
        print(f"      ✗  Too few clean rows ({len(df_clean)}) for {label}")
        return None

    split   = int(len(df_clean) * 0.70)
    X_train = df_clean[available].iloc[:split].values
    y_train = df_clean["target"].iloc[:split].values
    X_test  = df_clean[available].iloc[split:].values
    y_test  = df_clean["target"].iloc[split:].values

    sc      = StandardScaler()
    Xtr     = sc.fit_transform(X_train)
    Xte     = sc.transform(X_test)

    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42,
        verbosity=0,
    )
    model.fit(Xtr, y_train, eval_set=[(Xte, y_test)], verbose=False)

    acc   = accuracy_score(y_test, model.predict(Xte)) * 100
    probas = model.predict_proba(Xte)[:, 1]

    # Compute avg win / avg loss for Kelly sizing later (optional)
    closes  = df_clean["close"].iloc[split:].values if "close" in df_clean.columns else np.ones(len(y_test))
    mkt_ret = np.diff(closes) / closes[:-1] if len(closes) > 1 else np.zeros(len(probas) - 1)

    # Pad to match probas length
    if len(mkt_ret) < len(probas):
        mkt_ret = np.append(mkt_ret, 0.0)

    wins   = mkt_ret[(probas >= 0.55) & (mkt_ret > 0)]
    losses = mkt_ret[(probas >= 0.55) & (mkt_ret < 0)]
    avg_win  = float(np.mean(wins))   if len(wins) > 0   else 0.01
    avg_loss = float(abs(np.mean(losses))) if len(losses) > 0 else 0.01

    print(f"      {label}: acc={acc:.1f}%  n_train={split}  n_test={len(y_test)}  features={available}")

    return {
        "model":    model,
        "scaler":   sc,
        "features": available,
        "avg_win":  avg_win,
        "avg_loss": avg_loss,
        "accuracy": acc,
    }


def main():
    print(f"\n{'='*54}")
    print(f"  Train Simple Models (3-feature + 5-feature XGBoost)")
    print(f"  No Optuna — fast deterministic training")
    print(f"{'='*54}\n")

    for coin in COINS:
        feat_path = os.path.join(INPUT_DIR, f"{coin}_features.csv")
        if not os.path.exists(feat_path):
            print(f"  ✗  {feat_path} not found — run feature-engineering.py first\n")
            continue

        df = pd.read_csv(feat_path)
        print(f"  {coin}  ({len(df)} rows)")

        # ── Simple 3-feature model ──────────────────────────────────────────
        art3 = train_simple(df, FEATURES_3, "simple3")
        if art3:
            out3 = os.path.join(OUTPUT_DIR, f"{coin}_simple3.pkl")
            with open(out3, "wb") as f:
                pickle.dump(art3, f)
            print(f"      ✓  Saved → {out3}")

        # ── Simple 5-feature model ──────────────────────────────────────────
        art5 = train_simple(df, FEATURES_5, "simple5")
        if art5:
            out5 = os.path.join(OUTPUT_DIR, f"{coin}_simple5.pkl")
            with open(out5, "wb") as f:
                pickle.dump(art5, f)
            print(f"      ✓  Saved → {out5}")

        print()

    print(f"{'='*54}")
    print(f"  Done. Models saved to {OUTPUT_DIR}/")
    print(f"\n  Next steps:")
    print(f"  1. scp models/*_simple*.pkl ubuntu@server:~/crypto-pipeline/models/")
    print(f"  2. git push && ssh crypto 'cd ~/crypto-pipeline && git pull'")
    print(f"{'='*54}\n")


if __name__ == "__main__":
    main()