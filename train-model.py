"""
Model Training v5 — Optuna Hyperparameter Tuning + Profit Factor
Automatically finds the best hyperparameters for each model and coin.
Adds Profit Factor to the evaluation metrics.

Profit Factor = Total Gross Profit / Total Gross Loss
  > 2.0 = excellent
  1.5–2.0 = good
  1.0–1.5 = marginal
  < 1.0 = losing strategy

Requirements: pip install xgboost scikit-learn lightgbm pandas matplotlib optuna
Usage: python train-model.py
"""

import pandas as pd
import numpy as np
import os
import pickle
import warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression
from sklearn.feature_selection import SelectKBest, mutual_info_classif

INPUT_DIR  = "data"
OUTPUT_DIR = "models"
os.makedirs(OUTPUT_DIR, exist_ok=True)

COINS = ["BTC", "ETH", "SOL", "AVAX", "LINK"]

EXCLUDE_COLS = [
    "date", "open", "high", "low", "close", "volume",
    "coin", "target", "fg_label",
    "close_ma7", "close_ma30", "daily_return", "price_range",
]

N_FEATURES    = 25
N_FOLDS       = 5
MIN_TRAIN     = 200
CONF_THRESH   = 0.55
OPTUNA_TRIALS = 100  # trials per model — increase for better results (slower)


def get_feature_cols(df):
    return [c for c in df.columns
            if c not in EXCLUDE_COLS
            and df[c].dtype != object
            and not c.startswith("Unnamed")]


# ── Calibrated model wrapper ──────────────────────────────────────────────────

class _CalibratedModel:
    """Wraps a pre-fitted sklearn model with an IsotonicRegression calibrator.
    Exposes predict_proba() so it's a drop-in for the ensemble helpers."""
    def __init__(self, model, calibrator):
        self.model      = model
        self.calibrator = calibrator

    def predict_proba(self, X):
        raw = self.model.predict_proba(X)[:, 1]
        cal = self.calibrator.predict(raw)
        return np.column_stack([1 - cal, cal])


# ── Profit Factor ─────────────────────────────────────────────────────────────

def profit_factor(returns: np.ndarray) -> float:
    """
    Profit Factor = sum of winning returns / abs(sum of losing returns)
    Only counts periods where we had a position (signal=1).
    """
    wins   = returns[returns > 0].sum()
    losses = abs(returns[returns < 0].sum())
    if losses == 0:
        return float("inf") if wins > 0 else 1.0
    return round(wins / losses, 3)


def profit_factor_capped(returns: np.ndarray) -> float:
    """
    Profit factor for use as an Optuna objective — capped at 5.0 to prevent
    the optimiser chasing degenerate param sets that produce only a handful of wins.
    """
    active = returns[returns != 0]
    if len(active) == 0:
        return 1.0
    wins   = active[active > 0].sum()
    losses = abs(active[active < 0].sum())
    if losses == 0:
        return 5.0 if wins > 0 else 1.0
    return min(wins / losses, 5.0)


def kelly_fraction(p, avg_win, avg_loss, max_f=0.25):
    if avg_loss == 0:
        return 0.0
    b = avg_win / avg_loss
    f = (p * b - (1 - p)) / b
    return min(max(0.0, f / 2), max_f)


def ensemble_proba(models, scalers, X):
    probas = []
    for name, model in models.items():
        Xs = scalers[name].transform(X)
        probas.append(model.predict_proba(Xs)[:, 1])
    return np.mean(probas, axis=0)


def select_features(df, feature_cols):
    X = df[feature_cols].values
    y = df["target"].values
    selector = SelectKBest(mutual_info_classif, k=min(N_FEATURES, len(feature_cols)))
    selector.fit(X, y)
    scores = dict(zip(feature_cols, selector.scores_))
    for col in feature_cols:
        if col.startswith("mtf_"):
            scores[col] *= 1.2
    top = sorted(scores, key=lambda c: -scores[c])[:N_FEATURES]
    return top, scores


# ── Optuna objective functions ────────────────────────────────────────────────

def objective_xgb(trial, X_tr, y_tr, X_val, y_val, mkt_val):
    params = {
        "n_estimators":      trial.suggest_int("n_estimators", 50, 400),
        "max_depth":         trial.suggest_int("max_depth", 2, 6),
        "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "subsample":         trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight":  trial.suggest_int("min_child_weight", 1, 20),
        "reg_alpha":         trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda":        trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        "eval_metric": "logloss", "random_state": 42, "verbosity": 0,
    }
    sc  = StandardScaler()
    Xtr = sc.fit_transform(X_tr)
    Xvl = sc.transform(X_val)
    m   = XGBClassifier(**params)
    m.fit(Xtr, y_tr, eval_set=[(Xvl, y_val)], verbose=False)
    y_pred    = (m.predict_proba(Xvl)[:, 1] >= 0.5).astype(int)
    pos_rets  = np.where(y_pred == 1, mkt_val[:len(y_pred)], 0)
    return profit_factor_capped(pos_rets)


def objective_lgbm(trial, X_tr, y_tr, X_val, y_val, mkt_val):
    params = {
        "n_estimators":     trial.suggest_int("n_estimators", 50, 400),
        "max_depth":        trial.suggest_int("max_depth", 2, 6),
        "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
        "num_leaves":       trial.suggest_int("num_leaves", 10, 60),
        "min_child_samples":trial.suggest_int("min_child_samples", 5, 50),
        "reg_alpha":        trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda":       trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        "random_state": 42, "verbose": -1,
    }
    sc  = StandardScaler()
    Xtr = sc.fit_transform(X_tr)
    Xvl = sc.transform(X_val)
    m   = LGBMClassifier(**params)
    m.fit(Xtr, y_tr)
    y_pred   = (m.predict_proba(Xvl)[:, 1] >= 0.5).astype(int)
    pos_rets = np.where(y_pred == 1, mkt_val[:len(y_pred)], 0)
    return profit_factor_capped(pos_rets)


def objective_rf(trial, X_tr, y_tr, X_val, y_val, mkt_val):
    params = {
        "n_estimators":    trial.suggest_int("n_estimators", 100, 500),
        "max_depth":       trial.suggest_int("max_depth", 3, 10),
        "min_samples_leaf":trial.suggest_int("min_samples_leaf", 2, 30),
        "max_features":    trial.suggest_categorical("max_features", ["sqrt", "log2"]),
        "random_state": 42, "n_jobs": -1,
    }
    sc  = StandardScaler()
    Xtr = sc.fit_transform(X_tr)
    Xvl = sc.transform(X_val)
    m   = RandomForestClassifier(**params)
    m.fit(Xtr, y_tr)
    y_pred   = (m.predict_proba(Xvl)[:, 1] >= 0.5).astype(int)
    pos_rets = np.where(y_pred == 1, mkt_val[:len(y_pred)], 0)
    return profit_factor_capped(pos_rets)


def tune_model(name, objective_fn, X_tr, y_tr, X_val, y_val, mkt_val, n_trials):
    """Run Optuna tuning for one model type. Objective: Profit Factor (capped at 5)."""
    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda trial: objective_fn(trial, X_tr, y_tr, X_val, y_val, mkt_val),
        n_trials=n_trials,
        show_progress_bar=False,
    )
    return study.best_params, study.best_value


def build_tuned_model(name, best_params):
    """Instantiate a model with the best found params."""
    if name == "XGBoost":
        return XGBClassifier(**best_params, eval_metric="logloss",
                             random_state=42, verbosity=0)
    elif name == "LightGBM":
        return LGBMClassifier(**best_params, random_state=42, verbose=-1)
    else:
        return RandomForestClassifier(**best_params, random_state=42, n_jobs=-1)


# ── Walk-forward CV ───────────────────────────────────────────────────────────

def walk_forward_cv(df, feature_cols):
    n         = len(df)
    fold_size = n // (N_FOLDS + 1)
    accs, conf_accs, pfs = [], [], []

    print(f"\n  Walk-forward CV ({N_FOLDS} folds):")

    for fold in range(N_FOLDS):
        train_end  = fold_size * (fold + 1)
        test_start = train_end
        test_end   = min(test_start + fold_size, n)

        if train_end < MIN_TRAIN:
            continue

        X_tr = df[feature_cols].iloc[:train_end].values
        y_tr = df["target"].iloc[:train_end].values
        X_te = df[feature_cols].iloc[test_start:test_end].values
        y_te = df["target"].iloc[test_start:test_end].values
        mkt  = df["close"].pct_change().iloc[test_start:test_end].values

        # Use fixed params for CV (fast) — Optuna runs on final model only
        models  = {
            "XGBoost":     XGBClassifier(n_estimators=150, max_depth=4,
                               learning_rate=0.04, subsample=0.75,
                               colsample_bytree=0.75, eval_metric="logloss",
                               random_state=42, verbosity=0),
            "RandomForest":RandomForestClassifier(n_estimators=200, max_depth=5,
                               min_samples_leaf=8, random_state=42, n_jobs=-1),
            "LightGBM":    LGBMClassifier(n_estimators=150, max_depth=4,
                               learning_rate=0.04, subsample=0.75,
                               random_state=42, verbose=-1),
        }
        scalers = {}

        for name, model in models.items():
            sc = StandardScaler()
            Xtr_s = sc.fit_transform(X_tr)
            scalers[name] = sc
            if name == "XGBoost":
                Xte_s = sc.transform(X_te)
                model.fit(Xtr_s, y_tr, eval_set=[(Xte_s, y_te)], verbose=False)
            else:
                model.fit(Xtr_s, y_tr)

        avg_p  = ensemble_proba(models, scalers, X_te)
        y_pred = (avg_p >= 0.5).astype(int)
        acc    = accuracy_score(y_te, y_pred)

        conf_mask = (avg_p >= CONF_THRESH) | (avg_p <= (1 - CONF_THRESH))
        c_acc     = accuracy_score(y_te[conf_mask], y_pred[conf_mask]) if conf_mask.any() else 0

        # Profit Factor for this fold
        position_rets = np.where(y_pred == 1, mkt[:len(y_pred)], 0)
        pf = profit_factor(position_rets)

        accs.append(acc)
        conf_accs.append(c_acc)
        pfs.append(pf if pf != float("inf") else 3.0)

        print(f"    Fold {fold+1}: acc={acc*100:.1f}%  "
              f"conf_acc={c_acc*100:.1f}%  PF={pf:.2f}")

    mean_acc  = np.mean(accs)
    mean_cacc = np.mean(conf_accs)
    mean_pf   = np.mean(pfs)
    std_acc   = np.std(accs)
    print(f"\n  CV Accuracy:           {mean_acc*100:.1f}% ± {std_acc*100:.1f}%")
    print(f"  CV Confident Accuracy: {mean_cacc*100:.1f}%")
    print(f"  CV Profit Factor:      {mean_pf:.2f}")
    return mean_acc, std_acc, mean_cacc, mean_pf


# ── Optuna tuning + final model ───────────────────────────────────────────────

def train_final_with_optuna(df, feature_cols):
    sel_feats, scores = select_features(df, feature_cols)

    print(f"\n  Top {len(sel_feats)} features selected:")
    for f in sel_feats:
        prefix = ("MTF" if f.startswith("mtf_") else
                  "4H"  if f.startswith("h4_")  else
                  "1H"  if f.startswith("h1_")  else
                  "EXT" if any(f.startswith(p) for p in ["fear","fg_","fund","btc_dom"]) else "1D")
        print(f"    [{prefix}] {f:<35} {scores[f]:.4f}")

    split   = int(len(df) * 0.80)
    val_cut = int(split * 0.85)   # last 15% of train = validation for Optuna

    X_all   = df[sel_feats].values
    y_all   = df["target"].values

    X_tr_opt = X_all[:val_cut]
    y_tr_opt = y_all[:val_cut]
    X_val    = X_all[val_cut:split]
    y_val    = y_all[val_cut:split]

    X_train  = X_all[:split]
    y_train  = y_all[:split]
    X_test   = X_all[split:]
    y_test   = y_all[split:]

    print(f"\n  Running Optuna ({OPTUNA_TRIALS} trials × 3 models, objective: Profit Factor)...")

    # Market returns for the Optuna validation window (needed by PF objective)
    mkt_val = df["close"].pct_change().iloc[val_cut:split].values

    best_params = {}
    for name, obj_fn in [
        ("XGBoost",      objective_xgb),
        ("LightGBM",     objective_lgbm),
        ("RandomForest", objective_rf),
    ]:
        params, val_pf = tune_model(name, obj_fn, X_tr_opt, y_tr_opt,
                                    X_val, y_val, mkt_val, OPTUNA_TRIALS)
        best_params[name] = params
        print(f"    {name:<14} best val PF: {val_pf:.2f}  "
              f"params: {params}")

    # Train final ensemble with tuned params
    models  = {}
    scalers = {}

    for name, params in best_params.items():
        sc  = StandardScaler()
        Xtr = sc.fit_transform(X_train)
        scalers[name] = sc
        model = build_tuned_model(name, params)
        if name == "XGBoost":
            Xte = sc.transform(X_test)
            model.fit(Xtr, y_train, eval_set=[(Xte, y_test)], verbose=False)
        else:
            model.fit(Xtr, y_train)
        # Calibrate probabilities on the Optuna validation set (never used for fitting)
        # so that prob_up=0.62 actually means ~62% historical win rate → better Kelly sizing
        raw_val = model.predict_proba(sc.transform(X_val))[:, 1]
        ir = IsotonicRegression(out_of_bounds="clip")
        ir.fit(raw_val, y_val)
        models[name] = _CalibratedModel(model, ir)

    avg_p  = ensemble_proba(models, scalers, X_test)
    y_pred = (avg_p >= 0.5).astype(int)
    acc    = accuracy_score(y_test, y_pred)

    conf_mask = (avg_p >= CONF_THRESH) | (avg_p <= (1 - CONF_THRESH))
    conf_acc  = accuracy_score(y_test[conf_mask], y_pred[conf_mask]) if conf_mask.any() else 0
    conf_pct  = conf_mask.mean() * 100

    # Backtest with Kelly + Profit Factor
    test_df = df.iloc[split:].copy().reset_index(drop=True)
    test_df["prob_up"]  = avg_p
    test_df["signal"]   = y_pred
    test_df["mkt_ret"]  = test_df["close"].pct_change().fillna(0)

    wins   = test_df.loc[(test_df["signal"]==1) & (test_df["mkt_ret"]>0), "mkt_ret"]
    losses = test_df.loc[(test_df["signal"]==1) & (test_df["mkt_ret"]<0), "mkt_ret"].abs()
    avg_win  = wins.mean()   if len(wins)   > 0 else 0.01
    avg_loss = losses.mean() if len(losses) > 0 else 0.01

    test_df["kelly"] = test_df["prob_up"].apply(
        lambda p: kelly_fraction(p, avg_win, avg_loss) if p >= CONF_THRESH else 0.0
    )
    test_df["strat_ret"] = test_df["kelly"].shift(1).fillna(0) * test_df["mkt_ret"]

    cum_mkt  = (1 + test_df["mkt_ret"]).cumprod().iloc[-1] - 1
    cum_str  = (1 + test_df["strat_ret"]).cumprod().iloc[-1] - 1

    # Profit Factor on strategy returns
    strat_rets = test_df["strat_ret"].values
    pf_strategy = profit_factor(strat_rets[strat_rets != 0])

    # Profit Factor on raw signal returns (unweighted)
    position_rets = np.where(test_df["signal"].shift(1).fillna(0)==1,
                             test_df["mkt_ret"].values, 0)
    pf_signal = profit_factor(position_rets[position_rets != 0])

    # Max drawdown
    eq   = (1 + test_df["strat_ret"]).cumprod()
    peak = eq.expanding().max()
    max_dd = ((eq - peak) / peak).min() * 100

    # Win rate on trades
    trades     = test_df[test_df["signal"].shift(1).fillna(0) == 1]
    win_rate   = (trades["mkt_ret"] > 0).mean() * 100 if len(trades) > 0 else 0
    n_trades   = len(trades)

    return {
        "models":      models,
        "scalers":     scalers,
        "sel_feats":   sel_feats,
        "best_params": best_params,
        "accuracy":    acc,
        "conf_acc":    conf_acc,
        "conf_pct":    conf_pct,
        "cum_mkt":     cum_mkt,
        "cum_str":     cum_str,
        "avg_win":     avg_win,
        "avg_loss":    avg_loss,
        "pf_strategy": pf_strategy,
        "pf_signal":   pf_signal,
        "max_dd":      max_dd,
        "win_rate":    win_rate,
        "n_trades":    n_trades,
    }


# ── Plot ──────────────────────────────────────────────────────────────────────

def plot_results(result, ticker, df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"{ticker} — Tuned Ensemble Results", fontsize=13)

    # Feature importance
    ax1 = axes[0]
    xgb   = result["models"]["XGBoost"]
    # Unwrap calibration wrapper to get the raw XGBoost model
    xgb_raw = xgb.model if isinstance(xgb, _CalibratedModel) else xgb
    feats = result["sel_feats"]
    imp   = pd.Series(xgb_raw.feature_importances_, index=feats).sort_values()
    colors = []
    for f in imp.index:
        if f.startswith("mtf_"):   colors.append("#f59e0b")
        elif f.startswith("h4_"):  colors.append("#8b5cf6")
        elif f.startswith("h1_"):  colors.append("#06b6d4")
        elif any(f.startswith(p) for p in ["fear","fg_","fund","btc_"]): colors.append("#10b981")
        else:                      colors.append("#627eea")
    imp.plot(kind="barh", ax=ax1, color=colors)
    ax1.set_title("Feature Importance")
    ax1.set_xlabel("XGBoost importance")
    from matplotlib.patches import Patch
    ax1.legend(handles=[
        Patch(color="#f59e0b", label="MTF"),
        Patch(color="#8b5cf6", label="4H"),
        Patch(color="#06b6d4", label="1H"),
        Patch(color="#627eea", label="1D"),
        Patch(color="#10b981", label="External"),
    ], fontsize=7, loc="lower right")

    # Equity curve
    ax2 = axes[1]
    split   = int(len(df) * 0.80)
    test_df = df.iloc[split:].copy().reset_index(drop=True)
    sel     = result["sel_feats"]
    X_te    = test_df[sel].values
    avg_p   = ensemble_proba(result["models"], result["scalers"], X_te)

    test_df["prob_up"]  = avg_p
    test_df["mkt_ret"]  = test_df["close"].pct_change().fillna(0)
    test_df["signal"]   = (avg_p >= CONF_THRESH).astype(int)
    test_df["kelly"]    = test_df["prob_up"].apply(
        lambda p: kelly_fraction(p, result["avg_win"], result["avg_loss"])
        if p >= CONF_THRESH else 0.0
    )
    test_df["strat_ret"] = test_df["kelly"].shift(1).fillna(0) * test_df["mkt_ret"]

    eq_strat = (1 + test_df["strat_ret"]).cumprod() * 1000
    eq_bnh   = (1 + test_df["mkt_ret"]).cumprod()   * 1000

    x = range(len(test_df))
    ax2.plot(x, eq_strat, color="#10b981", linewidth=1.5, label="Kelly Strategy")
    ax2.plot(x, eq_bnh,   color="#64748b", linewidth=1,   label="Buy & Hold", linestyle="--")
    ax2.fill_between(x, eq_strat, 1000, alpha=0.08, color="#10b981")
    ax2.axhline(1000, color="#475569", linewidth=0.5, linestyle=":")
    ax2.set_title("Equity Curve (test period, $1000 start)")
    ax2.set_ylabel("Portfolio value ($)")
    ax2.legend(fontsize=8)

    # Add PF annotation
    ax2.text(0.02, 0.05,
             f"PF: {result['pf_strategy']:.2f}  "
             f"WR: {result['win_rate']:.1f}%  "
             f"MaxDD: {result['max_dd']:.1f}%",
             transform=ax2.transAxes, fontsize=8, color="#94a3b8")

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, f"{ticker}_results.png")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Plot → {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"\n{'='*56}")
    print(f"  Model Training v5 — Optuna + Profit Factor")
    print(f"  Models: XGBoost + RandomForest + LightGBM")
    print(f"  Optuna trials: {OPTUNA_TRIALS} per model")
    print(f"  Confidence threshold: {CONF_THRESH*100:.0f}%")
    print(f"{'='*56}")

    summary = []

    for ticker in COINS:
        path = os.path.join(INPUT_DIR, f"{ticker}_features.csv")
        if not os.path.exists(path):
            print(f"\n  ✗  {path} not found — run feature-engineering.py first")
            continue

        df           = pd.read_csv(path, parse_dates=["date"])
        feature_cols = get_feature_cols(df)
        df           = df.dropna(subset=["close","target"]).reset_index(drop=True)

        for col in feature_cols:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].median())

        feature_cols = [c for c in feature_cols if c in df.columns]

        print(f"\n  {'━'*50}")
        print(f"  {ticker}  ({len(df)} rows, {len(feature_cols)} features)")

        cv_acc, cv_std, cv_cacc, cv_pf = walk_forward_cv(df, feature_cols)
        result = train_final_with_optuna(df, feature_cols)

        print(f"\n  ── Final Model Results (last 20% of data) ──")
        print(f"  All-signal accuracy:   {result['accuracy']*100:.1f}%")
        print(f"  Confident-only acc:    {result['conf_acc']*100:.1f}%  "
              f"({result['conf_pct']:.0f}% of days signal)")
        print(f"  Buy-and-hold:          {result['cum_mkt']*100:+.1f}%")
        print(f"  Kelly strategy:        {result['cum_str']*100:+.1f}%")
        print(f"  ── Risk Metrics ────────────────────────────")
        print(f"  Profit Factor:         {result['pf_strategy']:.2f}  "
              f"(signal: {result['pf_signal']:.2f})")
        print(f"  Win Rate:              {result['win_rate']:.1f}%  "
              f"({result['n_trades']} trades)")
        print(f"  Max Drawdown:          {result['max_dd']:.1f}%")
        print(f"  Avg Win / Avg Loss:    {result['avg_win']*100:.2f}% / "
              f"{result['avg_loss']*100:.2f}%")

        plot_results(result, ticker, df)

        artifacts = {
            "models":      result["models"],
            "scalers":     result["scalers"],
            "features":    result["sel_feats"],
            "avg_win":     result["avg_win"],
            "avg_loss":    result["avg_loss"],
            "best_params": result["best_params"],
        }
        with open(os.path.join(OUTPUT_DIR, f"{ticker}_ensemble.pkl"), "wb") as f:
            pickle.dump(artifacts, f)
        print(f"  Saved → models/{ticker}_ensemble.pkl")

        summary.append({
            "ticker":   ticker,
            "cv_acc":   cv_acc,
            "cv_std":   cv_std,
            "cv_cacc":  cv_cacc,
            "cv_pf":    cv_pf,
            "conf_acc": result["conf_acc"],
            "conf_pct": result["conf_pct"],
            "mkt":      result["cum_mkt"],
            "strategy": result["cum_str"],
            "pf":       result["pf_strategy"],
            "win_rate": result["win_rate"],
            "max_dd":   result["max_dd"],
        })

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f"\n{'='*56}")
    print(f"  Final Summary")
    print(f"{'='*56}")
    print(f"  {'Coin':<6} {'CV Acc':>8} {'±':>5} {'Conf Acc':>9} "
          f"{'Signals':>8} {'B&H':>8} {'Kelly':>8} {'PF':>6} {'WR':>6} {'MaxDD':>7}")
    print(f"  {'─'*72}")
    for r in summary:
        pf_str = f"{r['pf']:.2f}" if r['pf'] != float("inf") else "∞"
        print(f"  {r['ticker']:<6} "
              f"{r['cv_acc']*100:>7.1f}% "
              f"{r['cv_std']*100:>4.1f}% "
              f"{r['conf_acc']*100:>8.1f}% "
              f"{r['conf_pct']:>7.0f}% "
              f"{r['mkt']*100:>+7.1f}% "
              f"{r['strategy']*100:>+7.1f}% "
              f"{pf_str:>6} "
              f"{r['win_rate']:>5.1f}% "
              f"{r['max_dd']:>6.1f}%")

    print(f"\n  PF = Profit Factor (gross profit / gross loss)")
    print(f"       > 2.0 excellent  |  1.5-2.0 good  |  < 1.5 marginal")
    print(f"  WR = Win Rate on trades where model issued a signal")
    print(f"\n  Next: python predict-today.py")
    print(f"{'='*56}\n")


if __name__ == "__main__":
    main()