#!/usr/bin/env python3
"""Fit a few sklearn models on top of the cheap PySCF descriptors and see if
they can predict the expensive GW/BSE numbers. Compared against a mean
baseline and against just trusting the cheap CIS S1 energy with no
correction, for comparison.

Run with your chem_ml environment:
  chem_ml_env/bin/python train_ml.py --target Singlet_binding
"""
from __future__ import annotations

import argparse
import json
import os

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import config as cfg

PYSCF_FEATURES = [
    "homo_lumo_gap_eV",
    "cis_s1_eV",
    "cis_s1_osc_strength",
    "dipole_debye",
    "n_atoms",
]
FEATURES = PYSCF_FEATURES + cfg.CHEAP_TABULATED_FEATURES


def evaluate(name, model, X, y, cv, results, use_scaler=True):
    pipe = make_pipeline(StandardScaler(), model) if use_scaler else model
    pred = cross_val_predict(pipe, X, y, cv=cv)
    rmse = float(np.sqrt(mean_squared_error(y, pred)))
    mae = float(mean_absolute_error(y, pred))
    r2 = float(r2_score(y, pred))
    results[name] = {"rmse": rmse, "mae": mae, "r2": r2, "pred": pred}
    print(f"{name:24s}  RMSE={rmse:.4f}  MAE={mae:.4f}  R2={r2:.3f}")
    return pipe


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--target", default=cfg.DEFAULT_TARGET, choices=cfg.TARGET_COLUMNS)
    args = p.parse_args()
    target = args.target

    df = pd.read_csv(cfg.RESULTS_DIR / "dataset.csv")
    df = df.dropna(subset=[target] + FEATURES).reset_index(drop=True)
    print(f"Dataset: {len(df)} PAH101 crystals with target={target}")

    X = df[FEATURES].to_numpy(dtype=float)
    y = df[target].to_numpy(dtype=float)
    cv = KFold(n_splits=min(cfg.N_CV_FOLDS, len(df)), shuffle=True, random_state=cfg.RANDOM_SEED)

    results: dict = {}
    evaluate("mean_baseline", DummyRegressor(strategy="mean"), X, y, cv, results, use_scaler=False)

    naive_pred = df["cis_s1_eV"].to_numpy(dtype=float)
    results["naive_cheap_S1_only"] = {
        "rmse": float(np.sqrt(mean_squared_error(y, naive_pred))),
        "mae": float(mean_absolute_error(y, naive_pred)),
        "r2": float(r2_score(y, naive_pred)),
        "pred": naive_pred,
    }
    r = results["naive_cheap_S1_only"]
    print(f"{'naive_cheap_S1_only':24s}  RMSE={r['rmse']:.4f}  MAE={r['mae']:.4f}  R2={r['r2']:.3f}")

    X_pyscf_only = df[PYSCF_FEATURES].to_numpy(dtype=float)
    evaluate(
        "ridge_pyscf_only",
        Ridge(alpha=1.0, random_state=cfg.RANDOM_SEED),
        X_pyscf_only, y, cv, results,
    )

    evaluate("ridge", Ridge(alpha=1.0, random_state=cfg.RANDOM_SEED), X, y, cv, results)
    evaluate(
        "random_forest",
        RandomForestRegressor(n_estimators=300, max_depth=6, random_state=cfg.RANDOM_SEED),
        X, y, cv, results, use_scaler=False,
    )
    evaluate(
        "gradient_boosting",
        GradientBoostingRegressor(n_estimators=200, max_depth=2, learning_rate=0.05, random_state=cfg.RANDOM_SEED),
        X, y, cv, results, use_scaler=False,
    )

    best_name = min(
        (k for k in results if k != "mean_baseline"), key=lambda k: results[k]["rmse"]
    )
    print(f"\nBest model: {best_name} (RMSE={results[best_name]['rmse']:.4f} eV)")

    rf_full = RandomForestRegressor(n_estimators=300, max_depth=6, random_state=cfg.RANDOM_SEED)
    rf_full.fit(X, y)
    importances = sorted(
        zip(FEATURES, rf_full.feature_importances_), key=lambda t: -t[1]
    )

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    ax = axes[0]
    order = [
        "mean_baseline", "naive_cheap_S1_only", "ridge_pyscf_only",
        "ridge", "random_forest", "gradient_boosting",
    ]
    rmses = [results[k]["rmse"] for k in order]
    colors = ["#adb5bd", "#e07a5f", "#f2cc8f", "#81b29a", "#3d5a80", "#293241"]
    ax.bar(order, rmses, color=colors)
    ax.set_ylabel(f"CV RMSE, {target} (eV)")
    ax.set_title("Model comparison")
    ax.tick_params(axis="x", rotation=35)
    for lbl in ax.get_xticklabels():
        lbl.set_ha("right")

    ax = axes[1]
    pred_best = results[best_name]["pred"]
    lo, hi = min(y.min(), pred_best.min()), max(y.max(), pred_best.max())
    pad = 0.05 * (hi - lo)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", lw=1, alpha=0.6)
    ax.scatter(y, pred_best, s=28, c="#3d5a80", alpha=0.8, edgecolors="none")
    ax.set_xlabel(f"GW/BSE {target} (eV), true")
    ax.set_ylabel("predicted (5-fold CV)")
    ax.set_title(f"Best model: {best_name}\nR$^2$={results[best_name]['r2']:.2f}")

    ax = axes[2]
    labels, vals = zip(*importances)
    ax.barh(labels[::-1], np.array(vals[::-1]), color="#3d5a80")
    ax.set_xlabel("Random-forest feature importance")
    ax.set_title(f"What predicts {target}?")

    fig.tight_layout()
    fig_path = cfg.RESULTS_DIR / f"fig_ml_{target}.png"
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {fig_path}")

    summary = {
        "target": target,
        "n_samples": len(df),
        "n_cv_folds": cv.get_n_splits(),
        "features": FEATURES,
        "metrics": {k: {kk: vv for kk, vv in v.items() if kk != "pred"} for k, v in results.items()},
        "best_model": best_name,
        "feature_importance_random_forest": [{"feature": f, "importance": float(v)} for f, v in importances],
    }
    out_json = cfg.RESULTS_DIR / f"ml_summary_{target}.json"
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"Wrote {out_json}")


if __name__ == "__main__":
    main()
