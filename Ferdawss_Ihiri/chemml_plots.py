#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import pandas as pd

import config as cfg

MODEL_COLORS = {
    "Ridge": "#3d5a80",
    "Lasso": "#81b29a",
    "ElasticNet": "#f2cc8f",
    "SVR": "#e07a5f",
    "DecisionTreeRegressor": "#98c1d9",
    "RandomForestRegressor": "#293241",
    "GradientBoostingRegressor": "#ee6c4d",
    "MLPRegressor": "#c1666b",
}


def short_name(model: str) -> str:
    return model.replace("Regressor", "")


def plot_target(csv_path: Path, target: str, out_path: Path) -> None:
    df = pd.read_csv(csv_path)
    df = df.sort_values("RMSE").reset_index(drop=True)

    fig, (ax_rmse, ax_r2) = plt.subplots(1, 2, figsize=(10, 4))
    colors = [MODEL_COLORS.get(m, "#888888") for m in df["Model"]]
    labels = [short_name(m) for m in df["Model"]]

    ax_rmse.barh(labels[::-1], df["RMSE"][::-1], color=colors[::-1])
    ax_rmse.set_xlabel(f"held-out RMSE, {target} (eV)")
    ax_rmse.set_title("ChemML GA model screen")

    ax_r2.barh(labels[::-1], df["r_squared"][::-1], color=colors[::-1])
    ax_r2.axvline(0, color="k", lw=0.8)
    ax_r2.set_xlabel("R$^2$")
    ax_r2.set_title(f"best: {short_name(df.iloc[0]['Model'])}")

    fig.suptitle(f"{target}: 8 sklearn regressors, GA-tuned by ChemML", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_overview(chemml_dir: Path, targets: list[str], out_path: Path) -> None:
    """One panel per target, RMSE bars, so you can eyeball all three at once."""
    fig, axes = plt.subplots(1, len(targets), figsize=(4.2 * len(targets), 4.2), sharey=False)
    if len(targets) == 1:
        axes = [axes]

    for ax, target in zip(axes, targets):
        csv_path = chemml_dir / f"{target}_best_models.csv"
        if not csv_path.is_file():
            ax.set_visible(False)
            continue
        df = pd.read_csv(csv_path).sort_values("RMSE").reset_index(drop=True)
        colors = [MODEL_COLORS.get(m, "#888888") for m in df["Model"]]
        labels = [short_name(m) for m in df["Model"]]
        ax.barh(labels[::-1], df["RMSE"][::-1], color=colors[::-1])
        ax.set_xlabel("RMSE (eV)")
        ax.set_title(target)

    fig.suptitle("ChemML AutoML screen, all three targets", y=1.03)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--chemml-dir", default=str(cfg.RESULTS_DIR / "chemml_rerun"))
    args = p.parse_args()

    chemml_dir = Path(args.chemml_dir)
    targets = cfg.TARGET_COLUMNS

    for target in targets:
        csv_path = chemml_dir / f"{target}_best_models.csv"
        if not csv_path.is_file():
            print(f"skipping {target}, no {csv_path.name}")
            continue
        plot_target(csv_path, target, chemml_dir / f"fig_chemml_{target}.png")

    plot_overview(chemml_dir, targets, chemml_dir / "fig_chemml_overview.png")


if __name__ == "__main__":
    main()
