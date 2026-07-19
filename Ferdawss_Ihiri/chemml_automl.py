#!/usr/bin/env python3
from __future__ import annotations

import argparse
import multiprocessing
import os
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

multiprocessing.cpu_count = lambda: 1
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")

import pandas as pd

import config as cfg


def _resolve_dataset(path: str | None):
    if path:
        return path
    for candidate in (
        cfg.RESULTS_DIR / "dataset.csv",
        cfg.RESULTS_DIR / "data" / "dataset.csv",
    ):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"dataset.csv not found under {cfg.RESULTS_DIR}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--target", default=cfg.DEFAULT_TARGET, choices=cfg.TARGET_COLUMNS)
    p.add_argument("--n-gen", type=int, default=2, help="GA generations per model")
    p.add_argument("--n-best", type=int, default=8)
    p.add_argument("--dataset", default=None)
    p.add_argument("--out-dir", default=None)
    args = p.parse_args()
    target = args.target

    from chemml.autoML import ModelScreener

    dataset_path = _resolve_dataset(args.dataset)
    df = pd.read_csv(dataset_path)
    feature_cols = [
        "homo_lumo_gap_eV", "cis_s1_eV", "cis_s1_osc_strength", "dipole_debye", "n_atoms",
    ] + cfg.CHEAP_TABULATED_FEATURES
    keep = feature_cols + [target]
    df = df[keep].dropna().reset_index(drop=True)
    print(f"ChemML ModelScreener on {len(df)} PAH101 crystals, target={target}")
    print(f"Dataset: {dataset_path}")
    print(f"Features: {feature_cols}")

    root_out = Path(args.out_dir) if args.out_dir else cfg.RESULTS_DIR / "chemml"
    out_dir = root_out / target
    out_dir.mkdir(parents=True, exist_ok=True)

    cwd = os.getcwd()
    os.chdir(out_dir)
    try:
        screener = ModelScreener(
            df,
            target=target,
            featurization=False,
            screener_type="regressor",
            n_gen=args.n_gen,
            output_file="ga_progress.txt",
        )
        best = screener.screen_models(n_best=args.n_best)
    finally:
        os.chdir(cwd)

    best_out = best.drop(columns=["E", "RE", "AE", "SE"], errors="ignore")
    print("\nBest models (by held-out RMSE):")
    print(best_out.to_string(index=False))

    summary = out_dir / "best_models_summary.csv"
    best_out.to_csv(summary, index=False)
    flat = root_out / f"{target}_best_models.csv"
    best_out.to_csv(flat, index=False)
    print(f"\nWrote {summary}")
    print(f"Wrote {flat}")


if __name__ == "__main__":
    main()
