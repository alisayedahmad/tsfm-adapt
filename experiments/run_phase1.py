"""
Phase 1 — BDG2 baselines + Chronos zero-shot.
Produces Table 1 (statistical baselines) and Table 2 (zero-shot TSFM).
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import time
import json
import numpy as np
import pandas as pd
from pathlib import Path

from src.data.sampling import select_buildings, wide_to_long
from src.eval.point_metrics import mase, smape
from src.models.baselines.statistical import run_statistical_baselines
from src.models.tsfm_wrappers.chronos_wrapper import ChronosForecaster


# ── config ──────────────────────────────────────────────────────────────
# adjust BDG2_PATH to wherever your electricity.csv lives
BDG2_PATH = "data/raw/electricity_cleaned.csv"

N_BUILDINGS = 20
HORIZON = 24        # 24h ahead
CONTEXT_LEN = 168   # 1 week of context for Chronos
STEP_SIZE = 168     # eval window every 7 days
N_WINDOWS = 8       # 8 rolling windows (~2 months of test)
SEASON = 24         # daily cycle for MASE denominator
RESULTS_DIR = Path("results/phase1")

# set True for a quick sanity check before the full run
QUICK_TEST = False
if QUICK_TEST:
    N_BUILDINGS = 3
    N_WINDOWS = 2


# ── data loading ────────────────────────────────────────────────────────
def load_bdg2(path):
    # wide format: DatetimeIndex, columns = building IDs, values = kWh
    # if your bdg2_loader.py returns a different format, adapt here
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index.freq = pd.infer_freq(df.index)
    print(f"[data] loaded {df.shape[1]} buildings, "
          f"{df.shape[0]} timestamps ({df.index[0]} -> {df.index[-1]})")
    print(f"[data] overall NaN rate: {df.isna().mean().mean():.1%}")
    return df


# ── chronos evaluation ──────────────────────────────────────────────────
def evaluate_chronos(df_wide, building_ids, cutoffs, horizon, context_len):
    model = ChronosForecaster()
    records = []
    total = len(building_ids) * len(cutoffs)
    done = 0

    for bid in building_ids:
        series = df_wide[bid].values.astype(np.float64).copy()
        timestamps = df_wide.index

        # forward-fill NaN, then back-fill any leading NaN ffill can't
        # reach — Chronos handles NaN in some versions but better safe
        # than sorry
        series = pd.Series(series).ffill().bfill().values

        for cutoff in cutoffs:
            cutoff_idx = timestamps.get_loc(cutoff)
            if isinstance(cutoff_idx, slice):
                cutoff_idx = cutoff_idx.stop

            start = cutoff_idx - context_len + 1
            end = cutoff_idx + horizon + 1

            if start < 0 or end > len(series):
                continue

            context = series[start : cutoff_idx + 1]
            y_true = series[cutoff_idx + 1 : cutoff_idx + 1 + horizon]

            if np.isnan(y_true).any() or len(y_true) < horizon:
                continue

            y_pred = model.predict(context, horizon=horizon)

            for h in range(horizon):
                records.append({
                    "unique_id": bid,
                    "ds": timestamps[cutoff_idx + 1 + h],
                    "cutoff": cutoff,
                    "y": y_true[h],
                    "Chronos": y_pred[h],
                })

            done += 1
            if done % 5 == 0:
                print(f"  [chronos] {done}/{total} windows done")

    return pd.DataFrame(records)


# ── metrics computation ─────────────────────────────────────────────────
def compute_metrics(cv_df, model_cols, df_wide, building_ids, season=24):
    first_cutoff = cv_df["cutoff"].min()
    results = []

    for model_name in model_cols:
        all_mase = []
        all_smape = []

        for bid in building_ids:
            bid_df = cv_df[cv_df["unique_id"] == bid]
            if bid_df.empty:
                continue

            # training data = everything strictly before first cutoff
            y_train = df_wide.loc[:first_cutoff, bid].ffill().dropna().values
            if len(y_train) < season * 2:
                continue

            for cutoff in bid_df["cutoff"].unique():
                window = bid_df[bid_df["cutoff"] == cutoff].sort_values("ds")
                y_true = window["y"].values
                y_pred = window[model_name].values

                if len(y_true) < HORIZON or np.isnan(y_pred).any():
                    continue

                m = mase(y_true, y_pred, y_train, season=season)
                s = smape(y_true, y_pred)

                if not np.isnan(m):
                    all_mase.append(m)
                if not np.isnan(s):
                    all_smape.append(s)

        results.append({
            "Method": model_name,
            "MASE_median": f"{np.median(all_mase):.3f}" if all_mase else "—",
            "MASE_mean": f"{np.mean(all_mase):.3f}" if all_mase else "—",
            "sMAPE_mean": f"{np.mean(all_smape):.1f}" if all_smape else "—",
            "n_evals": len(all_mase),
        })

    return pd.DataFrame(results)


# ── main ────────────────────────────────────────────────────────────────
def main():
    t0 = time.time()
    print("=" * 65)
    print("Phase 1 — BDG2 baselines + Chronos zero-shot")
    print("=" * 65)

    if QUICK_TEST:
        print(">>> QUICK_TEST mode — reduced scope <<<\n")

    # load
    df = load_bdg2(BDG2_PATH)
    buildings = select_buildings(df, n=N_BUILDINGS)

    # --- statistical baselines ---
    print("\n── Statistical baselines ──")
    df_long = wide_to_long(df, buildings)
    t1 = time.time()
    cv_stats = run_statistical_baselines(
        df_long, horizon=HORIZON, step_size=STEP_SIZE, n_windows=N_WINDOWS,
    )
    print(f"[baselines] wall time: {time.time() - t1:.0f}s")

    # extract cutoff timestamps for Chronos alignment
    cutoffs = sorted(cv_stats["cutoff"].unique())
    print(f"[eval] {len(cutoffs)} cutoff points: {cutoffs[0]} -> {cutoffs[-1]}")

    # --- chronos zero-shot ---
    print("\n── Chronos zero-shot ──")
    t2 = time.time()
    cv_chronos = evaluate_chronos(df, buildings, cutoffs, HORIZON, CONTEXT_LEN)
    print(f"[chronos] wall time: {time.time() - t2:.0f}s")

    # --- metrics ---
    print("\n── Computing metrics ──")
    stat_models = ["SeasonalNaive", "AutoETS", "AutoTheta"]
    metrics_stats = compute_metrics(cv_stats, stat_models, df, buildings, SEASON)
    metrics_chronos = compute_metrics(cv_chronos, ["Chronos"], df, buildings, SEASON)

    table = pd.concat([metrics_stats, metrics_chronos], ignore_index=True)

    # --- output ---
    print("\n" + "=" * 65)
    print(f"RESULTS — BDG2 electricity, {HORIZON}h-ahead, "
          f"{N_BUILDINGS} buildings, {N_WINDOWS} windows")
    print("=" * 65)
    print(table.to_string(index=False))
    print("=" * 65)
    print(f"Total wall time: {time.time() - t0:.0f}s")

    # save
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(RESULTS_DIR / "phase1_results.csv", index=False)

    # save raw forecasts for later analysis
    cv_stats.to_parquet(RESULTS_DIR / "cv_baselines.parquet", index=False)
    cv_chronos.to_parquet(RESULTS_DIR / "cv_chronos.parquet", index=False)

    # save config for reproducibility
    config = {
        "n_buildings": N_BUILDINGS, "horizon": HORIZON,
        "context_len": CONTEXT_LEN, "step_size": STEP_SIZE,
        "n_windows": N_WINDOWS, "season": SEASON,
        "buildings": buildings,
    }
    with open(RESULTS_DIR / "phase1_config.json", "w") as f:
        json.dump(config, f, indent=2, default=str)

    print(f"\nSaved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
