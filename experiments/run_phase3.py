"""
Phase 3 — does a domain gap exist at 168h horizon?

Same 20 buildings, same protocol as Phase 1, but forecasting 1 week ahead
instead of 24hour
If Chronos zero-shot falls clearly behind SeasonalNaive here,
there is finally something for the adapter to correct,so will see ...

No adapter in this script. Gap check only.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.eval.point_metrics import mase, smape
from src.models.tsfm_wrappers.chronos_wrapper import ChronosForecaster

BDG2_PATH   = ROOT / "data" / "raw" / "electricity_cleaned.csv"
PHASE1_CFG  = ROOT / "results" / "phase1" / "phase1_config.json"
RESULTS_DIR = ROOT / "results" / "phase3"

CONTEXT_LEN = 336   # 2 weeks of context for a 1 week forecast
HORIZON     = 168   # 1 week ahead
N_WINDOWS   = 8
STEP_SIZE   = 168
SEASON      = 24

QUICK_TEST  = False

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_series(df, bname):
    return df[bname].ffill().bfill()


def build_cutoffs(index, n_windows, horizon, step):
    last = index[-1] - pd.Timedelta(hours=int(horizon))
    out = []
    for i in range(n_windows):
        c = last - pd.Timedelta(hours=int(step * (n_windows - 1 - i)))
        if index.get_loc(c) >= CONTEXT_LEN:
            out.append(c)
    return out


def seasonal_naive(ctx, horizon, season_len=168):
    # repeat the last full week
    last_cycle = ctx[-season_len:]
    reps = int(np.ceil(horizon / season_len))
    return np.tile(last_cycle, reps)[:horizon]


def daily_naive(ctx, horizon, season_len=24):
    # repeat the last day, tiled across the horizon
    last_cycle = ctx[-season_len:]
    reps = int(np.ceil(horizon / season_len))
    return np.tile(last_cycle, reps)[:horizon]


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("Phase 3 — gap check at 168h horizon")
    print("=" * 65)
    print(f"device: {DEVICE}  |  context: {CONTEXT_LEN}h  |  horizon: {HORIZON}h")

    df = pd.read_csv(BDG2_PATH, index_col=0, parse_dates=True)

    with open(PHASE1_CFG) as f:
        buildings = json.load(f)["buildings"]

    n_windows = N_WINDOWS
    if QUICK_TEST:
        buildings = buildings[:3]
        n_windows = 2
        print(">>> QUICK_TEST <<<")

    cutoffs = build_cutoffs(df.index, n_windows, HORIZON, STEP_SIZE)
    print(f"cutoffs: {cutoffs[0]} -> {cutoffs[-1]}  ({len(cutoffs)} windows)")

    chronos = ChronosForecaster(device=DEVICE)

    records = []

    for bi, bname in enumerate(buildings, 1):
        series = load_series(df, bname)
        values = series.to_numpy(dtype=np.float64)

        for cutoff in cutoffs:
            cidx = series.index.get_loc(cutoff)
            ctx_raw = values[cidx - CONTEXT_LEN + 1 : cidx + 1]
            tgt = values[cidx + 1 : cidx + 1 + HORIZON]

            if np.isnan(tgt).any() or np.isnan(ctx_raw).mean() > 0.1:
                continue

            ctx = pd.Series(ctx_raw).ffill().bfill().to_numpy(dtype=np.float32)
            tgt = tgt.astype(np.float32)

            preds = {
                "WeeklyNaive": seasonal_naive(ctx, HORIZON, 168),
                "DailyNaive": daily_naive(ctx, HORIZON, 24),
                "Chronos": chronos.predict(ctx, horizon=HORIZON),
            }

            for method, pred in preds.items():
                records.append({
                    "building": bname, "cutoff": cutoff, "method": method,
                    "mase": mase(tgt, pred, values, season=SEASON),
                    "smape": smape(tgt, pred),
                })

        print(f"  [{bi}/{len(buildings)}] {bname}")

    results = pd.DataFrame(records)
    results.to_parquet(RESULTS_DIR / "phase3_results.parquet")

    summary = (
        results.groupby("method")
        .agg(MASE_median=("mase", "median"), MASE_mean=("mase", "mean"),
             sMAPE_mean=("smape", "mean"), n_evals=("mase", "count"))
        .round(3)
        .sort_values("MASE_median")
    )
    summary.to_csv(RESULTS_DIR / "phase3_summary.csv")

    print()
    print("=" * 65)
    print(f"RESULTS — BDG2, {HORIZON}h-ahead, {len(buildings)} buildings, {len(cutoffs)} windows")
    print("=" * 65)
    print(summary.to_string())

    best_naive = min(
        summary.loc["WeeklyNaive", "MASE_median"],
        summary.loc["DailyNaive", "MASE_median"],
    )
    chr_mase = summary.loc["Chronos", "MASE_median"]
    gap = (chr_mase - best_naive) / best_naive * 100

    print(f"\nChronos vs best naive: {chr_mase:.3f} vs {best_naive:.3f}  ({gap:+.1f}%)")
    if gap > 15:
        print("Gap found. Adapter has something to correct here.")
    elif gap < -15:
        print("Chronos clearly wins. No gap to close on this setup.")
    else:
        print("No meaningful gap. Same situation as 24h — try a different domain.")

    print(f"\nSaved to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
