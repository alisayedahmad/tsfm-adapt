"""
Phase 4 — does a domain gap exist on solar generation?

Same protocol as Phase 1/3, but on NREL solar (137 PV plants, Alabama 2006,
resampled to hourly). Solar is the case where Chronos should struggle:
hard zeros at night, weather-driven ramps, day-to-day variance that no
amount of seasonal structure explains.

Gap check only. No adapter here.

Note on metrics: solar is ~50% zeros, so sMAPE is unreliable and is reported
for completeness only. MASE is the metric to read. nMAE (MAE divided by plant
capacity) is added as a scale-free sanity check that handles zeros cleanly.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.data.nrel_loader import load_hourly, select_plants
from src.eval.point_metrics import mase, smape
from src.models.tsfm_wrappers.chronos_wrapper import ChronosForecaster

DATA_DIR    = ROOT / "data" / "raw"
RESULTS_DIR = ROOT / "results" / "phase4"

N_PLANTS    = 20
SEED        = 42
CONTEXT_LEN = 168
HORIZON     = 24
N_WINDOWS   = 8
STEP_SIZE   = 168
SEASON      = 24

QUICK_TEST  = False

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def build_cutoffs(index, n_windows, horizon, step):
    last = index[-1] - pd.Timedelta(hours=int(horizon))
    out = []
    for i in range(n_windows):
        c = last - pd.Timedelta(hours=int(step * (n_windows - 1 - i)))
        if index.get_loc(c) >= CONTEXT_LEN:
            out.append(c)
    return out


def daily_naive(ctx, horizon, season_len=24):
    last_cycle = ctx[-season_len:]
    reps = int(np.ceil(horizon / season_len))
    return np.tile(last_cycle, reps)[:horizon]


def daily_mean_naive(ctx, horizon, season_len=24, n_days=7):
    # average of the last n_days at each hour of day, smooths out weather noise
    usable = min(n_days, len(ctx) // season_len)
    cycles = ctx[-usable * season_len:].reshape(usable, season_len)
    profile = cycles.mean(axis=0)
    reps = int(np.ceil(horizon / season_len))
    return np.tile(profile, reps)[:horizon]


def nmae(y_true, y_pred, capacity):
    return float(np.mean(np.abs(y_true - y_pred)) / capacity) if capacity > 0 else np.nan


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("Phase 4 — gap check on NREL solar")
    print("=" * 65)
    print(f"device: {DEVICE}  |  context: {CONTEXT_LEN}h  |  horizon: {HORIZON}h")

    df = load_hourly(DATA_DIR)
    plants = select_plants(df, n=N_PLANTS, seed=SEED)

    n_windows = N_WINDOWS
    if QUICK_TEST:
        plants = plants[:3]
        n_windows = 2
        print(">>> QUICK_TEST <<<")

    cutoffs = build_cutoffs(df.index, n_windows, HORIZON, STEP_SIZE)
    print(f"plants: {len(plants)}  |  cutoffs: {cutoffs[0]} -> {cutoffs[-1]} ({len(cutoffs)})")

    chronos = ChronosForecaster(device=DEVICE)

    records = []

    for pi, pname in enumerate(plants, 1):
        series = df[pname]
        values = series.to_numpy(dtype=np.float64)
        capacity = float(np.nanmax(values))

        for cutoff in cutoffs:
            cidx = series.index.get_loc(cutoff)
            ctx = values[cidx - CONTEXT_LEN + 1 : cidx + 1].astype(np.float32)
            tgt = values[cidx + 1 : cidx + 1 + HORIZON].astype(np.float32)

            if np.isnan(tgt).any() or np.isnan(ctx).any():
                continue

            preds = {
                "DailyNaive": daily_naive(ctx, HORIZON),
                "DailyMean7": daily_mean_naive(ctx, HORIZON),
                "Chronos": chronos.predict(ctx, horizon=HORIZON),
            }

            for method, pred in preds.items():
                records.append({
                    "plant": pname, "cutoff": cutoff, "method": method,
                    "mase": mase(tgt, pred, values, season=SEASON),
                    "smape": smape(tgt, pred),
                    "nmae": nmae(tgt, pred, capacity),
                })

        print(f"  [{pi}/{len(plants)}] {pname}")

    results = pd.DataFrame(records)
    results.to_parquet(RESULTS_DIR / "phase4_results.parquet")

    summary = (
        results.groupby("method")
        .agg(MASE_median=("mase", "median"), MASE_mean=("mase", "mean"),
             nMAE_mean=("nmae", "mean"), sMAPE_mean=("smape", "mean"),
             n_evals=("mase", "count"))
        .round(3)
        .sort_values("MASE_median")
    )
    summary.to_csv(RESULTS_DIR / "phase4_summary.csv")

    with open(RESULTS_DIR / "phase4_config.json", "w") as f:
        json.dump({
            "plants": plants, "seed": SEED, "context_len": CONTEXT_LEN,
            "horizon": HORIZON, "n_windows": len(cutoffs), "season": SEASON,
        }, f, indent=2)

    print()
    print("=" * 65)
    print(f"RESULTS — NREL solar, {HORIZON}h-ahead, {len(plants)} plants, {len(cutoffs)} windows")
    print("=" * 65)
    print(summary.to_string())

    best_naive = min(
        summary.loc["DailyNaive", "MASE_median"],
        summary.loc["DailyMean7", "MASE_median"],
    )
    chr_mase = summary.loc["Chronos", "MASE_median"]
    gap = (chr_mase - best_naive) / best_naive * 100

    print(f"\nChronos vs best naive: {chr_mase:.3f} vs {best_naive:.3f}  ({gap:+.1f}%)")
    if gap > 15:
        print("Gap found. This is the case the adapter was built for.")
    elif gap < -15:
        print("Chronos clearly wins. No gap to close here either.")
    else:
        print("No meaningful gap. Chronos holds up on solar too.")

    print(f"\nSaved to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
