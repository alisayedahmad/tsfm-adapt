"""
Re-run the gap check with step_size=48h so every day of the week gets covered
Phase 1 used step=168 which locked everything to Sunday

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

BDG2_PATH  = ROOT / "data" / "raw" / "electricity_cleaned.csv"
PHASE1_CFG = ROOT / "results" / "phase1" / "phase1_config.json"
RESULTS_DIR = ROOT / "results" / "phase5b"

CONTEXT_LEN = 168
HORIZON = 24
STEP_SIZE = 48    # covers all days of the week
SEASON = 24

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def build_cutoffs(index, horizon, step, context_len):
    last = index[-1] - pd.Timedelta(hours=int(horizon))
    cutoffs = []
    c = last
    while index.get_loc(c) >= context_len:
        cutoffs.append(c)
        c = c - pd.Timedelta(hours=int(step))
    return sorted(cutoffs)


def daily_naive(ctx, horizon):
    return np.tile(ctx[-24:], int(np.ceil(horizon / 24)))[:horizon]


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(BDG2_PATH, index_col=0, parse_dates=True)
    with open(PHASE1_CFG) as f:
        buildings = json.load(f)["buildings"]

    cutoffs = build_cutoffs(df.index, HORIZON, STEP_SIZE, CONTEXT_LEN)
    print(f"cutoffs: {len(cutoffs)} windows, step={STEP_SIZE}h")

    # check day coverage
    days_covered = set()
    for c in cutoffs:
        forecast_start = c + pd.Timedelta(hours=1)
        days_covered.add(forecast_start.day_name())
    print(f"days covered: {sorted(days_covered)}")

    chronos = ChronosForecaster(device=DEVICE)

    records = []
    for bi, bname in enumerate(buildings, 1):
        series = df[bname].ffill().bfill()
        values = series.to_numpy(dtype=np.float64)

        for cutoff in cutoffs:
            cidx = series.index.get_loc(cutoff)
            ctx = values[cidx - CONTEXT_LEN + 1 : cidx + 1].astype(np.float32)
            tgt = values[cidx + 1 : cidx + 1 + HORIZON].astype(np.float32)
            if np.isnan(tgt).any() or np.isnan(ctx).mean() > 0.1:
                continue

            ctx_clean = pd.Series(ctx).ffill().bfill().to_numpy(dtype=np.float32)

            forecast_start = cutoff + pd.Timedelta(hours=1)
            dow = forecast_start.day_name()
            is_weekend = forecast_start.dayofweek >= 5

            for method, pred in [
                ("Chronos", chronos.predict(ctx_clean, horizon=HORIZON)),
                ("DailyNaive", daily_naive(ctx_clean, HORIZON)),
            ]:
                records.append({
                    "building": bname, "cutoff": cutoff,
                    "method": method, "dow": dow, "is_weekend": is_weekend,
                    "mase": mase(tgt, pred, values, season=SEASON),
                    "smape": smape(tgt, pred),
                })

        print(f"  [{bi}/{len(buildings)}] {bname}")

    results = pd.DataFrame(records)
    results.to_parquet(RESULTS_DIR / "gap_by_day.parquet")

    # overall
    summary = (
        results.groupby("method")
        .agg(MASE_median=("mase", "median"), MASE_mean=("mase", "mean"),
             sMAPE_mean=("smape", "mean"), n_evals=("mase", "count"))
        .round(3)
        .sort_values("MASE_median")
    )
    print()
    print("=" * 60)
    print("OVERALL")
    print("=" * 60)
    print(summary.to_string())

    # by day of week
    print()
    print("=" * 60)
    print("BY DAY OF WEEK")
    print("=" * 60)
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    for day in day_order:
        sub = results[results["dow"] == day]
        if sub.empty:
            continue
        chr_mase = sub[sub["method"] == "Chronos"]["mase"].median()
        nai_mase = sub[sub["method"] == "DailyNaive"]["mase"].median()
        n = len(sub) // 2
        gap = (chr_mase - nai_mase) / nai_mase * 100 if nai_mase > 0 else 0
        tag = "GAP" if gap > 15 else ("wins" if gap < -15 else "tied")
        print(f"  {day:10s}  Chronos={chr_mase:.3f}  Naive={nai_mase:.3f}  {gap:+6.1f}%  {tag}  (n={n})")

    # weekday vs weekend
    print()
    print("=" * 60)
    print("WEEKDAY vs WEEKEND")
    print("=" * 60)
    for label, mask in [("weekday", ~results["is_weekend"]), ("weekend", results["is_weekend"])]:
        sub = results[mask]
        chr_mase = sub[sub["method"] == "Chronos"]["mase"].median()
        nai_mase = sub[sub["method"] == "DailyNaive"]["mase"].median()
        gap = (chr_mase - nai_mase) / nai_mase * 100 if nai_mase > 0 else 0
        print(f"  {label:10s}  Chronos={chr_mase:.3f}  Naive={nai_mase:.3f}  {gap:+6.1f}%")

    print(f"\nSaved to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
