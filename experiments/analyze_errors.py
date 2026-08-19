"""
Do the worst Chronos errors fall on holidays / weekends?

If yes: covariable correction has a real lever.
If no: the errors are random and no correction will help.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.eval.point_metrics import mase
from src.models.tsfm_wrappers.chronos_wrapper import ChronosForecaster

BDG2_PATH  = ROOT / "data" / "raw" / "electricity_cleaned.csv"
PHASE1_CFG = ROOT / "results" / "phase1" / "phase1_config.json"

CONTEXT_LEN = 168
HORIZON = 24
N_WINDOWS = 8
STEP_SIZE = 168
SEASON = 24

# US federal holidays in the BDG2 test window (Nov-Dec 2017)
HOLIDAYS = {
    pd.Timestamp("2017-11-23"),  # Thanksgiving
    pd.Timestamp("2017-11-24"),  # Black Friday
    pd.Timestamp("2017-12-25"),  # Christmas
    pd.Timestamp("2017-12-26"),  # day after Christmas
    pd.Timestamp("2017-12-31"),  # NYE
    pd.Timestamp("2017-12-24"),  # Christmas Eve
}


def main():
    df = pd.read_csv(BDG2_PATH, index_col=0, parse_dates=True)
    with open(PHASE1_CFG) as f:
        buildings = json.load(f)["buildings"]

    last = df.index[-1] - pd.Timedelta(hours=int(HORIZON))
    cutoffs = [
        last - pd.Timedelta(hours=int(STEP_SIZE * (N_WINDOWS - 1 - i)))
        for i in range(N_WINDOWS)
        if df.index.get_loc(last - pd.Timedelta(hours=int(STEP_SIZE * (N_WINDOWS - 1 - i)))) >= CONTEXT_LEN
    ]

    import torch
    chronos = ChronosForecaster(device="cuda" if torch.cuda.is_available() else "cpu")

    records = []
    for bname in buildings:
        series = df[bname].ffill().bfill()
        values = series.to_numpy(dtype=np.float64)

        for cutoff in cutoffs:
            cidx = series.index.get_loc(cutoff)
            ctx = values[cidx - CONTEXT_LEN + 1 : cidx + 1].astype(np.float32)
            tgt = values[cidx + 1 : cidx + 1 + HORIZON].astype(np.float32)
            if np.isnan(tgt).any() or np.isnan(ctx).mean() > 0.1:
                continue

            ctx_clean = pd.Series(ctx).ffill().bfill().to_numpy(dtype=np.float32)
            pred = chronos.predict(ctx_clean, horizon=HORIZON)
            m = mase(tgt, pred, values, season=SEASON)

            forecast_date = cutoff + pd.Timedelta(hours=1)
            is_weekend = forecast_date.dayofweek >= 5
            is_holiday = forecast_date.normalize() in HOLIDAYS
            dow = forecast_date.day_name()

            records.append({
                "building": bname, "cutoff": cutoff,
                "forecast_date": forecast_date, "dow": dow,
                "is_weekend": is_weekend, "is_holiday": is_holiday,
                "mase": m,
            })

    results = pd.DataFrame(records)

    # split into worst 20% and rest
    threshold = results["mase"].quantile(0.8)
    worst = results[results["mase"] >= threshold]
    rest = results[results["mase"] < threshold]

    print("=" * 60)
    print("Error analysis: do the worst windows have a pattern?")
    print("=" * 60)

    print(f"\nall windows: {len(results)}")
    print(f"worst 20% threshold: MASE >= {threshold:.3f}")
    print(f"worst windows: {len(worst)}")

    print(f"\n--- Weekend rate ---")
    print(f"  worst 20%: {worst['is_weekend'].mean():.1%}")
    print(f"  rest:      {rest['is_weekend'].mean():.1%}")
    print(f"  baseline:  {2/7:.1%}")

    print(f"\n--- Holiday rate ---")
    print(f"  worst 20%: {worst['is_holiday'].mean():.1%}")
    print(f"  rest:      {rest['is_holiday'].mean():.1%}")

    print(f"\n--- Day of week distribution (worst 20%) ---")
    dow_counts = worst["dow"].value_counts()
    for day in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]:
        n = dow_counts.get(day, 0)
        pct = n / len(worst) * 100
        print(f"  {day:10s} {n:3d}  ({pct:.0f}%)")

    print(f"\n--- Cutoff distribution (worst 20%) ---")
    for cutoff, count in worst["cutoff"].value_counts().sort_index().items():
        date = pd.Timestamp(cutoff)
        tag = ""
        if date.normalize() + pd.Timedelta(hours=1) in HOLIDAYS or (date + pd.Timedelta(hours=1)).normalize() in HOLIDAYS:
            tag = " <-- near holiday"
        if date.dayofweek >= 4:
            tag += " (weekend/friday)"
        print(f"  {cutoff}  {count:3d} worst windows{tag}")

    print(f"\n--- Top 10 worst individual windows ---")
    for _, row in worst.nlargest(10, "mase").iterrows():
        tags = []
        if row["is_weekend"]: tags.append("weekend")
        if row["is_holiday"]: tags.append("HOLIDAY")
        tag = " [" + ", ".join(tags) + "]" if tags else ""
        print(f"  {row['building']:30s} {row['forecast_date']}  {row['dow']:9s}  MASE={row['mase']:.2f}{tag}")


if __name__ == "__main__":
    main()
