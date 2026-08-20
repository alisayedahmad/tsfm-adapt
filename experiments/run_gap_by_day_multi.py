"""
Gap check by day of week, one model per run.

  .venv          -> python experiments/run_gap_by_day_multi.py --model chronos
  .venv-tsfm     -> python experiments/run_gap_by_day_multi.py --model timesfm
  .venv-moirai   -> python experiments/run_gap_by_day_multi.py --model moirai

Then aggregate: python experiments/run_gap_by_day_multi.py --aggregate
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.eval.point_metrics import mase, smape

BDG2_PATH   = ROOT / "data" / "raw" / "electricity_cleaned.csv"
PHASE1_CFG  = ROOT / "results" / "phase1" / "phase1_config.json"
RESULTS_DIR = ROOT / "results" / "gap_by_day"

CONTEXT_LEN = 168
HORIZON = 24
STEP_SIZE = 48
SEASON = 24


def build_model(name):
    if name == "chronos":
        import torch
        from src.models.tsfm_wrappers.chronos_wrapper import ChronosForecaster
        inner = ChronosForecaster(device="cuda" if torch.cuda.is_available() else "cpu")
        return "Chronos", lambda c, h: np.asarray(inner.predict(c, horizon=h), dtype=np.float64)
    if name == "timesfm":
        from src.models.tsfm_wrappers.timesfm_wrapper import TimesFMForecaster
        m = TimesFMForecaster()
        return "TimesFM", lambda c, h: m.predict(c, h)
    if name == "moirai":
        from src.models.tsfm_wrappers.moirai_wrapper import MoiraiForecaster
        m = MoiraiForecaster()
        return "Moirai", lambda c, h: m.predict(c, h)
    raise ValueError(name)


def build_cutoffs(index):
    last = index[-1] - pd.Timedelta(hours=int(HORIZON))
    cutoffs = []
    c = last
    while index.get_loc(c) >= CONTEXT_LEN:
        cutoffs.append(c)
        c = c - pd.Timedelta(hours=int(STEP_SIZE))
    return sorted(cutoffs)


def daily_naive(ctx, horizon):
    return np.tile(ctx[-24:], int(np.ceil(horizon / 24)))[:horizon]


def run_model(args):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(BDG2_PATH, index_col=0, parse_dates=True)
    with open(PHASE1_CFG) as f:
        buildings = json.load(f)["buildings"]

    cutoffs = build_cutoffs(df.index)
    model_name, predict = build_model(args.model)

    print(f"{model_name} on BDG2, step={STEP_SIZE}h, {len(cutoffs)} cutoffs")

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
                (model_name, predict(ctx_clean, HORIZON)),
                ("DailyNaive", daily_naive(ctx_clean, HORIZON)),
            ]:
                records.append({
                    "building": bname, "cutoff": str(cutoff),
                    "method": method, "dow": dow, "is_weekend": is_weekend,
                    "mase": mase(tgt, pred, values, season=SEASON),
                    "smape": smape(tgt, pred),
                })

        print(f"  [{bi}/{len(buildings)}] {bname}")

    results = pd.DataFrame(records)
    path = RESULTS_DIR / f"gap_{args.model}.parquet"
    results.to_parquet(path)
    print(f"saved {path.name}")

    print_summary(results, model_name)


def print_summary(results, model_name):
    print()
    print("=" * 60)
    print(f"OVERALL — {model_name}")
    print("=" * 60)
    summary = (
        results.groupby("method")
        .agg(MASE_median=("mase", "median"), MASE_mean=("mase", "mean"),
             n_evals=("mase", "count"))
        .round(3)
        .sort_values("MASE_median")
    )
    print(summary.to_string())

    print()
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    for day in day_order:
        sub = results[results["dow"] == day]
        if sub.empty:
            continue
        m_tsfm = sub[sub["method"] == model_name]["mase"].median()
        m_naive = sub[sub["method"] == "DailyNaive"]["mase"].median()
        gap = (m_tsfm - m_naive) / m_naive * 100 if m_naive > 0 else 0
        tag = "GAP" if gap > 15 else ("wins" if gap < -15 else "tied")
        print(f"  {day:10s}  {model_name}={m_tsfm:.3f}  Naive={m_naive:.3f}  {gap:+6.1f}%  {tag}")


def aggregate():
    files = sorted(RESULTS_DIR.glob("gap_*.parquet"))
    if not files:
        print(f"nothing in {RESULTS_DIR}")
        return

    frames = [pd.read_parquet(f).reset_index(drop=True) for f in files]

    # deduplicate naive rows
    all_data = pd.concat(frames, ignore_index=True)
    all_data = all_data.drop_duplicates(subset=["building", "cutoff", "method"])

    models = [m for m in all_data["method"].unique() if m != "DailyNaive"]

    print(f"\nmodels found: {sorted(models)}")
    print()
    print("=" * 60)
    print("OVERALL")
    print("=" * 60)
    summary = (
        all_data.groupby("method")
        .agg(MASE_median=("mase", "median"), MASE_mean=("mase", "mean"))
        .round(3)
        .sort_values("MASE_median")
    )
    print(summary.to_string())

    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

    print()
    print("=" * 60)
    print("BY DAY — all models")
    print("=" * 60)
    header = f"{'Day':10s}  {'Naive':>7s}"
    for m in sorted(models):
        header += f"  {m:>8s}"
    print(header)

    for day in day_order:
        sub = all_data[all_data["dow"] == day]
        if sub.empty:
            continue
        naive = sub[sub["method"] == "DailyNaive"]["mase"].median()
        line = f"  {day:10s}  {naive:7.3f}"
        for m in sorted(models):
            val = sub[sub["method"] == m]["mase"].median()
            gap = (val - naive) / naive * 100 if naive > 0 else 0
            line += f"  {val:5.3f} ({gap:+.0f}%)"
        print(line)

    out = RESULTS_DIR / "gap_by_day_combined.csv"
    all_data.to_csv(out, index=False)
    print(f"\nwrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["chronos", "timesfm", "moirai"])
    ap.add_argument("--aggregate", action="store_true")
    args = ap.parse_args()

    if args.aggregate:
        aggregate()
    elif args.model:
        run_model(args)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
