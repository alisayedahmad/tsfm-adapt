"""
Phase 2b — Output correction vs Chronos zero-shot on BDG2.
Collects (Chronos prediction, actual) pairs on adaptation data,
fits per-step affine correction by OLS. No proxy head, no gradients.
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
from src.models.adapter.output_correction import OutputCorrection

# ── config ────────────────────────────────────────────────────────────────
BDG2_PATH    = ROOT / "data" / "raw" / "electricity_cleaned.csv"
PHASE1_CFG   = ROOT / "results" / "phase1" / "phase1_config.json"
RESULTS_DIR  = ROOT / "results" / "phase2b"

CONTEXT_LEN  = 168   # test-time Chronos context (1 week)
CAL_CTX_LEN  = 72    # calibration context
HORIZON      = 24
N_WINDOWS    = 8
STEP_SIZE    = 168
SEASON       = 24
N_ADAPT_DAYS = 7
CAL_STEP     = 1     # max overlap for calibration windows

QUICK_TEST   = False

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_series(df, bname):
    return df[bname].ffill().bfill()


def build_cutoffs(index, n_windows, horizon, step):
    last = index[-1] - pd.Timedelta(hours=int(horizon))
    return [
        last - pd.Timedelta(hours=int(step * (n_windows - 1 - i)))
        for i in range(n_windows)
        if index.get_loc(last - pd.Timedelta(hours=int(step * (n_windows - 1 - i)))) >= CONTEXT_LEN
    ]


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("Phase 2b — Output correction vs Zero-shot BDG2")
    print("=" * 65)
    print(f"device: {DEVICE}  |  adapt_days: {N_ADAPT_DAYS}  |  "
          f"cal_ctx: {CAL_CTX_LEN}h  |  test_ctx: {CONTEXT_LEN}h")

    df = pd.read_csv(BDG2_PATH, index_col=0, parse_dates=True)

    with open(PHASE1_CFG) as f:
        cfg = json.load(f)
    buildings = cfg["buildings"]

    n_windows = N_WINDOWS
    if QUICK_TEST:
        buildings = buildings[:3]
        n_windows = 2
        print(">>> QUICK_TEST <<<")

    cutoffs = build_cutoffs(df.index, n_windows, HORIZON, STEP_SIZE)
    print(f"cutoffs: {cutoffs[0]} → {cutoffs[-1]}  ({len(cutoffs)} windows)")

    chronos = ChronosForecaster(device=DEVICE)

    all_records = []

    for bname in buildings:
        series = load_series(df, bname)
        values = series.to_numpy(dtype=np.float64)

        # -- calibration: week immediately before first test cutoff --
        first_cutoff_idx = series.index.get_loc(cutoffs[0])
        cal_end = first_cutoff_idx
        cal_start = cal_end - N_ADAPT_DAYS * 24

        cal_preds, cal_actuals = [], []
        for start in range(cal_start, cal_end - CAL_CTX_LEN - HORIZON + 1, CAL_STEP):
            ctx_raw = values[start : start + CAL_CTX_LEN]
            tgt = values[start + CAL_CTX_LEN : start + CAL_CTX_LEN + HORIZON]
            if np.isnan(ctx_raw).mean() > 0.1 or np.isnan(tgt).any():
                continue
            ctx = pd.Series(ctx_raw).ffill().bfill().to_numpy(dtype=np.float32)
            pred = chronos.predict(ctx, horizon=HORIZON)
            cal_preds.append(pred)
            cal_actuals.append(tgt.astype(np.float64))

        if len(cal_preds) < 10:
            print(f"  {bname}: skip — only {len(cal_preds)} cal pairs")
            continue

        correction = OutputCorrection(horizon=HORIZON)
        correction.fit(np.array(cal_preds), np.array(cal_actuals))

        # -- evaluate --
        for cutoff in cutoffs:
            cidx = series.index.get_loc(cutoff)
            ctx_raw = values[cidx - CONTEXT_LEN + 1 : cidx + 1].astype(np.float32)
            tgt = values[cidx + 1 : cidx + 1 + HORIZON].astype(np.float32)
            if np.isnan(tgt).any() or np.isnan(ctx_raw).mean() > 0.1:
                continue
            ctx = pd.Series(ctx_raw).ffill().bfill().to_numpy(dtype=np.float32)
            zs_pred = chronos.predict(ctx, horizon=HORIZON)
            cor_pred = correction.correct(zs_pred)

            all_records.append({
                "building": bname, "cutoff": cutoff, "method": "zero_shot",
                "mase": mase(tgt, zs_pred, values, season=SEASON),
                "smape": smape(tgt, zs_pred),
            })
            all_records.append({
                "building": bname, "cutoff": cutoff, "method": "corrected",
                "mase": mase(tgt, cor_pred, values, season=SEASON),
                "smape": smape(tgt, cor_pred),
            })

        zs_m = np.median([r["mase"] for r in all_records if r["building"] == bname and r["method"] == "zero_shot"])
        co_m = np.median([r["mase"] for r in all_records if r["building"] == bname and r["method"] == "corrected"])
        delta = (zs_m - co_m) / max(zs_m, 1e-9) * 100
        print(f"  {bname}: zs={zs_m:.3f}  cor={co_m:.3f}  Δ={delta:+.1f}%  ({len(cal_preds)} cal pairs)")

    results = pd.DataFrame(all_records)
    results.to_parquet(RESULTS_DIR / "phase2b_results.parquet")

    summary = (
        results.groupby("method")
        .agg(MASE_median=("mase", "median"), MASE_mean=("mase", "mean"),
             sMAPE_mean=("smape", "mean"), n_evals=("mase", "count"))
        .round(3)
    )
    summary.to_csv(RESULTS_DIR / "phase2b_summary.csv")

    print()
    for bname in buildings:
        bdf = results[results["building"] == bname]
        zs_m = bdf.loc[bdf["method"] == "zero_shot", "mase"].median()
        co_m = bdf.loc[bdf["method"] == "corrected", "mase"].median()
        delta = (zs_m - co_m) / max(zs_m, 1e-9) * 100
        print(f"  {bname}: zs={zs_m:.3f}  cor={co_m:.3f}  Δ={delta:+.1f}%")

    print()
    print("=" * 65)
    print(f"RESULTS — BDG2, {HORIZON}h-ahead, {len(buildings)} buildings, {len(cutoffs)} windows")
    print("=" * 65)
    print(summary.to_string())

    zs  = summary.loc["zero_shot", "MASE_median"]
    cor = summary.loc["corrected", "MASE_median"]
    print(f"\nMedian MASE: zero_shot={zs:.3f}  corrected={cor:.3f}  "
          f"Δ={((zs - cor) / zs * 100):+.1f}%")
    print(f"Results saved to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
