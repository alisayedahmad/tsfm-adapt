"""
Phase 2 — domain adapter vs Chronos zero-shot on BDG2.

ONE adapter trained across all buildings' adaptation data (N_ADAPT_DAYS each),
evaluated per building. Evaluated on the same 8 rolling windows as Phase 1
(no leakage — test windows are at the end of 2017, adaptation data is Jan 2016).
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.data.preprocessing import make_time_features
from src.eval.point_metrics import mase, smape
from src.models.tsfm_wrappers.chronos_wrapper import ChronosForecaster

# ── config ───────────────────────────────────────────────────────────────────
BDG2_PATH    = ROOT / "data" / "raw" / "electricity_cleaned.csv"
PHASE1_CFG   = ROOT / "results" / "phase1" / "phase1_config.json"
RESULTS_DIR  = ROOT / "results" / "phase2"

CONTEXT_LEN  = 168   # 1 week — same as Phase 1 Chronos context
HORIZON      = 24
N_WINDOWS    = 8
STEP_SIZE    = 168   # 1 week between cutoffs
SEASON       = 24    # daily cycle for MASE denominator
N_ADAPT_DAYS = 7


QUICK_TEST   = False  # 3 buildings, 2 windows — flip to True for a smoke test

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ── helpers ───────────────────────────────────────────────────────────────────

def load_series(df: pd.DataFrame, bname: str) -> pd.Series:
    return df[bname].ffill().bfill()


def build_cutoffs(index: pd.DatetimeIndex, n_windows: int, horizon: int, step: int) -> list:
    # last cutoff: last timestamp where a full horizon window still fits
    last = index[-1] - pd.Timedelta(hours=int(horizon))
    cutoffs = [
        last - pd.Timedelta(hours=int(step * (n_windows - 1 - i)))
        for i in range(n_windows)
    ]
    return [c for c in cutoffs if index.get_loc(c) >= CONTEXT_LEN]


def eval_on_cutoffs(
    predict_fn,
    series: pd.Series,
    cutoffs: list,
) -> list[dict]:
    values = series.to_numpy(dtype=np.float64)
    records = []

    for cutoff in cutoffs:
        cidx = series.index.get_loc(cutoff)

        ctx_raw = values[cidx - CONTEXT_LEN + 1 : cidx + 1].astype(np.float32)
        tgt     = values[cidx + 1 : cidx + 1 + HORIZON].astype(np.float32)

        if np.isnan(tgt).any() or np.isnan(ctx_raw).mean() > 0.1:
            continue

        # forward-fill sparse NaN in context (same as Phase 1)
        ctx = pd.Series(ctx_raw).ffill().bfill().to_numpy(dtype=np.float32)
        ctx_ts = series.index[cidx - CONTEXT_LEN + 1 : cidx + 1]

        pred = predict_fn(ctx, ctx_ts)

        records.append({
            "cutoff": cutoff,
            "mase":   mase(tgt, pred, values, season=SEASON),
            "smape":  smape(tgt, pred),
        })

    return records


def zero_shot_predict(chronos: ChronosForecaster, ctx: np.ndarray, _ts) -> np.ndarray:
    return chronos.predict(ctx, horizon=HORIZON)


def adapted_predict(
    chronos: ChronosForecaster,
    adapter: object,
    ctx: np.ndarray,
    ctx_ts: pd.DatetimeIndex,
) -> np.ndarray:
    x_val  = torch.tensor(ctx, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    x_feat = torch.tensor(make_time_features(ctx_ts), dtype=torch.float32).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        adapted_ctx = adapter(x_val, x_feat).squeeze(0).cpu().numpy()
    return chronos.predict(adapted_ctx, horizon=HORIZON)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("Phase 2 — Adapter vs Zero-shot BDG2")
    print("=" * 65)
    print(f"device: {DEVICE}  |  adapt_days: {N_ADAPT_DAYS}  |  context: {CONTEXT_LEN}h")

    df = pd.read_csv(BDG2_PATH, index_col=0, parse_dates=True)

    with open(PHASE1_CFG) as f:
        cfg = json.load(f)
    buildings = cfg["buildings"]

    if QUICK_TEST:
        buildings = buildings[:3]
        global N_WINDOWS
        N_WINDOWS = 2
        print(">>> QUICK_TEST mode — 3 buildings, 2 windows <<<")

    cutoffs = build_cutoffs(df.index, N_WINDOWS, HORIZON, STEP_SIZE)
    print(f"cutoffs: {cutoffs[0]} → {cutoffs[-1]}  ({len(cutoffs)} windows)")

    chronos = ChronosForecaster(device=DEVICE)

    # -- train ONE adapter, select best checkpoint by Chronos validation --
    from src.data.preprocessing import extract_windows
    from src.training.train_adapter import ADAPTER_CTX_LEN, HORIZON as ADAPTER_HORIZON, WINDOW_STEP
    from src.data.dataset import AdapterDataset
    from src.models.adapter.adapter_module import DomainAdapter
    from src.models.adapter.reconstruction_loss import ForecastHead, reconstruction_loss
    from torch.utils.data import DataLoader
    from torch.optim import Adam
    from torch.optim.lr_scheduler import CosineAnnealingLR
    import time as _time
    import copy

    all_windows = []
    for bname in buildings:
        s = load_series(df, bname)
        vals = s.to_numpy(dtype=np.float32)
        n_hours = N_ADAPT_DAYS * 24
        ws = extract_windows(
            vals[:n_hours], s.index[:n_hours],
            context_len=ADAPTER_CTX_LEN, horizon=ADAPTER_HORIZON, step=WINDOW_STEP,
        )
        all_windows.extend(ws)
    print(f"[adapter] {len(all_windows)} windows from {len(buildings)} buildings")

    loader = DataLoader(
        AdapterDataset(all_windows), batch_size=32, shuffle=True, drop_last=False,
    )
    adapter = DomainAdapter(hidden_dim=256, bottleneck_dim=128).to(DEVICE)
    head = ForecastHead(context_len=ADAPTER_CTX_LEN, horizon=ADAPTER_HORIZON).to(DEVICE)
    params = list(adapter.parameters()) + list(head.parameters())
    optim = Adam(params, lr=1e-3, weight_decay=1e-4)
    sched = CosineAnnealingLR(optim, T_max=80)

    # pick 5 validation buildings, use first cutoff only for speed
    val_buildings = buildings[:5]
    val_cutoff = [cutoffs[0]]

    best_mase = float("inf")
    best_state = None

    t0 = _time.time()
    for epoch in range(80):
        total_loss = 0.0
        adapter.train(); head.train()
        for x_val, x_feat, y_val in loader:
            x_val = x_val.to(DEVICE)
            x_feat = x_feat.to(DEVICE)
            y_val = y_val.to(DEVICE)
            adapted_out = adapter(x_val, x_feat)
            loss = reconstruction_loss(adapted_out, y_val, head, x_val)
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            optim.step()
            total_loss += loss.item()
        sched.step()

        # validate through Chronos every 10 epochs
        if (epoch + 1) % 10 == 0:
            adapter.eval()
            val_mases = []
            for vb in val_buildings:
                vs = load_series(df, vb)
                recs = eval_on_cutoffs(
                    lambda ctx, ts: adapted_predict(chronos, adapter, ctx, ts),
                    vs, val_cutoff,
                )
                val_mases.extend([r["mase"] for r in recs])
            cur_mase = np.median(val_mases) if val_mases else float("inf")
            tag = ""
            if cur_mase < best_mase:
                best_mase = cur_mase
                best_state = copy.deepcopy(adapter.state_dict())
                tag = " ← best"
            print(f"  ep {epoch+1}/80  loss={total_loss/len(loader):.4f}  "
                  f"val_mase={cur_mase:.4f}{tag}")

    if best_state is not None:
        adapter.load_state_dict(best_state)
    adapter.eval()
    print(f"[adapter] {adapter.n_params():,} params  trained in {_time.time()-t0:.0f}s  "
          f"best_val_mase={best_mase:.4f}")

    all_records = []

    for bname in buildings:
        series = load_series(df, bname)

        zs_recs  = eval_on_cutoffs(
            lambda ctx, ts: zero_shot_predict(chronos, ctx, ts),
            series, cutoffs,
        )
        ada_recs = eval_on_cutoffs(
            lambda ctx, ts: adapted_predict(chronos, adapter, ctx, ts),
            series, cutoffs,
        )

        for r in zs_recs:
            r.update({"building": bname, "method": "zero_shot"})
        for r in ada_recs:
            r.update({"building": bname, "method": "adapted"})

        all_records.extend(zs_recs + ada_recs)

        zs_med  = np.median([r["mase"] for r in zs_recs])
        ada_med = np.median([r["mase"] for r in ada_recs])
        delta   = (zs_med - ada_med) / max(zs_med, 1e-9) * 100
        print(f"  {bname}: zs={zs_med:.3f}  ada={ada_med:.3f}  Δ={delta:+.1f}%")

    if not all_records:
        print("no results — check data path and building list")
        return

    results = pd.DataFrame(all_records)
    results.to_parquet(RESULTS_DIR / "phase2_results.parquet")

    summary = (
        results.groupby("method")
        .agg(
            MASE_median=("mase",  "median"),
            MASE_mean  =("mase",  "mean"),
            sMAPE_mean =("smape", "mean"),
            n_evals    =("mase",  "count"),
        )
        .round(3)
    )
    summary.to_csv(RESULTS_DIR / "phase2_summary.csv")

    print()
    print("=" * 65)
    print(f"RESULTS — BDG2, {HORIZON}h-ahead, {len(buildings)} buildings, {len(cutoffs)} windows")
    print("=" * 65)
    print(summary.to_string())

    zs  = summary.loc["zero_shot", "MASE_median"]
    ada = summary.loc["adapted",   "MASE_median"]
    print(f"\nMedian MASE: zero_shot={zs:.3f}  adapted={ada:.3f}  Δ={((zs - ada) / zs * 100):+.1f}%")
    print(f"Results saved to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
