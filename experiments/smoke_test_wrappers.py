"""
Smoke test for the TSFM wrappers. Run this before run_phase5.

Loads each model one at a time on a synthetic daily-seasonal series and checks
that predict() returns a sane shape and sane values. Catches install problems
and API drift before a two hour experiment does.

Models that fail are reported and skipped, not fatal.
"""

import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

CONTEXT_LEN = 168
HORIZON = 24


def make_test_series(n=CONTEXT_LEN + HORIZON, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    daily = 10 * np.sin(2 * np.pi * t / 24)
    trend = 0.01 * t
    return (50 + daily + trend + rng.normal(0, 1, n)).astype(np.float32)


def check(name, build_fn):
    print(f"\n[{name}]")
    try:
        t0 = time.time()
        model = build_fn()
        print(f"  loaded in {time.time() - t0:.0f}s")

        series = make_test_series()
        ctx, tgt = series[:CONTEXT_LEN], series[CONTEXT_LEN:]

        t0 = time.time()
        pred = model.predict(ctx, horizon=HORIZON)
        dt = time.time() - t0

        assert pred.shape == (HORIZON,), f"bad shape {pred.shape}, expected ({HORIZON},)"
        assert np.isfinite(pred).all(), "prediction contains nan or inf"

        mae = np.mean(np.abs(pred - tgt))
        naive_mae = np.mean(np.abs(ctx[-24:] - tgt))
        print(f"  predict: {dt * 1000:.0f}ms  MAE={mae:.2f}  (naive {naive_mae:.2f})")
        print(f"  range: [{pred.min():.1f}, {pred.max():.1f}]  actual [{tgt.min():.1f}, {tgt.max():.1f}]")

        if mae > 3 * naive_mae:
            print("  WARNING: much worse than naive on a clean sine, check the wrapper")
        else:
            print("  OK")

        model.unload()
        return True

    except Exception as e:
        print(f"  FAILED: {type(e).__name__}: {e}")
        traceback.print_exc(limit=3)
        return False


def main():
    print("=" * 60)
    print("TSFM wrapper smoke test")
    print("=" * 60)
    print(f"cuda: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"gpu: {props.name}  {props.total_memory / 1e9:.1f}GB")

    results = {}

    try:
        from src.models.tsfm_wrappers.chronos_wrapper import ChronosForecaster
        results["Chronos"] = check("Chronos", lambda: _wrap_chronos(ChronosForecaster))
    except ImportError as e:
        print(f"\n[Chronos]\n  not installed: {e}")
        results["Chronos"] = False

    try:
        from src.models.tsfm_wrappers.timesfm_wrapper import TimesFMForecaster
        results["TimesFM"] = check("TimesFM", lambda: TimesFMForecaster())
    except ImportError as e:
        print(f"\n[TimesFM]\n  not installed: {e}")
        results["TimesFM"] = False

    try:
        from src.models.tsfm_wrappers.moirai_wrapper import MoiraiForecaster
        results["Moirai"] = check("Moirai", lambda: MoiraiForecaster())
    except ImportError as e:
        print(f"\n[Moirai]\n  not installed: {e}")
        results["Moirai"] = False

    print()
    print("=" * 60)
    for name, ok in results.items():
        print(f"  {name:10s} {'ok' if ok else 'FAILED'}")
    n_ok = sum(results.values())
    print(f"\n{n_ok}/{len(results)} models usable")
    if n_ok < 2:
        print("Need at least 2 models for the comparison to mean anything.")


class _ChronosAdapter:
    # existing Chronos wrapper predates BaseTSFM, adapt it here rather than touch phase 1 code
    name = "Chronos"

    def __init__(self, inner):
        self.inner = inner

    def predict(self, context, horizon):
        return np.asarray(self.inner.predict(context, horizon=horizon), dtype=np.float64)

    def unload(self):
        import gc
        self.inner = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _wrap_chronos(cls):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return _ChronosAdapter(cls(device=device))


if __name__ == "__main__":
    main()
