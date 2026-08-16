import numpy as np
import pandas as pd
import pytest
import torch

from src.data.preprocessing import make_time_features, extract_windows, N_TIME_FEATURES
from src.data.dataset import AdapterDataset
from src.models.adapter.adapter_module import DomainAdapter
from src.models.adapter.reconstruction_loss import ForecastHead, reconstruction_loss


# ── fixtures ──────────────────────────────────────────────────────────────────

def make_timestamps(n: int, freq="h") -> pd.DatetimeIndex:
    return pd.date_range("2017-01-01", periods=n, freq=freq)


def make_series(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    return (np.sin(2 * np.pi * t / 24) + rng.normal(0, 0.1, n)).astype(np.float32)


# ── time features ─────────────────────────────────────────────────────────────

def test_time_features_shape():
    ts = make_timestamps(100)
    feats = make_time_features(ts)
    assert feats.shape == (100, N_TIME_FEATURES)


def test_time_features_range():
    ts = make_timestamps(48)
    feats = make_time_features(ts)
    assert feats.min() >= -1.0 - 1e-6
    assert feats.max() <=  1.0 + 1e-6


# ── window extraction ─────────────────────────────────────────────────────────

def test_extract_windows_count():
    n = 200
    ctx, hz, step = 72, 24, 12
    series = make_series(n)
    ts = make_timestamps(n)
    windows = extract_windows(series, ts, context_len=ctx, horizon=hz, step=step)
    expected = (n - ctx - hz) // step + 1
    assert len(windows) == expected


def test_extract_windows_shapes():
    series = make_series(300)
    ts = make_timestamps(300)
    windows = extract_windows(series, ts, context_len=72, horizon=24, step=24)
    assert len(windows) > 0
    w = windows[0]
    assert w["x_values"].shape  == (72,)
    assert w["x_features"].shape == (72, N_TIME_FEATURES)
    assert w["y_values"].shape   == (24,)


def test_extract_windows_skips_high_nan():
    series = make_series(300)
    series[:50] = np.nan   # first 50 are all NaN
    ts = make_timestamps(300)
    windows = extract_windows(series, ts, context_len=72, horizon=24, step=24)
    # windows starting before index 50 should be skipped
    assert all(not np.isnan(w["x_values"]).any() for w in windows)


# ── adapter I/O shapes ────────────────────────────────────────────────────────

@pytest.mark.parametrize("ctx_len", [72, 168])
def test_adapter_output_shape(ctx_len):
    B = 4
    adapter = DomainAdapter(hidden_dim=32, bottleneck_dim=16)
    x_val  = torch.randn(B, ctx_len)
    x_feat = torch.randn(B, ctx_len, N_TIME_FEATURES)
    out = adapter(x_val, x_feat)
    assert out.shape == (B, ctx_len)


def test_adapter_near_identity_at_init():
    # output_proj is init'd near zero, so adapter(x) ≈ x at start
    adapter = DomainAdapter(hidden_dim=32, bottleneck_dim=16)
    x_val  = torch.randn(2, 72)
    x_feat = torch.randn(2, 72, N_TIME_FEATURES)
    out = adapter(x_val, x_feat)
    # mean absolute deviation should be small (< 10% of input std)
    mad = (out - x_val).abs().mean().item()
    assert mad < 0.5, f"adapter deviates too much from identity at init: MAD={mad:.4f}"


def test_adapter_gradient_flows():
    adapter = DomainAdapter(hidden_dim=32, bottleneck_dim=16)
    head    = ForecastHead(context_len=72, horizon=24)
    x_val  = torch.randn(4, 72, requires_grad=False)
    x_feat = torch.randn(4, 72, N_TIME_FEATURES)
    y_val  = torch.randn(4, 24)

    adapted = adapter(x_val, x_feat)
    loss = reconstruction_loss(adapted, y_val, head)
    loss.backward()

    has_grad = all(
        p.grad is not None for p in adapter.parameters() if p.requires_grad
    )
    assert has_grad, "some adapter params have no gradient"


# ── param count sanity ────────────────────────────────────────────────────────

def test_adapter_param_count_default():
    adapter = DomainAdapter()   # default: hidden=256, bottleneck=128
    n = adapter.n_params()
    assert 50_000 < n < 500_000, f"unexpected param count: {n:,}"


def test_forecast_head_shapes():
    head = ForecastHead(context_len=72, horizon=24)
    x = torch.randn(8, 72)
    out = head(x)
    assert out.shape == (8, 24)


# ── dataset ───────────────────────────────────────────────────────────────────

def test_adapter_dataset_len():
    series = make_series(300)
    ts = make_timestamps(300)
    windows = extract_windows(series, ts, context_len=72, horizon=24, step=24)
    ds = AdapterDataset(windows)
    assert len(ds) == len(windows)


def test_adapter_dataset_dtypes():
    series = make_series(300)
    ts = make_timestamps(300)
    windows = extract_windows(series, ts, context_len=72, horizon=24, step=24)
    ds = AdapterDataset(windows)
    x_val, x_feat, y_val = ds[0]
    assert x_val.dtype  == torch.float32
    assert x_feat.dtype == torch.float32
    assert y_val.dtype  == torch.float32
