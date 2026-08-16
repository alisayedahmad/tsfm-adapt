import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from src.eval.point_metrics import mase, smape, naive_seasonal_forecast


def test_mase_perfect_prediction():
    rng = np.random.default_rng(0)
    # seasonal pattern + noise so naive denominator > 0
    y_train = np.tile(np.arange(24, dtype=float), 10) + rng.normal(0, 1, 240)
    y_true = np.ones(24)
    y_pred = np.ones(24)
    assert mase(y_true, y_pred, y_train) == 0.0


def test_mase_equals_one_for_naive():
    # if prediction == seasonal naive, MASE should be ~1
    rng = np.random.default_rng(0)
    y_train = np.tile(np.arange(24, dtype=float), 30) + rng.normal(0, 0.1, 720)
    context = y_train[-48:]
    y_true = y_train[-24:]
    y_pred = naive_seasonal_forecast(context[:-24], horizon=24, season=24)
    m = mase(y_true, y_pred, y_train[:-24])
    # not exactly 1 because of noise, but close
    assert 0.5 < m < 2.0


def test_mase_constant_series_returns_nan():
    y_train = np.ones(200)
    y_true = np.ones(24)
    y_pred = np.ones(24) * 2
    assert np.isnan(mase(y_true, y_pred, y_train))


def test_smape_perfect():
    y = np.array([1.0, 2.0, 3.0])
    assert smape(y, y) == 0.0


def test_smape_range():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([3.0, 4.0, 5.0])
    s = smape(y_true, y_pred)
    assert 0 < s < 200


def test_smape_symmetric():
    y_true = np.array([1.0, 2.0])
    y_pred = np.array([3.0, 4.0])
    assert smape(y_true, y_pred) == smape(y_pred, y_true)


def test_naive_seasonal_forecast_shape():
    context = np.arange(168, dtype=float)
    pred = naive_seasonal_forecast(context, horizon=24, season=24)
    assert len(pred) == 24


def test_naive_seasonal_forecast_repeats_last_day():
    # last 24h = [0..23], forecast should repeat that
    context = np.zeros(168)
    context[-24:] = np.arange(24)
    pred = naive_seasonal_forecast(context, horizon=24, season=24)
    np.testing.assert_array_equal(pred, np.arange(24))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
