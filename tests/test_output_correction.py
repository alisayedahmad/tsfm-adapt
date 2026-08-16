import numpy as np
import pytest

from src.models.adapter.output_correction import OutputCorrection


def test_identity_on_perfect_predictions():
    cor = OutputCorrection(horizon=4)
    preds = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.float64)
    cor.fit(preds, preds.copy())
    np.testing.assert_allclose(cor.scale, 1.0, atol=1e-6)
    np.testing.assert_allclose(cor.bias, 0.0, atol=1e-6)


def test_learns_bias():
    cor = OutputCorrection(horizon=3)
    rng = np.random.default_rng(42)
    preds = rng.normal(10, 2, (200, 3))
    cor.fit(preds, preds + 5.0)
    np.testing.assert_allclose(cor.scale, 1.0, atol=0.05)
    np.testing.assert_allclose(cor.bias, 5.0, atol=0.3)


def test_learns_scale():
    cor = OutputCorrection(horizon=3)
    rng = np.random.default_rng(42)
    preds = rng.normal(10, 2, (200, 3))
    cor.fit(preds, 2.0 * preds)
    np.testing.assert_allclose(cor.scale, 2.0, atol=0.05)
    np.testing.assert_allclose(cor.bias, 0.0, atol=0.5)


def test_correct_applies_transform():
    cor = OutputCorrection(horizon=2)
    cor.scale = np.array([2.0, 0.5])
    cor.bias = np.array([1.0, -1.0])
    result = cor.correct(np.array([3.0, 4.0]))
    np.testing.assert_allclose(result, [7.0, 1.0])


def test_handles_constant_predictions():
    cor = OutputCorrection(horizon=2)
    preds = np.ones((50, 2)) * 5.0
    actuals = np.ones((50, 2)) * 10.0
    cor.fit(preds, actuals)
    result = cor.correct(np.array([5.0, 5.0]))
    np.testing.assert_allclose(result, [10.0, 10.0], atol=0.1)
