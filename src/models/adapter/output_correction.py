import numpy as np


class OutputCorrection:
    """
    Per-horizon-step affine correction on Chronos zero-shot output.
    Learns scale[h] and bias[h] for each step h by OLS on calibration pairs.
    48 params. Closed-form. Can't overfit on 1000+ pairs.
    """

    def __init__(self, horizon: int = 24):
        self.horizon = horizon
        self.scale = np.ones(horizon, dtype=np.float64)
        self.bias = np.zeros(horizon, dtype=np.float64)

    def fit(self, preds: np.ndarray, actuals: np.ndarray):
        # preds, actuals: (N, horizon)
        for h in range(self.horizon):
            x, y = preds[:, h], actuals[:, h]
            xvar = x.var(ddof=1) if len(x) > 1 else 0.0
            if xvar < 1e-10:
                # constant predictions — can only learn bias
                self.scale[h] = 1.0
                self.bias[h] = y.mean() - x.mean()
            else:
                self.scale[h] = np.cov(x, y)[0, 1] / xvar
                self.bias[h] = y.mean() - self.scale[h] * x.mean()

    def correct(self, pred: np.ndarray) -> np.ndarray:
        return self.scale * pred + self.bias

    def summary(self) -> str:
        return (f"scale [{self.scale.min():.3f}, {self.scale.max():.3f}]  "
                f"bias [{self.bias.min():.3f}, {self.bias.max():.3f}]")
