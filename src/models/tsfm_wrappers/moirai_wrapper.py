import numpy as np
import pandas as pd

from src.models.tsfm_wrappers.base_wrapper import BaseTSFM


class MoiraiForecaster(BaseTSFM):
    """
    Moirai via uni2ts. Install with:
        pip install uni2ts

    Supports Moirai 2.0 (quantile head) and falls back to 1.1 (mixture head).
    Inference goes through gluonts, so each context is wrapped in a one-row
    PandasDataset. Slower than Chronos but the batch is one window anyway.
    """

    name = "Moirai"

    def __init__(self, version: str = "2.0", size: str = "small", context_length: int = 1000):
        self.version = version
        self.context_length = context_length

        if version.startswith("2"):
            from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module

            self._forecast_cls = Moirai2Forecast
            self._module = Moirai2Module.from_pretrained(f"Salesforce/moirai-2.0-R-{size}")
        else:
            from uni2ts.model.moirai import MoiraiForecast, MoiraiModule

            self._forecast_cls = MoiraiForecast
            self._module = MoiraiModule.from_pretrained(f"Salesforce/moirai-1.1-R-{size}")

        self._predictor = None
        self._built_for = None

    def _build(self, horizon: int, ctx_len: int):
        # predictor is tied to a prediction length, rebuild when horizon changes
        if self._built_for == (horizon, ctx_len):
            return

        kwargs = dict(
            module=self._module,
            prediction_length=horizon,
            context_length=ctx_len,
            target_dim=1,
            feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=0,
        )
        if not self.version.startswith("2"):
            kwargs["num_samples"] = 100

        self._predictor = self._forecast_cls(**kwargs).create_predictor(batch_size=1)
        self._built_for = (horizon, ctx_len)

    def predict(self, context: np.ndarray, horizon: int) -> np.ndarray:
        from gluonts.dataset.pandas import PandasDataset

        ctx = np.asarray(context, dtype=np.float32)
        if len(ctx) > self.context_length:
            ctx = ctx[-self.context_length:]

        self._build(horizon, len(ctx))

        df = pd.DataFrame(
            {"target": ctx},
            index=pd.date_range("2000-01-01", periods=len(ctx), freq="h"),
        )
        ds = PandasDataset(df, target="target")

        fc = next(iter(self._predictor.predict(ds)))

        # 2.0 exposes quantiles, 1.x exposes samples
        if hasattr(fc, "samples") and fc.samples is not None:
            out = np.median(fc.samples, axis=0)
        else:
            out = fc.quantile(0.5)

        return np.asarray(out, dtype=np.float64)[:horizon]
