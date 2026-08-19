import numpy as np

from src.models.tsfm_wrappers.base_wrapper import BaseTSFM


class TimesFMForecaster(BaseTSFM):
    """
    TimesFM 2.5 200M. Install with:
        pip install git+https://github.com/google-research/timesfm.git

    The 2.5 API differs from 2.0: model is compiled once with a ForecastConfig,
    then forecast() takes a list of contexts and returns (batch, horizon).
    """

    name = "TimesFM"

    def __init__(self, max_context: int = 1024, max_horizon: int = 256, device: str | None = None):
        import timesfm

        self.model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
            "google/timesfm-2.5-200m-pytorch"
        )
        self.model.compile(
            timesfm.ForecastConfig(
                max_context=max_context,
                max_horizon=max_horizon,
                normalize_inputs=True,
                use_continuous_quantile_head=True,
                fix_quantile_crossing=True,
                infer_is_positive=True,
            )
        )
        self.max_context = max_context

    def predict(self, context: np.ndarray, horizon: int) -> np.ndarray:
        ctx = np.asarray(context, dtype=np.float32)
        if len(ctx) > self.max_context:
            ctx = ctx[-self.max_context:]

        point, _ = self.model.forecast(horizon=horizon, inputs=[ctx])
        return np.asarray(point[0], dtype=np.float64)[:horizon]
