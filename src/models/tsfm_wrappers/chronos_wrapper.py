import numpy as np
import torch
from chronos import ChronosPipeline


class ChronosForecaster:
    def __init__(self, model_id="amazon/chronos-t5-small", device=None,                 dtype=torch.float32):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipeline = ChronosPipeline.from_pretrained(
            model_id, device_map=device, torch_dtype=dtype,
        )

    def predict(self, context, horizon=24, num_samples=100):
        # context: 1D numpy array, returns 1D numpy array (median forecast)
        ctx = torch.tensor(context, dtype=torch.float32).unsqueeze(0)
        torch.manual_seed(42)
        with torch.no_grad():
            samples = self.pipeline.predict(
                ctx, prediction_length=horizon, num_samples=num_samples,
            )
        # samples shape: (1, num_samples, horizon)
        return samples.median(dim=1).values.squeeze(0).cpu().numpy()

    def predict_quantiles(self, context, horizon=24, num_samples=100,
                          quantiles=(0.1, 0.5, 0.9)):
        # for later — probabilistic eval
        ctx = torch.tensor(context, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            samples = self.pipeline.predict(
                ctx, prediction_length=horizon, num_samples=num_samples,
            )
        samples_np = samples.squeeze(0).cpu().numpy()  # (num_samples, horizon)
        return {q: np.quantile(samples_np, q, axis=0) for q in quantiles}
