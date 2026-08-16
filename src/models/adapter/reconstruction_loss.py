import torch.nn as nn
import torch.nn.functional as F


class ForecastHead(nn.Module):
    """
    Lightweight MLP used ONLY during adapter training.
    Provides a gradient path from forecast error back to the adapter.
    Dropped at inference — Chronos takes over.

    Why not backprop through Chronos: Chronos tokenises continuous values
    into discrete bins, which kills the gradient. This head acts as a
    differentiable surrogate with the same prediction task.
    """

    def __init__(self, context_len: int, horizon: int = 24, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(context_len, hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, horizon),
        )

    def forward(self, x):
        # x: (B, context_len) — adapted series values
        return self.net(x)


def reconstruction_loss(adapted: object, targets: object, head: ForecastHead,
                         x_original: object, lambda_id: float =1.0):
    preds = head(adapted)
    # Huber: less sensitive to the outlier buildings in BDG2 than plain MSE
    forecast_loss = F.huber_loss(preds, targets, delta=1.0)
    # penalise large deviations from input — prevents distortion on unseen data
    identity_loss = F.mse_loss(adapted, x_original)
    return forecast_loss + lambda_id * identity_loss
