import time

import numpy as np
import pandas as pd
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from src.data.dataset import AdapterDataset
from src.data.preprocessing import extract_windows
from src.models.adapter.adapter_module import DomainAdapter
from src.models.adapter.reconstruction_loss import ForecastHead, reconstruction_loss

# adapter is trained on shorter windows (fits within 7-day adaptation budget)
# and applied at inference on the longer Chronos context (168h) —
# safe because all ops are per-timestep or use local kernels
ADAPTER_CTX_LEN = 72   # 3 days
HORIZON = 24
WINDOW_STEP = 1        # max overlap — critical with only 7 days of data


def train_building_adapter(
    values: np.ndarray,
    timestamps: pd.DatetimeIndex,
    n_adapt_days: int = 7,
    hidden_dim: int = 256,
    bottleneck_dim: int = 128,
    epochs: int = 50,
    lr: float = 1e-3,
    batch_size: int = 16,
    device: str | None = None,
    verbose: bool = False,
) -> tuple[DomainAdapter, float]:
    """
    Train an adapter on `n_adapt_days` of unlabeled target-domain data.
    Returns (trained adapter, wall-clock seconds).
    ForecastHead is discarded — only the adapter is returned.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    n_hours = n_adapt_days * 24
    windows = extract_windows(
        values[:n_hours], timestamps[:n_hours],
        context_len=ADAPTER_CTX_LEN, horizon=HORIZON, step=WINDOW_STEP,
    )
    if not windows:
        raise ValueError(
            f"no usable windows for adaptation "
            f"(need ≥{ADAPTER_CTX_LEN + HORIZON}h, got {n_hours}h with high NaN?)"
        )

    loader = DataLoader(
        AdapterDataset(windows), batch_size=batch_size,
        shuffle=True, drop_last=False,
    )

    adapter = DomainAdapter(hidden_dim=hidden_dim, bottleneck_dim=bottleneck_dim).to(device)
    head = ForecastHead(context_len=ADAPTER_CTX_LEN, horizon=HORIZON).to(device)

    params = list(adapter.parameters()) + list(head.parameters())
    optim = Adam(params, lr=lr, weight_decay=1e-4)
    sched = CosineAnnealingLR(optim, T_max=epochs)

    t0 = time.time()
    for epoch in range(epochs):
        total_loss = 0.0
        adapter.train()
        head.train()

        for x_val, x_feat, y_val in loader:
            x_val = x_val.to(device)
            x_feat = x_feat.to(device)
            y_val = y_val.to(device)

            adapted = adapter(x_val, x_feat)
            loss = reconstruction_loss(adapted, y_val, head, x_val)

            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            optim.step()
            total_loss += loss.item()

        sched.step()
        if verbose and (epoch + 1) % 10 == 0:
            print(f"    ep {epoch + 1:3d}/{epochs}  loss={total_loss / len(loader):.4f}")

    adapter.eval()
    return adapter, time.time() - t0
