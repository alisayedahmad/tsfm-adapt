import torch
import torch.nn as nn

from src.data.preprocessing import N_TIME_FEATURES

# raw value (1) + cyclical time features (6)
_N_INPUT = 1 + N_TIME_FEATURES


class DomainAdapter(nn.Module):
    """
    Bottleneck adapter that transforms a raw time series into a version
    the frozen TSFM backbone can read better.

    Architecture:
      - instance normalisation (RevIN-style, learned affine)
      - per-timestep linear projection
      - depthwise temporal conv (local context, 7-wide)
      - bottleneck MLP residual
      - output: residual correction → denormalise to original scale

    Starts near-identity (output_proj init ≈ 0) so training is stable
    from the first step — same principle as LoRA's B=0 init.

    Works for any context length at inference; trained on shorter windows
    (ADAPTER_CTX_LEN) because the per-timestep ops are length-agnostic.
    """

    def __init__(self, hidden_dim: int = 256, bottleneck_dim: int = 128):
        super().__init__()

        self.input_proj = nn.Linear(_N_INPUT, hidden_dim)

        # depthwise: mixes neighboring timesteps without cross-channel interference
        self.temporal = nn.Conv1d(
            hidden_dim, hidden_dim,
            kernel_size=7, padding=3, groups=hidden_dim,
        )

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.bottleneck = nn.Sequential(
            nn.Linear(hidden_dim, bottleneck_dim),
            nn.GELU(),
            nn.Linear(bottleneck_dim, hidden_dim),
        )

        self.output_proj = nn.Linear(hidden_dim, 1)

        # learned affine after correction — start as identity
        self.gain = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))

        # near-zero init so adapter ≈ identity at epoch 0
        nn.init.normal_(self.output_proj.weight, std=0.01)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, x_values: torch.Tensor, x_features: torch.Tensor) -> torch.Tensor:
        """
        x_values:   (B, T)       raw time series
        x_features: (B, T, 6)    cyclical time covariates
        returns:    (B, T)       adapted series in original scale
        """
        mu = x_values.mean(dim=-1, keepdim=True)
        sigma = x_values.std(dim=-1, keepdim=True).clamp(min=1e-6)
        x_norm = (x_values - mu) / sigma

        x_in = torch.cat([x_norm.unsqueeze(-1), x_features], dim=-1)  # (B, T, 7)

        h = self.norm1(self.input_proj(x_in))                           # (B, T, H)
        h = h + self.temporal(h.transpose(1, 2)).transpose(1, 2)        # temporal mixing
        h = h + self.bottleneck(self.norm2(h))                          # bottleneck residual

        delta = self.output_proj(h).squeeze(-1)                         # (B, T)

        x_adapted = (x_norm + delta) * self.gain + self.bias
        return x_adapted * sigma + mu

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
