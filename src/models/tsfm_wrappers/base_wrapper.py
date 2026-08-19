from abc import ABC, abstractmethod

import numpy as np


class BaseTSFM(ABC):
    """
    Common interface for every TSFM wrapper.
    Keeps the experiment scripts model-agnostic: they only ever call predict().
    """

    name: str = "base"

    @abstractmethod
    def predict(self, context: np.ndarray, horizon: int) -> np.ndarray:
        """context: (T,) float array. Returns (horizon,) point forecast."""
        ...

    def unload(self):
        # release GPU memory between models, 4GB does not leave much room
        import gc

        import torch

        for attr in ("model", "pipeline", "module"):
            if hasattr(self, attr):
                setattr(self, attr, None)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
