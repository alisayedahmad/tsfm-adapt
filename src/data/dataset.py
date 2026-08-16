import torch
from torch.utils.data import Dataset


class AdapterDataset(Dataset):
    def __init__(self, windows: list[dict]):
        self.windows = windows

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        w = self.windows[idx]
        return (
            torch.from_numpy(w["x_values"].copy()),
            torch.from_numpy(w["x_features"].copy()),
            torch.from_numpy(w["y_values"].copy()),
        )
