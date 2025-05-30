# ------------------------------------------------------
# mini_metro RL – unified models & dataset utilities
#   • Drop‑in replacement for the old Q‑table implementation
#   • Requires: torch≥2.2, torchvision (only for transforms)
# ------------------------------------------------------

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Tuple, Union

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ======================================================
# 1.  Neural network policies
# ======================================================

class CNNPolicy(nn.Module):
    """Vision backbone → logits for discrete action space.

    Args
    ----
    n_actions:  total number of *macro* actions your env exposes.
    in_channels: screenshot channels (RGB = 3).
    """

    def __init__(self, n_actions: int, in_channels: int = 3):
        super().__init__()

        # (B, 3, 96, 96) → (B, 64, 8, 8)
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),  # 96→23
            nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),           # 23→10
            nn.ReLU(True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),           # 10→8
            nn.ReLU(True),
            nn.Flatten(),                                         # 64*8*8 = 4096
        )

        self.head = nn.Sequential(
            nn.Linear(4096, 512),
            nn.ReLU(True),
            nn.Linear(512, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # expecting float32 in [0,1] shape (B, C, H, W)
        x = self.backbone(x)
        logits = self.head(x)
        return logits


class SymbolicPolicy(nn.Module):
    """Tiny MLP when you feed symbolic, low‑D features instead of raw pixels."""

    def __init__(self, input_dim: int, n_actions: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(True),
            nn.Linear(128, 128), nn.ReLU(True),
            nn.Linear(128, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ======================================================
# 2.  Behaviour‑Cloning helper
# ======================================================

def behavioural_clone(
    net: nn.Module,
    dataset: Dataset,
    lr: float = 1e-4,
    epochs: int = 3,
    batch_size: int = 64,
    device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu",
) -> None:
    """One simple supervised pass over aggregated demonstrations."""

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    net.to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    cce = nn.CrossEntropyLoss()

    net.train()
    for _ in range(epochs):
        for X, y in loader:
            X = X.to(device)  # (B, ...)
            y = y.to(device)
            logits = net(X)
            loss = cce(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()


# ======================================================
# 3.  DAgger JSON dataset utilities
# ======================================================

class DaggerDataset(Dataset):
    """Reads **multiple** iter_XX.json files and keeps everything in RAM."""

    def __init__(self, data_dir: Union[str, Path]):
        self.samples: List[Tuple[torch.Tensor, int]] = []
        data_dir = Path(data_dir)
        json_files = sorted(data_dir.glob("iter_*.json"))
        if not json_files:
            raise FileNotFoundError(f"No iter_*.json files found in {data_dir}")

        for fp in json_files:
            with fp.open() as f:
                for line in f:
                    if not line.strip():
                        continue
                    item = json.loads(line)
                    state = torch.tensor(item["state"], dtype=torch.float32)
                    action = int(item["action"])
                    self.samples.append((state, action))

    # -------- torch hooks --------
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        x, a = self.samples[idx]
        return x, torch.tensor(a, dtype=torch.long)


def make_loader(
    data_dir: Union[str, Path],
    batch_size: int = 64,
    shuffle: bool = True,
    **dl_kwargs,
) -> DataLoader:
    """Shortcut for `torch.utils.data.DataLoader` on the aggregated DAgger set."""
    ds = DaggerDataset(data_dir)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, **dl_kwargs)


# ======================================================
# 4.  Tiny smoke‑test (python -m mini_metro_models_and_dataset)
# ======================================================

if __name__ == "__main__":
    # Quick import / forward sanity‑check – run `python mini_metro_models_and_dataset.py`.
    n_actions = 32

    # Vision model
    vision_net = CNNPolicy(n_actions)
    dummy_img = torch.rand(4, 3, 96, 96)  # 4 random screenshots
    print("CNNPolicy logits:", vision_net(dummy_img).shape)

    # Symbolic model
    sym_net = SymbolicPolicy(20, n_actions)
    dummy_sym = torch.rand(4, 20)
    print("SymbolicPolicy logits:", sym_net(dummy_sym).shape)

    # Dataset loader (will fail if no iter_*.json present – that is expected)
    try:
        dl = make_loader("data", batch_size=8)
        print("Dataset size:", len(dl.dataset))
    except FileNotFoundError as e:
        print("[Info]", e)
