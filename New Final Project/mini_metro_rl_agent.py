# mini_metro_rl_agent.py
"""Reusable imitation‑first RL agent for Mini Metro.

This file replaces the old tabular‑Q agent.  It supports two operating
modes:

* ``mode="pixels"``     – the observation is an RGB game frame (H×W×3).
* ``mode="symbolic"``   – the observation is a 1‑D numpy array of
                           engineered features produced by your
                           detectors.

Both modes share the same high‑level API so the environment does *not*
need to know which one is being used.

Typical usage
-------------
>>> from mini_metro_rl_agent import MiniMetroRLAgent, load_pairs_json
>>> agent = MiniMetroRLAgent(n_actions=47, mode="pixels")
>>> agent.behavioural_clone(load_pairs_json("dagger_dataset/iter_00.json"))
>>> a = agent.act(obs)  # during play

The file also contains convenience wrappers for saving/loading the model
and building an in‑memory dataset loader for DAgger.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterable, List, Literal, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
#  Neural network back‑ends
# ---------------------------------------------------------------------------

class CNNPolicy(nn.Module):
    """Simple DQN‑style conv‑net → logits."""

    def __init__(self, n_actions: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Flatten(),
        )
        # 128×128 input → (128‑8)/4+1 = 31; 31→(31‑4)/2+1 = 14
        flat = 32 * 14 * 14  # ≈ 6272
        self.head = nn.Sequential(
            nn.Linear(flat, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B,3,H,W) → (B,n_actions)
        x = x.float() / 255.0  # normalise pixels once here
        x = self.features(x)
        return self.head(x)


class SymbolicPolicy(nn.Module):
    """2‑layer MLP for a low‑dimensional continuous state vector."""

    def __init__(self, input_dim: int, n_actions: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B,D) → (B,n_actions)
        return self.net(x.float())

# ---------------------------------------------------------------------------
#  Dataset helpers
# ---------------------------------------------------------------------------

class PairDataset(Dataset):
    """(state, action_index) pairs created by the DAgger recorder."""

    def __init__(self, pairs: Sequence[Tuple[np.ndarray, int]], mode: Literal["pixels", "symbolic"]):
        self.pairs = pairs
        self.mode = mode

    def __len__(self) -> int:  # pragma: no cover
        return len(self.pairs)

    def __getitem__(self, idx: int):
        obs, act = self.pairs[idx]
        obs = preprocess(obs, self.mode)
        return obs, act


def make_loader_from_pairs(pairs: Sequence[Tuple[np.ndarray, int]], *, mode: Literal["pixels", "symbolic"], batch_size: int = 32, shuffle: bool = True) -> DataLoader:
    ds = PairDataset(pairs, mode)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)


# ---------------------------------------------------------------------------
#  Public agent class
# ---------------------------------------------------------------------------

class MiniMetroRLAgent:
    """Policy network + convenience wrappers (act / BC / save / load)."""

    def __init__(
        self,
        n_actions: int,
        *,
        mode: Literal["pixels", "symbolic"] = "pixels",
        feature_dim: int | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        self.mode = mode
        device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.device = device

        if mode == "pixels":
            self.net: nn.Module = CNNPolicy(n_actions)
        else:
            if feature_dim is None:
                raise ValueError("feature_dim must be given when mode='symbolic'")
            self.net = SymbolicPolicy(feature_dim, n_actions)

        self.net.to(device)
        self.net.eval()

    # ------------------------------------------------------------------
    #  Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def choose_action(self, obs: np.ndarray, *, epsilon: float = 0.0) -> int:
        """Return an action index; optional ε‑greedy exploration."""
        if random.random() < epsilon:
            return random.randrange(self.n_actions)

        if isinstance(obs, np.ndarray):
            x = torch.from_numpy(obs)
        else:
            x = torch.from_numpy(preprocess(obs, self.mode))
        logits = self.net(x)
        return int(logits.argmax(dim=-1).item())

    # ------------------------------------------------------------------
    #  Behavioural cloning
    # ------------------------------------------------------------------

    def behavioural_clone(
        self,
        pairs: Sequence[Tuple[np.ndarray, int]],
        *,
        epochs: int = 3,
        lr: float = 1e-4,
        batch_size: int = 32,
        val_split: float = 0.05,
        print_every: int = 100,
    ) -> None:
        """Supervised training on (state, action) pairs."""
        if len(pairs) == 0:
            raise ValueError("Behavioural cloning received an empty dataset.")

        # train‑val split -------------------------------------------------
        rnd = np.random.RandomState(0)
        idx = rnd.permutation(len(pairs))
        cut = int(len(pairs) * (1 - val_split))
        train_pairs = [pairs[i] for i in idx[:cut]]
        val_pairs = [pairs[i] for i in idx[cut:]]

        train_loader = make_loader_from_pairs(train_pairs, mode=self.mode, batch_size=batch_size, shuffle=True)
        val_loader = make_loader_from_pairs(val_pairs, mode=self.mode, batch_size=batch_size, shuffle=False)

        # training loop ---------------------------------------------------
        self.net.train()
        optimiser = optim.Adam(self.net.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(epochs):
            running = 0.0
            for step, (x, y) in enumerate(train_loader, 1):
                x = x.to(self.device)
                y = y.to(self.device)

                optimiser.zero_grad()
                logits = self.net(x)
                loss = criterion(logits, y)
                loss.backward()
                optimiser.step()

                running += loss.item()
                if step % print_every == 0:
                    print(f"[epoch {epoch+1}/{epochs}] step {step:4d}/{len(train_loader)}  loss={running/print_every:.4f}")
                    running = 0.0
            # quick validation ------------------------------------------
            self.net.eval()
            correct = total = 0
            with torch.no_grad():
                for x, y in val_loader:
                    x = x.to(self.device)
                    y = y.to(self.device)
                    logits = self.net(x)
                    pred = logits.argmax(dim=-1)
                    correct += (pred == y).sum().item()
                    total += y.numel()
            print(f"→ val acc: {correct/total:.3%}")
            self.net.train()

        self.net.eval()

    # ------------------------------------------------------------------
    #  I/O helpers
    # ------------------------------------------------------------------

    @property
    def n_actions(self) -> int:  # pragma: no cover
        return self.net.head[-1].out_features if isinstance(self.net, CNNPolicy) else self.net.net[-1].out_features

    def save(self, ckpt_path: str | Path) -> None:
        ckpt = {
            "state_dict": self.net.state_dict(),
            "mode": self.mode,
            "feature_dim": getattr(self.net, "net", None)[0].in_features if self.mode == "symbolic" else None,
            "n_actions": self.n_actions,
        }
        torch.save(ckpt, Path(ckpt_path))

    @classmethod
    def load(cls, ckpt_path: str | Path, device: torch.device | str | None = None) -> "MiniMetroRLAgent":
        data = torch.load(ckpt_path, map_location="cpu")
        agent = cls(n_actions=data["n_actions"], mode=data["mode"], feature_dim=data["feature_dim"], device=device)
        agent.net.load_state_dict(data["state_dict"])
        agent.net.eval()
        return agent

# ---------------------------------------------------------------------------
#  Utility: load dataset JSON created by expert recorder
# ---------------------------------------------------------------------------

Pair = Tuple[np.ndarray, int]

def load_pairs_json(path: str | Path) -> List[Pair]:
    """Load one of the dagger_dataset/iter_XX.json files."""
    pairs: List[Pair] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            obs = np.asarray(rec["state"], dtype=np.uint8)
            act = int(rec["action"])
            pairs.append((obs, act))
    return pairs

# ---------------------------------------------------------------------------
#  Generic preprocessing (stub)
# ---------------------------------------------------------------------------

def preprocess(obs: np.ndarray, mode: Literal["pixels", "symbolic"]) -> np.ndarray:
    """Convert observation to the tensor format expected by the net.

    This stub *must* match the exact shape the policy was trained on.
    Feel free to swap it for your own implementation.
    """
    if mode == "pixels":
        # Expect obs as H×W×C uint8. Resize/crop here if you need to.
        if obs.ndim != 3 or obs.shape[2] != 3:
            raise ValueError("Pixel input must be H×W×3 uint8 array")
        # rearrange to C×H×W for PyTorch
        return np.transpose(obs, (2, 0, 1))  # (C,H,W)
    else:
        # symbolic vector → leave as‑is
        return obs

# ---------------------------------------------------------------------------
#  Quick smoke‑test ----------------------------------------------------------
# ---------------------------------------------------------------------------

if __name__ == "__main__":  # pragma: no cover
    dummy = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
    agent = MiniMetroRLAgent(n_actions=47)
    print("random action:", agent.act(dummy))
