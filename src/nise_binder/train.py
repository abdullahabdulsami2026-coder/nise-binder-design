"""Train LASEr-lite and Fold-lite on synthetic binder complexes."""

from __future__ import annotations

from pathlib import Path

import torch

from .dataset import build_dataset
from .fold import FoldLite, FoldTrainConfig, train_fold
from .laser import LaserLite, LaserTrainConfig, train_laser


def train_pair(
    n_data: int = 40,
    laser_steps: int = 80,
    fold_steps: int = 80,
    d_model: int = 48,
    helix_len: int = 14,
    seed: int = 0,
    ckpt_dir: str | Path | None = None,
) -> tuple[LaserLite, FoldLite, list]:
    poses = build_dataset(n=n_data, seed=seed, helix_len=helix_len)
    laser = LaserLite(d_model=d_model, n_encoder=2, n_decoder=1, k_neighbors=8)
    fold = FoldLite(d_model=d_model, n_layers=3, k_neighbors=8)
    laser_loss = train_laser(laser, poses, LaserTrainConfig(steps=laser_steps), seed=seed)
    fold_loss = train_fold(fold, poses, FoldTrainConfig(steps=fold_steps), seed=seed + 1)
    if ckpt_dir is not None:
        path = Path(ckpt_dir)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(laser.state_dict(), path / "laser_lite.pt")
        torch.save(fold.state_dict(), path / "fold_lite.pt")
    return laser, fold, poses
