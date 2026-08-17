"""Train LASEr-lite and Fold-lite on synthetic binder complexes."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from .dataset import build_dataset
from .fold import FoldLite, FoldTrainConfig, train_fold
from .geometry import Pose
from .laser import LaserLite, LaserTrainConfig, train_laser

WEIGHTS_DIR = Path(__file__).parent / "weights"
MODEL_SPEC = {
    "d_model": 48,
    "n_encoder": 2,
    "n_decoder": 1,
    "k_neighbors": 8,
    "n_rbf": 16,
    "n_layers": 3,
}


def build_models(d_model: int | None = None) -> tuple[LaserLite, FoldLite]:
    spec = {**MODEL_SPEC, "d_model": d_model or MODEL_SPEC["d_model"]}
    laser = LaserLite(
        d_model=spec["d_model"],
        n_encoder=spec["n_encoder"],
        n_decoder=spec["n_decoder"],
        k_neighbors=spec["k_neighbors"],
        n_rbf=spec["n_rbf"],
    )
    fold = FoldLite(
        d_model=spec["d_model"],
        n_layers=spec["n_layers"],
        k_neighbors=spec["k_neighbors"],
        n_rbf=spec["n_rbf"],
    )
    return laser, fold


def save_pair(laser: LaserLite, fold: FoldLite, ckpt_dir: str | Path) -> Path:
    path = Path(ckpt_dir)
    path.mkdir(parents=True, exist_ok=True)
    torch.save(laser.state_dict(), path / "laser_lite.pt")
    torch.save(fold.state_dict(), path / "fold_lite.pt")
    (path / "spec.json").write_text(json.dumps(MODEL_SPEC, indent=2), encoding="utf-8")
    return path


def load_pair(ckpt_dir: str | Path | None = None) -> tuple[LaserLite, FoldLite]:
    path = Path(ckpt_dir) if ckpt_dir is not None else WEIGHTS_DIR
    laser, fold = build_models()
    map_loc = torch.device("cpu")
    try:
        laser.load_state_dict(torch.load(path / "laser_lite.pt", map_location=map_loc, weights_only=True))
        fold.load_state_dict(torch.load(path / "fold_lite.pt", map_location=map_loc, weights_only=True))
    except TypeError:
        laser.load_state_dict(torch.load(path / "laser_lite.pt", map_location=map_loc))
        fold.load_state_dict(torch.load(path / "fold_lite.pt", map_location=map_loc))
    laser.eval()
    fold.eval()
    return laser, fold


def bundled_weights_available() -> bool:
    return (WEIGHTS_DIR / "laser_lite.pt").exists() and (WEIGHTS_DIR / "fold_lite.pt").exists()


def train_pair(
    n_data: int = 40,
    laser_steps: int = 80,
    fold_steps: int = 80,
    d_model: int = 48,
    helix_len: int = 14,
    seed: int = 0,
    ckpt_dir: str | Path | None = None,
    poses: list[Pose] | None = None,
) -> tuple[LaserLite, FoldLite, list]:
    poses = poses if poses is not None else build_dataset(n=n_data, seed=seed, helix_len=helix_len)
    laser, fold = build_models(d_model=d_model)
    train_laser(laser, poses, LaserTrainConfig(steps=laser_steps), seed=seed)
    train_fold(fold, poses, FoldTrainConfig(steps=fold_steps), seed=seed + 1)
    if ckpt_dir is not None:
        save_pair(laser, fold, ckpt_dir)
    return laser, fold, poses
