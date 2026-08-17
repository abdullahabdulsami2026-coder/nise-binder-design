"""Physics-based iterative selection–expansion baseline (energy ISE)."""

from __future__ import annotations

import numpy as np

from .geometry import Pose, pairwise_min_distance


def _lj(r: np.ndarray, sigma: float = 3.4, eps: float = 0.2) -> np.ndarray:
    x = np.clip(sigma / np.clip(r, 1.4, None), 0, 3.0)
    return 4 * eps * (x**12 - x**6)


def ligand_energy(pose: Pose) -> float:
    """Simple protein–ligand van der Waals + polar contact score (Rosetta-like stand-in)."""
    d = np.linalg.norm(pose.ca[:, None, :] - pose.ligand.xyz[None, :, :], axis=-1)
    vdw = float(_lj(d).sum())
    polar_bonus = 0.0
    if pose.sequence is not None:
        from .constants import NEGATIVE, POLAR, POSITIVE

        for i, aa in enumerate(pose.sequence):
            near = d[i] < 5.0
            if not np.any(near):
                continue
            lig_polar = pose.ligand.polar[near]
            if aa in NEGATIVE | POSITIVE | POLAR:
                polar_bonus -= 0.6 * float(lig_polar.sum())
            else:
                polar_bonus += 0.15 * float(lig_polar.sum())
    clash = float(np.maximum(2.4 - pairwise_min_distance(pose.ca, pose.ligand.xyz), 0).sum())
    return vdw + 3.0 * clash + polar_bonus


def minimize_pose(pose: Pose, steps: int = 25, lr: float = 0.05) -> Pose:
    """Move the ligand (and slightly the backbone) to lower ligand_energy."""
    out = pose.copy()
    rng = np.random.default_rng(0)
    best = out.copy()
    best_e = ligand_energy(best)
    for _ in range(steps):
        trial = best.copy()
        trial.ligand.xyz = best.ligand.xyz + rng.normal(0, lr, size=best.ligand.xyz.shape)
        trial.ca = best.ca + rng.normal(0, 0.15 * lr, size=best.ca.shape)
        e = ligand_energy(trial)
        if e < best_e:
            best, best_e = trial, e
    best.meta = dict(best.meta)
    best.meta["ligand_energy"] = float(best_e)
    return best
