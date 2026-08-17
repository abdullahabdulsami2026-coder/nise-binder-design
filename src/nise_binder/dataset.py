"""Synthetic protein–ligand complexes with rule-based native sequences."""

from __future__ import annotations

import numpy as np

from .constants import AA, HYDROPHOBIC_CONTACT, SURFACE
from .geometry import (
    Ligand,
    Pose,
    apixaban_like,
    binding_site_mask,
    burial,
    exatecan_like,
    four_helix_bundle,
    pairwise_min_distance,
)


def assign_native_sequence(pose: Pose, rng: np.random.Generator | None = None) -> str:
    """Design a plausible native sequence from pocket geometry.

    Binding-site residues match ligand chemistry; the core is hydrophobic;
    the surface is polar. A little mutation noise keeps the distribution
    from being a deterministic lookup table.
    """
    rng = rng or np.random.default_rng()
    d_lig = pairwise_min_distance(pose.ca, pose.ligand.xyz)
    buried = burial(pose.ca)

    seq = []
    for i in range(pose.n_res):
        nearest = int(np.argmin(np.linalg.norm(pose.ca[i] - pose.ligand.xyz, axis=1)))
        elem = pose.ligand.elements[nearest]
        if d_lig[i] < 6.0:
            if elem == "N":
                alphabet = tuple("DE")
            elif elem == "O":
                alphabet = tuple("NQST")
            elif elem == "F":
                alphabet = tuple("FILM")
            else:
                alphabet = HYDROPHOBIC_CONTACT
        elif buried[i] > 0.58:
            alphabet = tuple("AILV")
        else:
            alphabet = SURFACE
        aa = rng.choice(alphabet)
        if rng.random() < 0.03:
            aa = rng.choice(list(AA))
        seq.append(aa)
    return "".join(seq)


def make_complex(
    seed: int,
    ligand: Ligand | None = None,
    helix_len: int = 16,
    loop_len: int = 3,
) -> Pose:
    rng = np.random.default_rng(seed)
    lig = ligand or (exatecan_like() if seed % 2 == 0 else apixaban_like())
    pose = four_helix_bundle(
        helix_len=helix_len,
        loop_len=loop_len,
        radius=float(rng.uniform(7.1, 7.7)),
        supercoil=float(rng.uniform(0.02, 0.07)),
        seed=seed,
        ligand=lig,
        ligand_jitter=0.18,
    )
    pose.sequence = assign_native_sequence(pose, rng)
    pose.meta["binding_site"] = binding_site_mask(pose)
    pose.name = f"{lig.name}_{seed}"
    return pose


def build_dataset(n: int = 48, seed: int = 0, helix_len: int = 16) -> list[Pose]:
    poses = [make_complex(seed + i, helix_len=helix_len) for i in range(n)]
    return poses
