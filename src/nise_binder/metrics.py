"""Self-consistency and pocket-quality metrics used to filter NISE designs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .constants import POLAR
from .geometry import Pose, binding_site_mask, pairwise_min_distance, rmsd


@dataclass
class Consistency:
    bb_rmsd: float
    ligand_rmsd: float
    ligand_plddt: float
    nll: float
    buried_unsatisfied: int
    self_consistent: bool


def buried_unsatisfied_polars(pose: Pose, sequence: str, cutoff: float = 4.5) -> int:
    """Count buried polar residues that do not sit near a ligand polar atom.

    This is the ranking term the paper uses after self-consistency filters.
    """
    if pose.helix_id is None:
        buried = np.linalg.norm(pose.ca - pose.ca.mean(0), axis=1) < 6.5
    else:
        buried = pose.helix_id >= 0
        buried = buried & (np.linalg.norm(pose.ca - pose.ca.mean(0), axis=1) < np.median(
            np.linalg.norm(pose.ca - pose.ca.mean(0), axis=1)
        ))
    d = pairwise_min_distance(pose.ca, pose.ligand.xyz)
    count = 0
    for i, aa in enumerate(sequence):
        if (not buried[i]) or (aa not in POLAR and aa not in set("DEKR")):
            continue
        if d[i] > cutoff:
            count += 1
            continue
        nearest = int(np.argmin(np.linalg.norm(pose.ca[i] - pose.ligand.xyz, axis=1)))
        if not pose.ligand.polar[nearest]:
            count += 1
    return count


def evaluate(
    predicted: Pose,
    reference: Pose,
    nll: float,
    bb_cut: float = 2.5,
    lig_cut: float = 2.5,
) -> Consistency:
    bb = rmsd(predicted.ca, reference.ca, superpose=True)
    lig = rmsd(predicted.ligand.xyz, reference.ligand.xyz, superpose=True)
    plddt = float(predicted.meta.get("ligand_plddt", 0.0))
    seq = predicted.sequence or reference.sequence or ("A" * predicted.n_res)
    unsat = buried_unsatisfied_polars(predicted, seq)
    return Consistency(
        bb_rmsd=bb,
        ligand_rmsd=lig,
        ligand_plddt=plddt,
        nll=float(nll),
        buried_unsatisfied=unsat,
        self_consistent=bb < bb_cut and lig < lig_cut,
    )


def pocket_match(pose: Pose, sequence: str, cutoff: float = 6.0) -> float:
    """Fraction of pocket residues whose chemistry matches the ligand atom they contact."""
    d = pairwise_min_distance(pose.ca, pose.ligand.xyz)
    hits = total = 0
    for i, aa in enumerate(sequence):
        if d[i] >= cutoff:
            continue
        total += 1
        nearest = int(np.argmin(np.linalg.norm(pose.ca[i] - pose.ligand.xyz, axis=1)))
        elem = pose.ligand.elements[nearest]
        ok = (
            (elem == "N" and aa in "DE")
            or (elem == "O" and aa in "NQST")
            or (elem == "F" and aa in "FILM")
            or (elem in "CS" and aa in "FILMWVA")
        )
        hits += int(ok)
    return hits / max(total, 1)


def site_recovery(pred: str, native: str, pose: Pose) -> float:
    mask = binding_site_mask(pose)
    hits = sum(p == t for p, t, m in zip(pred, native, mask) if m)
    return hits / max(int(mask.sum()), 1)
