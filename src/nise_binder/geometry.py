"""Geometry, helical-bundle scaffolds, toy ligands, and PDB I/O."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Iterable

import numpy as np

from .constants import ELEMENT_CHARGE, ELEMENT_TO_IDX, IDX_TO_ELEMENT

EPS = 1e-8


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / (n + EPS)


def orthonormal_basis(axis: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    z = _unit(np.asarray(axis, dtype=float))
    helper = np.array([1.0, 0.0, 0.0]) if abs(z[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    x = _unit(np.cross(z, helper))
    y = np.cross(z, x)
    return x, y, z


def kabsch(P: np.ndarray, Q: np.ndarray) -> tuple[np.ndarray, float]:
    """Rotate/translate P onto Q. Returns aligned P and RMSD."""
    P = np.asarray(P, dtype=float)
    Q = np.asarray(Q, dtype=float)
    p0 = P.mean(axis=0)
    q0 = Q.mean(axis=0)
    A = (P - p0).T @ (Q - q0)
    U, _, Vt = np.linalg.svd(A)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = Vt.T @ U.T
    aligned = (P - p0) @ R.T + q0
    rmsd = float(np.sqrt(((aligned - Q) ** 2).sum(axis=1).mean()))
    return aligned, rmsd


def rmsd(P: np.ndarray, Q: np.ndarray, superpose: bool = True) -> float:
    if superpose:
        _, value = kabsch(P, Q)
        return value
    return float(np.sqrt(((P - Q) ** 2).sum(axis=1).mean()))


def rbf(distances: np.ndarray, d_min: float = 0.0, d_max: float = 20.0, n_bins: int = 16) -> np.ndarray:
    centers = np.linspace(d_min, d_max, n_bins)
    sigma = (d_max - d_min) / n_bins
    d = np.asarray(distances, dtype=float)[..., None]
    return np.exp(-((d - centers) ** 2) / (2 * sigma**2 + EPS))


def knn_indices(xyz: np.ndarray, k: int) -> np.ndarray:
    d = np.linalg.norm(xyz[:, None, :] - xyz[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    k = min(k, len(xyz) - 1)
    return np.argpartition(d, kth=k, axis=1)[:, :k]


def pairwise_min_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1).min(axis=1)


@dataclass
class Ligand:
    name: str
    xyz: np.ndarray
    elements: tuple[str, ...]
    polar: np.ndarray
    comments: str = ""

    @property
    def n_atoms(self) -> int:
        return int(self.xyz.shape[0])

    @property
    def elem_idx(self) -> np.ndarray:
        return np.array([ELEMENT_TO_IDX[e] for e in self.elements], dtype=np.int64)

    @property
    def charges(self) -> np.ndarray:
        return np.array([ELEMENT_CHARGE[e] for e in self.elements], dtype=float)

    def transformed(self, rotation: np.ndarray, translation: np.ndarray) -> "Ligand":
        return replace(self, xyz=self.xyz @ rotation.T + translation)

    def copy(self) -> "Ligand":
        return replace(self, xyz=self.xyz.copy(), polar=self.polar.copy())


def _ring(n: int, radius: float, z: float, phase: float = 0.0) -> np.ndarray:
    ang = phase + np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.stack([radius * np.cos(ang), radius * np.sin(ang), np.full(n, z)], axis=1)


def exatecan_like() -> Ligand:
    """Planar pentacyclic-ish pharmacophore: aromatic core, lactone, amine, fluorine."""
    core = _ring(6, 1.4, 0.0)
    fused = _ring(6, 1.4, 0.0, phase=0.4) + np.array([2.2, 0.3, 0.0])
    lactone = np.array(
        [
            [3.8, 0.2, 0.2],  # C
            [4.6, 0.8, 0.6],  # O carbonyl
            [4.2, -0.9, -0.2],  # O ring
        ]
    )
    extras = np.array(
        [
            [-1.8, 0.4, 0.1],  # amine N
            [-2.4, 1.5, 0.4],  # F
            [1.1, 2.4, 0.0],  # methyl C
        ]
    )
    xyz = np.concatenate([core, fused[:4], lactone, extras], axis=0)
    elements = tuple(["C"] * 6 + ["C"] * 4 + ["C", "O", "O", "N", "F", "C"])
    polar = np.array([e in {"N", "O", "F"} for e in elements])
    return Ligand(
        name="exatecan-like",
        xyz=xyz,
        elements=elements,
        polar=polar,
        comments="Toy camptothecin-class pharmacophore (aromatic + lactone + amine + F).",
    )


def apixaban_like() -> Ligand:
    """Elongated two-ring anticoagulant-like pharmacophore."""
    ring_a = _ring(6, 1.39, 0.0)
    ring_b = _ring(6, 1.39, 0.2) + np.array([4.6, 0.0, 0.0])
    linker = np.array(
        [
            [2.3, 0.1, 0.1],  # C
            [2.9, 1.3, 0.4],  # N
            [5.8, 1.6, 0.6],  # carboxamide C
            [6.7, 1.9, 1.1],  # O
            [5.4, 2.5, 0.1],  # N
        ]
    )
    xyz = np.concatenate([ring_a[:5], ring_b[:5], linker], axis=0)
    elements = tuple(["C"] * 5 + ["C"] * 4 + ["N"] + ["C", "N", "C", "O", "N"])
    polar = np.array([e in {"N", "O"} for e in elements])
    return Ligand(
        name="apixaban-like",
        xyz=xyz,
        elements=elements,
        polar=polar,
        comments="Toy factor-Xa-inhibitor-like pharmacophore.",
    )


def ligand_by_name(name: str) -> Ligand:
    key = name.lower().replace("_", "-")
    if key.startswith("exatecan"):
        return exatecan_like()
    if key.startswith("apixaban"):
        return apixaban_like()
    raise KeyError(f"Unknown ligand '{name}'. Use 'exatecan-like' or 'apixaban-like'.")


@dataclass
class Pose:
    """A protein backbone plus a docked ligand."""

    ca: np.ndarray
    ligand: Ligand
    sequence: str | None = None
    n: np.ndarray | None = None
    c: np.ndarray | None = None
    o: np.ndarray | None = None
    helix_id: np.ndarray | None = None
    name: str = "design"
    meta: dict = field(default_factory=dict)

    @property
    def n_res(self) -> int:
        return int(self.ca.shape[0])

    def with_sequence(self, sequence: str) -> "Pose":
        if len(sequence) != self.n_res:
            raise ValueError("sequence length must match backbone")
        return replace(self, sequence=sequence)

    def copy(self) -> "Pose":
        return replace(
            self,
            ca=self.ca.copy(),
            ligand=self.ligand.copy(),
            n=None if self.n is None else self.n.copy(),
            c=None if self.c is None else self.c.copy(),
            o=None if self.o is None else self.o.copy(),
            helix_id=None if self.helix_id is None else self.helix_id.copy(),
            meta=dict(self.meta),
        )


def alpha_helix_ca(
    n_res: int,
    origin: np.ndarray,
    axis: np.ndarray,
    phase: float = 0.0,
    radius: float = 2.3,
    rise: float = 1.5,
    turn: float = np.deg2rad(100.0),
) -> np.ndarray:
    x, y, z = orthonormal_basis(axis)
    origin = np.asarray(origin, dtype=float)
    coords = []
    for i in range(n_res):
        ang = phase + i * turn
        coords.append(origin + i * rise * z + radius * (np.cos(ang) * x + np.sin(ang) * y))
    return np.stack(coords)


def _loop_ca(start: np.ndarray, end: np.ndarray, n: int, bulge: float = 3.2) -> np.ndarray:
    if n <= 0:
        return np.zeros((0, 3))
    t = np.linspace(0.0, 1.0, n + 2)[1:-1]
    mid = (start + end) / 2.0
    outward = _unit(np.array([mid[0], mid[1], 0.0]) + EPS)
    pts = []
    for ti in t:
        pts.append((1 - ti) * start + ti * end + bulge * np.sin(np.pi * ti) * outward)
    return np.stack(pts)


def four_helix_bundle(
    helix_len: int = 18,
    loop_len: int = 4,
    radius: float = 7.4,
    supercoil: float = 0.04,
    seed: int | None = 0,
    ligand: Ligand | None = None,
    ligand_jitter: float = 0.0,
) -> Pose:
    """Parametric four-helix bundle with a central pocket, as used for EPIC-style designs."""
    rng = np.random.default_rng(seed)
    r = radius + rng.normal(0, 0.12)
    height = helix_len * 1.5
    centers = [
        np.array([r, 0.0, 0.0]),
        np.array([0.0, r, 0.0]),
        np.array([-r, 0.0, 0.0]),
        np.array([0.0, -r, 0.0]),
    ]
    axes = [
        np.array([supercoil, 0.0, 1.0]),
        np.array([0.0, supercoil, -1.0]),
        np.array([-supercoil, 0.0, 1.0]),
        np.array([0.0, -supercoil, -1.0]),
    ]
    origins = [
        centers[0] + np.array([0.0, 0.0, 0.0]),
        centers[1] + np.array([0.0, 0.0, height]),
        centers[2] + np.array([0.0, 0.0, 0.0]),
        centers[3] + np.array([0.0, 0.0, height]),
    ]
    helices = []
    helix_id = []
    for h, (origin, axis) in enumerate(zip(origins, axes)):
        phase = rng.uniform(0, 0.6)
        helices.append(alpha_helix_ca(helix_len, origin, axis, phase=phase))
        helix_id.append(np.full(helix_len, h, dtype=int))

    ca_parts = [helices[0]]
    hid_parts = [helix_id[0]]
    for h in range(3):
        loop = _loop_ca(helices[h][-1], helices[h + 1][0], loop_len)
        ca_parts += [loop, helices[h + 1]]
        hid_parts += [np.full(len(loop), -1, dtype=int), helix_id[h + 1]]
    ca = np.concatenate(ca_parts, axis=0)
    hid = np.concatenate(hid_parts, axis=0)

    lig = (ligand or exatecan_like()).copy()
    # Drop the ligand into the pocket, slightly off-center like a COMBS dock.
    centroid = ca.mean(axis=0)
    lig.xyz = lig.xyz - lig.xyz.mean(axis=0) + centroid + np.array([0.15, -0.1, 0.2])
    if ligand_jitter > 0:
        rot_axis = _unit(rng.normal(size=3))
        angle = rng.normal(0, ligand_jitter)
        x, y, z = orthonormal_basis(rot_axis)
        K = np.stack([x, y, z]).T
        # small Rodrigues rotation around random axis
        c, s = np.cos(angle), np.sin(angle)
        R = np.eye(3) * c + s * np.array(
            [
                [0, -rot_axis[2], rot_axis[1]],
                [rot_axis[2], 0, -rot_axis[0]],
                [-rot_axis[1], rot_axis[0], 0],
            ]
        ) + (1 - c) * np.outer(rot_axis, rot_axis)
        lig.xyz = (lig.xyz - centroid) @ R.T + centroid + rng.normal(0, 0.15, size=3)
        _ = K  # kept for basis construction; rotation uses Rodrigues

    n, c, o = backbone_from_ca(ca)
    return Pose(
        ca=ca,
        ligand=lig,
        n=n,
        c=c,
        o=o,
        helix_id=hid,
        name=f"bundle_{lig.name}",
        meta={"radius": float(r), "helix_len": helix_len, "loop_len": loop_len},
    )


def backbone_from_ca(ca: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Approximate N/C/O from a CA trace (good enough for a lite GNN)."""
    L = len(ca)
    n = np.zeros_like(ca)
    c = np.zeros_like(ca)
    o = np.zeros_like(ca)
    for i in range(L):
        prev = ca[i - 1] if i > 0 else ca[i] - (ca[min(i + 1, L - 1)] - ca[i])
        nxt = ca[i + 1] if i < L - 1 else ca[i] + (ca[i] - ca[i - 1])
        n[i] = ca[i] + 0.52 * _unit(prev - ca[i]) + np.array([0.0, 0.0, 0.15])
        c[i] = ca[i] + 0.55 * _unit(nxt - ca[i])
        binormal = _unit(np.cross(nxt - ca[i], prev - ca[i]) + np.array([0.05, 0.0, 0.0]))
        o[i] = c[i] + 1.23 * binormal
    return n, c, o


def burial(ca: np.ndarray) -> np.ndarray:
    centroid = ca.mean(axis=0)
    d = np.linalg.norm(ca - centroid, axis=1)
    return (d.max() - d) / (d.max() - d.min() + EPS)


def binding_site_mask(pose: Pose, cutoff: float = 6.0) -> np.ndarray:
    return pairwise_min_distance(pose.ca, pose.ligand.xyz) < cutoff


def noise_frames(ca: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """Noise whole residue frames together (LASErMPNN-style), not atoms independently."""
    shift = rng.normal(0, sigma, size=ca.shape)
    return ca + shift


def write_pdb(path: str, pose: Pose, sequence: str | None = None) -> None:
    seq = sequence or pose.sequence or ("G" * pose.n_res)
    lines = [f"HEADER    NISE design {pose.name}", f"REMARK    ligand {pose.ligand.name}"]
    atom = 1
    for i, xyz in enumerate(pose.ca, start=1):
        aa = seq[i - 1]
        lines.append(
            f"ATOM  {atom:5d}  CA  {aa:>3s} A{i:4d}    "
            f"{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}  1.00 20.00           C"
        )
        atom += 1
    for e, xyz in zip(pose.ligand.elements, pose.ligand.xyz):
        lines.append(
            f"HETATM{atom:5d}  {e:<2s}  LIG L{1:4d}    "
            f"{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}  1.00 40.00          {e:>2s}"
        )
        atom += 1
    lines.append("END")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def write_fasta(path: str, records: Iterable[tuple[str, str]]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for name, seq in records:
            handle.write(f">{name}\n")
            for i in range(0, len(seq), 80):
                handle.write(seq[i : i + 80] + "\n")
