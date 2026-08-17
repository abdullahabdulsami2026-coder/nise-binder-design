from pathlib import Path

import numpy as np

from nise_binder.geometry import (
    exatecan_like,
    four_helix_bundle,
    kabsch,
    rmsd,
    write_pdb,
)


def test_identical_rmsd_is_zero():
    pose = four_helix_bundle(helix_len=10, loop_len=2, seed=1)
    assert rmsd(pose.ca, pose.ca) < 1e-6


def test_kabsch_recovers_rotation():
    rng = np.random.default_rng(0)
    P = rng.normal(size=(20, 3))
    R = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    Q = P @ R.T + np.array([3.0, -2.0, 1.0])
    aligned, value = kabsch(P, Q)
    assert value < 1e-6
    assert np.allclose(aligned, Q, atol=1e-6)


def test_bundle_is_single_chain_with_pocket():
    pose = four_helix_bundle(helix_len=12, loop_len=3, seed=2, ligand=exatecan_like())
    assert pose.n_res == 12 * 4 + 3 * 3
    assert pose.ligand.n_atoms > 8
    d = np.linalg.norm(pose.ca.mean(0) - pose.ligand.xyz.mean(0))
    assert d < 6.0


def test_write_pdb(tmp_path: Path):
    pose = four_helix_bundle(helix_len=8, loop_len=2, seed=3)
    path = tmp_path / "bundle.pdb"
    write_pdb(str(path), pose, "A" * pose.n_res)
    text = path.read_text()
    assert "ATOM" in text and "HETATM" in text and "LIG" in text
