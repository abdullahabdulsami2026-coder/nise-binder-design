from pathlib import Path

from nise_binder.pdbdata import parse_pose

MINI_PDB = """
HEADER    TEST
ATOM      1  CA  ALA A   1      0.000   0.000   0.000  1.00 20.00           C
ATOM      2  CA  ASP A   2      1.500   0.200   0.100  1.00 20.00           C
ATOM      3  CA  LEU A   3      3.000   0.000   0.000  1.00 20.00           C
HETATM    4  C1  LIG A  10      1.500   2.000   0.000  1.00 30.00           C
HETATM    5  O1  LIG A  10      1.500   3.200   0.000  1.00 30.00           O
HETATM    6  N1  LIG A  10      2.600   2.400   0.000  1.00 30.00           N
END
"""


def test_parse_mini_pdb(tmp_path: Path):
    path = tmp_path / "mini.pdb"
    path.write_text(MINI_PDB)
    pose = parse_pose(path, ligand="LIG")
    assert pose.n_res == 3
    assert pose.sequence == "ADL"
    assert pose.ligand.name == "LIG"
    assert pose.ligand.n_atoms == 3
    assert pose.ligand.polar.sum() == 2
