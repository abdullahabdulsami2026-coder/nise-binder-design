"""Download and parse public PDB/mmCIF protein–ligand complexes."""

from __future__ import annotations

import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .constants import ELEMENT_TO_IDX
from .geometry import Ligand, Pose, backbone_from_ca, binding_site_mask

SKIP_HET = frozenset(
    "HOH DOD WAT SO4 PO4 ACT EDO GOL PEG PG4 DMS CL NA MG ZN CA MN K FE "
    "MES TRS EPE FMT NO3 NH4 UNX".split()
)

THREE_TO_ONE = {
    "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F",
    "GLY": "G", "HIS": "H", "ILE": "I", "LYS": "K", "LEU": "L",
    "MET": "M", "ASN": "N", "PRO": "P", "GLN": "Q", "ARG": "R",
    "SER": "S", "THR": "T", "VAL": "V", "TRP": "W", "TYR": "Y",
    "MSE": "M", "SEC": "C",
}

USER_AGENT = "nise-binder/0.1 (https://github.com/abdullahabdulsami2026-coder/nise-binder-design)"


@dataclass(frozen=True)
class PublicComplex:
    pdb_id: str
    label: str
    ligand: str | None = None
    note: str = ""


# Coordinates and sequences are in the public PDB. Kd values in the notes are from papers.
PUBLIC_COMPLEXES = [
    PublicComplex("9NZE", "EPIC + exatecan", "A1B7L", "Fry et al. 2026; crystal of the NISE binder."),
    PublicComplex("9NZG", "EPIC Q51N + exatecan", "A1B7L", "Fry et al. 2026; higher-affinity mutant."),
    PublicComplex("6W70", "ABLE + apixaban", "GG2", "Polizzi & DeGrado 2020 helical-bundle binder."),
    PublicComplex("8TN6", "PiB + rucaparib", "RPB", "Lu et al. 2024 COMBS binder; held out in the NISE paper."),
    PublicComplex("4JNJ", "Streptavidin monomer + biotin", "BTN", "Paper's sequence-recovery test case."),
    PublicComplex("3PTB", "Trypsin + benzamidine", "BEN", "Classic public serine-protease complex."),
]


def cache_dir() -> Path:
    root = Path.home() / ".cache" / "nise-binder" / "pdb"
    root.mkdir(parents=True, exist_ok=True)
    return root


def fetch_structure(pdb_id: str, cache: Path | None = None) -> Path:
    """Download a PDB or mmCIF file from RCSB. Prefers legacy PDB, then mmCIF."""
    pdb_id = pdb_id.upper()
    cache = cache or cache_dir()
    for ext in ("pdb", "cif"):
        dest = cache / f"{pdb_id}.{ext}"
        if dest.exists() and dest.stat().st_size > 200:
            return dest
    last_err: Exception | None = None
    for ext in ("pdb", "cif"):
        url = f"https://files.rcsb.org/download/{pdb_id}.{ext}"
        dest = cache / f"{pdb_id}.{ext}"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=30) as resp:
                dest.write_bytes(resp.read())
            if dest.stat().st_size > 200:
                return dest
        except Exception as exc:  # noqa: BLE001 — try the other format
            last_err = exc
            continue
    raise FileNotFoundError(f"Could not download {pdb_id} from RCSB ({last_err})")


def parse_pose(path: str | Path, ligand: str | None = None, chain: str | None = None) -> Pose:
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    if Path(path).suffix.lower() == ".cif" or text.lstrip().startswith("data_"):
        atoms = _parse_cif_atoms(text)
    else:
        atoms = _parse_pdb_atoms(text)
    return _pose_from_atoms(atoms, Path(path).stem.upper(), ligand=ligand, chain=chain)


def load_public_complex(entry: PublicComplex | str, cache: Path | None = None) -> Pose:
    if isinstance(entry, str):
        match = next((c for c in PUBLIC_COMPLEXES if c.pdb_id.upper() == entry.upper()), None)
        entry = match or PublicComplex(entry.upper(), entry.upper())
    path = fetch_structure(entry.pdb_id, cache=cache)
    pose = parse_pose(path, ligand=entry.ligand)
    pose.name = f"{entry.pdb_id}_{pose.ligand.name}"
    pose.meta["pdb_id"] = entry.pdb_id
    pose.meta["public"] = True
    pose.meta["label"] = entry.label
    return pose


def load_public_set(cache: Path | None = None, ids: list[str] | None = None) -> list[Pose]:
    wanted = PUBLIC_COMPLEXES
    if ids:
        wanted = [c for c in PUBLIC_COMPLEXES if c.pdb_id in {x.upper() for x in ids}]
    poses = []
    errors = []
    for entry in wanted:
        try:
            poses.append(load_public_complex(entry, cache=cache))
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{entry.pdb_id}: {exc}")
    if not poses:
        raise RuntimeError("No public complexes could be loaded. " + "; ".join(errors))
    return poses


def _parse_pdb_atoms(text: str) -> list[dict]:
    atoms = []
    for line in text.splitlines():
        if not (line.startswith("ATOM") or line.startswith("HETATM")):
            continue
        if len(line) < 54:
            continue
        alt = line[16].strip()
        if alt not in {"", "A"}:
            continue
        name = line[12:16].strip()
        resn = line[17:20].strip()
        chain = line[21].strip() or "A"
        try:
            resi = int(line[22:26])
        except ValueError:
            continue
        x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
        elem = line[76:78].strip().upper() if len(line) >= 78 else name[0]
        elem = elem.replace("+", "").replace("-", "") or name[0]
        if elem == "D":
            elem = "H"
        atoms.append(
            {
                "het": line.startswith("HETATM"),
                "name": name,
                "resn": resn,
                "chain": chain,
                "resi": resi,
                "xyz": np.array([x, y, z], dtype=float),
                "elem": elem,
            }
        )
    return atoms


def _tokenize_cif(line: str) -> list[str]:
    tokens: list[str] = []
    buf = []
    quote = None
    for ch in line.strip():
        if quote:
            if ch == quote:
                tokens.append("".join(buf))
                buf, quote = [], None
            else:
                buf.append(ch)
            continue
        if ch in {"'", '"'}:
            quote = ch
            continue
        if ch.isspace():
            if buf:
                tokens.append("".join(buf))
                buf = []
            continue
        buf.append(ch)
    if buf:
        tokens.append("".join(buf))
    return tokens


def _parse_cif_atoms(text: str) -> list[dict]:
    lines = text.splitlines()
    start = None
    for i, line in enumerate(lines):
        if line.startswith("_atom_site.") or line.strip() == "_atom_site.group_PDB":
            # walk back to loop_
            j = i
            while j > 0 and not lines[j].startswith("loop_"):
                j -= 1
            start = j
            break
    if start is None:
        raise ValueError("No _atom_site loop in mmCIF")
    fields = []
    i = start + 1
    while i < len(lines) and lines[i].lstrip().startswith("_atom_site."):
        fields.append(lines[i].strip().split(".")[-1])
        i += 1
    col = {name: k for k, name in enumerate(fields)}
    n = len(fields)
    tokens: list[str] = []
    rows = []
    while i < len(lines):
        raw = lines[i].rstrip()
        if raw.startswith("#") or raw.startswith("loop_") or (raw.startswith("_") and not raw.startswith("_atom_site")):
            break
        if not raw.strip() or raw.strip().startswith(";"):
            i += 1
            continue
        tokens.extend(_tokenize_cif(raw))
        while len(tokens) >= n:
            row, tokens = tokens[:n], tokens[n:]
            rows.append(row)
        i += 1

    def get(row, *names, default="."):
        for name in names:
            if name in col:
                return row[col[name]]
        return default

    atoms = []
    for row in rows:
        group = get(row, "group_PDB")
        alt = get(row, "label_alt_id")
        if alt not in {".", "?", "A", ""}:
            continue
        name = get(row, "label_atom_id", "auth_atom_id").replace('"', "")
        resn = get(row, "label_comp_id", "auth_comp_id")
        chain = get(row, "auth_asym_id", "label_asym_id") or "A"
        seq = get(row, "label_seq_id", "auth_seq_id", default="0")
        try:
            resi = int(seq) if seq not in {".", "?"} else 0
        except ValueError:
            resi = 0
        try:
            xyz = np.array(
                [float(get(row, "Cartn_x")), float(get(row, "Cartn_y")), float(get(row, "Cartn_z"))],
                dtype=float,
            )
        except ValueError:
            continue
        elem = get(row, "type_symbol", default=name[:1]).upper()
        atoms.append(
            {
                "het": group == "HETATM",
                "name": name,
                "resn": resn,
                "chain": chain,
                "resi": resi,
                "xyz": xyz,
                "elem": elem,
            }
        )
    return atoms


def _pose_from_atoms(atoms: list[dict], pdb_id: str, ligand: str | None, chain: str | None) -> Pose:
    protein = [a for a in atoms if (not a["het"]) and a["name"].strip() == "CA" and a["resn"] in THREE_TO_ONE]
    if chain:
        protein = [a for a in protein if a["chain"] == chain]
    if not protein:
        raise ValueError(f"{pdb_id}: no protein CA atoms")

    het = [a for a in atoms if a["het"] and a["elem"] not in {"H", "D"} and a["resn"] not in SKIP_HET]
    lig_name = ligand
    if lig_name is None:
        counts: dict[tuple, int] = {}
        for a in het:
            key = (a["resn"], a["chain"], a["resi"])
            counts[key] = counts.get(key, 0) + 1
        if not counts:
            raise ValueError(f"{pdb_id}: no organic ligand (all hetero atoms skipped)")
        lig_name = max(counts, key=counts.get)[0]

    lig_atoms = [a for a in het if a["resn"] == lig_name]
    if not lig_atoms:
        raise ValueError(f"{pdb_id}: ligand {lig_name} not found")

    # Prefer the protein chain closest to the ligand.
    lig_xyz_all = np.stack([a["xyz"] for a in lig_atoms])
    by_chain: dict[str, list] = {}
    for a in protein:
        by_chain.setdefault(a["chain"], []).append(a)
    best_chain, best_d = next(iter(by_chain)), np.inf
    for ch, cas in by_chain.items():
        ca = np.stack([a["xyz"] for a in cas])
        dmin = float(np.linalg.norm(ca[:, None, :] - lig_xyz_all[None, :, :], axis=-1).min())
        if dmin < best_d:
            best_chain, best_d = ch, dmin
    if chain:
        best_chain = chain
        if best_chain not in by_chain:
            raise ValueError(f"{pdb_id}: chain {chain} not found")
    protein = sorted(by_chain[best_chain], key=lambda a: a["resi"])
    # ligand copy on the same chain if present
    same = [a for a in lig_atoms if a["chain"] == best_chain]
    if same:
        lig_atoms = same
    # one residue instance
    first_resi = lig_atoms[0]["resi"]
    first_chain = lig_atoms[0]["chain"]
    lig_atoms = [a for a in lig_atoms if a["resi"] == first_resi and a["chain"] == first_chain]

    ca = np.stack([a["xyz"] for a in protein])
    seq = "".join(THREE_TO_ONE[a["resn"]] for a in protein)
    elements = []
    for a in lig_atoms:
        e = a["elem"] if a["elem"] in ELEMENT_TO_IDX else a["elem"][:1]
        if e == "D":
            e = "H"
        if e not in ELEMENT_TO_IDX:
            e = "C"
        elements.append(e)
    ligand_obj = Ligand(
        name=lig_name,
        xyz=np.stack([a["xyz"] for a in lig_atoms]),
        elements=tuple(elements),
        polar=np.array([e in {"N", "O", "F"} for e in elements]),
        comments=f"{pdb_id} public PDB ligand {lig_name}",
    )
    n, c, o = backbone_from_ca(ca)
    pose = Pose(
        ca=ca,
        ligand=ligand_obj,
        sequence=seq,
        n=n,
        c=c,
        o=o,
        helix_id=np.zeros(len(seq), dtype=int),
        name=f"{pdb_id}_{lig_name}",
        meta={"pdb_id": pdb_id, "chain": best_chain, "ligand": lig_name},
    )
    pose.meta["binding_site"] = binding_site_mask(pose)
    return pose
