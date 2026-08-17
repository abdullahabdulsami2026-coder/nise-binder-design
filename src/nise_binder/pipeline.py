"""End-to-end demo used by the CLI and the Streamlit app."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .assays import fit_hydrolysis, fit_kd, synthetic_titration, two_state_open_fraction
from .constants import PAPER
from .dataset import make_complex
from .geometry import ligand_by_name, to_pdb, write_fasta, write_pdb
from .metrics import pocket_match, site_recovery
from .nise import NISEConfig, NISEResult, apply_mutations, energy_ise, nise, proofread
from .pdbdata import load_public_complex, load_public_set
from .plots import hydrolysis_paper, kd_curves, trajectories
from .train import bundled_weights_available, load_pair, train_pair


@dataclass
class DemoConfig:
    outdir: str = "results"
    steps: int = 80
    n_data: int = 24
    rounds: int = 5
    n_expand: int = 12
    n_select: int = 2
    d_model: int = 48
    helix_len: int = 14
    seed: int = 0
    ligand: str = "exatecan-like"
    temperature: float = 0.85
    write_files: bool = True
    source: str = "synthetic"  # or "public"
    pdb_id: str = "9NZE"
    use_pretrained: bool = True


@dataclass
class DemoResult:
    native_name: str
    n_residues: int
    ligand: str
    source: str
    table: list[dict]
    mutations: list[dict]
    pdbs: dict[str, str]
    fasta: str
    proofread_fasta: str
    nise_history: list[dict]
    energy_history: list[dict]
    kd_demo_nM: dict[str, float]
    hydrolysis_t_half_h: float
    outdir: str | None = None
    nise: NISEResult | None = field(default=None, repr=False)
    baseline: NISEResult | None = field(default=None, repr=False)
    native: object | None = field(default=None, repr=False)
    designs: list = field(default_factory=list, repr=False)
    laser: object | None = field(default=None, repr=False)


def _fasta(records: list[tuple[str, str]]) -> str:
    chunks = []
    for name, seq in records:
        chunks.append(f">{name}")
        for i in range(0, len(seq), 80):
            chunks.append(seq[i : i + 80])
    return "\n".join(chunks) + "\n"


def run_demo(config: DemoConfig | None = None) -> DemoResult:
    cfg = config or DemoConfig()
    out = Path(cfg.outdir)
    if cfg.write_files:
        out.mkdir(parents=True, exist_ok=True)
        ckpt = out / "checkpoints"
    else:
        ckpt = None

    if cfg.source == "public":
        public_poses = load_public_set()
        native = load_public_complex(cfg.pdb_id)
        if native.meta.get("pdb_id") not in {p.meta.get("pdb_id") for p in public_poses}:
            public_poses = [*public_poses, native]
        train_poses = public_poses
    else:
        native = make_complex(cfg.seed, ligand=ligand_by_name(cfg.ligand), helix_len=cfg.helix_len)
        train_poses = None

    if cfg.use_pretrained and bundled_weights_available():
        laser, fold = load_pair()
    elif cfg.source == "public":
        laser, fold, _trained = train_pair(
            laser_steps=cfg.steps,
            fold_steps=cfg.steps,
            d_model=cfg.d_model,
            seed=cfg.seed,
            ckpt_dir=ckpt,
            poses=train_poses,
        )
    else:
        laser, fold, _trained = train_pair(
            n_data=cfg.n_data,
            laser_steps=cfg.steps,
            fold_steps=cfg.steps,
            d_model=cfg.d_model,
            helix_len=cfg.helix_len,
            seed=cfg.seed,
            ckpt_dir=ckpt,
        )
    start = native.copy()
    start.sequence = None

    result = nise(
        laser,
        fold,
        start,
        NISEConfig(
            rounds=cfg.rounds,
            n_expand=cfg.n_expand,
            n_select=cfg.n_select,
            temperature=cfg.temperature,
            seed=cfg.seed,
        ),
    )
    baseline = energy_ise(
        laser,
        start,
        rounds=cfg.rounds,
        n_expand=cfg.n_expand,
        n_select=cfg.n_select,
        seed=cfg.seed + 7,
        fold=fold,
    )

    designs = result.designs[:4] or result.all_kept[:4]
    fasta_records = []
    pdbs: dict[str, str] = {"native": to_pdb(native, native.sequence)}
    table = []
    for i, cand in enumerate(designs, start=1):
        name = f"design_{i}"
        fasta_records.append((name, cand.sequence))
        pdbs[name] = to_pdb(cand.pose, cand.sequence)
        table.append(
            {
                "name": name,
                "ligand_plddt": round(cand.scores.ligand_plddt, 2),
                "nll": round(cand.scores.nll, 3),
                "bb_rmsd": round(cand.scores.bb_rmsd, 3),
                "ligand_rmsd": round(cand.scores.ligand_rmsd, 3),
                "buried_unsatisfied": cand.scores.buried_unsatisfied,
                "binding_site_recovery": round(site_recovery(cand.sequence, native.sequence, native), 3),
                "pocket_chemistry": round(pocket_match(cand.pose, cand.sequence), 3),
                "sequence": cand.sequence,
            }
        )

    mutations = []
    mut_objs = []
    proofread_fasta = ""
    if designs:
        mut_objs = proofread(laser, designs[0].pose, designs[0].sequence, top_k=6)
        mutations = [
            {
                "position_1based": m.position + 1,
                "mutation": f"{m.from_aa}{m.position + 1}{m.to_aa}",
                "delta_nll": round(m.delta_nll, 3),
            }
            for m in mut_objs
        ]
        double = apply_mutations(designs[0].sequence, mut_objs[:2]) if mut_objs else designs[0].sequence
        proofread_fasta = _fasta([("best", designs[0].sequence), ("proofread_top2", double)])

    ligand_conc = 5e-8
    curves = []
    seeds = {"NISE-like": 1, "proofread": 2, "COMBS-like": 3, "HSA": 4}
    for label, kd in [
        ("NISE-like", PAPER["exatecan"]["epic_kd_uM"] * 1e-6),
        ("proofread", PAPER["exatecan"]["epic_double_kd_nM"] * 1e-9),
        ("COMBS-like", PAPER["exatecan"]["combs_best_kd_uM"] * 1e-6),
        ("HSA", PAPER["exatecan"]["hsa_kd_uM"] * 1e-6),
    ]:
        p, y = synthetic_titration(kd, ligand=ligand_conc, seed=seeds[label])
        fit = fit_kd(p, y, ligand=ligand_conc, n_boot=80, seed=seeds[label])
        curves.append((label, p, y, fit, ligand_conc))

    t = np.linspace(0, 12, 40)
    hyd = fit_hydrolysis(t, two_state_open_fraction(t, 0.35, 0.06))

    if cfg.write_files:
        _plot_and_write(out, native, designs, fasta_records, result, baseline, curves, mut_objs)

    demo = DemoResult(
        native_name=native.name,
        n_residues=native.n_res,
        ligand=native.ligand.name,
        source=cfg.source,
        table=table,
        mutations=mutations,
        pdbs=pdbs,
        fasta=_fasta(fasta_records),
        proofread_fasta=proofread_fasta,
        nise_history=[h.__dict__ for h in result.history],
        energy_history=[h.__dict__ for h in baseline.history],
        kd_demo_nM={name: fit.kd * 1e9 for name, _, __, fit, ___ in curves},
        hydrolysis_t_half_h=hyd.t_half_h,
        outdir=str(out) if cfg.write_files else None,
        nise=result,
        baseline=baseline,
        native=native,
        designs=designs,
        laser=laser,
    )
    if cfg.write_files:
        (out / "summary.json").write_text(json.dumps(_summary(demo), indent=2), encoding="utf-8")
    return demo


def _summary(demo: DemoResult) -> dict:
    return {
        "paper": PAPER,
        "native": demo.native_name,
        "n_residues": demo.n_residues,
        "ligand": demo.ligand,
        "source": demo.source,
        "nise_history": demo.nise_history,
        "energy_history": demo.energy_history,
        "designs": [{k: v for k, v in row.items() if k != "sequence"} for row in demo.table],
        "proofread": demo.mutations,
        "kd_demo_nM": demo.kd_demo_nM,
        "hydrolysis_t_half_h": demo.hydrolysis_t_half_h,
        "note": (
            "Public mode uses RCSB coordinates and crystal sequences. "
            "Synthetic mode uses parametric helical bundles. "
            "Neither uses the official LASErMPNN weights; experimental EPIC/APEX affinities are published numbers, not this demo."
        ),
    }


def _plot_and_write(out, native, designs, fasta_records, result, baseline, curves, mut_objs) -> None:
    import matplotlib.pyplot as plt

    fig = trajectories(result.history, baseline.history)
    fig.savefig(out / "nise_vs_energy.png", dpi=160)
    plt.close(fig)
    fig = kd_curves(curves)
    fig.savefig(out / "kd_fits.png", dpi=160)
    plt.close(fig)
    fig = hydrolysis_paper()
    fig.savefig(out / "hydrolysis.png", dpi=160)
    plt.close(fig)
    write_pdb(out / "native.pdb", native, native.sequence)
    for i, cand in enumerate(designs, start=1):
        write_pdb(out / f"design_{i}.pdb", cand.pose, cand.sequence)
    write_fasta(out / "designs.fasta", fasta_records)
    if mut_objs:
        double = apply_mutations(designs[0].sequence, mut_objs[:2])
        write_fasta(out / "proofread.fasta", [("best", designs[0].sequence), ("proofread_top2", double)])
