"""Command-line interface."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from . import __version__
from .assays import (
    anisotropy_model,
    fit_hydrolysis,
    fit_kd,
    protected_open_fraction,
    synthetic_titration,
    two_state_open_fraction,
)
from .constants import PAPER
from .geometry import write_fasta, write_pdb
from .metrics import pocket_match, site_recovery
from .nise import NISEConfig, apply_mutations, energy_ise, nise, proofread
from .train import train_pair


def _plot_trajectories(nise_hist, energy_hist, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6), sharex=True)
    for ax, hist, title in (
        (axes[0], nise_hist, "NISE (neural fold)"),
        (axes[1], energy_hist, "Energy ISE (physics baseline)"),
    ):
        it = [h.iteration for h in hist]
        ax.plot(it, [h.q3_plddt for h in hist], color="#c0392b", lw=2, label="ligand confidence Q3")
        ax.set_ylabel("ligand confidence", color="#c0392b")
        ax2 = ax.twinx()
        ax2.plot(it, [h.q1_nll for h in hist], color="#2471a3", lw=2, label="sequence NLL Q1")
        ax2.set_ylabel("sequence NLL", color="#2471a3")
        ax.set_title(title)
        ax.set_xlabel("iteration")
        ax.spines["top"].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_kd(fits: list[tuple[str, np.ndarray, np.ndarray, object, float]], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    colors = ["#1f6feb", "#9b59b6", "#e67e22", "#27ae60"]
    for color, (name, p, y, fit, ligand) in zip(colors, fits):
        ax.scatter(p * 1e6, y, s=22, color=color, label=f"{name}  Kd={fit.kd*1e9:.2g} nM")
        grid = np.logspace(np.log10(p.min()), np.log10(p.max()), 120)
        ax.plot(grid * 1e6, anisotropy_model(grid, fit.kd, fit.a_free, fit.a_bound, ligand), color=color, lw=1.6)
    ax.set_xscale("log")
    ax.set_xlabel("protein (µM)")
    ax.set_ylabel("anisotropy")
    ax.set_title("1:1 fluorescence-anisotropy Kd fits")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_hydrolysis(path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = np.linspace(0, 50, 200)
    free = two_state_open_fraction(t, k_h=0.35, k_c=0.06)
    hsa = protected_open_fraction(t, 0.35, 0.06, kd=43e-6, protein=500e-6, ligand=20e-6, protection=1.2)
    epic = protected_open_fraction(t, 0.35, 0.06, kd=1.2e-9, protein=20e-6, ligand=20e-6, protection=200.0)
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    ax.plot(t, free, label="free exatecan", color="#c0392b", lw=2)
    ax.plot(t, hsa, label="HSA (500 µM)", color="#7f8c8d", lw=2)
    ax.plot(t, epic, label="EPIC-like binder (20 µM)", color="#1f6feb", lw=2)
    ax.set_xlabel("time (h)")
    ax.set_ylabel("open (carboxylate) fraction")
    ax.set_title("Lactone protection by burying the labile ring")
    ax.legend(frameon=False)
    ax.set_ylim(-0.02, 1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def cmd_demo(args: argparse.Namespace) -> None:
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    ckpt = out / "checkpoints"

    print("Training LASEr-lite + Fold-lite on synthetic 4-helix binder complexes...")
    laser, fold, poses = train_pair(
        n_data=args.n_data,
        laser_steps=args.steps,
        fold_steps=args.steps,
        d_model=args.d_model,
        helix_len=args.helix_len,
        seed=args.seed,
        ckpt_dir=ckpt,
    )
    native = poses[0]
    start = native.copy()
    start.sequence = None  # paper: strip the sequence, keep backbone + docked ligand

    print(f"Running NISE on {native.name} ({native.n_res} residues)...")
    result = nise(
        laser,
        fold,
        start,
        NISEConfig(
            rounds=args.rounds,
            n_expand=args.n_expand,
            n_select=args.n_select,
            temperature=0.85,
            seed=args.seed,
        ),
    )
    print("Running energy-based ISE baseline...")
    baseline = energy_ise(
        laser,
        start,
        rounds=args.rounds,
        n_expand=args.n_expand,
        n_select=args.n_select,
        seed=args.seed + 7,
        fold=fold,
    )

    _plot_trajectories(result.history, baseline.history, out / "nise_vs_energy.png")

    designs = result.designs[:4] or result.all_kept[:4]
    fasta_records = []
    table = []
    for i, cand in enumerate(designs, start=1):
        name = f"design_{i}"
        write_pdb(out / f"{name}.pdb", cand.pose, cand.sequence)
        fasta_records.append((name, cand.sequence))
        recov = site_recovery(cand.sequence, native.sequence, native)
        chem = pocket_match(cand.pose, cand.sequence)
        table.append(
            {
                "name": name,
                "ligand_plddt": round(cand.scores.ligand_plddt, 2),
                "nll": round(cand.scores.nll, 3),
                "bb_rmsd": round(cand.scores.bb_rmsd, 3),
                "ligand_rmsd": round(cand.scores.ligand_rmsd, 3),
                "buried_unsatisfied": cand.scores.buried_unsatisfied,
                "binding_site_recovery": round(recov, 3),
                "pocket_chemistry": round(chem, 3),
            }
        )
    write_fasta(out / "designs.fasta", fasta_records)
    write_pdb(out / "native.pdb", native, native.sequence)

    mutations = []
    if designs:
        mutations = proofread(laser, designs[0].pose, designs[0].sequence, top_k=6)
        if mutations:
            double = apply_mutations(designs[0].sequence, mutations[:2])
            write_fasta(
                out / "proofread.fasta",
                [("best", designs[0].sequence), ("proofread_top2", double)],
            )

    # Assay recreation using published Kd / hydrolysis numbers.
    ligand_conc = 5e-8
    curves = []
    for label, kd in [
        ("NISE-like", PAPER["exatecan"]["epic_kd_uM"] * 1e-6),
        ("proofread", PAPER["exatecan"]["epic_double_kd_nM"] * 1e-9),
        ("COMBS-like", PAPER["exatecan"]["combs_best_kd_uM"] * 1e-6),
        ("HSA", PAPER["exatecan"]["hsa_kd_uM"] * 1e-6),
    ]:
        p, y = synthetic_titration(kd, ligand=ligand_conc, seed=abs(hash(label)) % 10_000)
        fit = fit_kd(p, y, ligand=ligand_conc, n_boot=80)
        curves.append((label, p, y, fit, ligand_conc))
    _plot_kd(curves, out / "kd_fits.png")
    _plot_hydrolysis(out / "hydrolysis.png")

    t = np.linspace(0, 12, 40)
    y_free = two_state_open_fraction(t, 0.35, 0.06)
    hyd = fit_hydrolysis(t, y_free)

    summary = {
        "paper": PAPER,
        "native": native.name,
        "n_residues": native.n_res,
        "nise_history": [h.__dict__ for h in result.history],
        "energy_history": [h.__dict__ for h in baseline.history],
        "designs": table,
        "proofread": [
            {
                "position_1based": m.position + 1,
                "mutation": f"{m.from_aa}{m.position + 1}{m.to_aa}",
                "delta_nll": round(m.delta_nll, 3),
            }
            for m in mutations
        ],
        "kd_demo_nM": {name: fit.kd * 1e9 for name, _, __, fit, ___ in curves},
        "hydrolysis_t_half_h": hyd.t_half_h,
        "note": (
            "This demo learns a self-consistent synthetic world so the NISE loop is runnable on a laptop. "
            "It reproduces the algorithm in Fry et al., not the experimental EPIC/APEX sequences."
        ),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print()
    print(f"Wrote results to {out.resolve()}")
    if table:
        best = table[0]
        print(
            f"Best design  pLDDT={best['ligand_plddt']:.1f}  NLL={best['nll']:.2f}  "
            f"pocket chemistry={best.get('pocket_chemistry', 0):.0%}  "
            f"site recovery vs native={best['binding_site_recovery']:.0%}"
        )
    if mutations:
        print("Proofreading suggestions:", ", ".join(f"{m.from_aa}{m.position+1}{m.to_aa}" for m in mutations[:4]))
    print("Figures: nise_vs_energy.png, kd_fits.png, hydrolysis.png")


def cmd_fit_kd(args: argparse.Namespace) -> None:
    protein, signal = [], []
    with open(args.csv, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            protein.append(float(row[args.protein_col]))
            signal.append(float(row[args.signal_col]))
    fit = fit_kd(np.array(protein), np.array(signal), ligand=args.ligand, n_boot=args.bootstrap)
    print(f"Kd = {fit.kd:.4g} {fit.unit}   95% CI [{fit.kd_low:.4g}, {fit.kd_high:.4g}]")
    print(f"A_free={fit.a_free:.4f}  A_bound={fit.a_bound:.4f}")


def cmd_hydrolysis(args: argparse.Namespace) -> None:
    t, y = [], []
    with open(args.csv, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            t.append(float(row[args.time_col]))
            y.append(float(row[args.frac_col]))
    fit = fit_hydrolysis(np.array(t), np.array(y))
    print(f"k_h={fit.k_h:.4g} /h   k_c={fit.k_c:.4g} /h   t½={fit.t_half_h:.3g} h   Y_open∞={fit.y_open_inf:.3f}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="nise-binder", description=__doc__)
    p.add_argument("--version", action="version", version=f"nise-binder {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    demo = sub.add_parser("demo", help="Train tiny models, run NISE, proofread, and recreate paper analyses.")
    demo.add_argument("--outdir", default="results")
    demo.add_argument("--steps", type=int, default=80)
    demo.add_argument("--n-data", type=int, default=24)
    demo.add_argument("--rounds", type=int, default=5)
    demo.add_argument("--n-expand", type=int, default=12)
    demo.add_argument("--n-select", type=int, default=2)
    demo.add_argument("--d-model", type=int, default=48)
    demo.add_argument("--helix-len", type=int, default=14)
    demo.add_argument("--seed", type=int, default=0)
    demo.set_defaults(func=cmd_demo)

    kd = sub.add_parser("fit-kd", help="Fit a 1:1 Kd from a fluorescence-anisotropy titration CSV.")
    kd.add_argument("csv")
    kd.add_argument("--ligand", type=float, required=True, help="Total ligand concentration (M)")
    kd.add_argument("--protein-col", default="protein_M")
    kd.add_argument("--signal-col", default="anisotropy")
    kd.add_argument("--bootstrap", type=int, default=400)
    kd.set_defaults(func=cmd_fit_kd)

    hyd = sub.add_parser("hydrolysis", help="Fit closed/open lactone kinetics from a CSV.")
    hyd.add_argument("csv")
    hyd.add_argument("--time-col", default="hours")
    hyd.add_argument("--frac-col", default="open_fraction")
    hyd.set_defaults(func=cmd_hydrolysis)
    return p


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
