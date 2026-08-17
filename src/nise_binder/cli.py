"""Command-line interface."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

from . import __version__
from .assays import fit_hydrolysis, fit_kd
from .pipeline import DemoConfig, run_demo


def cmd_demo(args: argparse.Namespace) -> None:
    demo = run_demo(
        DemoConfig(
            outdir=args.outdir,
            steps=args.steps,
            n_data=args.n_data,
            rounds=args.rounds,
            n_expand=args.n_expand,
            n_select=args.n_select,
            d_model=args.d_model,
            helix_len=args.helix_len,
            seed=args.seed,
            ligand=args.ligand,
            source=args.source,
            pdb_id=args.pdb,
            use_pretrained=args.pretrained,
        )
    )
    print(f"Wrote results to {Path(demo.outdir).resolve()}")
    if demo.table:
        best = demo.table[0]
        print(
            f"Best design  pLDDT={best['ligand_plddt']:.1f}  NLL={best['nll']:.2f}  "
            f"pocket chemistry={best.get('pocket_chemistry', 0):.0%}  "
            f"site recovery vs native={best['binding_site_recovery']:.0%}"
        )
    if demo.mutations:
        print("Proofreading suggestions:", ", ".join(m["mutation"] for m in demo.mutations[:4]))
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


def cmd_app(args: argparse.Namespace) -> None:
    try:
        import streamlit  # noqa: F401
    except ImportError as exc:
        raise SystemExit("Install the UI extras first:  pip install -e '.[app]'") from exc
    import subprocess

    app = Path(__file__).with_name("app.py")
    cmd = [sys.executable, "-m", "streamlit", "run", str(app)]
    if args.port:
        cmd += ["--server.port", str(args.port)]
    raise SystemExit(subprocess.call(cmd))


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
    demo.add_argument("--ligand", default="exatecan-like", choices=["exatecan-like", "apixaban-like"])
    demo.add_argument("--source", default="synthetic", choices=["synthetic", "public"])
    demo.add_argument("--pdb", default="9NZE", help="PDB ID used when --source public")
    demo.add_argument(
        "--pretrained",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Load bundled public-PDB weights (default). Use --no-pretrained to train from scratch.",
    )
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

    app = sub.add_parser("app", help="Open the Streamlit UI.")
    app.add_argument("--port", type=int, default=8501)
    app.set_defaults(func=cmd_app)
    return p


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
