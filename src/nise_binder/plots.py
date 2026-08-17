"""Matplotlib figures shared by the CLI and the Streamlit app."""

from __future__ import annotations

import numpy as np

from .assays import anisotropy_model, protected_open_fraction, two_state_open_fraction


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["text.usetex"] = False
    plt.rcParams["mathtext.default"] = "regular"
    return plt


def _finish(fig):
    fig.subplots_adjust(left=0.14, right=0.96, bottom=0.16, top=0.88, wspace=0.45)
    return fig


def trajectories(nise_hist, energy_hist):
    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6), sharex=True)
    for ax, hist, title in (
        (axes[0], nise_hist, "NISE (neural fold)"),
        (axes[1], energy_hist, "Energy ISE (physics baseline)"),
    ):
        it = [h.iteration for h in hist]
        ax.plot(it, [h.q3_plddt for h in hist], color="#c0392b", lw=2)
        ax.set_ylabel("ligand confidence", color="#c0392b")
        ax2 = ax.twinx()
        ax2.plot(it, [h.q1_nll for h in hist], color="#2471a3", lw=2)
        ax2.set_ylabel("sequence NLL", color="#2471a3")
        ax.set_title(title)
        ax.set_xlabel("iteration")
        ax.spines["top"].set_visible(False)
    return _finish(fig)


def kd_curves(fits: list[tuple[str, np.ndarray, np.ndarray, object, float]]):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    colors = ["#1f6feb", "#9b59b6", "#e67e22", "#27ae60"]
    for color, (name, p, y, fit, ligand) in zip(colors, fits):
        ax.scatter(p * 1e6, y, s=22, color=color, label=f"{name}  Kd={fit.kd * 1e9:.2g} nM")
        grid = np.logspace(np.log10(p.min()), np.log10(p.max()), 120)
        ax.plot(grid * 1e6, anisotropy_model(grid, fit.kd, fit.a_free, fit.a_bound, ligand), color=color, lw=1.6)
    ax.set_xscale("log")
    ax.set_xlabel("protein (uM)")
    ax.set_ylabel("anisotropy")
    ax.set_title("1:1 fluorescence-anisotropy Kd fits")
    ax.legend(frameon=False, fontsize=8)
    return _finish(fig)


def kd_single(protein: np.ndarray, signal: np.ndarray, fit, ligand: float):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    ax.scatter(protein * 1e6, signal, s=28, color="#1f6feb", zorder=3)
    grid = np.logspace(np.log10(max(protein.min(), 1e-12)), np.log10(protein.max()), 160)
    ax.plot(grid * 1e6, anisotropy_model(grid, fit.kd, fit.a_free, fit.a_bound, ligand), color="#1f6feb", lw=1.8)
    ax.set_xscale("log")
    ax.set_xlabel("protein (uM)")
    ax.set_ylabel("anisotropy")
    ax.set_title(f"Kd = {fit.kd * 1e9:.3f} nM")
    return _finish(fig)


def hydrolysis_paper():
    plt = _plt()
    t = np.linspace(0, 50, 200)
    free = two_state_open_fraction(t, k_h=0.35, k_c=0.06)
    hsa = protected_open_fraction(t, 0.35, 0.06, kd=43e-6, protein=500e-6, ligand=20e-6, protection=1.2)
    epic = protected_open_fraction(t, 0.35, 0.06, kd=1.2e-9, protein=20e-6, ligand=20e-6, protection=200.0)
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    ax.plot(t, free, label="free exatecan", color="#c0392b", lw=2)
    ax.plot(t, hsa, label="HSA (500 uM)", color="#7f8c8d", lw=2)
    ax.plot(t, epic, label="EPIC-like binder (20 uM)", color="#1f6feb", lw=2)
    ax.set_xlabel("time (h)")
    ax.set_ylabel("open (carboxylate) fraction")
    ax.set_title("Lactone protection by burying the labile ring")
    ax.legend(frameon=False)
    ax.set_ylim(-0.02, 1.02)
    return _finish(fig)


def hydrolysis_fit(t: np.ndarray, y: np.ndarray, pred: np.ndarray):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    ax.scatter(t, y, s=28, color="#c0392b", zorder=3, label="data")
    ax.plot(t, pred, color="#1f6feb", lw=1.8, label="two-state fit")
    ax.set_xlabel("time (h)")
    ax.set_ylabel("open fraction")
    ax.legend(frameon=False)
    return _finish(fig)


def pose_3d(pose, sequence: str | None = None):
    plt = _plt()
    seq = sequence or pose.sequence or ("G" * pose.n_res)
    fig = plt.figure(figsize=(5.2, 4.4))
    ax = fig.add_subplot(111, projection="3d")
    ca = pose.ca
    ax.plot(ca[:, 0], ca[:, 1], ca[:, 2], color="#5b8def", lw=2.2)
    ax.scatter(ca[:, 0], ca[:, 1], ca[:, 2], c="#1f4e9b", s=12)
    lig = pose.ligand.xyz
    colors = ["#e74c3c" if p else "#f1c40f" for p in pose.ligand.polar]
    ax.scatter(lig[:, 0], lig[:, 1], lig[:, 2], c=colors, s=40, depthshade=True)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_title(f"{pose.ligand.name}  {len(seq)} aa")
    return fig
