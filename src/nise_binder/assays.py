"""Analysis tools from the paper: 1:1 Kd fitting and lactone hydrolysis kinetics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import curve_fit


def bound_fraction(p_tot: np.ndarray, l_tot: float, kd: float) -> np.ndarray:
    """Quadratic 1:1 binding isotherm (used for fluorescence anisotropy)."""
    s = p_tot + l_tot + kd
    disc = np.clip(s**2 - 4 * p_tot * l_tot, 0, None)
    pl = 0.5 * (s - np.sqrt(disc))
    return np.clip(pl / max(l_tot, 1e-12), 0, 1)


def anisotropy_model(p_tot: np.ndarray, kd: float, a_free: float, a_bound: float, l_tot: float) -> np.ndarray:
    fb = bound_fraction(p_tot, l_tot, kd)
    return a_free + (a_bound - a_free) * fb


@dataclass
class KdFit:
    kd: float
    kd_low: float
    kd_high: float
    a_free: float
    a_bound: float
    unit: str = "M"


def fit_kd(
    protein: np.ndarray,
    signal: np.ndarray,
    ligand: float,
    n_boot: int = 400,
    seed: int = 0,
) -> KdFit:
    """Fit a 1:1 Kd, then bootstrap residuals for a 95% interval (paper-style)."""
    protein = np.asarray(protein, dtype=float)
    signal = np.asarray(signal, dtype=float)

    def model(p, kd, a_free, a_bound):
        return anisotropy_model(p, kd, a_free, a_bound, ligand)

    p0 = [max(np.median(protein), 1e-7), float(np.min(signal)), float(np.max(signal))]
    bounds = ([1e-12, -np.inf, -np.inf], [np.max(protein) * 50 + 1e-6, np.inf, np.inf])
    popt, _ = curve_fit(model, protein, signal, p0=p0, bounds=bounds, maxfev=20000)
    resid = signal - model(protein, *popt)
    rng = np.random.default_rng(seed)
    kds = []
    for _ in range(n_boot):
        yb = model(protein, *popt) + rng.choice(resid, size=len(resid), replace=True)
        try:
            pb, _ = curve_fit(model, protein, yb, p0=popt, bounds=bounds, maxfev=8000)
            kds.append(pb[0])
        except RuntimeError:
            continue
    kds = np.array(kds) if kds else np.array([popt[0]])
    return KdFit(
        kd=float(popt[0]),
        kd_low=float(np.quantile(kds, 0.025)),
        kd_high=float(np.quantile(kds, 0.975)),
        a_free=float(popt[1]),
        a_bound=float(popt[2]),
    )


def synthetic_titration(kd: float, ligand: float = 5e-8, n: int = 16, noise: float = 0.015, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    protein = np.logspace(np.log10(kd / 80), np.log10(kd * 80), n)
    y = anisotropy_model(protein, kd, 0.05, 0.22, ligand)
    y = y + rng.normal(0, noise, size=n)
    return protein, y


def two_state_open_fraction(t: np.ndarray, k_h: float, k_c: float) -> np.ndarray:
    """Closed → open lactone hydrolysis, starting from 100% closed."""
    keq = k_h + k_c
    y_inf = k_h / max(keq, 1e-12)
    return y_inf * (1.0 - np.exp(-keq * t))


@dataclass
class HydrolysisFit:
    k_h: float
    k_c: float
    t_half_h: float
    y_open_inf: float


def fit_hydrolysis(t_hours: np.ndarray, y_open: np.ndarray) -> HydrolysisFit:
    t = np.asarray(t_hours, dtype=float)
    y = np.clip(np.asarray(y_open, dtype=float), 0, 1)

    def model(tt, k_h, k_c):
        return two_state_open_fraction(tt, k_h, k_c)

    popt, _ = curve_fit(model, t, y, p0=[0.3, 0.05], bounds=([1e-6, 1e-8], [10.0, 10.0]), maxfev=20000)
    k_h, k_c = float(popt[0]), float(popt[1])
    t_half = np.log(2) / max(k_h + k_c, 1e-12)
    return HydrolysisFit(k_h=k_h, k_c=k_c, t_half_h=float(t_half), y_open_inf=k_h / (k_h + k_c))


def protected_open_fraction(
    t: np.ndarray,
    k_h_free: float,
    k_c_free: float,
    kd: float,
    protein: float,
    ligand: float,
    protection: float = 80.0,
) -> np.ndarray:
    """Approximate protection: only the unbound fraction hydrolyses at the free-drug rate."""
    fb = float(bound_fraction(np.array([protein]), ligand, kd)[0])
    k_h = k_h_free * (1 - fb) + (k_h_free / protection) * fb
    k_c = k_c_free
    return two_state_open_fraction(t, k_h, k_c)
