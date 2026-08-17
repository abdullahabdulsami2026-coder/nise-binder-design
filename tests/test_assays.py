import numpy as np

from nise_binder.assays import bound_fraction, fit_hydrolysis, fit_kd, synthetic_titration, two_state_open_fraction
from nise_binder.constants import PAPER


def test_bound_fraction_limits():
    p = np.array([0.0, 1e-3])
    y = bound_fraction(p, l_tot=1e-6, kd=1e-6)
    assert y[0] == 0.0
    assert y[1] > 0.9


def test_fit_kd_recovers_truth():
    kd = 1.2e-7
    ligand = 5e-8
    p, y = synthetic_titration(kd, ligand=ligand, noise=0.008, seed=4)
    fit = fit_kd(p, y, ligand=ligand, n_boot=40, seed=4)
    rel = abs(fit.kd - kd) / kd
    assert rel < 0.35
    assert fit.kd_low < fit.kd < fit.kd_high or np.isclose(fit.kd_low, fit.kd_high)


def test_hydrolysis_half_life():
    t = np.linspace(0, 15, 50)
    k_h, k_c = 0.30, 0.05
    y = two_state_open_fraction(t, k_h, k_c)
    fit = fit_hydrolysis(t, y)
    assert abs(fit.k_h - k_h) / k_h < 0.15
    assert abs(fit.t_half_h - np.log(2) / (k_h + k_c)) < 0.2


def test_paper_constants_match_abstract():
    assert PAPER["exatecan"]["nise_hit_rate"] == 1.0
    assert PAPER["apixaban"]["nise_hit_rate"] == 5 / 6
    assert PAPER["exatecan"]["epic_double_kd_nM"] == 1.2
    assert PAPER["apixaban"]["apex_kd_pM"] == 80.0
