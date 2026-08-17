"""Shared constants, including published numbers from Fry, Slaw & Polizzi (Nature 2026)."""

from __future__ import annotations

AA = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {aa: i for i, aa in enumerate(AA)}
IDX_TO_AA = {i: aa for aa, i in AA_TO_IDX.items()}
MASK_IDX = 20  # extra token used during decoding

HYDROPHOBIC = frozenset("AILMFVW")
AROMATIC = frozenset("FWYH")
POLAR = frozenset("STNQ")
NEGATIVE = frozenset("DE")
POSITIVE = frozenset("KR")
CORE_PACKING = tuple("AILMFVW")
SURFACE = tuple("EKRDQSA")
POLAR_CONTACT = tuple("DENQST")
HYDROPHOBIC_CONTACT = tuple("FILMWVA")

ELEMENT_TO_IDX = {"C": 0, "N": 1, "O": 2, "F": 3, "S": 4, "P": 5, "CL": 6, "H": 7}
IDX_TO_ELEMENT = {i: e for e, i in ELEMENT_TO_IDX.items()}

# Toy partial charges used to pretrain the ligand encoder (a stand-in for SPICE).
ELEMENT_CHARGE = {
    "C": 0.05,
    "N": -0.35,
    "O": -0.45,
    "F": -0.20,
    "S": -0.12,
    "P": 0.40,
    "CL": -0.15,
    "H": 0.10,
}

# Published experimental numbers from the paper (for analysis demos / reference).
PAPER = {
    "citation": "Fry, Slaw & Polizzi, Nature 656, 237–249 (2026)",
    "doi": "10.1038/s41586-026-10670-w",
    "exatecan": {
        "nise_hit_rate": 1.00,
        "combs_hit_rate": 3 / 16,
        "epic_kd_uM": 0.12,
        "combs_best_kd_uM": 8.0,
        "hsa_kd_uM": 43.0,
        "epic_q51n_kd_nM": 8.0,
        "epic_m97l_kd_nM": 7.4,
        "epic_double_kd_nM": 1.2,
        "fl118_kd_uM": 6.0,
        "belotecan_kd_uM": 19.0,
        "camptothecin_kd_uM": 90.0,
        "pdb": {"EPIC": "9NZE", "EPIC_Q51N": "9NZG"},
    },
    "apixaban": {
        "nise_hit_rate": 5 / 6,
        "ligandmpnn_hit_rate": 4 / 9024,
        "apex_kd_pM": 80.0,
        "prior_best_kd_nM": 680.0,
        "factor_xa_ki_pM": (80.0, 700.0),
    },
}
