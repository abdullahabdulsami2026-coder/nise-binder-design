"""Neural iterative selection–expansion (NISE) and neural proofreading."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from .energy import ligand_energy, minimize_pose
from .fold import FoldLite
from .geometry import Pose
from .laser import LaserLite
from .metrics import Consistency, evaluate


@dataclass
class Candidate:
    pose: Pose
    sequence: str
    scores: Consistency


@dataclass
class RoundStats:
    iteration: int
    mean_plddt: float
    q3_plddt: float
    mean_nll: float
    q1_nll: float
    n_self_consistent: int
    n_total: int


@dataclass
class NISEResult:
    designs: list[Candidate]
    history: list[RoundStats] = field(default_factory=list)
    all_kept: list[Candidate] = field(default_factory=list)


@dataclass
class NISEConfig:
    rounds: int = 6
    n_expand: int = 24
    n_select: int = 3
    temperature: float = 1.1
    bb_cut: float = 2.5
    lig_cut: float = 2.5
    seed: int = 0


def _nll(laser: LaserLite, pose: Pose, sequence: str) -> float:
    with torch.no_grad():
        return float(laser.fast_nll(pose, sequence).mean().cpu())


def nise(
    laser: LaserLite,
    fold: FoldLite,
    start: Pose,
    config: NISEConfig | None = None,
) -> NISEResult:
    """Closed loop: sample sequences, predict co-structures, keep confident self-consistent poses."""
    cfg = config or NISEConfig()
    rng = np.random.default_rng(cfg.seed)
    pool = [start.copy()]
    history: list[RoundStats] = []
    kept_all: list[Candidate] = []

    laser.eval()
    fold.eval()

    for iteration in range(cfg.rounds):
        round_candidates: list[Candidate] = []
        nlls: list[float] = []
        plddts: list[float] = []
        for pose in pool:
            for k in range(cfg.n_expand):
                seq = laser.sample(pose, temperature=cfg.temperature, rng=rng)
                predicted = fold.predict(pose, seq)
                nll = _nll(laser, predicted, seq)
                scores = evaluate(predicted, pose, nll, bb_cut=cfg.bb_cut, lig_cut=cfg.lig_cut)
                cand = Candidate(pose=predicted, sequence=seq, scores=scores)
                round_candidates.append(cand)
                nlls.append(nll)
                plddts.append(scores.ligand_plddt)
                if scores.self_consistent:
                    kept_all.append(cand)

        consistent = [c for c in round_candidates if c.scores.self_consistent]
        ranked = sorted(
            consistent or round_candidates,
            key=lambda c: (-c.scores.ligand_plddt, c.scores.buried_unsatisfied, c.scores.nll),
        )
        pool = [c.pose.copy() for c in ranked[: cfg.n_select]]
        for p, c in zip(pool, ranked[: cfg.n_select]):
            p.sequence = c.sequence
            p.meta["nll"] = c.scores.nll

        history.append(
            RoundStats(
                iteration=iteration,
                mean_plddt=float(np.mean(plddts)),
                q3_plddt=float(np.quantile(plddts, 0.75)),
                mean_nll=float(np.mean(nlls)),
                q1_nll=float(np.quantile(nlls, 0.25)),
                n_self_consistent=len(consistent),
                n_total=len(round_candidates),
            )
        )

    final = sorted(
        kept_all,
        key=lambda c: (-c.scores.ligand_plddt, c.scores.buried_unsatisfied, c.scores.nll),
    )
    # de-duplicate sequences
    uniq: list[Candidate] = []
    seen: set[str] = set()
    for cand in final:
        if cand.sequence in seen:
            continue
        seen.add(cand.sequence)
        uniq.append(cand)
    return NISEResult(designs=uniq[:12], history=history, all_kept=uniq)


def energy_ise(
    laser: LaserLite,
    start: Pose,
    rounds: int = 6,
    n_expand: int = 24,
    n_select: int = 3,
    temperature: float = 1.1,
    seed: int = 1,
    fold: FoldLite | None = None,
) -> NISEResult:
    """Paper Fig. 1c control: expand with LASEr, select by ligand energy after minimization."""
    rng = np.random.default_rng(seed)
    pool = [start.copy()]
    history: list[RoundStats] = []
    kept: list[Candidate] = []
    laser.eval()
    for iteration in range(rounds):
        nlls, plddts = [], []
        round_cands: list[Candidate] = []
        for pose in pool:
            for _ in range(n_expand):
                seq = laser.sample(pose, temperature=temperature, rng=rng)
                trial = pose.with_sequence(seq)
                refined = minimize_pose(trial, steps=12, lr=0.08)
                if fold is not None:
                    scored = fold.predict(refined, seq)
                    refined.meta["ligand_plddt"] = scored.meta["ligand_plddt"]
                else:
                    refined.meta["ligand_plddt"] = 0.0
                nll = _nll(laser, refined, seq)
                e = ligand_energy(refined)
                scores = evaluate(refined, pose, nll, bb_cut=3.5, lig_cut=3.5)
                cand = Candidate(pose=refined, sequence=seq, scores=scores)
                cand.pose.meta["ligand_energy"] = e
                round_cands.append(cand)
                nlls.append(nll)
                plddts.append(scores.ligand_plddt)
        ranked = sorted(round_cands, key=lambda c: c.pose.meta["ligand_energy"])
        pool = [c.pose.copy() for c in ranked[:n_select]]
        kept.extend(ranked[:n_select])
        history.append(
            RoundStats(
                iteration=iteration,
                mean_plddt=float(np.mean(plddts)),
                q3_plddt=float(np.quantile(plddts, 0.75)),
                mean_nll=float(np.mean(nlls)),
                q1_nll=float(np.quantile(nlls, 0.25)),
                n_self_consistent=sum(c.scores.self_consistent for c in round_cands),
                n_total=len(round_cands),
            )
        )
    ranked_final = sorted(kept, key=lambda c: c.pose.meta.get("ligand_energy", 0))
    return NISEResult(designs=ranked_final[:12], history=history, all_kept=ranked_final)


@dataclass
class Mutation:
    position: int  # 0-based
    from_aa: str
    to_aa: str
    delta_nll: float
    nll_mut: float
    nll_wt: float
    on_ligand: bool


def proofread(
    laser: LaserLite,
    pose: Pose,
    sequence: str | None = None,
    top_k: int = 8,
) -> list[Mutation]:
    """Suggest binding-site substitutions that lower LASEr NLL with the rest of the sequence fixed."""
    from .constants import AA
    from .geometry import binding_site_mask

    seq = sequence or pose.sequence
    if seq is None:
        raise ValueError("proofread needs a sequence")
    laser.eval()
    site = binding_site_mask(pose)
    suggestions: list[Mutation] = []
    with torch.no_grad():
        for i, keep in enumerate(site):
            if not keep:
                continue
            logits = laser.conditional_logits(pose, seq, i)
            logp = logits - logits.max()
            logp = logp - np.log(np.exp(logp).sum())
            wt = seq[i]
            nll_wt = float(-logp[ord_index(wt)])
            for aa_i, aa in enumerate(AA):
                if aa == wt:
                    continue
                nll_mut = float(-logp[aa_i])
                delta = nll_mut - nll_wt
                if delta < 0:
                    suggestions.append(
                        Mutation(
                            position=i,
                            from_aa=wt,
                            to_aa=aa,
                            delta_nll=delta,
                            nll_mut=nll_mut,
                            nll_wt=nll_wt,
                            on_ligand=True,
                        )
                    )
    suggestions.sort(key=lambda m: m.delta_nll)
    return suggestions[:top_k]


def ord_index(aa: str) -> int:
    from .constants import AA_TO_IDX

    return AA_TO_IDX[aa]


def apply_mutations(sequence: str, mutations: list[Mutation]) -> str:
    chars = list(sequence)
    for mut in mutations:
        chars[mut.position] = mut.to_aa
    return "".join(chars)
