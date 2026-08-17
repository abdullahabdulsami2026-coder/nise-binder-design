"""Co-structure predictor (Fold-lite).

Stands in for RFAA / Boltz-2 in the paper: given a sequence and ligand identity,
it refines backbone + ligand coordinates and emits a ligand confidence (pLDDT-like).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .constants import AA_TO_IDX, ELEMENT_TO_IDX
from .geometry import Pose, knn_indices, noise_frames, rbf
from .laser import MLP, scatter_mean

N_AA = 20
N_ELEM = len(ELEMENT_TO_IDX)


class CoordLayer(nn.Module):
    """EGNN-style coordinate update over a kNN graph."""

    def __init__(self, d_model: int, n_rbf: int):
        super().__init__()
        self.msg = MLP(2 * d_model + n_rbf, d_model)
        self.coord = nn.Sequential(nn.Linear(d_model, d_model), nn.SiLU(), nn.Linear(d_model, 1))
        self.upd = MLP(2 * d_model, d_model)

    def forward(
        self,
        h: torch.Tensor,
        xyz: torch.Tensor,
        src: torch.Tensor,
        dst: torch.Tensor,
        edge: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        m = self.msg(torch.cat([h[src], h[dst], edge], dim=-1))
        disp = xyz[src] - xyz[dst]
        coord_w = torch.tanh(self.coord(m))
        delta = scatter_mean(coord_w * disp, dst, xyz.size(0))
        xyz = xyz + 0.15 * delta
        agg = scatter_mean(m, dst, h.size(0))
        h = h + self.upd(torch.cat([h, agg], dim=-1))
        return h, xyz


class FoldLite(nn.Module):
    def __init__(self, d_model: int = 64, n_rbf: int = 16, k_neighbors: int = 12, n_layers: int = 4):
        super().__init__()
        self.n_rbf = n_rbf
        self.k_neighbors = k_neighbors
        self.aa = nn.Embedding(N_AA, d_model)
        self.elem = nn.Embedding(N_ELEM, d_model)
        self.layers = nn.ModuleList([CoordLayer(d_model, n_rbf) for _ in range(n_layers)])
        self.conf = nn.Sequential(nn.Linear(d_model, d_model), nn.SiLU(), nn.Linear(d_model, 1))

    def _graph(self, xyz_np: np.ndarray, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        src, dst = [], []
        nn_idx = knn_indices(xyz_np, min(self.k_neighbors, len(xyz_np) - 1))
        for i, nbrs in enumerate(nn_idx):
            for j in nbrs:
                src.append(i)
                dst.append(int(j))
        src_t = torch.as_tensor(src, device=device, dtype=torch.long)
        dst_t = torch.as_tensor(dst, device=device, dtype=torch.long)
        dist = np.linalg.norm(xyz_np[np.array(src)] - xyz_np[np.array(dst)], axis=-1)
        edge = torch.as_tensor(rbf(dist, n_bins=self.n_rbf), device=device, dtype=torch.float32)
        return src_t, dst_t, edge

    def forward(self, pose: Pose, sequence: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = next(self.parameters()).device
        L = pose.n_res
        aa_idx = torch.as_tensor([AA_TO_IDX[a] for a in sequence], dtype=torch.long, device=device)
        elem_idx = torch.as_tensor(pose.ligand.elem_idx, dtype=torch.long, device=device)
        aa_h = self.aa(aa_idx)
        h = torch.cat([aa_h, self.elem(elem_idx)], dim=0)
        xyz_np = np.concatenate([pose.ca, pose.ligand.xyz], axis=0)
        xyz = torch.as_tensor(xyz_np, dtype=torch.float32, device=device)
        src, dst, edge = self._graph(xyz_np, device)
        for layer in self.layers:
            h, xyz = layer(h, xyz, src, dst, edge)
            xyz_np = xyz.detach().cpu().numpy()
            src, dst, edge = self._graph(xyz_np, device)
        d_lig = torch.cdist(xyz[:L].detach(), xyz[L:].detach())
        w = torch.softmax(-d_lig.T, dim=1)
        nearby = w @ aa_h
        conf = torch.sigmoid(self.conf(nearby + self.elem(elem_idx))).squeeze(-1)
        return xyz[:L], xyz[L:], conf

    @torch.no_grad()
    def predict(self, pose: Pose, sequence: str) -> Pose:
        ca, lig_xyz, conf = self.forward(pose, sequence)
        out = pose.copy()
        out.ca = ca.cpu().numpy()
        out.ligand.xyz = lig_xyz.cpu().numpy()
        out.sequence = sequence
        out.meta = dict(out.meta)
        out.meta["ligand_plddt"] = float(conf.mean().cpu()) * 100.0
        out.meta["ligand_plddt_atoms"] = conf.cpu().numpy() * 100.0
        return out


@dataclass
class FoldTrainConfig:
    steps: int = 120
    lr: float = 1e-3
    noise: float = 0.6
    mismatch_frac: float = 0.5


def _random_sequence(n: int, rng: np.random.Generator) -> str:
    from .constants import AA

    return "".join(rng.choice(list(AA), size=n))


def train_fold(model: FoldLite, poses: list[Pose], config: FoldTrainConfig | None = None, seed: int = 1) -> list[float]:
    """Train Fold-lite to refine native poses and to assign low confidence to mismatches."""
    cfg = config or FoldTrainConfig()
    rng = np.random.default_rng(seed)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    model.train()
    losses = []
    for _ in range(cfg.steps):
        native = poses[int(rng.integers(0, len(poses)))]
        assert native.sequence is not None
        noisy = native.copy()
        noisy.ca = noise_frames(native.ca, cfg.noise, rng)
        noisy.ligand.xyz = native.ligand.xyz + rng.normal(0, cfg.noise, size=native.ligand.xyz.shape)

        mismatch = rng.random() < cfg.mismatch_frac
        seq = _random_sequence(native.n_res, rng) if mismatch else native.sequence
        ca_pred, lig_pred, conf = model.forward(noisy, seq)

        target_ca = torch.as_tensor(native.ca, dtype=torch.float32, device=ca_pred.device)
        target_lig = torch.as_tensor(native.ligand.xyz, dtype=torch.float32, device=ca_pred.device)
        if mismatch:
            coord_loss = 0.05 * ((ca_pred - torch.as_tensor(noisy.ca, dtype=ca_pred.dtype, device=ca_pred.device)) ** 2).mean()
            target_conf = torch.full_like(conf, 0.18)
        else:
            coord_loss = ((ca_pred - target_ca) ** 2).mean() + ((lig_pred - target_lig) ** 2).mean()
            target_conf = torch.full_like(conf, 0.88)
        conf_loss = 2.5 * F.mse_loss(conf, target_conf)
        loss = coord_loss + conf_loss
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(float(loss.item()))
    model.eval()
    return losses
