"""Ligand-aware sequence design network (LASEr-lite).

Faithful to the paper's recipe, simplified:
  * heterograph over protein residues + ligand atoms
  * pretrained ligand encoder (partial-charge readout)
  * autoregressive decoding in a random order
  * whole-frame backbone noise during training
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .constants import AA, AA_TO_IDX, ELEMENT_TO_IDX, MASK_IDX
from .geometry import Pose, binding_site_mask, knn_indices, noise_frames, pairwise_min_distance, rbf

N_AA = 20
N_ELEM = len(ELEMENT_TO_IDX)


def scatter_mean(src: torch.Tensor, index: torch.Tensor, n: int) -> torch.Tensor:
    out = torch.zeros(n, src.size(-1), device=src.device, dtype=src.dtype)
    out.index_add_(0, index, src)
    cnt = torch.zeros(n, 1, device=src.device, dtype=src.dtype)
    cnt.index_add_(0, index, torch.ones(src.size(0), 1, device=src.device, dtype=src.dtype))
    return out / cnt.clamp(min=1.0)


class MLP(nn.Module):
    def __init__(self, din: int, dout: int, hidden: int | None = None):
        super().__init__()
        hidden = hidden or max(din, dout)
        self.net = nn.Sequential(nn.Linear(din, hidden), nn.SiLU(), nn.Linear(hidden, dout))
        self.out_norm = nn.LayerNorm(dout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out_norm(self.net(x))


class MPNNLayer(nn.Module):
    def __init__(self, d_model: int, d_edge: int):
        super().__init__()
        self.msg = MLP(2 * d_model + d_edge, d_model)
        self.upd = MLP(2 * d_model, d_model)

    def forward(self, h: torch.Tensor, src: torch.Tensor, dst: torch.Tensor, edge: torch.Tensor) -> torch.Tensor:
        m = self.msg(torch.cat([h[src], h[dst], edge], dim=-1))
        agg = scatter_mean(m, dst, h.size(0))
        return h + self.upd(torch.cat([h, agg], dim=-1))


def _edges(xyz: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nn_idx = knn_indices(xyz, k)
    src = np.repeat(np.arange(len(xyz)), nn_idx.shape[1])
    dst = nn_idx.reshape(-1)
    dist = np.linalg.norm(xyz[src] - xyz[dst], axis=-1)
    return src, dst, dist


class LigandEncoder(nn.Module):
    def __init__(self, d_model: int = 64, n_rbf: int = 12):
        super().__init__()
        self.n_rbf = n_rbf
        self.elem = nn.Embedding(N_ELEM, d_model)
        self.layer = MPNNLayer(d_model, n_rbf)
        self.charge_head = nn.Linear(d_model, 1)

    def forward(self, xyz: torch.Tensor, elem_idx: torch.Tensor) -> torch.Tensor:
        h = self.elem(elem_idx)
        if xyz.size(0) == 1:
            return h
        src, dst, dist = _edges(xyz.detach().cpu().numpy(), k=min(6, xyz.size(0) - 1))
        src_t = torch.as_tensor(src, device=xyz.device, dtype=torch.long)
        dst_t = torch.as_tensor(dst, device=xyz.device, dtype=torch.long)
        edge = torch.as_tensor(rbf(dist, n_bins=self.n_rbf), device=xyz.device, dtype=h.dtype)
        return self.layer(h, src_t, dst_t, edge)

    def charges(self, h: torch.Tensor) -> torch.Tensor:
        return self.charge_head(h).squeeze(-1)


class LaserLite(nn.Module):
    def __init__(
        self,
        d_model: int = 64,
        n_rbf: int = 16,
        k_neighbors: int = 12,
        n_encoder: int = 3,
        n_decoder: int = 2,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_rbf = n_rbf
        self.k_neighbors = k_neighbors
        self.ligand_encoder = LigandEncoder(d_model=d_model, n_rbf=min(n_rbf, 12))
        self.res_in = MLP(2 * n_rbf + N_ELEM + 4, d_model)
        self.enc_layers = nn.ModuleList([MPNNLayer(d_model, n_rbf) for _ in range(n_encoder)])
        self.aa_embed = nn.Embedding(N_AA + 1, d_model)
        self.aa_head = nn.Linear(d_model, N_AA)

    def _residue_features(self, pose: Pose) -> torch.Tensor:
        d_lig = np.linalg.norm(pose.ca[:, None, :] - pose.ligand.xyz[None, :, :], axis=-1)
        nearest = d_lig.argmin(axis=1)
        d_min = d_lig.min(axis=1)
        polar_xyz = pose.ligand.xyz[pose.ligand.polar]
        apolar_xyz = pose.ligand.xyz[~pose.ligand.polar]
        d_pol = pairwise_min_distance(pose.ca, polar_xyz) if len(polar_xyz) else np.full(pose.n_res, 20.0)
        d_apo = pairwise_min_distance(pose.ca, apolar_xyz) if len(apolar_xyz) else np.full(pose.n_res, 20.0)
        elem_oh = np.zeros((pose.n_res, N_ELEM), dtype=float)
        elem_oh[np.arange(pose.n_res), pose.ligand.elem_idx[nearest]] = 1.0
        hid = pose.helix_id if pose.helix_id is not None else np.zeros(pose.n_res)
        feats = np.concatenate(
            [
                rbf(d_pol, d_max=12.0, n_bins=self.n_rbf),
                rbf(d_apo, d_max=12.0, n_bins=self.n_rbf),
                elem_oh,
                (d_min / 12.0)[:, None],
                (hid >= 0).astype(float)[:, None],
                np.sin(np.arange(pose.n_res)[:, None] * 0.15),
                np.cos(np.arange(pose.n_res)[:, None] * 0.15),
            ],
            axis=1,
        )
        return torch.as_tensor(feats, dtype=torch.float32)

    def encode(self, pose: Pose, device: torch.device | None = None) -> torch.Tensor:
        device = device or next(self.parameters()).device
        res_h = self.res_in(self._residue_features(pose).to(device))
        lig_xyz = torch.as_tensor(pose.ligand.xyz, dtype=torch.float32, device=device)
        lig_elem = torch.as_tensor(pose.ligand.elem_idx, dtype=torch.long, device=device)
        lig_h = self.ligand_encoder(lig_xyz, lig_elem)
        h = torch.cat([res_h, lig_h], dim=0)

        xyz_all = np.concatenate([pose.ca, pose.ligand.xyz], axis=0)
        src, dst, dist = _edges(xyz_all, k=min(self.k_neighbors, len(xyz_all) - 1))
        src_t = torch.as_tensor(src, device=device, dtype=torch.long)
        dst_t = torch.as_tensor(dst, device=device, dtype=torch.long)
        edge = torch.as_tensor(rbf(dist, n_bins=self.n_rbf), device=device, dtype=torch.float32)
        for layer in self.enc_layers:
            h = layer(h, src_t, dst_t, edge)
        h = torch.cat([res_h + h[: pose.n_res], h[pose.n_res :]], dim=0)
        return h

    def structure_logits(self, pose: Pose, encoded: torch.Tensor | None = None) -> torch.Tensor:
        """P(aa | backbone, ligand) with no sequence leak — used for training and sampling."""
        device = next(self.parameters()).device
        encoded = encoded if encoded is not None else self.encode(pose, device=device)
        return self.aa_head(encoded[: pose.n_res])

    def _decode_logits(
        self,
        pose: Pose,
        encoded: torch.Tensor,
        seq_idx: torch.Tensor,
        order: np.ndarray,
    ) -> torch.Tensor:
        """Logits with optional sequence context. MASK tokens add no identity information."""
        _ = order
        L = pose.n_res
        ctx = self.aa_embed(seq_idx)
        mask = seq_idx == MASK_IDX
        ctx = torch.where(mask[:, None], torch.zeros_like(ctx), ctx)
        return self.aa_head(encoded[:L] + ctx)

    def fast_nll(self, pose: Pose, sequence: str) -> torch.Tensor:
        device = next(self.parameters()).device
        true_idx = torch.as_tensor([AA_TO_IDX[a] for a in sequence], dtype=torch.long, device=device)
        logits = self.structure_logits(pose)
        return F.cross_entropy(logits, true_idx, reduction="none")

    @torch.no_grad()
    def sample(
        self,
        pose: Pose,
        temperature: float = 1.0,
        rng: np.random.Generator | None = None,
        autoregressive: bool = False,
    ) -> str:
        rng = rng or np.random.default_rng()
        device = next(self.parameters()).device
        L = pose.n_res
        encoded = self.encode(pose, device=device)
        if not autoregressive:
            logits = self.structure_logits(pose, encoded=encoded)
            if temperature <= 1e-6:
                idx = torch.argmax(logits, dim=-1).cpu().numpy()
            else:
                probs = torch.softmax(logits / temperature, dim=-1).cpu().numpy()
                idx = np.array([rng.choice(N_AA, p=p / p.sum()) for p in probs])
            return "".join(AA[int(i)] for i in idx)

        known = torch.full((L,), MASK_IDX, dtype=torch.long, device=device)
        order = rng.permutation(L)
        for pos in order:
            logits = self._decode_logits(pose, encoded, known, order)[pos]
            if temperature <= 1e-6:
                aa = int(torch.argmax(logits))
            else:
                probs = torch.softmax(logits / temperature, dim=-1).cpu().numpy()
                aa = int(rng.choice(N_AA, p=probs / probs.sum()))
            known[pos] = aa
        return "".join(AA[int(i)] for i in known.cpu().numpy())

    @torch.no_grad()
    def conditional_logits(self, pose: Pose, sequence: str, position: int) -> np.ndarray:
        """P(A_i | A_{-i}, structure, ligand) — used for neural proofreading."""
        device = next(self.parameters()).device
        idx = torch.as_tensor([AA_TO_IDX[a] for a in sequence], dtype=torch.long, device=device)
        idx[position] = MASK_IDX
        encoded = self.encode(pose, device=device)
        logits = self._decode_logits(pose, encoded, idx, np.arange(pose.n_res))
        return logits[position].cpu().numpy()

    def charge_loss(self, pose: Pose) -> torch.Tensor:
        device = next(self.parameters()).device
        xyz = torch.as_tensor(pose.ligand.xyz, dtype=torch.float32, device=device)
        elem = torch.as_tensor(pose.ligand.elem_idx, dtype=torch.long, device=device)
        pred = self.ligand_encoder.charges(self.ligand_encoder(xyz, elem))
        target = torch.as_tensor(pose.ligand.charges, dtype=torch.float32, device=device)
        return F.mse_loss(pred, target)


@dataclass
class LaserTrainConfig:
    steps: int = 120
    lr: float = 1e-3
    temperature_noise: float = 0.25
    charge_weight: float = 0.2


def train_laser(model: LaserLite, poses: list[Pose], config: LaserTrainConfig | None = None, seed: int = 0) -> list[float]:
    cfg = config or LaserTrainConfig()
    rng = np.random.default_rng(seed)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    model.train()
    losses = []
    for step in range(cfg.steps):
        pose = poses[int(rng.integers(0, len(poses)))].copy()
        pose.ca = noise_frames(pose.ca, cfg.temperature_noise, rng)
        assert pose.sequence is not None
        opt.zero_grad()
        nll_i = model.fast_nll(pose, pose.sequence)
        site = torch.as_tensor(binding_site_mask(pose), dtype=nll_i.dtype, device=nll_i.device)
        weights = 1.0 + 4.0 * site
        nll = (nll_i * weights).sum() / weights.sum()
        loss = nll + cfg.charge_weight * model.charge_loss(pose)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(float(loss.item()))
    model.eval()
    return losses


def sequence_recovery(pred: str, true: str, mask: np.ndarray | None = None) -> float:
    if mask is None:
        mask = np.ones(len(true), dtype=bool)
    hits = sum(p == t for p, t, m in zip(pred, true, mask) if m)
    return hits / max(int(mask.sum()), 1)


def sample_many(model: LaserLite, pose: Pose, n: int, temperature: float, seed: int = 0) -> list[str]:
    rng = np.random.default_rng(seed)
    return [model.sample(pose, temperature=temperature, rng=rng) for _ in range(n)]
