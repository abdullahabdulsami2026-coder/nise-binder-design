import numpy as np
import pytest

from nise_binder.dataset import make_complex
from nise_binder.fold import FoldLite, FoldTrainConfig, train_fold
from nise_binder.laser import LaserLite, LaserTrainConfig, sequence_recovery, train_laser
from nise_binder.metrics import buried_unsatisfied_polars
from nise_binder.nise import NISEConfig, nise, proofread


def test_native_sequence_length():
    pose = make_complex(seed=3, helix_len=10)
    assert pose.sequence is not None
    assert len(pose.sequence) == pose.n_res
    assert buried_unsatisfied_polars(pose, pose.sequence) >= 0


def test_laser_and_nise_run():
    torch = pytest.importorskip("torch")
    poses = [make_complex(seed=i, helix_len=8) for i in range(6)]
    laser = LaserLite(d_model=32, n_encoder=1, n_decoder=1, k_neighbors=6, n_rbf=8)
    fold = FoldLite(d_model=32, n_layers=2, k_neighbors=6, n_rbf=8)
    train_laser(laser, poses, LaserTrainConfig(steps=8, lr=2e-3), seed=0)
    train_fold(fold, poses, FoldTrainConfig(steps=8, lr=2e-3), seed=1)

    native = poses[0]
    sampled = laser.sample(native, temperature=0.4)
    assert len(sampled) == native.n_res
    rec = sequence_recovery(sampled, native.sequence, native.meta["binding_site"])
    assert 0.0 <= rec <= 1.0

    pred = laser.sample(native, temperature=0.0)
    assert len(pred) == native.n_res

    start = native.copy()
    start.sequence = None
    result = nise(
        laser,
        fold,
        start,
        NISEConfig(rounds=2, n_expand=4, n_select=2, temperature=1.0, seed=0),
    )
    assert result.history
    assert result.designs or result.all_kept
    best = (result.designs or result.all_kept)[0]
    muts = proofread(laser, best.pose, best.sequence, top_k=3)
    assert isinstance(muts, list)
    _ = torch
