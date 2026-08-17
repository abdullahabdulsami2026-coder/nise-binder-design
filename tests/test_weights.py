from nise_binder.train import bundled_weights_available, load_pair


def test_bundled_weights_load():
    assert bundled_weights_available()
    laser, fold = load_pair()
    assert next(laser.parameters()).shape[0] > 0
    assert next(fold.parameters()).numel() > 0
