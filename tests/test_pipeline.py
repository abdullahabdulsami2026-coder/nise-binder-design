from nise_binder.pipeline import DemoConfig, run_demo


def test_run_demo_tiny(tmp_path):
    demo = run_demo(
        DemoConfig(
            outdir=str(tmp_path),
            steps=6,
            n_data=4,
            rounds=2,
            n_expand=3,
            n_select=2,
            d_model=32,
            helix_len=8,
            seed=0,
            ligand="exatecan-like",
            write_files=True,
            use_pretrained=False,
        )
    )
    assert demo.n_residues > 20
    assert demo.table
    assert "design_1" in demo.pdbs
    assert (tmp_path / "summary.json").exists()
    assert demo.nise is not None
    assert demo.nise.history
