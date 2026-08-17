"""Streamlit UI for running NISE and fitting binding / hydrolysis assays."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import streamlit as st
import streamlit.components.v1 as components

from nise_binder.assays import fit_hydrolysis, fit_kd, two_state_open_fraction
from nise_binder.constants import PAPER
from nise_binder.pipeline import DemoConfig, run_demo
from nise_binder.pdbdata import PUBLIC_COMPLEXES
from nise_binder.plots import hydrolysis_fit, hydrolysis_paper, kd_single, pose_3d, trajectories

st.set_page_config(page_title="NISE binder design", layout="wide", page_icon="🧬")

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"
ON_CLOUD = Path("/mount/src").exists() or os.environ.get("STREAMLIT_CLOUD") == "1"


def _examples() -> Path | None:
    for cand in (Path.cwd() / "examples", EXAMPLES):
        if (cand / "epic_fp.csv").exists():
            return cand
    return None


def _viewer(pdb: str, height: int = 420) -> None:
    payload = json.dumps(pdb)
    html = f"""
    <div id="mol" style="width:100%;height:{height}px;position:relative;"></div>
    <script src="https://3Dmol.org/build/3Dmol-min.js"></script>
    <script>
      const element = document.getElementById("mol");
      const viewer = $3Dmol.createViewer(element, {{backgroundColor: "white"}});
      viewer.addModel({payload}, "pdb");
      viewer.setStyle({{hetflag: false}}, {{cartoon: {{color: "spectrum"}}}});
      viewer.setStyle({{hetflag: true}}, {{stick: {{colorscheme: "Jmol"}}}});
      viewer.zoomTo();
      viewer.render();
    </script>
    """
    components.html(html, height=height + 16)


def _read_table(upload, fallback: Path | None) -> tuple[list[dict], list[str]]:
    import csv
    import io

    if upload is not None:
        text = upload.getvalue().decode("utf-8")
    elif fallback is not None and fallback.exists():
        text = fallback.read_text(encoding="utf-8")
    else:
        return [], []
    rows = list(csv.DictReader(io.StringIO(text)))
    headers = list(rows[0].keys()) if rows else []
    return rows, headers


def _selected_pose(demo, choice: str):
    if choice == "native":
        return demo.native, demo.native.sequence
    idx = int(choice.split("_")[-1]) - 1
    cand = demo.designs[idx]
    return cand.pose, cand.sequence


def main() -> None:
    st.title("NISE binder design")
    st.info(
        "Educational recreation of Fry, Slaw & Polizzi, *Nature* 2026. "
        "Public mode loads real RCSB coordinates (e.g. EPIC 9NZE). "
        "Designed sequences come from a small open model, not the paper's production LASErMPNN/Boltz-2 stack."
    )
    st.caption(
        "Laptop-scale NISE loop (Fry, Slaw & Polizzi, Nature 2026). "
        "Public PDB mode uses real crystal coordinates and sequences from RCSB. "
        "Sequence design still uses the small LASEr-lite model, not official LASErMPNN/Boltz-2 weights."
    )

    with st.sidebar:
        st.header("Design")
        source = st.radio("Data", ["Public PDB (RCSB)", "Synthetic toy world"], index=0)
        public = source.startswith("Public")
        pdb_id = "9NZE"
        ligand = "exatecan-like"
        helix_len = 12
        if public:
            labels = {f"{c.pdb_id} — {c.label}": c.pdb_id for c in PUBLIC_COMPLEXES}
            choice = st.selectbox("Complex", list(labels))
            pdb_id = labels[choice]
            custom = st.text_input("Or PDB ID", value="").strip().upper()
            if custom:
                pdb_id = custom
            st.caption(next((c.note for c in PUBLIC_COMPLEXES if c.pdb_id == pdb_id), "Downloaded from RCSB on demand."))
        else:
            ligand = st.selectbox("Ligand", ["exatecan-like", "apixaban-like"])
            helix_len = st.slider("Helix length", 8, 18, 12)
        steps = st.slider("Training steps (if retraining)", 20, 160, 60, 10)
        rounds = st.slider("NISE rounds", 2, 8, 3 if ON_CLOUD else 5)
        n_expand = st.slider("Sequences per pose", 4, 16, 6 if ON_CLOUD else 10)
        seed = st.number_input("Seed", min_value=0, value=0, step=1)
        run = st.button("Run NISE", type="primary", use_container_width=True)
        retrain = st.checkbox("Retrain models (slow)", value=False)
        st.caption("Default uses bundled weights trained on the public PDB complexes above.")
        st.markdown(
            "Official models: "
            "[LASErMPNN](https://github.com/polizzilab/LASErMPNN) · "
            "[NISE](https://github.com/polizzilab/NISE)"
        )

    if run:
        msg = "Running NISE on the selected complex…"
        if retrain:
            msg = "Training LASEr-lite + Fold-lite, then running NISE…"
        elif public:
            msg = "Loading public structure (RCSB if uncached), then running NISE…"
        with st.spinner(msg):
            demo = run_demo(
                DemoConfig(
                    ligand=ligand,
                    steps=int(steps),
                    rounds=int(rounds),
                    n_expand=int(n_expand),
                    helix_len=int(helix_len),
                    seed=int(seed),
                    n_data=20,
                    write_files=False,
                    outdir="results",
                    source="public" if public else "synthetic",
                    pdb_id=pdb_id,
                    use_pretrained=not retrain,
                )
            )
        st.session_state["demo"] = demo
        origin = "RCSB " + demo.native_name if public else "synthetic"
        st.success(f"Finished · {origin} · ligand {demo.ligand} · {demo.n_residues} residues")

    tabs = st.tabs(["Designs", "Trajectory", "Proofread", "Fit Kd", "Hydrolysis"])
    demo = st.session_state.get("demo")

    with tabs[0]:
        if demo is None:
            st.info("Set parameters in the sidebar and run NISE.")
        else:
            best = demo.table[0]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Ligand pLDDT", f"{best['ligand_plddt']:.1f}")
            c2.metric("Sequence NLL", f"{best['nll']:.2f}")
            c3.metric("Pocket chemistry", f"{best['pocket_chemistry']:.0%}")
            c4.metric("Site recovery", f"{best['binding_site_recovery']:.0%}")
            rows = [{k: v for k, v in row.items() if k != "sequence"} for row in demo.table]
            st.dataframe(rows, use_container_width=True, hide_index=True)
            names = list(demo.pdbs)
            choice = st.selectbox("Structure", names, index=min(1, len(names) - 1))
            pose, seq = _selected_pose(demo, choice)
            left, right = st.columns((1.2, 1))
            with left:
                _viewer(demo.pdbs[choice])
                with st.expander("Offline Cα trace"):
                    st.pyplot(pose_3d(pose, seq), clear_figure=True)
            with right:
                st.download_button("Download PDB", demo.pdbs[choice], file_name=f"{choice}.pdb", mime="chemical/x-pdb")
                st.download_button("Download FASTA", demo.fasta, file_name="designs.fasta")
                st.code(seq or "", language=None)
            st.caption("Interactive view uses 3Dmol (network). Polar ligand atoms are sticks.")

    with tabs[1]:
        if demo is None:
            st.info("Run NISE to see ligand confidence vs sequence NLL over iterations.")
        else:
            st.pyplot(trajectories(demo.nise.history, demo.baseline.history), clear_figure=True)
            st.caption(
                "Paper Fig. 1c idea: NISE should jointly raise ligand confidence and lower sequence NLL. "
                "Energy ISE selects by ligand energy and typically does not."
            )

    with tabs[2]:
        if demo is None:
            st.info("Run NISE to score pocket substitutions.")
        elif not demo.mutations:
            st.warning("No substitutions lowered NLL at pocket positions.")
        else:
            st.dataframe(demo.mutations, use_container_width=True, hide_index=True)
            st.download_button("Download proofread FASTA", demo.proofread_fasta, file_name="proofread.fasta")
            st.caption("Top two substitutions are combined in `proofread_top2`, analogous to EPIC Q51N/M97L.")

    with tabs[3]:
        st.write(
            "Quadratic 1:1 fit of a fluorescence-anisotropy titration. "
            f"Paper EPIC Kd = {PAPER['exatecan']['epic_kd_uM']} µM; proofread double mutant = "
            f"{PAPER['exatecan']['epic_double_kd_nM']} nM."
        )
        upload = st.file_uploader("CSV with protein_M, anisotropy", type=["csv"], key="kd_csv")
        ligand_M = st.number_input("Total ligand (M)", value=5e-8, format="%.2e")
        example = (_examples() / "epic_fp.csv") if _examples() else None
        rows, headers = _read_table(upload, example)
        if rows:
            pcol = st.selectbox("Protein column", headers, index=headers.index("protein_M") if "protein_M" in headers else 0)
            ycol = st.selectbox("Signal column", headers, index=headers.index("anisotropy") if "anisotropy" in headers else min(1, len(headers) - 1))
            protein = np.array([float(r[pcol]) for r in rows])
            signal = np.array([float(r[ycol]) for r in rows])
            fit = fit_kd(protein, signal, ligand=float(ligand_M), n_boot=200)
            k1, k2, k3 = st.columns(3)
            k1.metric("Kd", f"{fit.kd * 1e9:.3g} nM")
            k2.metric("95% CI low", f"{fit.kd_low * 1e9:.3g} nM")
            k3.metric("95% CI high", f"{fit.kd_high * 1e9:.3g} nM")
            st.pyplot(kd_single(protein, signal, fit, float(ligand_M)), clear_figure=True)

    with tabs[4]:
        st.write("Two-state closed ⇌ open lactone kinetics. Free exatecan t½ is about 2 h in the paper.")
        st.pyplot(hydrolysis_paper(), clear_figure=True)
        upload = st.file_uploader("CSV with hours, open_fraction", type=["csv"], key="hyd_csv")
        example = (_examples() / "hydrolysis_free.csv") if _examples() else None
        rows, headers = _read_table(upload, example)
        if rows:
            tcol = st.selectbox("Time column", headers, index=headers.index("hours") if "hours" in headers else 0)
            fcol = st.selectbox("Open-fraction column", headers, index=headers.index("open_fraction") if "open_fraction" in headers else min(1, len(headers) - 1))
            t = np.array([float(r[tcol]) for r in rows])
            y = np.array([float(r[fcol]) for r in rows])
            fit = fit_hydrolysis(t, y)
            h1, h2, h3 = st.columns(3)
            h1.metric("t½", f"{fit.t_half_h:.2g} h")
            h2.metric("k_h", f"{fit.k_h:.3g} /h")
            h3.metric("Y_open ∞", f"{fit.y_open_inf:.2f}")
            pred = two_state_open_fraction(t, fit.k_h, fit.k_c)
            st.pyplot(hydrolysis_fit(t, y, pred), clear_figure=True)


if __name__ == "__main__":
    main()
