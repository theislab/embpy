"""verify_full_ecosystem.py

Ecosystem-Wide Verification Harness for the embpy Model Context Protocol (MCP) Server.
Demonstrates procedural execution across:
- Phase A: Preprocessing & Resolution Tools (embpy.pp)
- Phase B: Universal BioEmbedder & Model Registry (~150 Models)
- Phase C: Universal Annotation Tools (embpy.annotate)
- Phase D: Analysis, Visualisation & Benchmarking (embpy.tl / embpy.pl)

Strictly verifies:
1. Exact Pydantic/JSON schemas with zero **kwargs.
2. Gene/molecule resolution that respects the requested species (detection is opt-in).
3. Exact row-alignment preservation across duplicate indices and blank entries.
4. Strict enforcement of preprocessing thresholds.
5. Dynamic model registry catalog (~150 models).
6. Multi-modality routing across cells, molecules, and proteins.
7. Token-saving rule compliance (compact metadata payloads, no leaked matrices).
8. Multi-domain biological annotation with strict failure error reporting.
9. 2D UMAP projection and scIB / perturbation benchmark evaluations.
"""

from __future__ import annotations

import inspect
import json
import os
import sys
import tempfile
import unittest.mock as mock

import anndata as ad
import numpy as np
import pandas as pd

from embpy.mcp import tools
from embpy.mcp.server import create_server


def print_banner(title: str) -> None:
    sep = "=" * 80
    print(f"\n{sep}\n{title}\n{sep}")


def print_section(title: str) -> None:
    print(f"\n>>> {title}\n" + "-" * 60)


def run_phase_a(tmpdir: str) -> dict[str, str]:
    print_banner("PHASE A: Preprocessing & Resolution Tools (embpy.pp)")

    # ---------------------------------------------------------
    # 1. Schema Integrity Check
    # ---------------------------------------------------------
    print_section("A.1 Schema Integrity Check (FastMCP JSON Schema & inspect.signature)")
    server = create_server()
    pp_tools = ["inspect_data", "resolve_identifiers", "preprocess_counts"]

    for tool_name in pp_tools:
        fn = getattr(tools, tool_name)
        sig = inspect.signature(fn)
        has_varkw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
        assert not has_varkw, f"VIOLATION: {tool_name} declares **kwargs in signature!"

        # Fetch FastMCP tool parameter schema
        mcp_tool = next(t for t in server._tool_manager.list_tools() if t.name == tool_name)
        param_schema = mcp_tool.parameters

        print(f"Tool: {tool_name}")
        print(f"  Python Signature : {fn.__name__}{sig}")
        print(f"  Strict No-Kwargs : [VERIFIED - NO **kwargs]")
        print(f"  Properties Count : {len(param_schema.get('properties', {}))}")
        print(f"  JSON Schema Snippet:\n{json.dumps(param_schema, indent=2)[:400]} ...\n")

    # ---------------------------------------------------------
    # 2. Multi-Entity Resolution
    # ---------------------------------------------------------
    print_section("A.2 Multi-Entity Resolution (explicit species, opt-in detection)")

    def _resolve(label: str, **kwargs):
        res = tools.resolve_identifiers(**kwargs)
        keys = ("status", "error_type", "sample_mapping", "organism_used", "notes", "target_id_type")
        print(f"{label}: {json.dumps({k: res[k] for k in keys if k in res})}")
        return res

    human = _resolve("Human (organism='human')", identifiers=["TP53", "EGFR"], organism="human")
    assert human["status"] == "success", human
    assert human["sample_mapping"] == {"TP53": "ENSG00000141510", "EGFR": "ENSG00000146648"}

    # Capitalization never changes an explicitly requested species.
    egfr = _resolve("'Egfr' with organism='human'", identifiers=["Egfr"], organism="human")
    assert egfr["sample_mapping"]["Egfr"] == "ENSG00000146648", egfr
    assert "case-insensitive" in egfr["notes"]["Egfr"]

    mouse = _resolve("Mouse (organism='mouse')", identifiers=["Trp53"], organism="mouse")
    assert mouse["sample_mapping"]["Trp53"] == "ENSMUSG00000059552", mouse

    conflict = _resolve("Mouse Ensembl id with organism='human'", identifiers=["ENSMUSG00000059552"], organism="human")
    assert conflict["status"] == "error" and conflict["error_type"] == "SpeciesMismatchError", conflict

    auto = _resolve("Opt-in detection (organism='auto')", identifiers=["TP53", "Trp53"], organism="auto")
    assert auto["organism_used"] == {"human": 1, "mouse": 1}, auto

    aspirin = _resolve("Molecule name (entity_type='molecule')", identifiers=["Aspirin"], entity_type="molecule")
    assert aspirin["target_id_type"] == "smiles", aspirin
    assert "CC(=O)Oc1ccccc1C(=O)O" in aspirin["sample_mapping"].get("Aspirin", ""), aspirin

    banana = _resolve("Unsupported target", identifiers=["TP53"], target_id_type="banana")
    assert banana["status"] == "error", banana
    print("\n[VERIFIED] Species are respected, detection is opt-in and reported, and targets are validated.")

    # ---------------------------------------------------------
    # 3. Row-Aligned Integrity
    # ---------------------------------------------------------
    print_section("A.3 Row-Aligned Integrity (Duplicates, Blanks, and Custom Index)")
    raw_h5ad = os.path.join(tmpdir, "aligned_input.h5ad")
    resolved_h5ad = os.path.join(tmpdir, "aligned_output.h5ad")

    # Construct DataFrame with duplicate indices, blanks, and custom row names
    index_names = ["feature_1", "feature_2", "feature_1", "feature_blank"]
    symbols = ["TP53", "EGFR", "TP53", ""]
    var_df = pd.DataFrame({"symbol": symbols, "channel": [1, 2, 3, 4]}, index=index_names)
    X_mat = np.random.randn(3, 4).astype(np.float32)
    adata_aligned = ad.AnnData(X=X_mat, var=var_df)
    adata_aligned.obs_names = [f"cell_{i}" for i in range(3)]
    adata_aligned.write_h5ad(raw_h5ad)

    print(f"Original var_names : {list(adata_aligned.var_names)}")
    print(f"Original symbols   : {symbols}")

    res_row = tools.resolve_identifiers(
        path=raw_h5ad,
        var_column="symbol",
        output_path=resolved_h5ad,
    )
    print("\nResolution Output Metadata:")
    print(json.dumps(res_row, indent=2))

    # Reopen and inspect row alignment
    reopened = ad.read_h5ad(resolved_h5ad)
    print("\nReopened .var table:")
    print(reopened.var)

    assert list(reopened.var_names) == index_names, "Index ordering or duplicate names were scrambled!"
    mapped_series = reopened.var["symbol_mapped"]
    assert mapped_series.iloc[0] == "ENSG00000141510", f"Row 0 mismatch: {mapped_series.iloc[0]}"
    assert mapped_series.iloc[1] == "ENSG00000146648", f"Row 1 mismatch: {mapped_series.iloc[1]}"
    assert mapped_series.iloc[2] == "ENSG00000141510", f"Row 2 duplicate mismatch: {mapped_series.iloc[2]}"
    assert pd.isna(mapped_series.iloc[3]) or mapped_series.iloc[3] is None, f"Row 3 empty row not None: {mapped_series.iloc[3]}"
    print("\n[VERIFIED] Exact row alignment, original order, duplicates, and empty rows strictly preserved.")

    # ---------------------------------------------------------
    # 4. Strict Preprocessing Settings
    # ---------------------------------------------------------
    print_section("A.4 Strict Preprocessing (Enforcing min_genes=200 on 40-Gene AnnData)")
    prep_raw_path = os.path.join(tmpdir, "40_genes_dataset.h5ad")
    prep_out_path = os.path.join(tmpdir, "preprocessed_strict.h5ad")

    # 15 cells, 40 genes
    X_small = np.random.poisson(lam=2.0, size=(15, 40)).astype(np.float32)
    adata_small = ad.AnnData(X=X_small)
    adata_small.obs_names = [f"cell_{i}" for i in range(15)]
    adata_small.var_names = [f"gene_{j}" for j in range(40)]
    adata_small.write_h5ad(prep_raw_path)

    print(f"Input AnnData: {adata_small.n_obs} cells x {adata_small.n_vars} genes.")
    print("Invoking preprocess_counts with min_genes=200 and auto_adjust=False...")

    res_prep = tools.preprocess_counts(
        path=prep_raw_path,
        output_path=prep_out_path,
        min_genes=200,
        auto_adjust=False,
        scale=True,
    )
    print("\nPreprocessing Result Payload:")
    print(json.dumps(res_prep, indent=2))

    assert res_prep["status"] == "success"
    assert res_prep["n_cells"] == 0, f"Expected 0 cells, got: {res_prep['n_cells']}"
    saved_prep = ad.read_h5ad(prep_out_path)
    assert saved_prep.n_obs == 0, f"Saved dataset cell count is {saved_prep.n_obs}, expected 0!"
    print("\n[VERIFIED] min_genes=200 strictly honored; zero cells survived without silent threshold attenuation.")

    return {
        "aligned_h5ad": resolved_h5ad,
        "prep_raw_h5ad": prep_raw_path,
    }


def run_phase_b(tmpdir: str) -> dict[str, str]:
    print_banner("PHASE B: Universal BioEmbedder & Model Registry (~150 Models)")

    # ---------------------------------------------------------
    # 1. Dynamic Catalog Verification
    # ---------------------------------------------------------
    print_section("B.1 Dynamic Catalog Verification (~150 Native Registered Models)")
    reg_summary = tools.list_model_registry()
    print("Catalog Summary Payload:")
    print(json.dumps({k: v for k, v in reg_summary.items() if k != "all_registered_models"}, indent=2))

    total_models = reg_summary["total_models_available"]
    recommended = reg_summary["recommended_models"]
    print(f"\nTotal Registered Models Detected: {total_models}")
    print(f"Recommended Models by Modality   : {json.dumps(recommended, indent=2)}")

    assert total_models >= 100, f"Expected >= 100 models, found: {total_models}"
    assert "molformer_base" in recommended["molecule"], "molformer_base must be in recommended molecule models"
    assert "molformer" not in recommended["molecule"], "Invalid placeholder 'molformer' detected!"
    assert any("esm2" in m for m in recommended["protein"]), "ESM-2 must be present in protein recommendations"
    assert "pca" in recommended["single_cell"], "PCA must be present in single_cell recommendations"
    print("\n[VERIFIED] Catalog is dynamically populated from internal registries with verified keys.")

    # ---------------------------------------------------------
    # 2. Modality Routing Across 3 Distinct Modalities
    # ---------------------------------------------------------
    print_section("B.2 Modality Routing & B.3 Token-Saving Proof")

    # Modality 1: Cells / Genes via PCA
    print("\n--- Modality 1: Single Cells (Model: 'pca') ---")
    cell_raw_path = os.path.join(tmpdir, "cells_raw.h5ad")
    cell_emb_path = os.path.join(tmpdir, "cells_embedded.h5ad")

    # 50 cells x 20 genes with cell_type and batch annotations
    rng = np.random.RandomState(42)
    X_cells = rng.poisson(lam=3.0, size=(50, 20)).astype(np.float32)
    obs_cells = pd.DataFrame(
        {
            "cell_type": ["B_cell" if i < 25 else "T_cell" for i in range(50)],
            "batch": ["batch_1" if i % 2 == 0 else "batch_2" for i in range(50)],
            "perturbation": ["ctrl" if i < 20 else ("TP53" if i < 35 else "BRCA1") for i in range(50)],
            "split": ["train" if i % 2 == 0 else "test" for i in range(50)],
        },
        index=[f"cell_{i}" for i in range(50)],
    )
    adata_cells = ad.AnnData(X=X_cells, obs=obs_cells)
    adata_cells.var_names = [f"gene_{j}" for j in range(20)]
    adata_cells.write_h5ad(cell_raw_path)

    res_cells = tools.compute_embeddings(
        path=cell_raw_path,
        output_path=cell_emb_path,
        models="pca",
        entity_type="cell",
        preprocessing="none",
    )
    print("Cells Embedding Payload (Token-Saving Metadata Only):")
    print(json.dumps(res_cells, indent=2))
    assert res_cells["status"] == "success"
    assert "X_pca" in res_cells["embedding_keys"]
    assert "X" not in res_cells and "matrix" not in res_cells, "Raw matrix leaked into response!"

    # Modality 2: Molecules via Morgan Fingerprints (or ChemBerta/MolFormer)
    print("\n--- Modality 2: Small Molecules (Model: 'morgan_fp') ---")
    mol_raw_path = os.path.join(tmpdir, "molecules_raw.h5ad")
    mol_emb_path = os.path.join(tmpdir, "molecules_embedded.h5ad")

    mol_smiles = [
        "CC(=O)Oc1ccccc1C(=O)O",  # Aspirin
        "CN(C)C(=N)NC(=N)N",      # Metformin
        "c1ccccc1",                # Benzene
    ]
    adata_mols = ad.AnnData(
        X=np.zeros((len(mol_smiles), 2), dtype=np.float32),
        obs=pd.DataFrame({"smiles": mol_smiles}, index=mol_smiles),
    )
    adata_mols.write_h5ad(mol_raw_path)

    res_mols = tools.compute_embeddings(
        path=mol_raw_path,
        output_path=mol_emb_path,
        models="morgan_fp",
        entity_type="molecule",
    )
    print("Molecules Embedding Payload (Token-Saving Metadata Only):")
    print(json.dumps(res_mols, indent=2))
    assert res_mols["status"] == "success"
    assert any("morgan_fp" in k for k in res_mols["embedding_keys"])
    assert "matrix" not in res_mols, "Raw molecule matrix leaked into response!"

    # Modality 3: Proteins via lightweight ESM2 (8M)
    print("\n--- Modality 3: Proteins (Model: 'esm2_8M') ---")
    prot_raw_path = os.path.join(tmpdir, "proteins_raw.h5ad")
    prot_emb_path = os.path.join(tmpdir, "proteins_embedded.h5ad")

    # Protein features in .var
    adata_prots = ad.AnnData(
        X=np.zeros((2, 2), dtype=np.float32),
        var=pd.DataFrame(index=["TP53", "EGFR"]),
    )
    adata_prots.write_h5ad(prot_raw_path)

    res_prots = tools.compute_embeddings(
        path=prot_raw_path,
        output_path=prot_emb_path,
        models="esm2_8M",
        entity_type="protein",
    )
    print("Proteins Embedding Payload (Token-Saving Metadata Only):")
    print(json.dumps(res_prots, indent=2))
    assert res_prots["status"] == "success"
    assert any("esm2_8M" in k for k in res_prots["embedding_keys"])
    assert "matrix" not in res_prots, "Raw protein matrix leaked into response!"

    print("\n[VERIFIED] All 3 biological modalities routed successfully; zero raw matrices leaked to stdout.")

    return {
        "cell_embedded_h5ad": cell_emb_path,
        "mol_embedded_h5ad": mol_emb_path,
        "prot_embedded_h5ad": prot_emb_path,
    }


def run_phase_c(tmpdir: str) -> None:
    print_banner("PHASE C: Universal Annotation Tools (embpy.annotate)")

    # ---------------------------------------------------------
    # 1. Multi-Domain Annotation
    # ---------------------------------------------------------
    print_section("C.1 Multi-Domain Annotation (Gene Perturbations & Molecule Pathways)")

    # Domain 1: Genes (Perturbations)
    print("\n--- Domain 1: Gene Annotations (GO Terms & Knowledge Base) ---")
    gene_ann_in = os.path.join(tmpdir, "gene_ann_in.h5ad")
    gene_ann_out = os.path.join(tmpdir, "gene_ann_out.h5ad")
    adata_gene = ad.AnnData(
        X=np.zeros((10, 5), dtype=np.float32),
        obs=pd.DataFrame({"perturbation": ["TP53", "BRCA1"] * 5}),
    )
    adata_gene.write_h5ad(gene_ann_in)

    res_gene_ann = tools.annotate_metadata(
        path=gene_ann_in,
        output_path=gene_ann_out,
        entity_type="gene",
        annotation_types=["go", "pathways"],
        column="perturbation",
    )
    print("Gene Annotation Result:")
    print(json.dumps(res_gene_ann, indent=2))
    assert res_gene_ann["status"] in ("success", "partial_success")
    assert res_gene_ann["per_type"]["go"]["annotated"] > 0, "GO annotation must actually retrieve GO terms"
    assert res_gene_ann["per_type"]["pathways"]["annotated"] > 0

    # Domain 2: Molecules (Structural Properties & Bioactivity)
    print("\n--- Domain 2: Molecule Annotations (RDKit & ChEMBL Metadata) ---")
    mol_ann_in = os.path.join(tmpdir, "mol_ann_in.h5ad")
    mol_ann_out = os.path.join(tmpdir, "mol_ann_out.h5ad")
    adata_mol = ad.AnnData(
        X=np.zeros((3, 2), dtype=np.float32),
        obs=pd.DataFrame({"smiles": ["CC(=O)Oc1ccccc1C(=O)O", "CN(C)C(=N)NC(=N)N", "c1ccccc1"]}),
    )
    adata_mol.write_h5ad(mol_ann_in)

    res_mol_ann = tools.annotate_metadata(
        path=mol_ann_in,
        output_path=mol_ann_out,
        entity_type="molecule",
        column="smiles",
        annotation_types=["properties", "interactions"],
    )
    print("Molecule Annotation Result:")
    print(json.dumps(res_mol_ann, indent=2))
    assert res_mol_ann["per_type"]["properties"]["annotated"] == 3, res_mol_ann
    assert "mol_molecular_weight" in res_mol_ann["new_columns_added"]
    assert "mol_logp" in res_mol_ann["new_columns_added"]

    # ---------------------------------------------------------
    # 2. Strict Failure Handling
    # ---------------------------------------------------------
    print_section("C.2 Strict Failure Handling on Service Timeout/Drop")
    fail_h5ad = os.path.join(tmpdir, "fail_input.h5ad")
    fail_out_h5ad = os.path.join(tmpdir, "fail_out.h5ad")
    adata_gene.write_h5ad(fail_h5ad)

    import requests

    print("Simulating an outage of every annotation service...")
    outage = requests.ConnectionError("Connection timeout to MyGene reference service")
    with mock.patch("requests.get", side_effect=outage), mock.patch("requests.post", side_effect=outage):
        res_fail = tools.annotate_metadata(
            path=fail_h5ad,
            output_path=fail_out_h5ad,
            entity_type="gene",
            annotation_types=["go", "pathways"],
            column="perturbation",
        )
    print("Failure Response Payload:")
    print(json.dumps(res_fail, indent=2))
    assert res_fail["status"] == "error", f"Must return status='error', got {res_fail['status']}"
    assert res_fail["error_type"] == "AnnotationServiceError"
    assert res_fail["successful_count"] == 0
    assert "Connection timeout" in res_fail["service_errors_sample"][0]["error"]
    assert not os.path.exists(fail_out_h5ad), "Failed execution must not produce partial/placeholder file!"

    print("Simulating services that answer with no annotations...")
    empty = mock.MagicMock(status_code=200, ok=True)
    empty.json.return_value = {"hits": [], "results": []}
    with mock.patch("requests.get", return_value=empty), mock.patch("requests.post", return_value=empty):
        res_empty = tools.annotate_metadata(
            path=fail_h5ad,
            output_path=fail_out_h5ad,
            entity_type="gene",
            annotation_types=["go", "pathways"],
            column="perturbation",
        )
    print(json.dumps({k: res_empty.get(k) for k in ("status", "error_type", "per_type")}, indent=2))
    assert res_empty["status"] == "error" and res_empty["error_type"] == "NoAnnotationsFound"
    assert not os.path.exists(fail_out_h5ad)
    print("\n[VERIFIED] Outages and empty answers are reported as distinct errors; nothing is written.")


def run_phase_d(tmpdir: str, cell_embedded_h5ad: str) -> None:
    print_banner("PHASE D: Analysis, Visualisation & Benchmarking (embpy.tl / embpy.pl)")

    # ---------------------------------------------------------
    # 1. 2D Visualisation (UMAP)
    # ---------------------------------------------------------
    print_section("D.1 2D Visualisation (UMAP)")
    umap_out_path = os.path.join(tmpdir, "cells_umap.h5ad")
    print(f"Computing UMAP on embedding key 'X_pca' from {cell_embedded_h5ad}...")

    res_umap = tools.compute_umap(
        path=cell_embedded_h5ad,
        output_path=umap_out_path,
        embedding_key="X_pca",
        n_neighbors=10,
    )
    print("UMAP Calculation Payload:")
    print(json.dumps(res_umap, indent=2))

    assert res_umap["status"] == "success"
    assert res_umap["umap_shape"] == [50, 2]
    saved_umap = ad.read_h5ad(umap_out_path)
    assert res_umap["umap_key"] in saved_umap.obsm
    assert saved_umap.obsm[res_umap["umap_key"]].shape == (50, 2)
    print("\n[VERIFIED] UMAP projected coordinates stored strictly to disk; return payload returned dimensions only.")

    # ---------------------------------------------------------
    # 2. Explicit Benchmarking (scIB)
    # ---------------------------------------------------------
    print_section("D.2 Explicit Benchmarking (scIB Conservation & Batch)")

    # Test 2.1: Reject ambiguous / missing label
    print("Testing 2.1: Rejection of ambiguous/non-existent label key 'test'...")
    res_scib_fail = tools.benchmark_scib(
        path=cell_embedded_h5ad,
        embedding_keys=["X_pca"],
        label_key="test",  # Non-existent in adata.obs
    )
    print("Rejection Response Payload:")
    print(json.dumps(res_scib_fail, indent=2))
    assert res_scib_fail["status"] == "error"
    assert res_scib_fail["error_type"] == "KeyError"
    assert "label_key 'test' not found" in res_scib_fail["error"]
    print("  [PASS] Ambiguous label key successfully rejected.")

    # Test 2.2: Explicit valid benchmarking
    print("\nTesting 2.2: Computing scIB benchmark with explicit ground truth 'cell_type'...")
    res_scib_pass = tools.benchmark_scib(
        path=cell_embedded_h5ad,
        embedding_keys=["X_pca"],
        label_key="cell_type",
        batch_key="batch",
    )
    print("scIB Benchmark Summary:")
    print(f"  Status        : {res_scib_pass['status']}")
    print(f"  Rankings      : {res_scib_pass['rankings']}")
    print(f"  Markdown Table:\n{res_scib_pass['markdown_table']}")

    assert res_scib_pass["status"] == "success"
    assert "X_pca" in res_scib_pass["rankings"]
    assert len(res_scib_pass["markdown_table"]) > 20
    print("\n[VERIFIED] scIB metrics generated clean, human-readable markdown table without raw array dumping.")

    # ---------------------------------------------------------
    # 3. Perturbation Benchmarking (ArcInstitute cell-eval)
    # ---------------------------------------------------------
    print_section("D.3 Perturbation Benchmarking (ArcInstitute cell-eval)")

    # Test 3.1: Reject ambiguous split inputs without explicit pred_label/ref_label
    print("Testing 3.1: Rejection of split_col without explicit pred_label & ref_label...")
    res_pert_fail = tools.benchmark_perturbations(
        path=cell_embedded_h5ad,
        control_pert="ctrl",
        pert_col="perturbation",
        split_col="split",
    )
    print("Ambiguous Split Rejection Payload:")
    print(json.dumps(res_pert_fail, indent=2))
    assert res_pert_fail["status"] == "error"
    assert res_pert_fail["error_type"] == "ValueError"
    assert "Ambiguous benchmark inputs" in res_pert_fail["error"]
    print("  [PASS] Ambiguous split input strictly rejected.")

    # Test 3.2: Explicit perturbation evaluation
    print("\nTesting 3.2: Explicit perturbation evaluation with pred_label='test' & ref_label='train'...")
    pert_csv = os.path.join(tmpdir, "pert_benchmark.csv")
    res_pert_pass = tools.benchmark_perturbations(
        path=cell_embedded_h5ad,
        control_pert="ctrl",
        pert_col="perturbation",
        split_col="split",
        pred_label="test",
        ref_label="train",
        output_csv=pert_csv,
    )
    print("Perturbation Benchmark Output:")
    print(f"  Status          : {res_pert_pass['status']}")
    print(f"  Perturbations   : {res_pert_pass['n_perturbations']}")
    print(f"  Output CSV      : {res_pert_pass['output_csv']}")
    print(f"  Markdown Table  :\n{res_pert_pass['markdown_table']}")

    assert res_pert_pass["status"] == "success"
    assert res_pert_pass["n_perturbations"] >= 1
    assert os.path.exists(pert_csv)
    print("\n[VERIFIED] ArcInstitute cell-eval executed cleanly and output formatted differential metrics.")


def main():
    print_banner("EMBPY MCP SERVER: COMPREHENSIVE ECOSYSTEM-WIDE VERIFICATION HARNESS")
    print("Initializing isolated verification environment...")

    with tempfile.TemporaryDirectory() as tmpdir:
        print(f"Temporary Verification Root: {tmpdir}")
        try:
            phase_a_artifacts = run_phase_a(tmpdir)
            phase_b_artifacts = run_phase_b(tmpdir)
            run_phase_c(tmpdir)
            run_phase_d(tmpdir, phase_b_artifacts["cell_embedded_h5ad"])

            print_banner("ALL PHASES (A, B, C, D) PASSED FLAWLESSLY WITH ZERO DEFECTS")
            print("1. All schemas are strictly typed with NO **kwargs.")
            print("2. Multi-entity resolution functions across Human, Mouse, and Molecules.")
            print("3. Row-alignment integrity is 100% preserved.")
            print("4. Preprocessing strictly enforces user cutoffs.")
            print("5. Dynamic catalog exposes ~150 native models.")
            print("6. Modality routing verified for cells, molecules, and proteins.")
            print("7. Token-saving rule strictly obeyed (no raw arrays leaked).")
            print("8. Multi-domain annotations and error reporting verified.")
            print("9. UMAP, scIB, and perturbation benchmarking verified.")
            print("=" * 80)
            sys.exit(0)
        except Exception as exc:
            print(f"\n[FATAL ERROR IN ECOSYSTEM VERIFICATION]: {exc}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            sys.exit(1)


if __name__ == "__main__":
    main()
