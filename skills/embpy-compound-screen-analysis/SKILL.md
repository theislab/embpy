---
name: embpy-compound-screen-analysis
description: Work with small-molecule screens and compound lists in embpy - resolve drug names to canonical SMILES, compute molecular fingerprints or ChemBERTa/MoLFormer embeddings, add RDKit properties, ChEMBL targets and clinical records, and find structurally similar compounds. Use when the user has drug names or SMILES in an .h5ad (a compound screen, sci-Plex, LINCS-style data) and wants them standardized, embedded, annotated with mechanism of action or development phase, or compared by structure.
---

# Compound screen analysis

Expected input: an `.h5ad` whose `.obs` has a compound column (names or SMILES), here
`compound`, often with one row per well or cell.

## 1. Standardize to canonical SMILES

```python
resolve_identifiers(path="data/compounds.h5ad", obs_column="compound", entity_type="molecule", target_id_type="smiles", output_path="data/compounds_smiles.h5ad")
```

- Writes the canonical SMILES to `.obs["compound_mapped"]`. Names are looked up online
  (PubChem / CIR); SMILES inputs are canonicalized locally with RDKit.
- Controls such as `DMSO` or `vehicle` do not map; check that `unmapped_sample` contains only
  controls before continuing, and tell the user about any real compound that failed.

## 2. Embed

```python
compute_embeddings(path="data/compounds_smiles.h5ad", models="morgan_fp", entity_type="molecule", obs_column="compound_mapped", missing="nan", output_path="data/compounds_embedded.h5ad")
```

- `morgan_fp`, `rdkit_fp` and `maccs_fp` run on CPU with no download. `chemberta2MTR` and
  `molformer_base` are learned embeddings (download, slower). `list_model_registry(entity_type="molecule")`
  lists all options.
- `missing="nan"` keeps rows whose compound did not resolve (controls) as NaN rows.

## 3. Annotate

```python
annotate_metadata(path="data/compounds_embedded.h5ad", entity_type="molecule", annotation_types=["properties"], column="compound_mapped", output_path="data/compounds_annotated.h5ad")
```

- `properties` adds RDKit columns (`mol_molecular_weight`, `mol_logp`, `mol_qed`, ...).
  `interactions` (ChEMBL targets and mechanism), `pathways` (KEGG) and `diseases` (PubChem)
  query web services and take longer.
- For the ChEMBL clinical record (development phase, indications, mechanism, targets), use the
  Python API:

```python
import anndata as ad
import embpy as ep

adata = ad.read_h5ad("data/compounds_annotated.h5ad")
adata = ep.tl.annotate_drug_perturbations(adata, column="compound", sources=["development", "mechanisms"])
print(adata.obs.filter(like="drug_").drop_duplicates().head())
```

This adds `drug_*` columns to `.obs` and the full records to `adata.uns["chembl_annotations"]`.

## 4. Structurally similar compounds

```python
key = "X_emb__molecule__morgan_fp__pool_mean"
treated = adata[adata.obs["compound_mapped"].notna()].copy()
per_compound = ep.tl.pseudobulk_embeddings(treated, group_col="compound", obsm_key=key)
print(ep.tl.rank_perturbations(per_compound, "aspirin", obsm_key=key, top_k=3))
```

Fingerprint similarity is structural similarity, not shared biology; say so when reporting.
For phenotype-based similarity, embed the measured response instead (see
`embpy-perturbation-screen-analysis`).
