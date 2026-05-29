# Technical Guide

This page holds the details that are useful when building on top of embpy, but
too heavy for the GitHub README.

For per-function documentation, see the [API reference](api.md).

## Package Layout

This repository contains two Python packages:

| Package | Purpose | Source |
| --- | --- | --- |
| `embpy` | Embeddings, resolvers, annotations, preprocessing, plotting, and analysis utilities. | `src/embpy/` |
| `world_model` | Perturbation world-model training, evaluation, Slurm submission, and report workflows. | `src/world_model/` |

`world_model` depends on `embpy`. `embpy` should not import from
`world_model`.

The world-model documentation lives in
[`src/world_model/README.md`](../src/world_model/README.md).

## Standardized Output Contract

`BioEmbedder.embed(...)` is the preferred user-facing embedding entry point.
It accepts direct identifiers, pandas objects, AnnData objects, and supported
input files such as CSV, TSV, and Parquet.

The output can be:

- `output="anndata"` for scverse workflows
- `output="table"` for embedding tables
- `output="payload"` for canonical IDs, matrix, aliases, and provenance

Generated embeddings are never placed in `.X`.

| Entity type | Default AnnData location | Notes |
| --- | --- | --- |
| genes | `.varm` | Feature-aligned by default, canonicalized to gene IDs when possible. |
| gene perturbation labels | `.obsm` | Set `is_perturbation=True` when genes are row-level action labels. |
| proteins | `.varm` | Feature-aligned, usually keyed by UniProt/canonical protein IDs. |
| molecules | `.obsm` | Observation-aligned, canonical SMILES used when possible. |
| text and sequences | `.obsm` | Observation-aligned raw inputs. |
| cells | `.obsm` | One vector per cell. |
| perturbation morphology/actions | `.uns` | Entity-aligned payload; materialize to `.obsm` only when a downstream model needs one vector per observation. |

For gene perturbation/action embeddings, attach directly to rows:

```python
adata = embedder.embed(
    adata,
    entity_type="gene",
    obs_column="perturbation",
    id_type="symbol",
    model="esm2_650M",
    output="anndata",
    is_perturbation=True,
    key="X_pert_esm2_650M",
)
```

For entity-aligned perturbation payloads, such as morphology embeddings,
keep the source-of-truth table in `.uns` and materialize a row-aligned
matrix when needed:

```python
from embpy.io import materialize_perturbation_obsm

adata = materialize_perturbation_obsm(
    adata,
    embedding_key="X_pert_esm2_650M",
    perturbation_key="perturbation",
    obsm_key="X_pert_esm2_650M",
)
```

## Input Normalization

`BioEmbedder.embed(...)` normalizes inputs before model inference:

- Python lists, tuples, NumPy arrays, and pandas Series are treated as
  identifier collections.
- Multi-column pandas DataFrames require `identifier_column=...`.
- AnnData identifiers can come from `.obs_names`, `.var_names`,
  `obs_column=...`, or `var_column=...`.
- File inputs can be CSV, TSV, or Parquet.

Canonical biological IDs are used as primary keys where possible. Original
symbols, names, SMILES, and user inputs are preserved as aliases and metadata.

## Single-Cell Preprocessing Policy

Cell embeddings go through the single-cell preprocessing policy owned by
`embpy.pp`.

Use:

```python
embedder.embed(
    adata,
    entity_type="cell",
    model="pca",
    preprocessing="auto",
    output="anndata",
)
```

`preprocessing="auto"` resolves the appropriate preparation for the requested
model. The policy can preserve raw counts, create `.layers["counts"]`, create
`.layers["log_normalized"]`, select highly variable genes, and record the
resolved preprocessing choices in `.uns["embpy_cell_embeddings"]`.

Users can still override the important knobs:

- `n_top_genes`
- `select_hvg`
- `target_sum`
- `log_transform`
- `scale`
- `min_genes`
- `min_cells`
- `max_pct_mito`
- `pca_use_hvg`

## Environments and Installation

Pixi is the recommended development path:

```bash
pixi install -e default
pixi run -e default verify
```

Common environments:

```bash
pixi install -e default   # CPU/dev
pixi install -e mps       # Apple Silicon/PyTorch MPS
pixi install -e gpu       # Linux CUDA GPU
pixi install -e boltz     # Boltz-specific examples
```

Pip users can start with:

```bash
pip install embpy
```

Optional model families may require extra dependencies. Large GPU models should
be installed in an environment with compatible PyTorch/CUDA wheels.

## Model Keys

The active model registry depends on the installed optional dependencies.

Use:

```python
from embpy import BioEmbedder

embedder = BioEmbedder(device="auto")
embedder.list_available_models()
```

for the model keys available in the current environment.

The registry covers DNA, protein, molecule, single-cell, morphology, text, and
structure-backed models.

## Annotations

High-level annotation helpers live in `embpy.tl`:

```python
from embpy import tl

tl.annotate_gene_perturbations(adata, column="gene")
tl.annotate_proteins(adata, column="protein")
tl.annotate_molecules(adata, column="smiles")
```

Resource-level annotators live in `embpy.resources`:

```python
from embpy.resources import CellLineAnnotator

annotator = CellLineAnnotator()
annotator.annotate_adata(adata, column="cell_line")
```

## Plotting and Analysis

Analysis helpers live in `embpy.tl`; plotting helpers live in `embpy.pl`.

Common functions include:

- `tl.compute_similarity`
- `tl.compute_distance_matrix`
- `tl.compute_knn_overlap`
- `tl.rank_perturbations`
- `pl.plot_embedding_space`
- `pl.knn_overlap`
- `pl.cross_embedding_correlation`
- `pl.embedding_norms`

See the [API reference](api.md) for the full list.

## World Model Integration

The world-model workflows consume AnnData files with explicit state and action
embedding keys.

Typical cross-modality setup:

- cell state: `.obsm["X_stack"]`
- support action: `.obsm["X_pert_esm2_650M"]`
- query action: `.obsm["X_pert_subcell_mae_rybg"]`

The world-model package has its own README with the current Slurm submission,
output-layout, baseline, and report workflows:

[`src/world_model/README.md`](../src/world_model/README.md)

## Developer Notes

Keep user-facing examples in the README and notebooks centered on
`BioEmbedder.embed(...)`. Lower-level helpers may remain public and documented,
but tutorials should introduce them only when they clarify a real workflow.
