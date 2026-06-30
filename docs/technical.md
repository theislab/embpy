# Technical Guide

This page holds the details that are useful when building on top of embpy, but
too heavy for the GitHub README.

For per-function documentation, see the [API reference](api.md).

## Package Layout

This repository contains the public `embpy` Python package:

| Package | Purpose | Source |
| --- | --- | --- |
| `embpy` | Embeddings, resolvers, annotations, preprocessing, plotting, and analysis utilities. | `src/embpy/` |

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

### pip / uv extras

`pip install embpy` installs a **lightweight core** that always builds without a
compiler or GPU: numpy, pandas, anndata, scikit-learn, rdkit, matplotlib and the
lightweight identifier-resolution libraries. The heavy / build-fragile
dependencies are split into optional extras and imported lazily, so you only
install what a given workflow needs:

| Extra | Adds | For |
| --- | --- | --- |
| `models` | torch, transformers, torch-geometric, sentencepiece | Computing embeddings via `BioEmbedder` and the model wrappers |
| `bio` | biopython | FASTA/sequence parsing in the gene/protein resolvers |
| `genome` | pysam, pyensembl | Local indexed-genome access |
| `scanpy` | scanpy | Single-cell preprocessing |
| `scib` | scib, scanpy | scIB metrics for single-cell embeddings (`tl.compute_scib_metrics`) |
| `benchmark` | xgboost | The optional xgboost regressor in `tl.benchmark_embeddings` |
| `seqmodels`, `esm3`, `ppi`, `morphology`, ... | see `pyproject.toml` | Specific model families |
| `all` | every pip-installable extra above | Full CPU feature set |

```bash
pip install embpy                  # lightweight core
pip install "embpy[models]"        # + embedding backends
pip install "embpy[models,scib]"   # + backends + scIB metrics
```

Each extra resolves independently; nothing forces a heavy dependency on a user
who does not need it.

### Pixi environments

Each pixi environment is self-contained and installed independently — you do
**not** need every environment to solve in order to use one:

```bash
pixi install -e default   # CPU/dev
pixi run -e default verify
pixi install -e mps       # Apple Silicon/PyTorch MPS
pixi install -e gpu       # Linux CUDA GPU
pixi install -e helical-gpu  # single-cell foundation models (scGPT, Geneformer, ...)
pixi install -e boltz     # Boltz-specific examples
```

Pixi keeps a single lockfile for the whole workspace, so regenerating it solves
every environment. When an unrelated environment (for example a CUDA-only one)
cannot solve on your machine, or you simply want to skip the re-solve, install
the one you want directly from the committed lock:

```bash
pixi install -e default --frozen
```

The CPU environments (`default`, `dev`, `docs`) share one solve group and are
kept independent from the GPU/CUDA-only ones, so a failure in a GPU environment
never blocks a CPU install.

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

## Developer Notes

Keep user-facing examples in the README and notebooks centered on
`BioEmbedder.embed(...)`. Lower-level helpers may remain public and documented,
but tutorials should introduce them only when they clarify a real workflow.
