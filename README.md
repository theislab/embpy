# embpy

[![Tests][badge-tests]][tests]
[![Documentation][badge-docs]][documentation]

[badge-tests]: https://img.shields.io/github/actions/workflow/status/theislab/embpy/test.yaml?branch=main
[badge-docs]: https://img.shields.io/readthedocs/embpy
[tests]: https://github.com/theislab/embpy/actions/workflows/test.yaml
[documentation]: https://embpy.readthedocs.io/

**embpy** is a Python toolkit for generating biological embeddings with one
unified API.

Use it to embed genes, proteins, small molecules, morphology perturbations, and
single cells; annotate the resulting objects; and compare embeddings with
scverse-friendly plotting and analysis utilities.

<p align="center">
  <img src="docs/embpy_architecture.svg" alt="embpy architecture" width="900"/>
</p>

## What embpy Does

- Embeds biological entities through `BioEmbedder.embed(...)`.
- Resolves biological identifiers into model-ready inputs, such as gene
  sequences, protein sequences, SMILES strings, and morphology images.
- Returns AnnData, tables, or payloads with provenance and canonical IDs.
- Stores generated embeddings outside `.X`, using `.obsm`, `.varm`, or `.uns`
  according to the entity type.
- Adds real metadata annotations for genes, proteins, molecules, and cell
  lines.
- Provides plotting and comparison helpers for embedding quality checks.

## Install

### pip / uv

There are **two supported installs**. Both work with `pip` and `uv` — these are
standard extras, not tool-specific.

```bash
pip install "embpy[cpu]"
```

```bash
pip install "embpy[gpu]"
```

`[cpu]` pulls CPU-only PyTorch plus every backend that runs on CPU — enough for all
the tutorials. `[gpu]` pulls CUDA PyTorch and the same backends, GPU-accelerated.
Pick `[gpu]` on a machine with an NVIDIA card, `[cpu]` otherwise.

Installing the bare package (`pip install embpy`) gives the lightweight core —
resolvers, IO, annotation, plotting and analysis — but no embedding backends. It
exists so the docs build and CI stay fast; for real use, pick `[cpu]` or `[gpu]`.

<details>
<summary>Backends that must be installed separately</summary>

A few backends cannot be installed by pip on an arbitrary machine, so they are in
neither extra. embpy degrades gracefully without them and the warning it logs names
the command to run:

| Backend | Install | Why it is separate |
| --- | --- | --- |
| Caduceus | `pip install "embpy[caduceus]"` | `mamba-ssm` needs a CUDA toolchain at build time |
| Scooby | `pip install "embpy[scooby]"` | installs from a git URL |
| Evo / Evo2 | `pip install "embpy[evo]"` | CUDA toolchain |
| Boltz-2 | `pip install "embpy[boltz]"` | heavy structure stack |
| MiniMol | `pip install "embpy[minimol]"` | pinned graphium stack |
| AlphaGenome | `pip install "embpy[alphagenome]"` | API client, needs a key |
| single-cell FMs | `pip install "embpy[helical]"` | pins torch/transformers |

</details>

### On an HPC cluster (Slurm)

Verified on Helmholtz Munich HPC; adjust partition/QoS names for your site.

**1. Grab an interactive CPU node.** The tutorials are small enough for CPU:

```bash
srun --partition=interactive_cpu_p --qos=interactive_cpu --cpus-per-task=8 --mem=16G --time=04:00:00 --pty bash
```

If you see `QOSMaxMemoryPerJob`, that is a **policy cap, not a shortage** — ask for
less memory, not more. 16G is enough here.

**2. Create and activate an environment:**

```bash
export UV_LINK_MODE=copy && uv venv .venv --python 3.12 && source .venv/bin/activate
```

**3. Install:**

```bash
uv pip install -e ".[cpu]" ipykernel jupyter
```

**4. Run the notebooks headless:**

```bash
jupyter nbconvert --to notebook --execute docs/notebooks/01_embed_any_model.ipynb --output /tmp/out.ipynb
```

**Or run them interactively.** Start a server on the compute node:

```bash
jupyter lab --no-browser --ip=0.0.0.0 --port=8899 --IdentityProvider.token=embpy
```

Then point your notebook client at `http://<compute-node>:8899/lab?token=embpy`
(`hostname` gives the node). Do not background it with `Ctrl+Z` — that suspends the
process, which keeps the port open while answering nothing; use `nohup ... &` if you
need the prompt back.

## Quick Start

```python
from embpy import BioEmbedder

embedder = BioEmbedder(device="auto", organism="human")
```

Embed genes with multiple model families:

```python
gene_adata = embedder.embed(
    ["TP53", "EGFR", "MYC"],
    entity_type="gene",
    id_type="symbol",
    model=["hyenadna_tiny_1k", "esm2_8M", "minilm_l6_v2"],
    output="anndata",
)

gene_adata.varm.keys()
gene_adata.uns["embeddings"].keys()
```

Embed gene perturbation labels as row-aligned action embeddings:

```python
# pert_adata.obs["perturbation"] contains symbols such as TP53/MYC.
pert_adata = embedder.embed(
    pert_adata,
    entity_type="gene",
    obs_column="perturbation",
    id_type="symbol",
    model="esm2_650M",
    output="anndata",
    is_perturbation=True,
    key="X_pert_esm2_650M",
)

pert_adata.obsm["X_pert_esm2_650M"]
```

Embed proteins:

```python
protein_adata = embedder.embed(
    ["TP53", "EGFR", "BRCA1"],
    entity_type="protein",
    id_type="symbol",
    model="esm2_8M",
    output="anndata",
)
```

Embed small molecules:

```python
smiles = [
    "CC(=O)OC1=CC=CC=C1C(=O)O",  # aspirin
    "Cn1cnc2c1c(=O)n(C)c(=O)n2C",  # caffeine
]

molecule_adata = embedder.embed(
    smiles,
    entity_type="molecule",
    id_type="smiles",
    model="morgan_fp",
    output="anndata",
    key="X_morgan_fp",
)
```

Embed cells from AnnData with model-aware preprocessing:

```python
cell_adata = embedder.embed(
    adata,
    entity_type="cell",
    model="pca",
    preprocessing="auto",
    output="anndata",
    key="X_pca",
)

cell_adata.obsm["X_pca"]
cell_adata.uns["embpy_cell_embeddings"]
```

Annotate and plot:

```python
from embpy import tl, pl

molecule_adata.obs["smiles"] = molecule_adata.obs_names
molecule_adata = tl.annotate_molecules(
    molecule_adata,
    column="smiles",
    sources=["structural", "bioactivity", "ontology"],
)

pl.plot_embedding_space(
    molecule_adata,
    obsm_key="X_morgan_fp",
    method="pca",
    color="mol_logp",
)
```

## Tutorials

**Start here** — a short, goal-oriented path:

1. [Embed with any model](docs/notebooks/01_embed_any_model.ipynb) — the full model
   catalog, then one embedding call per family (DNA, protein, molecule, text, …).
2. [Where embeddings live](docs/notebooks/02_output_contract.ipynb) — the
   `.obsm` / `.varm` / `.uns` storage contract and provenance.
3. [Compare embedding spaces across models](docs/notebooks/03_compare_models.ipynb) —
   do two models see your biology the same way?
4. [Which model captures *my* biology?](docs/notebooks/04_benchmark_models.ipynb) —
   rank models on your own labelled task.
5. [Do these models encode the same information?](docs/notebooks/05_compare_representations.ipynb) —
   compare representations with TSI/QSI/CKA and pick a layer from evidence.
6. [Reading a model's attention](docs/notebooks/06_attention_weights.ipynb) —
   extract per-layer attention and reduce it to AnnData-ready summaries.

**By modality** — deeper, domain-specific workflows:

- [Genes](docs/notebooks/genes.ipynb)
- [Proteins](docs/notebooks/proteins.ipynb)
- [Small molecules](docs/notebooks/small_molecules.ipynb)
- [Cells](docs/notebooks/cells.ipynb)
- [Validating single-cell embeddings with scIB](docs/notebooks/scib_validation.ipynb)
- [Variant effects](docs/notebooks/variant_effects.ipynb)

Each notebook uses real `BioEmbedder.embed(...)` calls, real annotation APIs,
and embpy plotting/comparison utilities.

## Model Families

embpy supports models across:

- DNA and regulatory sequence models
- protein language and structure models
- small-molecule fingerprints and chemical language models
- single-cell foundation models and classical baselines
- morphology models for HPA and JUMP-style images
- text models for biological descriptions

Use:

```python
embedder.list_available_models()
```

for the model keys available in your environment.

## Output Contract

`BioEmbedder.embed(...)` follows a scverse-friendly output contract:

- genes are feature-like and live in `.varm` by default
- gene perturbation labels use `is_perturbation=True` and live in `.obsm`
- proteins are feature-like and live in `.varm`
- molecules, text, sequences, and cells are observation-like and live in `.obsm`
- perturbation/action embeddings can be kept entity-aligned in `.uns`
- `.X` remains expression/count-like data or a sparse placeholder

See the [technical guide](docs/technical.md#standardized-output-contract) for
the full contract.

## Documentation

- [API reference](docs/api.md): per-function reference generated from docstrings
- [Technical guide](docs/technical.md): output contract, install matrix, package
  layout, and developer notes
- [Contributing](docs/contributing.md)
- [Changelog](docs/changelog.md)

## Citation

If you use embpy in your work, please cite the repository for now. A formal
citation will be added when the package is released.

## Contact

For questions, issues, or feature requests, open a GitHub issue or contact the
maintainers listed in the package metadata.
