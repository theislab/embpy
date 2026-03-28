# embpy

[![Tests][badge-tests]][tests]
[![Documentation][badge-docs]][documentation]

[badge-tests]: https://img.shields.io/github/actions/workflow/status/grpinto/embpy/test.yaml?branch=main
[badge-docs]: https://img.shields.io/readthedocs/embpy

**embpy** is a Python package for generating embeddings of biological perturbations and cell lines using 60+ foundation models through a unified interface.

Given a perturbation (genetic or chemical) and/or single-cell expression data, embpy resolves the underlying biological sequences, routes them to the appropriate foundation models, and returns dense vector representations ready for downstream machine learning.

## Workflow

<p align="center">
  <img src="docs/_static/embpy_workflow.png" alt="embpy workflow" width="800"/>
</p>

```mermaid
flowchart LR
    subgraph input [Input]
        Gene["Gene Symbol / Ensembl ID"]
        Mol["SMILES / Drug Name"]
        Cells["AnnData with counts"]
    end

    subgraph resolve [Sequence Resolution]
        DNASeq["DNA Sequence"]
        ProtSeq["Protein Sequence"]
        MolStruct["Molecular Structure"]
        CellExpr["Expression Matrix"]
    end

    subgraph models [Foundation Models]
        DNAMod["DNA: Enformer, Borzoi, Flashzoi, Evo2, NT, HyenaDNA, Caduceus"]
        ProtMod["Protein: ESM-2, ESM-C, ESM3, ProtT5"]
        MolMod["Molecule: ChemBERTa, MolFormer, RDKit, MiniMol"]
        SCMod["Single-Cell: scGPT, Geneformer, UCE, PCA, scVI"]
    end

    subgraph output [Output]
        Embs["Embeddings in .obsm"]
        Annots["Annotations in .obs/.uns"]
    end

    Gene --> DNASeq & ProtSeq
    Mol --> MolStruct
    Cells --> CellExpr
    DNASeq --> DNAMod
    ProtSeq --> ProtMod
    MolStruct --> MolMod
    CellExpr --> SCMod
    DNAMod & ProtMod & MolMod & SCMod --> Embs
    Gene & Mol --> Annots
```

## Key Features

- **60+ foundation models** across DNA, protein, molecule, single-cell, and text modalities
- **Unified `BioEmbedder` interface** -- one class to access all models with automatic sequence resolution
- **`embed_adata()`** -- embed cells and perturbations together in a single call
- **Weighted protein embeddings** -- TPM-weighted isoform averaging, annotation-weighted residue pooling, expression-context concatenation
- **Multi-source annotation** -- `MoleculeAnnotator` (RDKit, ChEMBL, ChEBI, KEGG, PubChem), `GeneAnnotator` (MyGene, GTEx, STRING-DB, Open Targets, GWAS Catalog), `ProteinAnnotator` (UniProt functional metadata, InterPro domains)
- **GPU acceleration** via rapids_singlecell for preprocessing, PCA, UMAP, neighbors, and Leiden
- **Batch processing** with SLURM array job scripts for full-genome embedding
- **scverse integration** -- AnnData-native throughout, compatible with scanpy/scvi-tools/pertpy

## Quick Start

### Embed a gene with a DNA model

```python
from embpy.embedder import BioEmbedder

embedder = BioEmbedder(device="auto")

# DNA embedding (resolves gene -> genomic sequence -> model)
emb = embedder.embed_gene("TP53", model="enformer_human_rough", pooling_strategy="mean")
print(emb.shape)  # (3072,)
```

### Embed a protein with ESM-2

```python
# Protein embedding (resolves gene -> UniProt sequence -> model)
emb = embedder.embed_gene("TP53", model="esm2_650M", pooling_strategy="mean")
print(emb.shape)  # (1280,)

# All isoforms
isoforms = embedder.embed_protein("TP53", model="esm2_650M", isoform="all")
for iso_id, emb in isoforms.items():
    print(f"  {iso_id}: {emb.shape}")
```

### Embed a small molecule

```python
emb = embedder.embed_molecule("CC(=O)OC1=CC=CC=C1C(=O)O", model="chemberta2MTR")
print(emb.shape)  # (768,)
```

### Embed cells from an AnnData

```python
import anndata as ad

adata = ad.read_h5ad("perturbseq.h5ad")

result = embedder.embed_cells(
    adata,
    models=["pca", "scvi", "scgpt"],
    preprocessing="standard",
    n_pca_components=50,
    n_latent=30,
)
# result.obsm["X_pca"]   -> (n_cells, 50)
# result.obsm["X_scvi"]  -> (n_cells, 30)
# result.obsm["X_scgpt"] -> (n_cells, 512)
```

### Combined cell + perturbation embedding

```python
result = embedder.embed_adata(
    adata,
    cell_models=["pca", "scgpt"],
    perturbation_models=["esm2_650M"],
    perturbation_column="perturbation",
    perturbation_type="auto",
)
# Cell embeddings + perturbation embeddings side by side in .obsm
```

### Weighted perturbation embedding

```python
from embpy.tl import WeightedProteinEmbedder

wpe = WeightedProteinEmbedder(embedder)

# TPM-weighted isoform average
emb = wpe.embed_perturbation(
    "TP53", model="esm2_650M", strategy="tpm_weighted",
    tpm_values={"P04637": 45.2, "P04637-2": 12.8},
)

# Annotation-weighted: active/binding sites get 3x weight
emb = wpe.embed_perturbation(
    "TP53", model="esm2_650M", strategy="annotation_weighted",
    site_boost=3.0,
)
```

### Annotate perturbations

```python
from embpy.tl import annotate_molecules, annotate_gene_perturbations, annotate_proteins

# Molecule annotations (physicochemical, bioactivities, pathways, diseases)
adata = annotate_molecules(adata, column="drug_name")

# Gene annotations (pathways, tissue expression, PPI, diseases)
adata = annotate_gene_perturbations(adata, column="gene")

# Protein annotations (UniProt function, domains, PTMs, GO terms)
adata = annotate_proteins(adata, column="gene")
```

## Available Models

### DNA Models

| Model | Key | Parameters |
|---|---|---|
| Enformer | `enformer_human_rough` | 250M |
| Borzoi (4 replicates) | `borzoi_v0` -- `borzoi_v3` | 200M |
| Borzoi Mouse | `borzoi_v0_mouse` -- `borzoi_v3_mouse` | 200M |
| Flashzoi (4 replicates) | `flashzoi_v0` -- `flashzoi_v3` | 200M |
| Evo 1 / 1.5 | `evo1_8k`, `evo1_131k`, `evo1.5_8k` | 7B |
| Evo 2 | `evo2_7b`, `evo2_40b` | 7B / 40B |
| Nucleotide Transformer v1/v2 | `nt_500m_human_ref`, `nt_v2_500m`, ... | 50M -- 2.5B |
| Nucleotide Transformer v3 | `ntv3_100m_pre`, `ntv3_650m_pos`, ... | 8M -- 650M |
| HyenaDNA | `hyenadna_tiny_1k` -- `hyenadna_large_1m` | 1.6M -- 6.6M |
| GENA-LM | `gena_lm_bert_base`, `gena_lm_bert_large`, ... | 110M -- 336M |
| Caduceus | `caduceus_ph_131k`, `caduceus_ps_131k` | 16M |

### Protein Models

| Model | Key | Parameters |
|---|---|---|
| ESM-1b | `esm1b` | 650M |
| ESM-1v (5 seeds) | `esm1v_1` -- `esm1v_5` | 650M |
| ESM-2 | `esm2_8M` -- `esm2_15B` | 8M -- 15B |
| ESM-C | `esmc_300m`, `esmc_600m`, `esmc_6b` | 300M -- 6B |
| ESM3 | `esm3_small`, `esm3_medium`, `esm3_large` | 1.4B -- 98B |
| ProtT5 | `prot_t5_xl`, `prot_t5_xl_half` | 3B |

### Molecule Models

| Model | Key | Type |
|---|---|---|
| ChemBERTa | `chemberta2MTR`, `chemberta2MLM` | Transformer |
| MolFormer | `molformer_base` | Transformer |
| RDKit Fingerprints | `rdkit_fp`, `morgan_fp`, `maccs_fp`, ... | Classical |
| MiniMol | `minimol` | GNN |
| MHG-GNN | `mhg_gnn` | Hypergraph GNN |
| MolE | `mole` | Graph Transformer |

### Single-Cell Foundation Models

| Model | Key | Parameters |
|---|---|---|
| scGPT | `scgpt` | 51M |
| Geneformer v1/v2 | `geneformer_v1_6L` -- `geneformer_v2_18L` | 10M -- 316M |
| UCE | `uce` | 1.3B |
| TranscriptFormer | `transcriptformer_metazoa`, `transcriptformer_sapiens` | 368M -- 542M |
| Tahoe-x1 | `tahoe_70m`, `tahoe_1b`, `tahoe_3b` | 70M -- 3B |
| Cell2Sentence-Scale | `cell2sentence_2b`, `cell2sentence_27b` | 2B -- 27B |
| PCA | `pca` | -- |
| scVI / scANVI / totalVI | `scvi`, `scanvi`, `totalvi` | -- |

### Text Models

| Model | Key |
|---|---|
| MiniLM | `minilm_l6_v2` |
| BERT | `bert_base_uncased` |

## Installation

You need Python 3.11 or newer.

### 1. Base install (CPU)

```bash
pip install git+https://github.com/theislab/embpy.git@main
```

The base install includes all HuggingFace-based models (GENA-LM, Nucleotide
Transformer v1/v2/v3, HyenaDNA, ESM-2, ProtT5, ChemBERTa, Molformer, ...)
since `transformers` is a core dependency.

### 2. GPU install

```bash
# CUDA 12.1
pip install "embpy[torch-cu121]" --extra-index-url https://download.pytorch.org/whl/cu121

# CUDA 12.4 (most common on modern HPC clusters)
pip install "embpy[torch-cu124]" --extra-index-url https://download.pytorch.org/whl/cu124

# CUDA 12.8
pip install "embpy[torch-cu128]" --extra-index-url https://download.pytorch.org/whl/cu128
```

### 3. Optional extras

| Extra | Models enabled |
|---|---|
| *(base)* | GENA-LM, NT v1/v2/v3, HyenaDNA, ESM-2/C, ProtT5, ChemBERTa, MolFormer, RDKit |
| `caduceus` | Caduceus (SSM/Mamba DNA model) |
| `evo` | Evo v1/v1.5 |
| `evo2` | Evo 2 |
| `helical` | Single-cell foundation models (scGPT, Geneformer, UCE, ...) |
| `ppi` | PPI GNN encoder |
| `pertpy` | pertpy integration |
| `scanpy` | scanpy integration |

```bash
# Example: CUDA 12.4 + single-cell models + Evo2
pip install "embpy[torch-cu124,helical,evo2]" \
  --extra-index-url https://download.pytorch.org/whl/cu124

# Full GPU install (CUDA 12.4)
pip install "embpy[all-cu124]" \
  --extra-index-url https://download.pytorch.org/whl/cu124
```

## Tutorials

| # | Topic | Notebook |
|---|---|---|
| 01 | Identifiers and Preprocessing | [01_identifiers_and_preprocessing.ipynb](docs/tutorials_core_tools/01_identifiers_and_preprocessing.ipynb) |
| 02 | Gene (DNA) Embeddings | [02_gene_embeddings.ipynb](docs/tutorials_core_tools/02_gene_embeddings.ipynb) |
| 03 | Protein Embeddings + Isoforms | [03_protein_embeddings.ipynb](docs/tutorials_core_tools/03_protein_embeddings.ipynb) |
| 04 | Molecule Embeddings | [04_molecule_embeddings.ipynb](docs/tutorials_core_tools/04_molecule_embeddings.ipynb) |
| 05 | Text Embeddings | [05_text_embeddings.ipynb](docs/tutorials_core_tools/05_text_embeddings.ipynb) |
| 06 | PPI Embeddings | [06_ppi_embeddings.ipynb](docs/tutorials_core_tools/06_ppi_embeddings.ipynb) |
| 07 | Combined Analysis | [07_combined_analysis.ipynb](docs/tutorials_core_tools/07_combined_analysis.ipynb) |
| 08 | Visualization and Analysis | [08_visualization_and_analysis.ipynb](docs/tutorials_core_tools/08_visualization_and_analysis.ipynb) |
| 09 | Embedding Benchmark | [09_embedding_benchmark.ipynb](docs/tutorials_core_tools/09_embedding_benchmark.ipynb) |
| 10 | DNA Embeddings (Advanced) | [10_dna_embeddings.ipynb](docs/tutorials_core_tools/10_dna_embeddings.ipynb) |
| 11 | DepMap Analysis | [11_depmap_analysis.ipynb](docs/tutorials_core_tools/11_depmap_analysis.ipynb) |
| 12 | Single-Cell Foundation Models | [12_singlecell_foundation_models.ipynb](docs/tutorials_core_tools/12_singlecell_foundation_models.ipynb) |
| 13 | JUMP Cell Painting | [13_jump_cell_painting.ipynb](docs/tutorials_core_tools/13_jump_cell_painting.ipynb) |
| 14 | Unified Embedding (embed_adata) | [14_unified_embedding.ipynb](docs/tutorials_core_tools/14_unified_embedding.ipynb) |
| 15 | Molecule Annotation | [15_molecule_annotation.ipynb](docs/tutorials_core_tools/15_molecule_annotation.ipynb) |
| 16 | Gene/Protein Annotation + Weighted Embeddings | [16_gene_protein_annotation.ipynb](docs/tutorials_core_tools/16_gene_protein_annotation.ipynb) |

## Package Structure

```
embpy/
    embedder.py          # BioEmbedder -- unified embedding interface
    models/
        dna_models.py    # Enformer, Borzoi, Evo, NT, HyenaDNA, Caduceus, GENA-LM
        protein_models.py # ESM-2, ESM-C, ESM3, ProtT5
        molecule_models.py # ChemBERTa, MolFormer, RDKit, MiniMol, MHG-GNN, MolE
        singlecell_models.py # scGPT, Geneformer, UCE, PCA, scVI
    resources/
        gene_resolver.py      # Gene symbol <-> Ensembl ID <-> DNA sequence
        protein_resolver.py   # Gene -> UniProt canonical/isoform sequences
        molecule_annotator.py # Small molecule annotations (6 sources)
        gene_annotator.py     # Gene annotations (pathways, expression, PPI, diseases)
        protein_annotator.py  # Protein annotations (UniProt, InterPro)
        drug_resolver.py      # Drug name <-> SMILES resolution
    pp/
        sc_preprocessing.py   # Single-cell preprocessing (raw/standard pipelines)
        basic.py              # Perturbation embedding matrix construction
    tl/
        similarity.py         # Cosine/Pearson/Spearman similarity, KNN overlap
        dimred.py             # UMAP, t-SNE (CPU/GPU)
        clustering.py         # Leiden, k-means, spectral (CPU/GPU)
        weighted_protein_embedding.py # TPM-weighted, annotation-weighted, expression-context
        metrics.py            # Benchmarking metrics
        pipeline.py           # Automated evaluation pipelines
        metadata.py           # pertpy-based metadata annotation
```

## Release Notes

See the [changelog][].

## Contact

For questions and help requests, you can reach out in the [scverse discourse][].
If you found a bug, please use the [issue tracker][].

## Citation

> t.b.a

[mambaforge]: https://github.com/conda-forge/miniforge#mambaforge
[scverse discourse]: https://discourse.scverse.org/
[issue tracker]: https://github.com/theislab/embpy/issues
[tests]: https://github.com/theislab/embpy/actions/workflows/test.yml
[documentation]: https://embpy.readthedocs.io
[changelog]: https://embpy.readthedocs.io/en/latest/changelog.html
[api documentation]: https://embpy.readthedocs.io/en/latest/api.html
