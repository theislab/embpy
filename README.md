# embpy

[![Tests][badge-tests]][tests]
[![Documentation][badge-docs]][documentation]

[badge-tests]: https://img.shields.io/github/actions/workflow/status/grpinto/embpy/test.yaml?branch=main
[badge-docs]: https://img.shields.io/readthedocs/embpy

A package for biological embeddings in Python — DNA, protein, molecule and text
foundation models with a unified interface.

## Getting started

Please refer to the [documentation][], in particular the [API documentation][].

## Installation

You need Python 3.11 or newer.
If you don't have Python installed, we recommend [Mambaforge][].

### 1. Base install (CPU)

```bash
pip install git+https://github.com/grpinto/embpy.git@main
```

The base install includes all HuggingFace-based models (GENA-LM, Nucleotide
Transformer v1/v2/v3, HyenaDNA, ESM-2, ProtT5, ChemBERTa, Molformer, …) since
`transformers` is a core dependency.

### 2.  GPU install — pick your CUDA version

For some models, including Caduceus with Mamba-SSM, you may want to install torch using GPU. Standard installation with torch for CPU works for most models, however limits the inference speed. 

```bash
# CUDA 12.1
pip install "embpy[torch-cu121]" --extra-index-url https://download.pytorch.org/whl/cu121

# CUDA 12.4 
pip install "embpy[torch-cu124]" --extra-index-url https://download.pytorch.org/whl/cu124

# CUDA 12.8
pip install "embpy[torch-cu128]" --extra-index-url https://download.pytorch.org/whl/cu128

# CUDA 12.9
pip install "embpy[torch-cu128]" --extra-index-url https://download.pytorch.org/whl/cu129

# CUDA 13.0
pip install "embpy[torch-cu128]" --extra-index-url https://download.pytorch.org/whl/cu130

# CPU only
pip install "embpy[torch-cpu]"
```

> **Check your CUDA version:** `nvcc --version` or `nvidia-smi`

### 3. Optional model extras

Install only what you need:

| Extra | Models enabled | Notes |
|---|---|---|
| *(base)* | GENA-LM, NT v1/v2/v3, HyenaDNA, ESM-2/C, ProtT5, ChemBERTa, Molformer, RDKit | `transformers` already included |
| `caduceus` | Caduceus (SSM/Mamba DNA model) | Requires CUDA — install `torch-cu*` first |
| `evo` | Evo v1/v1.5 | |
| `evo2` | Evo 2 | |
| `ppi` | PPI GNN encoder | Requires `torch-cu*` |
| `mhg-gnn` | MHG-GNN molecule model | Requires `torch-cu*` |
| `mole` | MolE molecule model | Requires `torch-cu*` |
| `minimol` | MiniMol | |
| `pertpy` | pertpy integration | |
| `scanpy` | scanpy integration | |

```bash
# Example: GPU install (CUDA 12.8) CUDA 12.4 + Caduceus + Evo
pip install "embpy[torch-cu128,caduceus,evo]" \
  --extra-index-url https://download.pytorch.org/whl/cu128
```

### 4. Caduceus — special note on `mamba-ssm`

Caduceus uses the [Mamba SSM](https://github.com/state-spaces/mamba) CUDA
extension, which must be compiled against your exact CUDA + PyTorch versions.
Install the matching `torch-cu*` extra first, then:

```bash
pip install "embpy[caduceus]"
```

If the build fails (common on HPC clusters with non-standard CUDA paths), try:

```bash
CUDA_HOME=/usr/local/cuda pip install mamba-ssm --no-build-isolation
```

Or load the CUDA module first:

```bash
module load cuda
pip install mamba-ssm --no-build-isolation
```

### 5. SNP embedding utilities

The `snp_utils` module provides model-agnostic variant-effect embeddings.
It requires a reference genome FASTA — either download via the helper or
provide your own:

```python
from embpy.tl.snp_utils import download_hg38_per_chrom, SequenceProvider, SNPEmbedder

# Download chr17 only (~100 MB) for testing
fasta_dir = download_hg38_per_chrom("./hg38_chroms", chromosomes=["chr17"])

# Or download full hg38 (~3.1 GB uncompressed)
# from embpy.tl.snp_utils import download_hg38_single_fasta
# fasta = download_hg38_single_fasta("./hg38.fa")

provider = SequenceProvider(fasta_dir=fasta_dir)

from embpy.embedder import BioEmbedder
embedder = BioEmbedder(device="cuda")
wrapper  = embedder._get_model("nt_v2_100m")

snp_embedder = SNPEmbedder(wrapper)
result = snp_embedder.embed_snp_from_vcf_row(
    chrom="chr17", pos=7_674_220, ref="C", alt="T",
    chromosome_sequence=provider.get_chromosome("chr17"),
    context_window=256,
    variant_id="rs28934578",
)
print(result.delta_norms)   # L2 magnitude of variant effect
```

## Release notes

See the [changelog][].

## Contact

For questions and help requests, you can reach out in the [scverse discourse][].
If you found a bug, please use the [issue tracker][].

## Citation

> t.b.a

[mambaforge]: https://github.com/conda-forge/miniforge#mambaforge
[scverse discourse]: https://discourse.scverse.org/
[issue tracker]: https://github.com/grpinto/embpy/issues
[tests]: https://github.com/grpinto/embpy/actions/workflows/test.yml
[documentation]: https://embpy.readthedocs.io
[changelog]: https://embpy.readthedocs.io/en/latest/changelog.html
[api documentation]: https://embpy.readthedocs.io/en/latest/api.html
[pypi]: https://pypi.org/project/embpy
