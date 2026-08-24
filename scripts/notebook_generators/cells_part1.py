"""Part 1 of docs/notebooks/cells.ipynb -- intro, the two datasets, the catalogue.

Leaves for later parts: pd/np/plt/sc/warnings/time/Path/display, embpy/tl/pl/pp,
embedder, OUTPUT_DIR, DATA_DIR, the six contract constants, `adata` (Kang, with
its contract asserted), ROSTER, SKIPPED and CATALOGUE.
"""
from __future__ import annotations
import json, sys
from pathlib import Path

CELLS: list[tuple[str, str]] = []
def md(src: str) -> None: CELLS.append(("markdown", src.strip("\n")))
def code(src: str) -> None: CELLS.append(("code", src.strip("\n")))

# ============================================================ A. the opening
md(r"""
# Cells

**What you'll learn.** The numbered tutorials introduce one idea at a time on a
handful of entities. This notebook is the single-cell *deep dive*: every cell
model embpy can reach, run on the same cells, then scored three different ways.

The three scoring sections answer three different questions, and the difference
matters more than any individual number. There is no single "which model is
best" ranking, and a notebook that produced one would be lying to you.

| Section | Question | What it needs from the data |
| --- | --- | --- |
| [1. Do the models agree?](#1-do-the-models-agree) | do two models encode the same information? | two or more embeddings |
| [2. scIB](#2-scib-does-integration-preserve-the-biology) | does integration remove the batch and keep the biology? | cell-type labels plus a batch covariate |
| [3. cell-eval](#3-cell-eval-is-the-perturbation-preserved) | does the embedding preserve a perturbation effect? | many perturbation levels |

Because those three needs conflict, this notebook uses **two** datasets, both
from `pertpy`. Section 3 runs on a different object than sections 1 and 2, and
[the dataset section](#the-datasets-and-the-contract-you-can-swap) says why.

**Prerequisites.** [04_benchmark_models](04_benchmark_models.ipynb) is the short
version of sections 2 and 3 -- five models, one dataset, no batch covariate.
This notebook is the long version, and the difference is not length: nb04 had no
batch variable at all, so it could not ask the integration question, and it used
cell type as a stand-in perturbation, which made every discrimination score come
back exactly 1.000. Both of those are fixed here, and both are shown rather than
asserted.
""")

md(r"""
## Requirements

The single-cell backends do not coexist with embpy's other extras: `helical`
pins `numpy<2.3` and `transformers<=4.51.3`, `arc-state` wants
`transformers>=4.52.3`, and `cell-eval` wants `numpy>=2.4.2`. Those three
constraints are mutually unsatisfiable, so a single `uv pip install embpy[all]`
cannot work and no amount of retrying will make it.

The environment this notebook was executed in resolves that by installing
`helical` with `--no-deps` and adding its real runtime requirements by hand:

```bash
uv venv --python 3.13 .venv-sc
uv pip install --python .venv-sc/bin/python -e .
uv pip install --python .venv-sc/bin/python \
    scanpy scib scib-metrics cell-eval pdex scvi-tools
uv pip install --python .venv-sc/bin/python --no-deps helical
uv pip install --python .venv-sc/bin/python arc-state arc-stack
uv pip install --python .venv-sc/bin/python ipykernel
.venv-sc/bin/python -m ipykernel install --user \
    --name embpy-sc --display-name "Python (embpy-sc)"
```

Then select the `Python (embpy-sc)` kernel. Two practical points:

* **`.venv-sc` has no `pip`.** It was created by uv, so `python -m pip install`
  fails with `No module named pip`. Use `uv pip install --python
  .venv-sc/bin/python ...` for everything.
* **`pertpy` is not installed there, deliberately.** It is not an embpy
  dependency, and adding it to this environment risks moving `numpy` or
  `scanpy` in a resolution that took real work to get right. The loaders below
  read a staged `.h5ad` when one exists and only call `pertpy` as a fallback, so
  you stage the two files once from any environment that has pertpy.

> **What is installed is metrics, not methods.** `scib` and `scib-metrics` give
> you the scIB *scores*. `harmonypy`, `scanorama` and `bbknn` are not installed,
> so there is no explicit batch-correction step here to compare against. What
> section 2 measures is whatever integration each embedding model performs
> *implicitly*, which is a narrower claim than "we benchmarked integration
> methods" and worth keeping straight.
""")

code(r"""
import os
import time
import warnings
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from IPython.display import display

import embpy
from embpy import BioEmbedder, pl, pp, tl

# The single-cell stack is chatty: scib warns about the leidenalg backend on
# every resolution it tries, and scanpy warns when a copy densifies. Neither
# changes a result, and 40 repeats of each would bury the tables.
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
sc.settings.verbosity = 1

OUTPUT_DIR = Path("outputs")
DATA_DIR = Path("data")
OUTPUT_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

embedder = BioEmbedder(device="auto", organism="human")
print(f"embpy {embpy.__version__}")
""")

# ============================================================ B. the datasets
md(r"""
## The datasets, and the contract you can swap

Two objects, because the three scoring sections need differently shaped
experiments and no single dataset provides all of it:

| | `adata` -- an **atlas** | `pert_adata` -- `norman_2019` |
| --- | --- | --- |
| used by | the sweep, section 1, section 2, attention, decode | section 3 only |
| needs | cell-type labels plus a **technical** covariate | many perturbation levels plus a control |
| the covariate is | donor, lab, sequencing technology, 10x chemistry | -- |

**Section 2 needs an atlas, not a perturbation experiment.** scIB asks whether
merging batches removes a *technical* axis while preserving the *biological*
one. Point it at a treatment experiment and a model scores well for making
stimulated cells look like control cells -- which is not integration, it is
deleting the result. So the atlas is what sections 1, 2 and 6 run on, and
section 3 deliberately uses a different object.

The mirror of that rule applies to section 3. cell-eval scores agreement *per
perturbation*, so an atlas gives it nothing to work with, and a dataset with a
single treatment arm gives it almost nothing.
[04_benchmark_models](04_benchmark_models.ipynb) is the worked failure: with 7
stand-in perturbations it returned a discrimination score of exactly 1.000 for
every model, ranking nothing at all.

**Everything after this section reads five constants and nothing else about the
data.** To use your own atlas, repoint `ATLAS_PATH` or `load_atlas`, then check
it against the requirements this section asserts.
""")

code(r"""
# ------------------------------------------------------------ THE CONTRACT
# Repoint these to swap in your own data. Nothing downstream hardcodes a
# column name, so a rename here propagates through every later section.
LABEL_KEY = "cell_type"      # ground-truth biology; scIB bio-conservation
BATCH_KEY = "batch"          # TECHNICAL covariate; scIB batch-correction
COUNTS_LAYER = "counts"      # raw integer counts, kept across preprocessing

N_CELLS = 1500               # subsample size, to keep a run to minutes
SEED = 0
MIN_CELLS_PER_LABEL = 4      # below this, per-label statistics are meaningless

# The atlas slot. A dataset qualifies if it has cell-type labels, a technical
# batch covariate that is not nested inside those labels, raw integer counts,
# and unique gene symbols. Candidates that satisfy it:
#   scvi.data.heart_cell_atlas_subsampled()  -- donors as batch, cell_type labels
#   the scIB pancreas benchmark              -- 4 technologies, the canonical one
#   any published atlas with a donor / lab / technology column
ATLAS_PATH = DATA_DIR / "atlas.h5ad"


def load_atlas():
    # Staging is the intended path: it keeps this kernel free of dataset
    # dependencies and makes the choice of atlas explicit rather than a
    # silent default buried in a helper.
    if ATLAS_PATH.exists():
        return sc.read_h5ad(ATLAS_PATH)
    import scvi  # noqa: PLC0415 - fallback only, for whoever stages the file

    ad = scvi.data.heart_cell_atlas_subsampled()
    ad.write_h5ad(ATLAS_PATH)
    return ad


adata = load_atlas()
print(f"atlas: {adata.n_obs} cells x {adata.n_vars} genes")
print(f"obs columns: {list(adata.obs.columns)}")
""")

md(r"""
### Assert the contract, do not trust it

Six things every later section depends on, checked here rather than discovered
as a confusing failure eight sections later.

Raw integer counts are the strictest of them. `scvi` and `scanvi` declare
`input_layer="counts"`, and Geneformer tokenises a cell by gene *rank*, so a
pre-normalised matrix does not raise -- it quietly produces a different, wrong
answer. Atlases are often published already normalised, which is exactly the
case this assertion catches.
""")

code(r"""
# Resolve the contract columns against whatever this atlas calls them, so a
# dataset using "donor" or "tech" needs one edit here and none below.
COLUMN_ALIASES = {
    LABEL_KEY: ["cell_type", "cell_type_annotation", "celltype", "labels", "louvain"],
    BATCH_KEY: ["batch", "donor", "cell_source", "tech", "study", "sample"],
}
for canonical, candidates in COLUMN_ALIASES.items():
    if canonical in adata.obs.columns:
        continue
    found = next((c for c in candidates if c in adata.obs.columns), None)
    if found is None:
        raise KeyError(
            f"no column for {canonical!r}; tried {candidates}. "
            f"Available: {list(adata.obs.columns)}"
        )
    adata.obs[canonical] = adata.obs[found]
    print(f"mapped {canonical!r} <- {found!r}")

sc.pp.subsample(adata, n_obs=min(N_CELLS, adata.n_obs), random_state=SEED)
for key in (LABEL_KEY, BATCH_KEY):
    adata.obs[key] = adata.obs[key].astype(str).astype("category")

# Prefer an explicit counts layer over .X: published atlases frequently ship
# normalised values in .X and keep the counts beside them.
if COUNTS_LAYER in adata.layers:
    adata.X = adata.layers[COUNTS_LAYER].copy()
else:
    adata.layers[COUNTS_LAYER] = adata.X.copy()

X = adata.layers[COUNTS_LAYER]
integral = float(np.abs(X.data - np.round(X.data)).max()) == 0.0

assert integral, (
    "X must hold raw integer counts. scvi/scanvi declare input_layer='counts' "
    "and Geneformer tokenises by rank; a normalised matrix fails silently."
)
assert adata.obs[LABEL_KEY].nunique() >= 4, "scIB bio-conservation needs >= 4 labels"
assert adata.obs[BATCH_KEY].nunique() >= 2, "scIB batch metrics need >= 2 batches"
assert adata.var_names.is_unique, "gene symbols must be unique"
assert adata.n_vars >= 2000, (
    f"only {adata.n_vars} genes; the rank tokenisers (Geneformer, scGPT, UCE) "
    "need thousands before their embeddings mean anything"
)

print(f"counts integral, {1 - X.nnz / (X.shape[0] * X.shape[1]):.1%} zeros, "
      f"X.max() = {X.max():.0f}")
print(f"{LABEL_KEY}: {adata.obs[LABEL_KEY].nunique()} levels | "
      f"{BATCH_KEY}: {adata.obs[BATCH_KEY].nunique()} levels")
print(f"var_names: {list(adata.var_names[:4])}")
""")

md(r"""
The label-by-batch cross-tabulation decides whether section 2 can mean anything,
so read it rather than skipping it. If every cell type appears in every batch,
then "mix the batches" and "keep the cell types apart" are separable requests.
If a cell type lives in only one batch they are in direct conflict, no model can
satisfy both, and a low score would be measuring the experimental design rather
than the model.
""")

code(r"""
crosstab = pd.crosstab(adata.obs[LABEL_KEY], adata.obs[BATCH_KEY])
crosstab["total"] = crosstab.sum(axis=1)
display(crosstab.sort_values("total", ascending=False))

tiny = crosstab.index[crosstab["total"] < MIN_CELLS_PER_LABEL].tolist()
print(f"labels below {MIN_CELLS_PER_LABEL} cells: {tiny or 'none'}")

# How nested is the design? 0 means every label is spread evenly across
# batches; the printed maximum means each label sits in exactly one batch.
n_batches = adata.obs[BATCH_KEY].nunique()
shares = crosstab.drop(columns="total").div(crosstab["total"], axis=0)
NESTEDNESS = float((shares - 1.0 / n_batches).abs().sum(axis=1).mean())
print(f"nestedness {NESTEDNESS:.3f}  "
      f"(0 = balanced, {2 * (1 - 1 / n_batches):.2f} = fully nested)")
""")

md(r"""
### Why a real dataset, and not a simulation

The obvious way to write this notebook is to simulate cells: a few hundred rows,
a Poisson count matrix, a handful of marker genes. The notebook this one
replaces did exactly that over 24 genes, and every number in it was noise.

The reason is structural rather than a matter of scale. scGPT, Geneformer, UCE,
STATE and STACK all tokenise a cell by the *rank order of thousands of genes*.
Give them 24 genes and every cell yields nearly the same token sequence, so
every cell gets nearly the same embedding -- and comparing near-constant
embeddings compares rounding error. That is what the `n_vars >= 2000` assertion
above guards.

Fabricating the *covariates* does not work either. Three designs were built and
measured on `pbmc3k` before this notebook committed to real data. Silhouette on
PCA(30), 600 cells:

| Design | biology | injected covariate | verdict |
| --- | --- | --- | --- |
| `pbmc3k`, untouched | +0.101 | -- | no batch or perturbation exists to measure |
| injected batch, sigma = 0.5 | +0.064 | +0.034 | the covariate barely registers |
| injected batch, sigma = 1.5 | **-0.010** | +0.118 | visible only by destroying the biology |
| injected perturbation, 8x on 18 genes | +0.101 | -0.005 | invisible at every strength tried |
| library-depth tertile as batch | +0.101 | -0.067 | and 0.714 confounded with cell type -- it *is* cell type |

Two reasons those failed, both of which generalise past this notebook:

* **`normalize_total` removes a library-size shift by construction.** Half of
  the fabricated batch effect was a per-cell scale factor, and normalising is
  the first thing every one of these models wants. Injecting signal into a
  quantity the pipeline exists to divide out cannot survive the pipeline.
* **An effect on 18 genes out of 32,738 does not survive HVG selection.** Real
  signatures are hundreds of genes wide *and* cell-type dependent. A fixed short
  list scaled by a constant is neither, so PCA never sees it.
""")

md(r"""
### Is this atlas a real integration benchmark?

Section 2 scores every embedding on how well it removes `BATCH_KEY`. That score
is only interesting if there is a batch effect to remove, so measure it here,
before any model runs, and let the measurement set expectations.

A covariate that separates cells *globally* but not *within a cell type* is
confounded with biology rather than technical, and removing it would mean
merging cell types. So measure both, and treat a near-zero batch silhouette as
what it is: a weak benchmark, on which every model will score well and the
column will rank nothing.
""")

code(r"""
from sklearn.metrics import silhouette_score

# A throwaway PCA purely to size the problem. This is not one of the
# embeddings under test; it exists so the prediction below is measured
# rather than assumed.
probe = adata.copy()
sc.pp.normalize_total(probe, target_sum=1e4)
sc.pp.log1p(probe)
sc.pp.highly_variable_genes(probe, n_top_genes=2000)
probe = probe[:, probe.var.highly_variable].copy()
sc.pp.scale(probe, max_value=10)
sc.tl.pca(probe, n_comps=30)
E = probe.obsm["X_pca"]

big_labels = [
    lab for lab, n in probe.obs[LABEL_KEY].value_counts().items() if n >= 40
][:4]

rows = [{
    "covariate": f"{LABEL_KEY} (the biology to keep)",
    "global": silhouette_score(E, probe.obs[LABEL_KEY].astype(str)),
}]
batch_row = {
    "covariate": f"{BATCH_KEY} (the axis to remove)",
    "global": silhouette_score(E, probe.obs[BATCH_KEY].astype(str)),
}
for lab in big_labels:
    mask = (probe.obs[LABEL_KEY] == lab).to_numpy()
    batch_row[f"within {lab}"] = (
        silhouette_score(E[mask], probe.obs[BATCH_KEY][mask].astype(str))
        if probe.obs[BATCH_KEY][mask].nunique() > 1 else np.nan
    )
rows.append(batch_row)

COVARIATE_STRENGTH = pd.DataFrame(rows).set_index("covariate")
display(COVARIATE_STRENGTH.round(3))

# State the prediction now. Section 2 scores it rather than assuming it.
batch_global = float(batch_row["global"])
WEAK_BATCH = batch_global < 0.05
print(
    f"batch silhouette {batch_global:+.3f} -> "
    + ("WEAK integration benchmark: expect batch_correction high and near-tied "
       "for every model, ranking nothing."
       if WEAK_BATCH else
       "a real batch effect exists: expect batch_correction to separate the "
       "models.")
)
del probe, E
""")

# ============================================================ D. the catalogue
md(r"""
## Which cell models does embpy have?

Twenty-five entries live in the single-cell registry
(`src/embpy/models/singlecell_models.py`), and they are not twenty-five
interchangeable choices. They differ in four ways that decide whether a model
will work on *your* AnnData at all:

* **`vocab_type`** -- whether the model wants gene symbols, Ensembl IDs, or does
  not care. Get this wrong on a `symbol` model and you get zero matches and an
  empty embedding, with no error. [Part 2](#preprocess-once-per-requirement-not-once-per-model)
  measures the overlap before embedding for exactly that reason.
* **`input_layer`** -- raw counts, `.layers["counts"]`, or log-normalised
  expression. `preprocessing="auto"` reads this and prepares the right one.
* **`supports_decode`** -- whether the latent space can be projected back to
  expression. Five keys can; the rest are encoders only.
* **`supports_generation`** -- whether whole profiles can be synthesised. Exactly
  one key can.
""")

code(r"""
from embpy.models.singlecell_models import list_singlecell_models, singlecell_info

# The models this notebook attempts. The first five are the ones
# 04_benchmark_models.ipynb proved end to end; the next three have weights
# cached on this cluster and are attempted with failures reported, not raised.
ROSTER = [
    "pca", "scvi", "scgpt", "geneformer_v2_12L", "state",
    "uce", "transcriptformer_sapiens", "tahoe_70m",
]
# Not new models -- the same scvi-tools wrapper, told different things. These
# two exist to make section 2's comparison honest; see the tiers table there.
INTEGRATION_VARIANTS = ["scvi_batch", "scanvi"]
SKIPPED = ["stack", "cell2sentence_2b", "tahoe_1b", "tahoe_3b", "totalvi"]

rows = []
for key in ROSTER + SKIPPED:
    card = singlecell_info(key)
    rows.append({
        "model": key,
        "wrapper": card.wrapper_class_name,
        "dim": card.embedding_dim,
        "vocab": card.vocab_type,
        "input_layer": card.input_layer,
        "auto_prep": card.default_preprocessing,
        "decode": card.supports_decode,
        "generate": card.supports_generation,
        "attempted": key in ROSTER,
    })

CATALOGUE = pd.DataFrame(rows).set_index("model")
display(CATALOGUE)

print(f"registry holds {len(list_singlecell_models())} single-cell models; "
      f"this notebook attempts {len(ROSTER)}")
""")

md(r"""
**Skipped on purpose.** A catalogue that quietly omits what it could not run is
not a catalogue, so here is each exclusion with its reason:

* **`stack`** -- the only `supports_generation=True` entry, and its checkpoint
  *is* on disk. It imports cleanly and then fails inside arc-stack's own h5ad
  reader, upstream of anything embpy controls. Its absence is why this notebook
  has no generation section, which is a real gap rather than an oversight.
* **`cell2sentence_2b` / `_27b`** -- 4.9 GB of Gemma-2 weights, and the LLM
  tokenisation turns a cell into a gene *sentence*, so an attention tensor over
  it is enormous.
* **`tahoe_1b`, `tahoe_3b`** -- only the 70M checkpoint is cached; these would
  download.
* **`totalvi`** -- needs paired protein counts, which this dataset does not have.
* **the other eight Geneformer variants** -- v1 at 6L and 12L, the CZI
  fine-tune, v2 at 20L and 18L, and three cancer/104M variants. Only
  `v2/gf-12L-38M-i4096` is cached here, so `geneformer_v2_12L` is the one that
  runs without a download. The other eight are a real catalogue rather than
  padding: swapping to a 316M-parameter variant is a one-string change and a
  download.

The registry keys are worth cross-checking against what the embedder will
actually accept, because a key in the registry that the embedder cannot resolve
is a catalogue entry you cannot use.
""")

code(r"""
available = set(embedder.list_available_models("single_cell"))
registry = set(list_singlecell_models())

print(f"registry: {len(registry)} | embedder reports available: {len(available)}")
only_registry = sorted(registry - available)
only_available = sorted(available - registry)
print(f"in registry but not reported available: {only_registry or 'none'}")
print(f"reported available but not in registry: {only_available or 'none'}")

missing = [k for k in ROSTER if k not in available]
if missing:
    print(f"\nWARNING: roster entries the embedder cannot resolve: {missing}")
else:
    print(f"\nall {len(ROSTER)} roster entries resolve")
""")

Path(sys.argv[1] if len(sys.argv) > 1 else "cells_part1.json").write_text(json.dumps(CELLS))
print(f"part 1: {len(CELLS)} cells")
