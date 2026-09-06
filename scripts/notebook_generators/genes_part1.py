"""Generate part 1 of docs/notebooks/genes.ipynb -- intro, panel, catalog, resolution.

Companion to the numbered tutorials: those introduce one idea at a time, this
runs the whole gene-capable family through the whole metric set.
"""
from __future__ import annotations
import json, sys
from pathlib import Path

CELLS: list[tuple[str, str]] = []
def md(src: str) -> None: CELLS.append(("markdown", src.strip("\n")))
def code(src: str) -> None: CELLS.append(("code", src.strip("\n")))

# ===================================================================== intro
md(r"""
# Genes

**What you'll learn.** The numbered tutorials introduce one idea at a time on a
handful of entities. This notebook is the gene *deep dive*: every distinct
gene-capable model in embpy, compared with every metric embpy ships, plus the
readouts that only exist for genes -- exon-level attention, locus geometry, and
variant context.

Genes are the one modality where embpy offers two genuinely different kinds of
model, and they disagree in an informative way:

* **Sequence models** read the DNA of the locus itself. They know nothing about
  what the gene *does*.
* **Prior-knowledge tables** are static lookups distilled from text, co-expression,
  or CRISPR screens. They know what the gene does and nothing about its sequence.

It answers four questions in order:

| Section | Question |
| --- | --- |
| [Embed](#embed-every-gene-capable-model) | which gene models exist, and what do they cost? |
| [Compare](#1-global-geometry) | do they encode the same structure? |
| [Interpret](#does-the-geometry-track-biology) | does that structure track gene biology? |
| [Read inside](#reading-a-dna-models-attention) | what is the model actually looking at? |

Prerequisites: [Comparing embeddings](03_compare_embeddings.ipynb) for the metric
definitions and [Reading a model's attention](06_attention_weights.ipynb) for the
extract-then-summarise pattern. Both are used here without re-deriving.
Variant-level work has its own notebook -- [Variant effects](variant_effects.ipynb)
-- and is only touched on at the end.
""")

code(r"""
import json
import time
import warnings
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from IPython.display import display

from embpy import BioEmbedder, pl, tl

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 40)

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# Nucleotide Transformer and GENA-LM ship custom modelling code and report a
# "newly initialised pooler" notice on load. Mean pooling never touches that
# head, so the notice is noise here; quieting it keeps the tables readable.
from transformers import logging as hf_logging

hf_logging.set_verbosity_error()

embedder = BioEmbedder(device="auto", organism="human")
""")

# ===================================================================== panel
md(r"""
## A panel built to make the models disagree

Comparison metrics need enough entities to be meaningful, and interpretation
needs *labels that are not derived from the embeddings*. So the panel is 40 human
genes in five classes of eight -- a grouping from gene biology, not from any
model.

The classes are chosen to split along the fault line described above, because a
panel every model handles identically measures nothing:

* **Paralog families** (`tubulin`, `hox`) are held together by *shared
  sequence*. Their members are duplicates of one ancestral gene, so a DNA model
  should find them easily and a knowledge table has no special advantage.
* **Pathway classes** (`glycolysis`, `interferon`, `cell_cycle`) are held together
  by *shared function*. Their members are unrelated in sequence -- glycolysis
  alone spans half a dozen unrelated folds -- so a DNA model should struggle and a
  knowledge table should do well.

That is a prediction, and the [interpretation section](#does-the-geometry-track-biology)
scores it rather than assuming it.
""")

code(r"""
PANEL: dict[str, list[str]] = {
    # sequence-coherent: paralogs of one ancestral gene
    "tubulin":    ["TUBA1A", "TUBA1B", "TUBA1C", "TUBB", "TUBB2A", "TUBB3", "TUBB4B", "TUBB6"],
    "hox":        ["HOXA1", "HOXA2", "HOXA3", "HOXA5", "HOXA9", "HOXA10", "HOXA11", "HOXA13"],
    # function-coherent: shared pathway, unrelated sequence
    "glycolysis": ["GAPDH", "LDHA", "PKM", "ALDOA", "ENO1", "PGK1", "TPI1", "GPI"],
    "interferon": ["STAT1", "STAT2", "IRF1", "IRF9", "ISG15", "MX1", "OAS1", "IFIT1"],
    "cell_cycle": ["CDK1", "CDK2", "CCNA2", "CCNB1", "AURKA", "PLK1", "CHEK1", "BUB1"],
}

SEQUENCE_COHERENT = {"tubulin", "hox"}

symbols = [s for members in PANEL.values() for s in members]
families = [fam for fam, members in PANEL.items() for _ in members]

gene_space = ad.AnnData(
    X=np.zeros((len(symbols), 1), dtype=np.float32),
    obs=pd.DataFrame(
        {
            "symbol": symbols,
            "family": pd.Categorical(families),
            "coherence": pd.Categorical(
                ["sequence" if f in SEQUENCE_COHERENT else "function" for f in families]
            ),
        },
        index=pd.Index(symbols, name="gene"),
    ),
    var=pd.DataFrame(index=["placeholder"]),
)

print(f"{gene_space.n_obs} genes in {len(PANEL)} classes")
print(gene_space.obs.groupby(["coherence", "family"], observed=True).size().to_string())
""")

# =================================================================== catalog
md(r"""
## Which gene models does embpy have?

Genes are served by two families in `model_catalog`, and it is worth seeing both
before choosing anything.
""")

code(r"""
static_catalog = embedder.model_catalog("static")
dna_catalog = embedder.model_catalog("dna")
print(f"static lookup tables: {len(static_catalog)}    DNA sequence models: {len(dna_catalog)}")
display(static_catalog)
display(dna_catalog)
""")

md(r"""
Fifty DNA keys overstates the choice considerably: most are *size, species, or
seed variants* of the same trained model, and comparing `nt_v2_50m` against
`nt_v2_500m` measures scale, not architecture. The roster below takes **one
checkpoint per distinct architecture**, which is what makes a cross-model
comparison mean anything.

| Key | Model | Why it earns a slot |
| --- | --- | --- |
| `hyenadna_small_32k` | HyenaDNA | implicit long convolutions, 32 kb context, **no attention at all** |
| `nt_v2_100m` | Nucleotide Transformer v2 | the standard 6-mer BPE transformer baseline |
| `gena_lm_bert_base` | GENA-LM | BPE-tokenised BERT from a different group and corpus |
| `caduceus_ph_131k` | Caduceus | bi-directional Mamba SSM, reverse-complement equivariant |
| `enformer_human_rough` | Enformer | the odd one out -- *supervised* on genomic tracks, not a language model |

Skipped on purpose: `hyenadna_tiny_1k/medium_160k/large_1m`, `nt_v2_50m/250m/500m`,
`nt_500m_*`, `ntv3_*`, `gena_lm_*_large/multi/bigbird` (scale, corpus and
tokeniser variants); every `*_mouse` key (this panel is human); `borzoi_*` and
`flashzoi_*` (a 524 kb-context track predictor -- it has its own notebook,
[Variant effects](variant_effects.ipynb)); `evo1*`/`evo2_*` (need a GPU and
FlashAttention); `alphagenome` (a hosted API needing a key); `scooby_*` (installs
from a git URL).

On the static side the choice is smaller and every table that *works* earns a
slot, because they are built from genuinely different evidence. That qualifier is
doing real work: the roster is now derived from embpy's own download
specification, and one advertised key still points at a file the public data
repository does not carry. The [embed section](#embed-every-gene-capable-model)
checks that rather than glossing over it.

`get_model(..., load=False)` reads a wrapper's declared capabilities without
downloading weights, which is how the table below can mention a 40B checkpoint
cheaply.
""")

code(r"""
DNA_ROSTER = ["hyenadna_small_32k", "nt_v2_100m", "gena_lm_bert_base",
              "caduceus_ph_131k", "enformer_human_rough"]

rows = []
for key in DNA_ROSTER + ["borzoi_v0", "evo2_40b", "alphagenome", "scooby_neurips"]:
    try:
        w = embedder.get_model(key, load=False)
        rows.append({"model": key, "wrapper": type(w).__name__,
                     "has_attention": w.has_attention, "status": "available"})
    except Exception as exc:
        rows.append({"model": key, "wrapper": "-", "has_attention": None,
                     "status": f"{type(exc).__name__}"})

display(pd.DataFrame(rows).set_index("model"))
""")

md(r"""
`has_attention` is a class attribute, so that column costs nothing to read -- and
it is already telling you where the [attention section](#reading-a-dna-models-attention)
can go. **HyenaDNA and Caduceus report `False` by construction**: one is built
from implicit long convolutions and the other from a state-space model, so
neither computes an attention matrix that could be extracted. That is a property
of the architecture, not a gap in embpy. `docs/attention_extraction.md` records
the per-model reasoning.
""")

Path(sys.argv[1] if len(sys.argv) > 1 else "cells_part1.json").write_text(json.dumps(CELLS))
print(f"part 1: {len(CELLS)} cells")
