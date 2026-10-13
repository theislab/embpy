"""Generate docs/notebooks/proteins.ipynb -- the protein-modality deep dive.

Companion to the numbered tutorials: those introduce one idea at a time, this
runs the whole protein family through the whole metric set.
"""
from __future__ import annotations
import json, sys
from pathlib import Path

CELLS: list[tuple[str, str]] = []
def md(src: str) -> None: CELLS.append(("markdown", src.strip("\n")))
def code(src: str) -> None: CELLS.append(("code", src.strip("\n")))

# ===================================================================== intro
md(r"""
# Proteins

**What you'll learn.** The numbered tutorials introduce one idea at a time on a
handful of entities. This notebook is the protein *deep dive*: every distinct
protein language model in embpy, compared with every metric embpy ships, plus
the readouts that only exist for proteins -- residue-level attention,
ortholog conservation, and per-site weighting.

It answers four questions in order:

| Section | Question |
| --- | --- |
| [Embed](#embed-every-protein-model) | which protein models exist, and what do they cost? |
| [Compare](#1-global-geometry) | do they encode the same structure? |
| [Interpret](#does-the-geometry-track-biology) | does that structure track protein biology? |
| [Read inside](#reading-esm-2s-attention) | what is the model actually looking at? |

Prerequisites: [Comparing embeddings](03_compare_embeddings.ipynb) for the
metric definitions and [Reading a model's attention](06_attention_weights.ipynb)
for the extract-then-summarise pattern. Both are used here without re-deriving.
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

# Every ESM checkpoint reports "some weights were newly initialised" for its
# pooler head. Mean pooling never touches that head, so the notice is noise here;
# quieting it keeps the tables below readable.
from transformers import logging as hf_logging

hf_logging.set_verbosity_error()

embedder = BioEmbedder(device="auto", organism="human")
""")

# =================================================================== panel
md(r"""
## A panel with real structure in it

Comparison metrics need enough entities to be meaningful, and interpretation
needs *labels that are not derived from the embeddings*. So the panel is 40 human
proteins in five functional classes of eight — a grouping from protein biology,
not from any model. Every metric below is scored against it.

The classes are deliberately of mixed difficulty. Some are defined by a shared
fold (kinases, glycolytic enzymes), which a sequence model should find easily.
Others are defined by what the protein *does* rather than how it is built —
"protease" spans unrelated catalytic mechanisms, and "secreted" mixes 99-residue
chemokines with much larger growth factors. Whether that distinction shows up in
the embeddings is measured below rather than assumed.
""")

code(r"""
PANEL: dict[str, list[str]] = {
    "kinase":   ["CDK1", "CDK2", "MAPK1", "MAPK3", "AURKA", "PLK1", "SRC", "CHEK1"],
    "TF":       ["TP53", "JUN", "FOS", "IRF1", "GATA3", "FOXP3", "MYC", "STAT1"],
    "secreted": ["IL2", "IL6", "TNF", "VEGFA", "IGF1", "TGFB1", "IL4", "CXCL8"],
    "glycolysis": ["GAPDH", "LDHA", "PKM", "ALDOA", "ENO1", "PGK1", "TPI1", "GPI"],
    "protease": ["CASP3", "CASP8", "CASP9", "CTSB", "GZMB", "BAX", "BID", "MMP9"],
}

symbols = [s for members in PANEL.values() for s in members]
families = [fam for fam, members in PANEL.items() for _ in members]

protein_space = ad.AnnData(
    X=np.zeros((len(symbols), 1), dtype=np.float32),
    obs=pd.DataFrame(
        {"symbol": symbols, "family": pd.Categorical(families)},
        index=pd.Index(symbols, name="protein"),
    ),
    var=pd.DataFrame(index=["placeholder"]),
)

print(f"{protein_space.n_obs} proteins in {len(PANEL)} classes")
print(protein_space.obs["family"].value_counts().to_string())
""")

# ================================================================== catalog
md(r"""
## Which protein models does embpy have?

`model_catalog` is the inventory. 23 protein keys is misleading, though: most are
*size or seed variants* of the same trained model, and comparing `esm2_650M`
against `esm2_3B` measures scale, not architecture.
""")

code(r"""
catalog = embedder.model_catalog("protein")
display(catalog)
""")

md(r"""
So the roster below picks **one checkpoint per distinct model**, which is what
makes a cross-model comparison mean anything:

| Key | Model | Why it earns a slot |
| --- | --- | --- |
| `esm1b` | ESM-1b (2020) | the original large PLM; a genuine predecessor, not a smaller ESM-2 |
| `esm1v_1` | ESM-1v | trained for *variant effect*, not general representation — same size as ESM-1b, different objective |
| `esm2_650M` | ESM-2 | the modern default; one size stands in for 8M–15B |
| `esmc_300m` | ESM-C | later EvolutionaryScale line, separate codebase from ESM-2 |
| `prot_t5_xl_half` | ProtT5 | encoder–decoder from ProtTrans — a different lineage entirely |

Skipped on purpose: `esm2_8M/35M/150M/3B/15B` and `esm1v_2..5` (scale and seed
variants — `esm1v` ships as a 5-seed ensemble, and one seed is enough to show
where the space sits); `esm3_*` (needs a licence-gated checkpoint and a Forge
token); `boltz2*` (a structure predictor, and its `boltz` dependency pins
`numpy<2`, so it needs an environment of its own).

`get_model(..., load=False)` reads a wrapper's declared capabilities without
downloading weights — which is how the table below can mention a 15B checkpoint
cheaply.
""")

code(r"""
ROSTER = ["esm1b", "esm1v_1", "esm2_650M", "esmc_300m", "prot_t5_xl_half"]

rows = []
for key in ROSTER + ["esm2_15B", "esm3_small", "boltz2"]:
    try:
        w = embedder.get_model(key, load=False)
        rows.append({"model": key, "wrapper": type(w).__name__,
                     "has_attention": w.has_attention, "status": "available"})
    except Exception as exc:
        rows.append({"model": key, "wrapper": "-", "has_attention": None,
                     "status": f"{type(exc).__name__}"})

display(pd.DataFrame(rows).set_index("model"))
""")

# =============================================================== resolution
md(r"""
## Resolve first, and measure the sequences

Long proteins are the quiet failure mode of every protein language model, and the
way they fail is worse than it looks. Three things are true of ESM-2 at once:

* `ESM2Wrapper` tokenises with `truncation=True`, but the ESM tokenizer ships **no
  `model_max_length`** — so that flag is a no-op and nothing is trimmed. The
  tokenizer even says so: *"Asking to truncate to max_length but no maximum length
  is provided ... Default to no truncation."*
* The config advertises `max_position_embeddings = 1026`, but ESM-2 uses **rotary**
  position embeddings, which extrapolate rather than index a fixed table.
* So a 2000-residue protein raises **no error**, is **not truncated**, and returns
  a perfectly normal-looking vector — computed well outside the window the model
  was trained on.

No truncation, no exception, no warning about the actual problem. The only way to
know is to measure, so resolve the sequences explicitly and look at them before
trusting anything downstream.
""")

code(r"""
from embpy.resources.protein_resolver import ProteinResolver

resolver = ProteinResolver(organism="human")
sequences = {s: resolver.get_canonical_sequence(s, id_type="symbol") for s in symbols}

missing = [s for s, q in sequences.items() if not q]
if missing:
    print(f"unresolved (UniProt lookup failed): {missing}")

protein_space.obs["length"] = [len(sequences[s]) if sequences[s] else 0 for s in symbols]

lengths = protein_space.obs["length"]
print(f"length: min {lengths.min()}  median {int(lengths.median())}  max {lengths.max()}")
print("\nlongest five:")
print(protein_space.obs.nlargest(5, "length")[["family", "length"]].to_string())
""")

code(r"""
# ESM-2 was trained on a 1024-token window: 1022 residues plus <cls> and <eos>.
ESM_RESIDUE_LIMIT = 1022

probe_wrapper = embedder.get_model("esm2_650M")
print(f"tokenizer.model_max_length      : {probe_wrapper.tokenizer.model_max_length:.3g}"
      "   <- unset, so truncation=True does nothing")
print(f"config.max_position_embeddings  : {probe_wrapper.model.config.max_position_embeddings}"
      "   <- rotary, so not a hard ceiling")
embedder.clear_model_cache()

over = protein_space.obs.query("length > @ESM_RESIDUE_LIMIT")
if len(over):
    print(f"\n{len(over)} sequence(s) exceed the trained window -- they would be "
          "extrapolated, not truncated, and not flagged:")
    print(over[["family", "length"]].to_string())
else:
    print(f"\nAll {protein_space.n_obs} sequences fit the {ESM_RESIDUE_LIMIT}-residue "
          f"trained window (longest: {lengths.max()}), so nothing below is "
          "confounded by out-of-window extrapolation.")
""")

# ==================================================================== embed
md(r"""
## Embed every protein model

One `embed` call per model, each writing its own `.obsm` key. Two practical
points:

* **`clear_model_cache()` between models.** Five checkpoints at once is roughly
  11 GB of weights; embpy caches wrappers by default, which is the right choice
  when you re-embed but the wrong one when you sweep. Dropping each model after
  use keeps the peak at one model.
* **Optional backends degrade, they do not fail.** ESM-C needs the
  EvolutionaryScale SDK (`pip install esm --no-deps`); if it is missing, that row
  is skipped with a reason and everything downstream adapts.
""")

code(r"""
spaces: dict[str, str] = {}
timings = []

for key in ROSTER:
    obsm_key = f"X_{key}"
    t0 = time.perf_counter()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            protein_space = embedder.embed(
                protein_space, entity_type="protein", id_type="symbol",
                obs_column="symbol", model=key, output="anndata",
                attach_to="obs", key=obsm_key, pooling_strategy="mean",
            )
        elapsed = time.perf_counter() - t0
        spaces[key] = obsm_key
        timings.append({"model": key, "dim": protein_space.obsm[obsm_key].shape[1],
                        "seconds": round(elapsed, 1), "status": "ok"})
    except Exception as exc:
        timings.append({"model": key, "dim": None, "seconds": round(time.perf_counter() - t0, 1),
                        "status": f"skipped -- {type(exc).__name__}: {str(exc)[:60]}"})
    finally:
        embedder.clear_model_cache()

display(pd.DataFrame(timings).set_index("model"))
print("\nembedded spaces:", list(spaces.values()))
""")

code(r"""
# Everything below works off this dict, so a skipped model simply drops out.
embeddings = {key: np.asarray(protein_space.obsm[obsm]) for key, obsm in spaces.items()}
keys = list(embeddings)
obsm_keys = [spaces[k] for k in keys]
print({k: v.shape for k, v in embeddings.items()})
""")
Path(sys.argv[1] if len(sys.argv) > 1 else "cells_part1.json").write_text(json.dumps(CELLS))
print(f"part 1: {len(CELLS)} cells")
