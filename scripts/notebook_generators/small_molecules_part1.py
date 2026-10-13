"""Generate part 1 of docs/notebooks/small_molecules.ipynb -- intro, panel, catalogue.

Companion to the numbered tutorials: those introduce one idea at a time, this
runs every runnable molecule model in embpy through the whole metric set and
then against the public drug record.
"""
from __future__ import annotations
import json, sys
from pathlib import Path

CELLS: list[tuple[str, str]] = []
def md(src: str) -> None: CELLS.append(("markdown", src.strip("\n")))
def code(src: str) -> None: CELLS.append(("code", src.strip("\n")))

# ===================================================================== intro
md(r"""
# Small molecules

**What you'll learn.** The numbered tutorials introduce one idea at a time on a
handful of entities. This notebook is the small-molecule *deep dive*: every
runnable molecule model in embpy, compared with every metric embpy ships, and
then held against the public clinical record for the same 40 compounds.

Molecules are the modality where the two kinds of representation are furthest
apart, and where the gap is easiest to state precisely:

* **Structural fingerprints** are hand-specified bit vectors. A Morgan bit fires
  when a particular circular atom environment is present. Nothing about the
  vector was learned, and nothing in it knows what the molecule *does*.
* **Learned representations** are transformers trained on SMILES strings. They
  saw hundreds of millions of molecules and no pharmacology, so they know more
  chemistry than a bit vector and still nothing about mechanism.

Neither family was ever told what a drug is for. The public record was, and
comparing the two is the point of the second half.

It answers four questions in order:

| Section | Question |
| --- | --- |
| [Embed](#embed-every-molecule-model) | which molecule models exist, and what do they cost? |
| [Compare](#1-global-geometry) | do they encode the same structure? |
| [Interpret](#does-the-geometry-track-biology) | does that structure track the mechanism labels? |
| [Annotate](#the-clinical-record) | what does the public record say, and does any embedding recover it? |

Prerequisites: [Comparing embeddings](03_compare_embeddings.ipynb) for the metric
definitions and [Benchmarking models](04_benchmark_models.ipynb) for the
downstream protocol. Both are used here without re-deriving. There is no
attention section: the flag exists on the molecule transformers, the
[catalogue](#which-molecule-models-does-embpy-have) says why it is not used, and
residue-level reading belongs to [proteins](proteins.ipynb) and
[genes](genes.ipynb).
""")

code(r"""
import json
import time
import warnings
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display

import embpy
from embpy import BioEmbedder, pl, tl

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 40)

# The notebook is executed from docs/notebooks, so a relative path keeps the
# artifact next to the sibling notebooks' outputs rather than in the repo root.
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# MoLFormer loads custom remote code and reports newly initialised head weights
# that mean pooling never touches. ChemBERTa already quiets transformers around
# its own from_pretrained call, MoLFormer does not, so the global switch is what
# keeps the sweep output in the next part readable.
from transformers import logging as hf_logging

hf_logging.set_verbosity_error()

# No organism argument. A molecule has no species, and the constructor's default
# only configures the gene and protein resolvers, which this notebook never uses.
embedder = BioEmbedder(device="auto")

print(f"embpy {embpy.__version__} on device {embedder.device}")
""")

# ===================================================================== panel
md(r"""
## A panel where structure and mechanism disagree

Comparison metrics need enough entities to be meaningful, and interpretation
needs *labels that are not derived from the embeddings*. So the panel is 40
approved or once-approved drugs in five mechanism classes of eight -- a grouping
from pharmacology, not from any model.

A panel every model handles identically measures nothing. What makes this one
useful is that **structural similarity and mechanism similarity disagree across
it, and disagree by design**:

| Family | What it is | Prior expectation |
| --- | --- | --- |
| `statin` | HMG-CoA reductase inhibitors | easy: a congeneric series, though it splits into a fungal decalin subgroup (lovastatin, simvastatin, mevastatin, pravastatin) and a synthetic heptenoic-acid subgroup (atorvastatin, rosuvastatin, fluvastatin, pitavastatin) |
| `benzodiazepine` | GABA-A positive allosteric modulators | easiest: all eight are built on one 1,4-benzodiazepine ring, so any fingerprint should recover the family almost perfectly |
| `beta_blocker` | beta-adrenergic antagonists | mostly easy: six are aryloxypropanolamines, but labetalol is an arylethanolamine and carvedilol carries a carbazole, so both should sit further out |
| `kinase_inhibitor` | ATP-competitive protein kinase inhibitors | hard: one target family, unrelated scaffolds -- imatinib's phenylaminopyrimidine, sorafenib's diaryl urea, sunitinib's oxindole |
| `nsaid` | cyclooxygenase inhibitors | hardest: one mechanism spanning salicylate (aspirin), propionic acid (ibuprofen, naproxen), acetic acid (diclofenac, indomethacin), coxib (celecoxib, rofecoxib) and oxicam (meloxicam) chemistry |

The prediction that falls out of the table is specific enough to be wrong: a
structural method should rank the families roughly `benzodiazepine` >= `statin`
>= `beta_blocker` > `kinase_inhibitor` > `nsaid` on label purity, and a learned
representation has room to close some of that gap. If a fingerprint *cannot*
recover the benzodiazepines, that is a red flag about the model rather than
about the panel. That is a prediction, and the
[interpretation section](#does-the-geometry-track-biology) scores it rather than
assuming it.
""")

code(r"""
PANEL: dict[str, list[str]] = {
    # tight congeneric series: shared scaffold as well as shared mechanism
    "statin": ["atorvastatin", "simvastatin", "lovastatin", "pravastatin",
               "rosuvastatin", "fluvastatin", "pitavastatin", "mevastatin"],
    "benzodiazepine": ["diazepam", "lorazepam", "alprazolam", "clonazepam",
                       "midazolam", "temazepam", "oxazepam", "chlordiazepoxide"],
    "beta_blocker": ["propranolol", "atenolol", "metoprolol", "bisoprolol",
                     "carvedilol", "nadolol", "timolol", "labetalol"],
    # mechanism-coherent, structurally diverse: shared target class only
    "kinase_inhibitor": ["imatinib", "dasatinib", "nilotinib", "erlotinib",
                         "gefitinib", "sorafenib", "sunitinib", "lapatinib"],
    "nsaid": ["aspirin", "ibuprofen", "naproxen", "diclofenac", "celecoxib",
              "rofecoxib", "indomethacin", "meloxicam"],
}

COMPOUNDS = [name for members in PANEL.values() for name in members]
FAMILY = {name: fam for fam, members in PANEL.items() for name in members}

# The categorical is ordered, and ordered in PANEL order rather than
# alphabetically, so every later table and legend reads easy families first and
# hard families last without re-sorting at each call site.
mol_space = ad.AnnData(
    X=np.zeros((len(COMPOUNDS), 1), dtype=np.float32),
    obs=pd.DataFrame(
        {
            "compound": COMPOUNDS,
            "family": pd.Categorical(
                [FAMILY[name] for name in COMPOUNDS],
                categories=list(PANEL), ordered=True,
            ),
        },
        index=pd.Index(COMPOUNDS, name="compound"),
    ),
    var=pd.DataFrame(index=["placeholder"]),
)

# A repeated name would silently shrink the panel and quietly bias every purity
# score, so the count is printed rather than assumed.
print(f"{mol_space.n_obs} compounds in {len(PANEL)} families")
print(f"unique names: {len(set(COMPOUNDS))}/{len(COMPOUNDS)}")
print(mol_space.obs["family"].value_counts().sort_index().to_string())
""")

md(r"""
### Two compounds are in the panel for what they will expose later

Most of the 40 are here to fill out a family. Two are here for a specific reason,
and it is worth naming them now so their appearance later is not a surprise.

* **rofecoxib** is withdrawn. It was pulled worldwide in 2004 over
  cardiovascular risk, and it is the compound that makes
  [the safety record](#the-safety-record) show something rather than print eight
  rows of `None`. It also pairs with celecoxib -- same COX-2 mechanism,
  different scaffold -- for the
  [scaffold-hopping test](#molecule-only-scaffold-hopping).
* **imatinib** is the compound whose mechanism of action embpy used to report as
  empty. ChEMBL registers drug-level rows against the *parent* molecule, so a
  query filtered on `molecule_chembl_id` returns nothing for a drug curated
  against a salt.
  [Mechanism, and the parent-molecule trap](#mechanism-and-the-parent-molecule-trap)
  shows the count both ways.

> **Structural, not causal.** Every distance in this notebook is a distance
> between representations of a structure. A model putting two compounds close
> together is a statement about their SMILES strings, not evidence that they
> share a target, a dose, or a safety profile. The annotation sections exist
> precisely because that inference does not come for free.
""")

# ================================================================= catalogue
md(r"""
## Which molecule models does embpy have?

`model_catalog` is the inventory. Molecules are served by one family, and unlike
DNA there are few enough keys to print the whole thing.
""")

code(r"""
molecule_catalog = embedder.model_catalog("molecule")
print(f"molecule keys: {len(molecule_catalog)}")
display(molecule_catalog)
""")

md(r"""
Fourteen keys overstates the choice in two different ways, and they need separating
before anything is embedded.

Eight of the fourteen are `RDKitWrapper` in different clothes: the registry maps
the key to a `fingerprint_type` string and the wrapper dispatches on it, so
`morgan_fp` and `morgan_count_fp` are one algorithm read two ways. Three of the
remaining six need a backend that is not installed here. That leaves nine keys
worth running: five fingerprint algorithms and two transformer architectures,
plus one binary-versus-count contrast and one pre-training contrast.

| Key | Wrapper | Width | Why it earns a slot |
| --- | --- | --- | --- |
| `maccs_fp` | `RDKitWrapper` | 167 | 166 hand-written substructure keys -- the only space here a chemist can read bit by bit |
| `morgan_fp` | `RDKitWrapper` | 2048 | ECFP4 bits, what a cheminformatics baseline means by "fingerprint" |
| `morgan_count_fp` | `RDKitWrapper` | 2048 | the same bits weighted by how often each environment occurs |
| `rdkit_fp` | `RDKitWrapper` | 2048 | hashed topological paths, a Daylight-style alternative to circular environments |
| `atom_pair_fp` | `RDKitWrapper` | 2048 | atom pairs with their topological distance -- a global descriptor rather than a local one |
| `torsion_fp` | `RDKitWrapper` | 2048 | four-atom torsions; registry key `torsion_fp`, internal type `topological_torsion` |
| `chemberta2MTR` | `ChembertaWrapper` | 384 | RoBERTa over SMILES, pre-trained with multi-task regression on computed properties |
| `chemberta2MLM` | `ChembertaWrapper` | 384 | same architecture and tokeniser, masked-language pre-training only -- the supervision contrast |
| `molformer_base` | `MolformerWrapper` | 768 | linear-attention transformer from a different group and corpus, loaded with `trust_remote_code=True` |

The widths for the three transformers are what the checkpoints declare. The
[sweep](#embed-every-molecule-model) reports the width it actually got, which is
the only number worth trusting.

Skipped on purpose: `atom_pair_count_fp` and `torsion_count_fp`, because the
binary-versus-count contrast is made once with Morgan and two more pairs add cost
rather than insight; `minimol`, because it needs `embpy[minimol]`, whose
`torch-sparse` and `torch-scatter` wheels only build with torch already present;
`mhg_gnn`, because it installs from a git URL and has no `pyproject` extra;
`mole`, because the pre-trained weights have never been released publicly.

`get_model(..., load=False)` reads a wrapper's declared capabilities without
downloading weights, which is how the table below can mention all twelve keys
cheaply.
""")

code(r"""
ROSTER = ["maccs_fp", "morgan_fp", "morgan_count_fp", "rdkit_fp", "atom_pair_fp",
          "torsion_fp", "chemberta2MTR", "chemberta2MLM", "molformer_base"]

rows = []
for key in ROSTER + ["minimol", "mhg_gnn", "mole"]:
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
### Available, but not runnable

Read that table carefully, because it is more optimistic than the truth. All
twelve keys report `available`, including the three the roster excludes.

`load=False` constructs the wrapper and stops. Every molecule wrapper defers its
backend import to `load()`, so construction cannot fail for a missing dependency
and `available` here means *the class exists and the registry knows the key*. It
is not a claim that the model runs. The `status` column earns its name only when
something is actually loaded, which is what the next cell does.

The `has_attention` column needs the same care. `False` on the six fingerprints
is **declared**: a Morgan bit vector is hand-computed, so there is nothing to
attend with. `True` on ChemBERTa, MoLFormer and MolE is **inherited** -- the
default is `has_attention: bool = True` at `src/embpy/models/base.py:88`, and none
of the three molecule transformers overrides it. That makes the flag a statement
about the base class rather than a verified capability, which is one reason this
notebook reports it and never calls `extract_attention` on a molecule model.
""")

code(r"""
# Loading the three excluded keys is cheap and safe: each fails at its backend
# import, before any weight download, so this cell measures the failure mode
# rather than the model. The typed error is the point -- DependencyError names an
# install line, ModelLoadError means the backend was found and still refused.
from embpy.errors import DependencyError, ModelLoadError

loaded_anyway = []
for key in ["minimol", "mhg_gnn", "mole"]:
    try:
        embedder.get_model(key, load=True)
        loaded_anyway.append(key)
        print(f"{key:10s} loads on this machine -- the backend is installed here")
    except (DependencyError, ModelLoadError) as exc:
        detail = str(exc).splitlines()[0]
        pkg = getattr(exc, "package", None)
        print(f"{key:10s} {type(exc).__name__}(package={pkg!r})")
        print(f"{'':10s}   {detail}")
    except Exception as exc:      # anything else is a genuine surprise, so name it
        print(f"{key:10s} unexpected {type(exc).__name__}: {exc}")

# A key that did load is now cached, and the sweep should embed the roster and
# nothing else, so the cache goes back to empty before part two touches it.
if loaded_anyway:
    print(f"clearing {len(loaded_anyway)} wrapper(s) loaded only to test the "
          f"failure mode")
    embedder.clear_model_cache(which="other")
""")

md(r"""
Two of those three are ordinary missing dependencies, and the messages say what to
install. `mhg_gnn` is the noisier one: it has no entry in `WRAPPER_EXTRAS`, so
embpy cannot name an extra and falls back to `DependencyError(package="unknown")`
-- honest, but not advice you can paste into a shell.

`mole` is the interesting case, because installing the backend would not help.
`_get_model` constructs `MolEWrapper(model_path_or_name="mole")` and passes no
`checkpoint_path`, and `MolEWrapper.load` raises
`ValueError("MolE requires a pretrained checkpoint path")` at
`src/embpy/models/molecule_models.py:943` when that argument is `None`. There is
no way to supply it through `BioEmbedder`. On this machine the import fails first,
so the printed error names the missing package rather than the missing checkpoint,
but both roads end in the same place.

> **Listed is not runnable.** `mole` appears in
> `list_available_models("molecule")` and in `model_catalog("molecule")`, and
> cannot be embedded with anything embpy exposes. A catalogue that quietly
> included it in a "9 of 14 models" count would be flattering itself; the roster
> above is nine because nine is what runs.
""")

code(r"""
# The fingerprint geometry is knowable without loading anything, and two claims
# in the roster table are worth checking rather than repeating: that MACCS is
# 167 wide whatever n_bits says, and that the Morgan radius is fixed at 2 because
# the registry passes no radius at all.
fp_rows = []
for key in ROSTER[:6]:
    w = embedder.get_model(key, load=False)
    fp_rows.append({
        "model": key,
        "fingerprint_type": w.fingerprint_type,
        "width": w.n_bits,
        "radius": w.radius,
        "values": "binary bits" if w.is_binary_fingerprint else "integer counts",
        "pooling": ", ".join(w.available_pooling_strategies),
    })

fingerprints = pd.DataFrame(fp_rows).set_index("model")
display(fingerprints.round(3))
""")

md(r"""
Three things in that table shape the rest of the notebook.

**`radius` is 2 everywhere, and means something in exactly two rows.** It is a
constructor default the registry never overrides, and `_compute_fingerprint`
only reads it in the two Morgan branches. MACCS, RDKit, atom-pair and torsion
fingerprints ignore it, and MACCS ignores `n_bits` too -- its 167 slots are 166
hand-curated keys plus an unused bit 0.

**`pooling` is `flat` and the value is inert.** `RDKitWrapper.embed` accepts a
`pooling_strategy` argument and never looks at it. Passing one is legal and
changes nothing, which is worth knowing before you spend an afternoon tuning it.

**The widths do not match.** 167, 2048, 384 and 768 in one comparison is not a
detail to smooth over: most similarity measures are sensitive to dimensionality,
and a naive one would rank `maccs_fp` differently for having 167 columns rather
than for encoding anything differently. [Global geometry](#1-global-geometry)
handles that explicitly, and says which of its four metrics is dimension-agnostic
and which is not.

Nothing has been embedded yet, and nothing can be until the 40 names become
structures. That is the next part's job, and it starts by measuring the
resolution rather than trusting it.
""")

Path(sys.argv[1] if len(sys.argv) > 1 else "cells_part1.json").write_text(json.dumps(CELLS))
print(f"part 1: {len(CELLS)} cells")
