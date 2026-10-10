"""Part 2 of docs/notebooks/cells.ipynb -- preprocessing, vocabulary, the sweep."""
from __future__ import annotations
import json, sys
from pathlib import Path

CELLS: list[tuple[str, str]] = []
def md(s: str) -> None: CELLS.append(("markdown", s.strip("\n")))
def code(s: str) -> None: CELLS.append(("code", s.strip("\n")))

# ==================================================== A. preprocessing
md(r"""
## Preprocess once per requirement, not once per model

There is no single "preprocess the data" step here, because the models do not
want the same input. The registry records what each one eats, and the three
answers are incompatible:

* `pca` wants **log-normalised** expression, from `.layers["log_normalized"]`.
* `scvi`, `scanvi` and `totalvi` want **raw counts**, from `.layers["counts"]`.
* the seven transformers want **raw counts in `.X`**, which they then tokenise
  themselves.

`preprocessing="auto"` reads those cards and prepares whatever the requested
models need. `resolve_singlecell_preprocessing` is the function that decides,
and it is worth calling directly once so the decision is visible rather than
implicit.

The rule it applies is deliberately conservative: if *any* requested model needs
processed expression, the whole call is lifted to `"standard"`. That is safe in
one direction only, and the reason is worth stating -- the standard pipeline
*preserves* raw counts in `.X` and `.layers["counts"]`, so a raw-count model can
still find what it needs afterwards. The converse is not true, which is why the
lift goes this way and not the other.
""")

code(r"""
from embpy.models.singlecell_models import (
    resolve_singlecell_preprocessing,
    singlecell_info,
)

resolved, report = resolve_singlecell_preprocessing(ROSTER, requested="auto")
print(f"resolved preprocessing for the whole roster: {resolved!r}")
print(f"reason: {report['reason']}\n")

REQUIREMENTS = pd.DataFrame(report["requirements"]).T
REQUIREMENTS.index.name = "model"
display(REQUIREMENTS)
""")

md(r"""
Read the `input_layer` column against `default_preprocessing`. Those two are not
the same question: `default_preprocessing` is what the pipeline has to *produce*,
`input_layer` is what the wrapper then *reads*. `pca` is the only model here that
moves both away from raw, and it is the only one with `uses_hvg` set, which is
why one classical baseline drags the whole call to `"standard"`.

**Why this matters more than it looks.** Feed every model one log-normalised
matrix and nothing raises. Geneformer will happily rank log-normalised values,
scVI will fit a negative-binomial likelihood to non-integers, and both return
confident embeddings that are answering a different question than you asked.
Silent wrongness is the failure mode this section exists to prevent, which is
why the assertion in [part 1](#assert-the-contract-do-not-trust-it) checks
integrality rather than trusting the loader.
""")

code(r"""
# Preprocess once, into a copy, so `adata` keeps the untouched counts and every
# model can be given whichever representation its card asks for.
t0 = time.perf_counter()
prepared = pp.preprocess_counts(
    adata,
    pipeline=resolved,
    min_genes=0,       # QC already applied upstream; filtering here would
    min_cells=0,       # silently change n_obs and break the contract asserts
    target_sum=1e4,
    log_transform=True,
    n_top_genes=2000,
    select_hvg=True,
    scale=False,       # scaling is a PCA convenience, not a model requirement
    copy=True,
)
print(f"preprocess_counts({resolved!r}) took {time.perf_counter() - t0:.1f}s")
print(f"layers: {list(prepared.layers.keys())}")
print(f"var columns added: {[c for c in prepared.var.columns if c not in adata.var.columns]}")
print(f"highly variable genes: {int(prepared.var['highly_variable'].sum())}")

# The load-bearing check: raw counts must survive the standard pipeline,
# because five of the roster's models read them afterwards.
raw_after = prepared.layers[COUNTS_LAYER]
print(f"\n{COUNTS_LAYER!r} still integral after preprocessing: "
      f"{float(np.abs(raw_after.data - np.round(raw_after.data)).max()) == 0.0}")
print(f"log_normalized present: {'log_normalized' in prepared.layers}")
""")

# ==================================================== B. the vocabulary
md(r"""
## The vocabulary, and the silent failure it causes

Every transformer in the roster treats a cell as a sentence whose words are
genes. That sentence is drawn from a **vocabulary**: a fixed inventory of gene
tokens, decided at pre-training and frozen. A gene the model has no token for
cannot appear in the sentence at all.

Two separate things travel under that name, and they fail differently.

**The identifier convention.** Some models were trained on gene symbols
(`TP53`), others on Ensembl IDs (`ENSG00000141510`). The registry records this as
`vocab_type`:

| `vocab_type` | Models | Meaning |
| --- | --- | --- |
| `symbol` | `scgpt`, `uce`, `state`, `stack` | wants symbols; Ensembl IDs match nothing |
| `either` | `geneformer_*`, `transcriptformer_*`, `tahoe_*` | the wrapper maps internally |
| `any` | `pca`, `scvi`, `scanvi`, `totalvi` | identifier-agnostic; it never looks up a gene |

The registry docstring states the consequence of getting it wrong plainly:
Ensembl IDs handed to a `symbol` model *"produce zero matches and the model
silently returns empty embeddings"*. No exception, no warning from the model
itself -- just a matrix of nothing that flows into every downstream table.

**The gene set.** Even with the right convention, your genes may simply not be
in the model's inventory: a targeted panel, another species, or symbol aliases
that have since been renamed.
""")

code(r"""
# embpy detects the convention by sampling var_names and matching the Ensembl
# pattern: >= 80% means ensembl_id, <= 20% means symbol, in between is "mixed"
# (which it warns about and leaves alone -- a mixed index cannot be converted
# safely in either direction).
detected = BioEmbedder._detect_vocab_type(adata.var_names)
print(f"detected convention of adata.var_names: {detected!r}")
print(f"sample: {list(adata.var_names[:3])}\n")

rows = []
for key in ROSTER:
    card = singlecell_info(key)
    # _ensure_singlecell_vocabulary reports what it *would* do without
    # committing to it, which is what makes it usable as an audit.
    _, plan = embedder._ensure_singlecell_vocabulary(adata, key, organism="human")
    rows.append({
        "model": key,
        "vocab_type": card.vocab_type,
        "action": plan["action"],
        "reason": plan.get("reason", ""),
    })

VOCAB_PLAN = pd.DataFrame(rows).set_index("model")
display(VOCAB_PLAN)
""")

md(r"""
### Why the *size* of the overlap matters, not just its sign

A reader might reasonably assume that out-of-vocabulary genes are simply
dropped, costing you signal in proportion to how many were lost. For Geneformer
that is not what happens, and the real behaviour is worse and more interesting.

Geneformer ranks each cell's genes by expression divided by that gene's median
across its entire pre-training corpus, then takes the top 4096 as the token
sequence. An out-of-vocabulary gene therefore never competes for a rank slot.
Remove genes and you change *which* genes make the cut -- so the overlap changes
the representation of the genes that were in-vocabulary all along. It is not a
proportional loss of signal; it is a change of input.

That is the argument for measuring the overlap up front rather than reasoning
about it afterwards. The vocabularies ship as files next to the weights, so this
is a lookup rather than an inference.
""")

code(r"""
import glob
import json
import os
import pickle

HELICAL_CACHE = os.path.expanduser("~/.cache/helical/models")


def load_model_vocabulary(model_key):
    # Returns (set_of_genes, keyed_on, source_path) or (None, None, reason).
    # Vocabularies live beside the weights in formats that differ per model, so
    # this reads the two that ship a plain lookup table and reports honestly
    # for the rest rather than guessing.
    if model_key == "scgpt":
        hits = glob.glob(f"{HELICAL_CACHE}/scgpt/**/vocab.json", recursive=True)
        if not hits:
            return None, None, "vocab.json not in the helical cache"
        with open(hits[0]) as fh:
            vocab = json.load(fh)
        genes = {g for g in vocab if not g.startswith("<")}
        return genes, "symbol", hits[0]
    if model_key.startswith("geneformer"):
        hits = glob.glob(
            f"{HELICAL_CACHE}/geneformer/**/token_dictionary*.pkl", recursive=True
        )
        if not hits:
            return None, None, "token_dictionary.pkl not in the helical cache"
        with open(hits[0], "rb") as fh:
            tokens = pickle.load(fh)
        genes = {g for g in tokens if not str(g).startswith("<")}
        return genes, "ensembl_id", hits[0]
    return None, None, "no plain lookup table ships with this model"


rows = []
for key in ROSTER:
    genes, keyed_on, source = load_model_vocabulary(key)
    if genes is None:
        rows.append({"model": key, "vocab_size": np.nan, "keyed_on": "-",
                     "recognised": np.nan, "fraction": np.nan, "note": source})
        continue
    # Compare on the convention the vocabulary is keyed on, not ours.
    ours = set(adata.var_names)
    if keyed_on == "ensembl_id" and detected == "symbol":
        note = "our symbols vs an Ensembl vocabulary -- the wrapper maps these"
        overlap = np.nan
    else:
        overlap = len(ours & genes)
        note = ""
    rows.append({
        "model": key,
        "vocab_size": len(genes),
        "keyed_on": keyed_on,
        "recognised": overlap,
        "fraction": overlap / adata.n_vars if overlap is not np.nan else np.nan,
        "note": note,
    })

VOCAB_OVERLAP = pd.DataFrame(rows).set_index("model")
display(VOCAB_OVERLAP)
""")

md(r"""
Two things to take from that table.

**The vocabularies are not the same size, and not by a little.** scGPT carries
roughly three times as many gene tokens as Geneformer, and the extra entries are
largely clone-named lncRNAs (`RP5-973N23.5`, `AC008079.12`) rather than
protein-coding genes. Two models trained on the same species differ by tens of
thousands of genes in what they can even represent. Neither choice is wrong, but
they are answering questions about different transcriptomes.

**A blank `recognised` is not a failure.** Where a vocabulary is keyed on
Ensembl IDs and this AnnData carries symbols, comparing the two sets directly
would report a spurious zero. Those rows are the ones `vocab_type="either"`
covers, where the wrapper does the mapping internally with its own alias
dictionary -- 173,697 entries in Geneformer's case. Reporting NaN and saying why
is the honest answer; reporting 0 would be a bug dressed as a finding.

> **The models whose vocabulary cannot be read here are not thereby safe.**
> UCE and STATE represent genes by embeddings of their protein products, which
> is how they generalise across species, so there is no flat gene list to
> intersect. That means this audit cannot bound their coverage -- not that their
> coverage is complete.
""")

# ==================================================== C. the sweep
md(r"""
## Embed with every model that will run

One `embed_cells` call per model, each writing a row-aligned matrix to
`.obsm["X_<model>"]`, so one AnnData ends up holding several views of the same
cells.

Three rules this section follows, all of them reactions to how the previous
version of this notebook behaved:

* **No failure raises.** A missing backend, an unreachable checkpoint or an
  upstream bug goes into `FAILURES` and gets printed as a table. The notebook
  this replaces raised twice, so a reader without the optional stack got a
  traceback instead of a notebook.
* **The cache is cleared between the large transformers.** Wrappers are cached
  by `(model_key, device, kwargs)`, which is what makes chunked inference cheap,
  but eight foundation models resident at once is a way to run out of GPU
  memory rather than a speed-up.
* **Timings are recorded.** Not as a benchmark -- the hardware is whatever you
  are on -- but because a two-order-of-magnitude spread across the roster is
  itself a practical result when you are choosing what to run at scale.
""")

code(r"""
SWEEP: list[str] = []
FAILURES: dict[str, str] = {}
TIMINGS: dict[str, float] = {}

# Wrapper constructor kwargs, nested by model name. The nesting is not
# optional: model_kwargs is dict[str, dict[str, Any]] keyed by model, so a
# flat {"batch_key": ...} is silently ignored and the model never sees it.
MODEL_KWARGS: dict[str, dict[str, object]] = {}


def run_model(model_key, target=None, key=None, **kwargs):
    # Embed one model, timing it and recording any failure instead of
    # letting it stop the notebook. Returns True on success.
    obsm_key = key or f"X_{model_key}"
    t0 = time.perf_counter()
    try:
        embedder.embed_cells(
            target if target is not None else prepared,
            models=[model_key],
            preprocessing="none",   # part 2 already prepared the layers
            model_kwargs=MODEL_KWARGS or None,
            **kwargs,
        )
    except Exception as exc:  # noqa: BLE001 - a failed model is data, not a stop
        FAILURES[obsm_key] = f"{type(exc).__name__}: {exc}"
        TIMINGS[obsm_key] = time.perf_counter() - t0
        print(f"  {model_key:26} FAILED  {type(exc).__name__}")
        return False
    TIMINGS[obsm_key] = time.perf_counter() - t0
    SWEEP.append(obsm_key)
    print(f"  {model_key:26} ok      {TIMINGS[obsm_key]:6.1f}s")
    return True


print("classical baselines:")
for model_key in ("pca", "scvi"):
    run_model(model_key)
""")

md(r"""
### The foundation models

These are the ones with a vocabulary, and the ones that cost real time. Each is
attempted independently, so one unavailable backend costs you that row and
nothing else.
""")

code(r"""
print("foundation models:")
for model_key in ("scgpt", "geneformer_v2_12L", "uce", "transcriptformer_sapiens",
                  "tahoe_70m"):
    run_model(model_key, batch_size=8)
    embedder.clear_model_cache()   # see the note above on resident weights
""")

md(r"""
### STATE, and the constraint that comes with it

STATE is worth its own cell for two reasons that are easy to trip over.

Its `embed_cells` writes the AnnData to a temporary `.h5ad` on disk and hands
the *path* to Arc's inference code, so the object has to be h5ad-writable --
an in-memory-only view, or an `.obs` column holding an unserialisable object,
fails here and nowhere else in the roster.

And if no checkpoint is given, `load()` downloads `arcinstitute/SE-600M`, which
is roughly 12 GB. That is fine once and painful in a loop, so point it at a
local copy when you have one. The checkpoint goes in `model_kwargs` nested
under the model name, alongside anything else the wrapper constructor takes.
""")

code(r"""
# Point STATE at a local checkpoint when one exists, rather than triggering a
# 12 GB download. The nesting under "state" is what makes it reach the
# wrapper constructor at all.
STATE_CHECKPOINT = Path("data/checkpoints/state/SE-600M/se600m_epoch16.ckpt")
if STATE_CHECKPOINT.exists():
    MODEL_KWARGS["state"] = {"checkpoint": str(STATE_CHECKPOINT)}
    print(f"using local STATE checkpoint: {STATE_CHECKPOINT}")
else:
    print("no local STATE checkpoint; load() will download SE-600M (~12 GB)")

print("\nSTATE:")
run_model("state", batch_size=8)
embedder.clear_model_cache()
MODEL_KWARGS.pop("state", None)
""")

md(r"""
### The two scvi-tools variants that exist to keep section 1 honest

`scvi` has already run above, told nothing about the data beyond the counts.
Its wrapper also accepts `batch_key`, which it forwards into
`setup_anndata` -- so it can be *told* the very covariate section 1 will score
it on removing.

Running it both ways gives the notebook its one controlled experiment: two
embeddings from the same wrapper, the same architecture and the same data,
differing in exactly one thing. Whatever gap appears between them in section 1
is the value of being told, and nothing else.

`scanvi` goes further and has no choice about it: it **raises** without
`labels_key`, so it necessarily sees the cell-type labels that
bio-conservation is scored against. That does not make it a bad model, it makes
its scores incomparable to the unsupervised ones, and section 1 marks the row
rather than averaging it in.
""")

code(r"""
print("integration variants:")

MODEL_KWARGS["scvi"] = {"batch_key": BATCH_KEY}
run_model("scvi", key="X_scvi_batch")
# embed_cells names the slot after the model, so rename to keep both runs.
if "X_scvi" in prepared.obsm and "X_scvi_batch" in SWEEP:
    prepared.obsm["X_scvi_batch"] = prepared.obsm.pop("X_scvi")
MODEL_KWARGS.pop("scvi", None)

# scANVI without labels_key: show the raise rather than describing it.
try:
    embedder.embed_cells(prepared, models=["scanvi"], preprocessing="none")
except Exception as exc:  # noqa: BLE001 - the point is the message
    print(f"\nscanvi with no labels_key -> {type(exc).__name__}: {exc}")

MODEL_KWARGS["scanvi"] = {"batch_key": BATCH_KEY, "labels_key": LABEL_KEY}
run_model("scanvi")
MODEL_KWARGS.pop("scanvi", None)
embedder.clear_model_cache()
""")

md(r"""
## What the sweep actually produced

The failures table is a first-class result, not an appendix. A model absent from
`SWEEP` is absent from every comparison for the rest of the notebook, and each
later table names which ones those are rather than quietly showing fewer rows.
""")

code(r"""
EMBEDDINGS = [k for k in SWEEP if k in prepared.obsm]

shapes = pd.DataFrame(
    [{"obsm_key": k, "dim": prepared.obsm[k].shape[1],
      "seconds": round(TIMINGS.get(k, float("nan")), 1),
      "finite": bool(np.isfinite(prepared.obsm[k]).all())}
     for k in EMBEDDINGS]
).set_index("obsm_key")
display(shapes.sort_values("seconds"))

print(f"{len(EMBEDDINGS)} of {len(ROSTER) + len(INTEGRATION_VARIANTS)} attempted "
      f"models produced an embedding")
if FAILURES:
    print("\nfailures, verbatim:")
    display(pd.Series(FAILURES, name="error").to_frame())
else:
    print("no failures")
""")

md(r"""
The dimensionality spread in that table is the reason [section
1](#1-do-the-models-agree) leads with a rank-based metric. The roster runs from
10 dimensions to 1280, and most similarity measures are not comparable across
widths -- a 1280-dimensional space simply has more room in which to be far
apart. Any comparison that ignores that is measuring model width as much as
model content.

One more check before comparing anything: an embedding of the right shape can
still be degenerate. A model whose vocabulary matched almost nothing returns a
near-constant matrix, which has a perfectly reasonable shape and no information
in it at all. Rank is the cheap test.
""")

code(r"""
rows = []
for key in EMBEDDINGS:
    M = np.asarray(prepared.obsm[key], dtype=np.float64)
    # A near-constant embedding is the signature of a vocabulary miss: the
    # shape is fine, the variance is not.
    per_dim_std = M.std(axis=0)
    rows.append({
        "obsm_key": key,
        "dim": M.shape[1],
        "effective_rank": int(np.linalg.matrix_rank(M, tol=1e-6)),
        "dead_dims": int((per_dim_std < 1e-8).sum()),
        "mean_std": per_dim_std.mean(),
    })

DEGENERACY = pd.DataFrame(rows).set_index("obsm_key")
DEGENERACY["rank_fraction"] = DEGENERACY["effective_rank"] / DEGENERACY["dim"]
display(DEGENERACY.round(4))

suspect = DEGENERACY.index[DEGENERACY["rank_fraction"] < 0.5].tolist()
print(f"embeddings using under half their dimensions: {suspect or 'none'}")
""")

Path(sys.argv[1] if len(sys.argv) > 1 else "cells_part2.json").write_text(json.dumps(CELLS))
print(f"part 2: {len(CELLS)} cells")
