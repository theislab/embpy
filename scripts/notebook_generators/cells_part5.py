"""Part 5 of docs/notebooks/cells.ipynb -- attention, decode, annotate, close."""
from __future__ import annotations
import json, sys
from pathlib import Path

CELLS: list[tuple[str, str]] = []
def md(s: str) -> None: CELLS.append(("markdown", s.strip("\n")))
def code(s: str) -> None: CELLS.append(("code", s.strip("\n")))

# ================================================== A. attention
md(r"""
## Reading a cell model's attention

Every number so far came from a pooled vector: one row per cell, the model's
whole opinion compressed. Attention is the layer underneath -- which *genes* the
model looked at while forming that opinion.

For a cell model this is unusually interpretable, because the tokens are genes.
An attention row is a distribution over the genes in one cell, so asking "did
this cell's attention concentrate on its own marker genes?" is a question with a
checkable answer.

Two models in the roster can answer it, for different reasons:

* **`geneformer_v2_12L`** is a HuggingFace BERT underneath, so
  `extract_attention` takes the native `output_attentions` path and returns
  `(batch, heads, seq, seq)`.
* **`tahoe_70m`** is not HuggingFace, so it goes through a forward pre-hook that
  flips `need_weights` back on. It is the one single-cell wrapper whose capture
  is *verified* on this cluster rather than inferred -- `(4, 8, 1606, 1606)` from
  a four-cell embed.

The indexing helpers matter here and are easy to get backwards.
`tl.block_to_attention_index(b) == b`, but
`tl.block_to_hidden_state_index(b) == b + 1`, because hidden-state index 0 is the
embedding layer before any transformer block has run.
""")

code(r"""
ATTENTION_MODEL = "geneformer_v2_12L"

# A small slice: attention is (batch, heads, seq, seq), and seq is the number of
# ranked gene tokens, so the tensor grows quadratically in a hurry.
N_ATTN_CELLS = 4
attn_slice = prepared[:N_ATTN_CELLS].copy()

ATTENTION = None
if f"X_{ATTENTION_MODEL}" in EMBEDDINGS:
    wrapper = embedder._get_or_load_singlecell_wrapper(
        ATTENTION_MODEL, batch_size=N_ATTN_CELLS, device_str="auto"
    )
    try:
        ATTENTION = wrapper.extract_attention(attn_slice, layers=None)
    except Exception as exc:  # noqa: BLE001 - report, do not stop
        print(f"extract_attention failed: {type(exc).__name__}: {exc}")
    else:
        blocks = sorted(ATTENTION)
        first = ATTENTION[blocks[0]]
        print(f"{ATTENTION_MODEL}: {len(blocks)} blocks captured, keys {blocks}")
        print(f"per-block shape: {tuple(first.shape)}  "
              f"(batch, heads, seq, seq)")
        print(f"block -> attention index: {tl.block_to_attention_index(0)}")
        print(f"block -> hidden-state index: {tl.block_to_hidden_state_index(0)}"
              f"  (0 is the embedding layer)")
else:
    print(f"{ATTENTION_MODEL} is not in EMBEDDINGS; attention section skipped")
""")

md(r"""
### What the heads are doing

Raw attention tensors are too large to read. `tl` provides summaries that
collapse them into something inspectable, and the two most useful ask opposite
questions: how *spread out* is each head's attention, and how much attention does
each gene *receive*.

A head with near-maximal entropy is attending to everything equally, which is
another way of saying it is attending to nothing. A head with low entropy has
picked a few genes. Both exist in a trained model, and the mix per layer is
informative.
""")

code(r"""
if ATTENTION:
    blocks = sorted(ATTENTION)
    rows = []
    for block in blocks:
        entropy = tl.attention_entropy(ATTENTION[block])
        uniformity = tl.head_uniformity(ATTENTION[block])
        rows.append({
            "block": block,
            "mean_entropy": float(np.mean(entropy)),
            "min_entropy": float(np.min(entropy)),
            "head_uniformity": float(np.mean(uniformity)),
        })
    ATTN_SUMMARY = pd.DataFrame(rows).set_index("block")
    display(ATTN_SUMMARY.round(3))

    ranked = tl.rank_layers(ATTENTION)
    print(f"layers ranked by informativeness: {ranked}")
""")

md(r"""
### Does attention land on the marker genes?

The sharper test. Derive a marker set from the data -- not a hand-picked
signature, which would not transfer when the atlas is swapped -- and ask whether
cells of that label concentrate attention on it, using another label as the
contrast.

Two honest caveats before the number appears. This is four cells, so it is an
illustration and not a measurement. And **attention is not attribution**: a head
attending to a gene does not establish that the gene drove the embedding. It
says where the model looked, which is a weaker and different claim.
""")

code(r"""
if ATTENTION:
    # Markers from the data, so this survives a change of dataset.
    marker_source = prepared.copy()
    sc.tl.rank_genes_groups(
        marker_source, groupby=LABEL_KEY, method="wilcoxon", n_genes=25
    )
    label_of_interest = str(prepared.obs[LABEL_KEY].value_counts().index[0])
    MARKERS = [
        str(g) for g in
        marker_source.uns["rank_genes_groups"]["names"][label_of_interest]
    ]
    print(f"markers for {label_of_interest!r}: {MARKERS[:8]} ...")

    block = sorted(ATTENTION)[-1]     # the last block, closest to the output
    to_set = tl.attention_to_gene_set(
        ATTENTION[block], gene_names=list(attn_slice.var_names), gene_set=MARKERS
    )
    received = tl.received_attention(ATTENTION[block])
    print(f"\nattention mass on the marker set, per cell: "
          f"{np.round(np.asarray(to_set).ravel(), 4).tolist()}")
    print(f"labels of those cells: "
          f"{attn_slice.obs[LABEL_KEY].astype(str).tolist()}")
    print(f"received-attention vector length: {np.asarray(received).size}")
    del marker_source
""")

md(r"""
### Where attention is not available, and why

Two models in the roster return `has_attention = False`, and both are honest
refusals with a specific cause rather than an unimplemented feature:

* **`scgpt`** builds `FlashMHA` unconditionally (`singlecell_models.py:724`).
  Flash attention fuses the softmax into the kernel and never materialises the
  weight matrix, so there is nothing to capture -- the weights do not exist at
  any point.
* **`state`** uses a fused `F.scaled_dot_product_attention`
  (`singlecell_models.py:980`), for the same reason.

That is a real architectural trade-off rather than an embpy limitation: the
kernels that make these models fast are the kernels that discard the weights.
""")

md(r"""
### Two flags that promise what they cannot deliver

Keeping the catalogue honest cuts both ways, so here is where embpy's own
metadata is wrong.

`SingleCellWrapper` sets `has_attention = True` as its class default
(`singlecell_models.py:504`), and only `scgpt` and `state` override it. So
**`PCAEmbedding` and `ScVIToolsWrapper` both advertise attention**, and neither
can deliver it -- there is no attention in a PCA or a VAE to begin with.

The two fail differently, and the difference matters:

* **`PCAEmbedding`** raises `NotImplementedError`, which is the right answer
  reached for the wrong reason. `torch_module()` reads `self._model`, and PCA
  keeps its fit state in `self._pca`, so the module lookup returns `None`. The
  outcome is correct; only the advertised flag is wrong.
* **`ScVIToolsWrapper`** raises the same error while a real `torch.nn.Module`
  sits one attribute away. `torch_module()` reads `self._model`, but
  `embed_cells` assigns the trained model to `self._trained_model`
  (`singlecell_models.py:1856`). The module is reachable at
  `wrapper._trained_model.module` -- the same object `decode_cells` uses from
  line 1913 onward. So the introspection API cannot see a module that
  demonstrably exists.

`StackWrapper` inherits `True` with no `_get_layer_modules` override and no
recorded evidence either way, which makes it an unverified promise rather than a
known failure.

Show the raise, then show the module, because a claim like this is only worth
making if it is demonstrated.
""")

code(r"""
rows = []
for model_key in ("pca", "scvi"):
    if f"X_{model_key}" not in EMBEDDINGS:
        continue
    w = embedder._get_or_load_singlecell_wrapper(model_key, batch_size=8,
                                                 device_str="auto")
    try:
        w.extract_attention(prepared[:2].copy(), layers=None)
        outcome = "returned something"
    except NotImplementedError as exc:
        outcome = f"NotImplementedError: {str(exc)[:60]}"
    except Exception as exc:  # noqa: BLE001
        outcome = f"{type(exc).__name__}: {str(exc)[:60]}"

    # Is a torch module actually reachable, by any route?
    via_api = w.torch_module() is not None
    via_attr = getattr(getattr(w, "_trained_model", None), "module", None) is not None
    rows.append({
        "model": model_key,
        "has_attention": w.has_attention,
        "torch_module() finds one": via_api,
        "one exists at _trained_model.module": via_attr,
        "extract_attention": outcome,
    })

if rows:
    display(pd.DataFrame(rows).set_index("model"))
    print("has_attention=True with no reachable module is a metadata bug, "
          "not a missing feature.")
""")

# ================================================== B. decode
md(r"""
## Decode: from latent back to expression

Most of the roster is an encoder only. Five registry keys declare
`supports_decode` -- `pca`, `scvi`, `scanvi`, `totalvi` and `state` -- and for
those the latent space can be projected back to gene expression.

That makes a check available which no similarity metric provides: **how much of
the original cell survives the round trip?** An embedding that reconstructs
expression well has kept the information; one that reconstructs badly has thrown
some away, whatever its scIB score says.

`stack` is the only `supports_generation` entry in the registry -- it can
synthesise whole profiles rather than merely reconstruct them. It fails inside
arc-stack's own h5ad reader on this environment, which is why this notebook has
no generation section. That is a real gap, not an omission.
""")

code(r"""
DECODERS = [k for k in ("pca", "scvi", "state")
            if f"X_{k}" in EMBEDDINGS and singlecell_info(k).supports_decode]
print(f"decode-capable models present: {DECODERS}")

rows = []
for model_key in DECODERS:
    obsm_key = f"X_{model_key}"
    try:
        decoded = embedder.decode_cells(
            prepared, model=model_key, obsm_key=obsm_key,
            write_layer=f"{obsm_key}_reconstructed",
        )
    except Exception as exc:  # noqa: BLE001
        print(f"  {model_key:8} decode failed: {type(exc).__name__}: {exc}")
        continue

    # Compare against the log-normalised layer, which is what the decoders
    # target -- not the raw counts.
    target = prepared.layers.get("log_normalized", prepared.X)
    target = np.asarray(target.todense() if hasattr(target, "todense") else target)
    recon = np.asarray(decoded.todense() if hasattr(decoded, "todense") else decoded)

    # Per-gene correlation, then the median across genes: a single global
    # correlation is dominated by the mean expression profile and looks
    # flatteringly high for any method.
    per_gene = []
    for j in range(0, target.shape[1], max(1, target.shape[1] // 500)):
        a, b = target[:, j], recon[:, j]
        if a.std() > 1e-8 and b.std() > 1e-8:
            per_gene.append(float(np.corrcoef(a, b)[0, 1]))
    rows.append({
        "model": model_key,
        "dim": prepared.obsm[obsm_key].shape[1],
        "median_gene_r": float(np.median(per_gene)) if per_gene else np.nan,
        "genes_sampled": len(per_gene),
    })

if rows:
    RECONSTRUCTION = pd.DataFrame(rows).set_index("model")
    display(RECONSTRUCTION.round(3))
    print("Median per-gene correlation. A global correlation would be dominated "
          "by the mean expression profile and flatter every method.")
""")

# ================================================== C. annotate
md(r"""
## Annotating the same object

The embeddings, the scIB scores and the reconstructions all live on one AnnData,
and the annotation layer attaches to the same object. Two sides to it, and only
one applies cleanly to a cell-by-gene matrix.

**The `.var` side works.** Rows are cells but columns are genes, so gene-level
annotation lands in `.var` and describes the features every model consumed.

**The `.obs` side mostly does not.** `tl.annotate_gene_perturbations` and
`tl.annotate_drug_perturbations` expect one row per *perturbation*, not one row
per cell. On this atlas the rows are cells, so there is nothing for them to key
on. [05_annotate_entities](05_annotate_entities.ipynb) covers that layer properly
and this notebook does not duplicate it.

> **embpy has no cell-type annotator.** No marker-based classifier, no
> reference-mapping method. Every `cell_type` label in this notebook arrived with
> the dataset, and scIB scores the embeddings *against* those labels. If your
> data has no labels, section 1 cannot run at all -- which is a genuine
> limitation of the metric, not something a better embedding would fix.
""")

code(r"""
# Gene-level annotation goes to .var and describes the features, not the cells.
# Take the top HVGs rather than all 19k: the annotators are network-bound and
# the point is the shape of the result, not a full sweep.
N_GENES_TO_ANNOTATE = 12
top_genes = (
    prepared.var.sort_values("dispersions_norm", ascending=False).index[:N_GENES_TO_ANNOTATE]
    if "dispersions_norm" in prepared.var.columns
    else prepared.var_names[:N_GENES_TO_ANNOTATE]
)
print(f"annotating {len(top_genes)} genes: {list(top_genes[:6])} ...")

gene_frame = ad.AnnData(
    X=np.zeros((len(top_genes), 1), dtype=np.float32),
    obs=pd.DataFrame({"symbol": list(top_genes)}, index=list(top_genes)),
)
try:
    gene_frame = tl.annotate_gene_perturbations(
        gene_frame, column="symbol", sources=["pathways"]
    )
except Exception as exc:  # noqa: BLE001 - live APIs, degrade to a report
    print(f"gene annotation unavailable: {type(exc).__name__}: {exc}")
else:
    cols = [c for c in gene_frame.obs.columns if c.startswith("gene_")]
    display(gene_frame.obs[cols].head(8))

    # Carry the result back onto .var, where it describes the features.
    for col in cols:
        prepared.var[col] = gene_frame.obs[col].reindex(prepared.var_names)
    print(f"columns added to .var: {cols}")
""")

md(r"""
Note the shape of that call: a list of gene symbols had to be wrapped in an
AnnData with a placeholder `np.zeros` matrix. The `tl.annotate_*` family is
AnnData-in, AnnData-out -- it joins on an identifier column and writes prefixed
columns into `.obs`, and never reads `.X` at all. There is no list-in,
frame-out form, so the zero matrix is boilerplate rather than data.
""")

# ================================================== D. save and close
md(r"""
## Save the artifact

An artifact you cannot read back is not an artifact. Write it, reload it, and
check that the embeddings *and* the provenance survived -- `.obsm` matrices are
the easy part, and `.uns` is where round trips usually break, because h5ad has
to serialise a nested dictionary of mixed types.
""")

code(r"""
ARTIFACT = OUTPUT_DIR / "cell_embeddings.h5ad"
prepared.write_h5ad(ARTIFACT)
print(f"wrote {ARTIFACT} ({ARTIFACT.stat().st_size / 1e6:.1f} MB)")

reloaded = sc.read_h5ad(ARTIFACT)
obsm_survived = [k for k in EMBEDDINGS if k in reloaded.obsm]
print(f"\n.obsm keys recovered: {len(obsm_survived)} of {len(EMBEDDINGS)}")
missing = sorted(set(EMBEDDINGS) - set(obsm_survived))
if missing:
    print(f"LOST: {missing}")

# Values, not just keys: a key that survives with mangled values is worse than
# one that is missing, because nothing downstream will notice.
identical = all(
    np.allclose(np.asarray(prepared.obsm[k]), np.asarray(reloaded.obsm[k]))
    for k in obsm_survived
)
print(f"matrices identical after the round trip: {identical}")

prov = reloaded.uns.get("embpy_cell_embeddings")
if prov is None:
    print("provenance LOST: .uns['embpy_cell_embeddings'] did not survive")
else:
    print(f"provenance keys: {sorted(prov)[:8]}")
    if "__preprocessing__" in prov:
        display(pd.Series(dict(prov["__preprocessing__"])).to_frame("preprocessing"))
""")

md(r"""
## What we found

**The measured verdict**, reprinted so the conclusion sits beside the evidence
rather than being asserted from memory:
""")

code(r"""
print(BATCH_VERDICT if "BATCH_VERDICT" in globals() else "section 1 did not run")
print()
if "SCIB" in globals() and not SCIB.empty:
    print("scIB, best by total:")
    display(SCIB[["bio_conservation", "batch_correction", "total"]].round(3))
if "CELL_EVAL" in globals() and not CELL_EVAL.empty:
    print("cell-eval, averaged over perturbations:")
    display(CELL_EVAL.round(3))
""")

md(r"""
**Six things worth carrying away.**

1. **The two metric families need different experiments.** scIB is an atlas
   integration benchmark and cell-eval is a perturbation-prediction benchmark.
   Neither substitutes for the other, and running either on the wrong dataset
   produces numbers rather than answers -- which is why this notebook loads two
   objects instead of one.
2. **Check the covariate before you score it.** A batch variable with a
   near-zero silhouette gives every model a high `batch_correction` and ranks
   nothing. Part 1 measures it up front so section 1's column can be read for
   what it is.
3. **A high batch score is only good if the covariate was a nuisance.** Rewarding
   a model for erasing a difference you care about is the same arithmetic as
   rewarding it for removing one you do not.
4. **The comparison is not apples to apples unless you say what each model was
   told.** `scvi` can be handed the batch key; `scanvi` cannot run without the
   labels. `X_scvi` against `X_scvi_batch` measures the difference that makes.
5. **Vocabulary overlap bounds everything downstream.** A `symbol`-vocabulary
   model given Ensembl IDs returns empty embeddings with no error, and for a
   rank tokeniser a partial overlap changes which genes make the cut -- so it
   alters the representation of the genes that *were* recognised.
6. **Point every pair metric at the embedding, or check which did not get
   pointed.** An unconfigured cell-eval metric scores `.X` and sits in the same
   table looking identical. One of them cannot be configured at all.

**Where to go next.**

* [04_benchmark_models](04_benchmark_models.ipynb) -- the short version of both
  sections, five models on one dataset.
* [05_annotate_entities](05_annotate_entities.ipynb) -- the annotation layer, on
  entities where it applies row-wise.
* [genes](genes.ipynb), [proteins](proteins.ipynb),
  [small molecules](small_molecules.ipynb) -- the other modality deep dives.
  Those are where embpy's own comparison metrics do the work, because no
  community standard exists for them.
""")

Path(sys.argv[1] if len(sys.argv) > 1 else "cells_part5.json").write_text(json.dumps(CELLS))
print(f"part 5: {len(CELLS)} cells")
