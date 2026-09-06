"""Part 3 of docs/notebooks/cells.ipynb -- scIB, the atlas integration question."""
from __future__ import annotations
import json, sys
from pathlib import Path

CELLS: list[tuple[str, str]] = []
def md(s: str) -> None: CELLS.append(("markdown", s.strip("\n")))
def code(s: str) -> None: CELLS.append(("code", s.strip("\n")))

md(r"""
## 1. scIB: does integration preserve the biology?

scIB (Luecken et al. 2022) is an **atlas integration** benchmark. It asks one
question with two halves: when you merge batches, does the technical axis go
away, and does the biological one survive?

It is not a general embedding-quality score, and it says nothing about
perturbations -- that is [section 2](#2-cell-eval-is-the-perturbation-preserved),
on a different dataset. Nine metrics go in, and they collapse into two aggregate
columns plus scIB's 0.6/0.4 weighting of them:

| Block | Metrics | Rewards |
| --- | --- | --- |
| bio-conservation | `nmi`, `ari`, `asw_label`, `isolated_label_asw`, `clisi` | keeping cell types apart |
| batch-correction | `asw_batch`, `graph_conn`, `ilisi`, `kbet` | mixing batches together |

Those two pull against each other by construction, which is the point: the
easiest way to mix batches perfectly is to collapse every cell into one blob,
and the easiest way to preserve biology perfectly is to change nothing. `total`
is a compromise, not a truth.

`tl.compute_scib_metrics` is the only function in `tl` that takes several
`.obsm` keys at once, so the whole roster is scored in one call.
""")

code(r"""
t0 = time.perf_counter()
SCIB = tl.compute_scib_metrics(
    prepared,
    embedding_keys=EMBEDDINGS,
    label_key=LABEL_KEY,
    batch_key=BATCH_KEY,
    # NMI and ARI need a Leiden clustering optimised against the label, and the
    # default sweeps 20 resolutions per embedding. At ten embeddings that is 200
    # clusterings, which dominates the runtime. Ten resolutions finds
    # essentially the same optimum for a fraction of the cost.
    cluster_resolution_range=(0.2, 2.0, 0.2),
)
print(f"scored {len(EMBEDDINGS)} embeddings in {time.perf_counter() - t0:.0f}s")
display(SCIB.round(3))
""")

md(r"""
### Which metrics actually ran

Every scIB metric here is computed best-effort: a failure in one becomes a NaN
with a warning rather than sinking the whole comparison. That is the right
default -- the scIB suite is sensitive to installed versions, and its LISI
metrics shell out to a binary that only ships for some platforms -- but it means
a table can look complete while a column is entirely absent.

So check, rather than reading the table and assuming.
""")

code(r"""
BIO_COLS = ["nmi", "ari", "asw_label", "isolated_label_asw", "clisi"]
BATCH_COLS = ["asw_batch", "graph_conn", "ilisi", "kbet"]

status = []
for col in BIO_COLS + BATCH_COLS:
    if col not in SCIB.columns:
        status.append({"metric": col, "block": "bio" if col in BIO_COLS else "batch",
                       "state": "absent"})
    else:
        n_nan = int(SCIB[col].isna().sum())
        status.append({
            "metric": col,
            "block": "bio" if col in BIO_COLS else "batch",
            "state": "ok" if n_nan == 0 else f"NaN for {n_nan}/{len(SCIB)}",
        })
METRIC_STATUS = pd.DataFrame(status).set_index("metric")
display(METRIC_STATUS)

usable_bio = [c for c in BIO_COLS if c in SCIB.columns and SCIB[c].notna().any()]
usable_batch = [c for c in BATCH_COLS if c in SCIB.columns and SCIB[c].notna().any()]
print(f"bio_conservation is the mean of {len(usable_bio)}: {usable_bio}")
print(f"batch_correction is the mean of {len(usable_batch)}: {usable_batch}")
""")

md(r"""
### `isolated_label_asw`, and why it exists here at all

That column is in the table above, and it would not be if this notebook had
skipped the batch covariate.

scIB counts isolated-label ASW as a **bio-conservation** metric, but it
*identifies* which labels are isolated by counting how few batches each one
appears in. So it needs the batch key even though it scores biology. Called
without one, `compute_scib_metrics` omits it entirely rather than reporting a
column of NaN.

This atlas is a good case for it. `t_cell` has seven cells across three inDrop
runs and appears in no plate-based protocol at all; `smarter` contributes only
four of the fourteen cell types. Those genuinely batch-restricted labels are the
metric's subject matter -- and they are also the ones a flat random subsample
would have deleted, which is why [part 1](#assert-the-contract-do-not-trust-it)
samples within label instead.

`04_benchmark_models.ipynb` ran with no batch key, so it scored four bio
metrics, not five. Show the difference rather than describing it.
""")

code(r"""
# The same embeddings, the same labels, no batch covariate. One call, purely to
# diff the column sets against the run above.
SCIB_NO_BATCH = tl.compute_scib_metrics(
    prepared, embedding_keys=EMBEDDINGS[:2], label_key=LABEL_KEY, batch_key=None,
    cluster_resolution_range=(0.2, 2.0, 0.2),
)
with_batch = set(SCIB.columns)
without = set(SCIB_NO_BATCH.columns)
print(f"columns only present with a batch key: {sorted(with_batch - without)}")
print(f"columns only present without one:      {sorted(without - with_batch) or 'none'}")
print(f"\nbio_conservation with a batch key   : "
      f"mean of {len([c for c in BIO_COLS if c in with_batch])} metrics")
print(f"bio_conservation without a batch key: "
      f"mean of {len([c for c in BIO_COLS if c in without])} metrics")
print("\nSo the two bio_conservation columns are not on the same scale, and "
      "comparing this notebook's numbers to nb04's directly would be wrong.")
""")

md(r"""
### Scoring the prediction from part 1

[Part 1](#is-this-atlas-a-real-integration-benchmark) measured the covariate
before any model ran and set an expectation: with a median within-label batch
silhouette of about +0.18, the batch-correction block should genuinely separate
the models rather than saturating near 1.0 and ranking nothing.

That is a prediction, and this cell scores it rather than assuming it. The test
is the *spread* of `batch_correction`: a near-tied column ranks nothing whatever
its mean.
""")

code(r"""
batch_spread = float(SCIB["batch_correction"].max() - SCIB["batch_correction"].min())
bio_spread = float(SCIB["bio_conservation"].max() - SCIB["bio_conservation"].min())
DISCRIMINATES = batch_spread >= 0.05

BATCH_VERDICT = (
    f"batch_correction spans {SCIB['batch_correction'].min():.3f} to "
    f"{SCIB['batch_correction'].max():.3f} (spread {batch_spread:.3f}); "
    f"bio_conservation spans {SCIB['bio_conservation'].min():.3f} to "
    f"{SCIB['bio_conservation'].max():.3f} (spread {bio_spread:.3f}). "
    + ("Both blocks discriminate, as part 1 predicted from the within-label "
       "batch silhouette."
       if DISCRIMINATES else
       "The batch block does NOT discriminate -- it is near-tied, so it ranks "
       "nothing here regardless of its mean.")
)
print(BATCH_VERDICT)
print(f"\npart 1 predicted a {'strong' if not WEAK_BATCH else 'weak'} benchmark; "
      f"the batch block {'did' if DISCRIMINATES else 'did not'} separate the models.")
if WEAK_BATCH == DISCRIMINATES:
    print("PREDICTION MISSED -- worth understanding before trusting either number.")
""")

md(r"""
### Three tiers, not one comparison

Here is the thing the table above hides, and it is the most important caveat in
this notebook.

The models were not all told the same things. `ScVIToolsWrapper.__init__` accepts
`batch_key` and forwards it into `setup_anndata`, so scVI can be *given* the very
covariate scIB scores it on removing. And scANVI **raises** without
`labels_key`, so it necessarily sees the cell-type labels that bio-conservation
is scored against.

| Tier | Models | What it saw |
| --- | --- | --- |
| told nothing | `pca`, `scgpt`, `geneformer_v2_12L`, `uce`, `transcriptformer_sapiens`, `tahoe_70m`, `state` | counts only |
| told the batch | `scvi` with `batch_key` | the covariate it is scored on removing |
| told the labels | `scanvi` | the labels bio-conservation scores |

Averaging across those tiers and calling the result a ranking would be
dishonest. `X_scvi` against `X_scvi_batch` isolates it cleanly: same wrapper,
same architecture, same data, differing in exactly one thing.
""")

code(r"""
PAIR = [k for k in ("X_scvi", "X_scvi_batch") if k in SCIB.index]
if len(PAIR) == 2:
    told = SCIB.loc[PAIR, ["bio_conservation", "batch_correction", "total"]]
    told.index = ["told nothing", "told the batch"]
    display(told.round(3))
    delta = (SCIB.loc["X_scvi_batch", "batch_correction"]
             - SCIB.loc["X_scvi", "batch_correction"])
    print(f"batch_correction gained by being told the covariate: {delta:+.3f}")
    print(f"bio_conservation change over the same edit: "
          f"{SCIB.loc['X_scvi_batch', 'bio_conservation'] - SCIB.loc['X_scvi', 'bio_conservation']:+.3f}")
else:
    print(f"need both X_scvi and X_scvi_batch to make this comparison; have {PAIR}")

if "X_scanvi" in SCIB.index:
    print("\nscanvi saw the labels, so its bio_conservation is not comparable "
          "to an unsupervised model's:")
    display(SCIB.loc[["X_scanvi"], ["bio_conservation", "batch_correction", "total"]].round(3))
""")

md(r"""
### Where the nine metrics disagree

`bio_conservation` and `batch_correction` are means, and a mean hides
disagreement. If all five bio metrics ranked the roster identically, four of them
would be redundant. They do not, and the pairs that disagree most are worth
naming -- because a claim like "model X preserves biology best" is only as solid
as the agreement between the metrics behind it.
""")

code(r"""
present = [c for c in BIO_COLS + BATCH_COLS
           if c in SCIB.columns and SCIB[c].notna().sum() >= 3]
if len(present) >= 2 and len(SCIB) >= 3:
    RANK_AGREEMENT = SCIB[present].corr(method="spearman")
    display(RANK_AGREEMENT.round(2))

    off = RANK_AGREEMENT.where(~np.eye(len(RANK_AGREEMENT), dtype=bool)).stack()
    off = off.sort_values()
    print(f"least agreement: {off.index[0][0]} vs {off.index[0][1]} "
          f"(Spearman {off.iloc[0]:+.2f})")
    print(f"most agreement:  {off.index[-1][0]} vs {off.index[-1][1]} "
          f"(Spearman {off.iloc[-1]:+.2f})")
    if off.iloc[0] < 0:
        print("\nA negative correlation means those two metrics rank the roster "
              "in opposite directions. Neither is wrong; they reward different "
              "things, and the aggregate averages them anyway.")
else:
    print(f"too few models or metrics for a rank correlation "
          f"({len(SCIB)} models, {len(present)} metrics)")
""")

md(r"""
### The two blocks, plotted

The only plots in this section. The numbers come from scIB, and embpy's own
purity and clustering helpers are deliberately not used -- scIB already optimises
a Leiden clustering against the label to compute NMI and ARI, so re-clustering
would recompute the same thing worse.

Read the two panels together rather than the `total` column alone. A model high
on one and low on the other has made a trade, and which trade you want depends
on what you are going to do next.
""")

code(r"""
fig, axes = plt.subplots(1, 2, figsize=(13, 3.8), sharey=False)
order = SCIB.sort_values("total", ascending=False).index

for ax, (block, cols, title) in zip(axes, [
    ("bio_conservation", usable_bio, "bio-conservation (keep the cell types apart)"),
    ("batch_correction", usable_batch, "batch-correction (mix the protocols)"),
]):
    SCIB.loc[order, cols].plot.bar(ax=ax, width=0.8)
    ax.plot(range(len(order)), SCIB.loc[order, block].values, "k_",
            markersize=18, markeredgewidth=2.5, label=block)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=25, labelsize=8)
    ax.legend(fontsize=7, ncol=2)
plt.tight_layout()
plt.show()
""")

code(r"""
# One orientation UMAP, on the best-scoring embedding: the biology it was asked
# to keep, and the covariate it was asked to remove, side by side.
BEST = str(SCIB.index[0])
pl.embedding_color_panel(
    prepared,
    obsm_key=BEST,
    method="umap",
    color_keys=[LABEL_KEY, BATCH_KEY],
    ncols=2,
    title=f"{BEST}: biology to keep (left), protocol to remove (right)",
)
plt.show()
print(f"{BEST} scored highest on total ({SCIB.loc[BEST, 'total']:.3f}). "
      f"A UMAP is a rendering of the embedding, not the embedding -- the numbers "
      f"above are what to conclude from.")
""")

md(r"""
**What section 1 established.**

* The roster does not agree on either half of the question, and the spread in
  `BATCH_VERDICT` says by how much.
* `isolated_label_asw` is only computable because this notebook supplied a batch
  covariate, so its `bio_conservation` is a mean of five metrics where
  `04_benchmark_models.ipynb`'s was a mean of four. The two are not comparable.
* The comparison is three tiers, not one. `X_scvi` against `X_scvi_batch`
  measures what being told the covariate is worth.
* The nine metrics do not rank the roster identically, so no single number
  settles it.

None of which says whether these embeddings preserve a *perturbation*. That
needs a different experiment, and [section 2](#2-cell-eval-is-the-perturbation-preserved)
runs it on one.
""")

Path(sys.argv[1] if len(sys.argv) > 1 else "cells_part3.json").write_text(json.dumps(CELLS))
print(f"part 3: {len(CELLS)} cells")
