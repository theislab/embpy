"""Part 3 of docs/notebooks/cells.ipynb -- do the models agree with each other."""
from __future__ import annotations
import json, sys
from pathlib import Path

CELLS: list[tuple[str, str]] = []
def md(s: str) -> None: CELLS.append(("markdown", s.strip("\n")))
def code(s: str) -> None: CELLS.append(("code", s.strip("\n")))

# ==================================================== A. why this comes first
md(r"""
## 1. Do the models agree?

Everything in this section compares the embeddings **to each other**, with no
labels involved at all. That is deliberate, and the ordering matters: this is a
prior question to [section 2](#2-scib-does-integration-preserve-the-biology).

Before asking which model is *right*, it is worth knowing whether the models
even differ. Two possibilities, and they lead to different work:

* **They agree.** Then running eight of them is waste, and the cheapest one
  wins by default. Nothing downstream will separate them either.
* **They disagree.** Then the disagreement localises where the interesting
  biology -- or the interesting failure -- lives, and a label-based score has
  something to adjudicate.

Running this before scIB also buys a cross-check worth having. If two
embeddings turn out near-identical here and section 2 hands them very different
scores, the suspicion should fall on the *metric*, not the models. Run the
sections the other way round and that check is unavailable.
""")

md(r"""
### Why a rank-based metric leads

The [sweep](#what-the-sweep-actually-produced) produced embeddings from 10
dimensions to 1280. That spread breaks most similarity measures, and not
subtly: a 1280-dimensional space has far more room in which two points can be
far apart, so raw distances are not on a common scale and neither is anything
built directly from them.

`tl.alignment_matrix` offers four metrics, and they differ in exactly this
respect:

| Metric | What it compares | Null value | Dimension-safe |
| --- | --- | --- | --- |
| `tsi` | distance *orderings* | fixed 0.5 | yes |
| `qsi` | quantile-binned orderings | fixed 0.5 | yes |
| `linear_cka` | subspace alignment | 0 | mostly |
| `mutual_knn` | shared neighbour sets | k/n | yes |

TSI leads because its null is a fixed 0.5 whatever the width. Reporting all
four is not padding: where they disagree, the disagreement is informative,
because they are sensitive to different things -- ordering versus subspace
versus neighbourhood.

Two mechanical points that are easy to get wrong. `alignment_matrix` takes a
plain `Mapping` of name to matrix, **not** an AnnData, so the dict is built
from `.obsm` by hand. And `metric` and `distance` are different arguments:
`metric` selects which alignment metric runs, `distance` is the distance used
*inside* it -- and `distance` is ignored entirely for `linear_cka`.
""")

code(r"""
# alignment_matrix reads a Mapping, not an AnnData, so build the dict here.
# Keeping it explicit also makes it obvious that only surviving models appear.
SPACES = {key: np.asarray(prepared.obsm[key]) for key in EMBEDDINGS}
print(f"comparing {len(SPACES)} spaces: {list(SPACES)}")
print(f"dimensions: {{k: v.shape[1] for k, v in SPACES.items()}}".replace("'", ""))

ALIGN = {}
for metric in ("tsi", "qsi", "linear_cka", "mutual_knn"):
    t0 = time.perf_counter()
    ALIGN[metric] = tl.alignment_matrix(SPACES, metric=metric)
    print(f"  {metric:12} {time.perf_counter() - t0:5.1f}s")

display(ALIGN["tsi"].round(3))
""")

md(r"""
Read the TSI matrix as: **0.5 means the two spaces order distances no better
than chance relative to each other, 1.0 means they agree completely.** The
diagonal is 1.0 by construction and carries no information.

The off-diagonal block structure is what to look at. Models sharing an
architecture family, or a training corpus, tend to agree with each other more
than with outsiders -- and where that expectation breaks is more interesting
than where it holds.
""")

code(r"""
# Flatten the four matrices into one tidy table: easier to sort, and it makes
# the metrics directly comparable pair by pair.
rows = []
keys = list(SPACES)
for i, a in enumerate(keys):
    for b in keys[i + 1:]:
        row = {"space_a": a, "space_b": b}
        for metric, frame in ALIGN.items():
            row[metric] = float(frame.loc[a, b])
        rows.append(row)

PAIRS = pd.DataFrame(rows)
display(PAIRS.sort_values("tsi", ascending=False).round(3).head(12))

print(f"most similar pair by TSI:  {PAIRS.loc[PAIRS['tsi'].idxmax(), 'space_a']} / "
      f"{PAIRS.loc[PAIRS['tsi'].idxmax(), 'space_b']} = {PAIRS['tsi'].max():.3f}")
print(f"least similar pair by TSI: {PAIRS.loc[PAIRS['tsi'].idxmin(), 'space_a']} / "
      f"{PAIRS.loc[PAIRS['tsi'].idxmin(), 'space_b']} = {PAIRS['tsi'].min():.3f}")
""")

md(r"""
### Where the four metrics disagree

If all four metrics ranked the pairs identically, three of them would be
redundant and this notebook should drop them. Correlating the columns says
whether that is the case, and it is a cheaper question than it looks: a low
correlation between two metrics means they are picking up genuinely different
structure, and any single-number summary of "similarity" is hiding a choice.
""")

code(r"""
METRIC_AGREEMENT = PAIRS[["tsi", "qsi", "linear_cka", "mutual_knn"]].corr(
    method="spearman"
)
display(METRIC_AGREEMENT.round(3))

# Name the least-agreeing metric pair explicitly rather than leaving it to
# the reader to scan a symmetric matrix.
off = METRIC_AGREEMENT.where(~np.eye(len(METRIC_AGREEMENT), dtype=bool))
flat = off.stack().sort_values()
print(f"least-agreeing metrics: {flat.index[0][0]} vs {flat.index[0][1]} "
      f"(Spearman {flat.iloc[0]:+.3f})")
print(f"most-agreeing metrics:  {flat.index[-1][0]} vs {flat.index[-1][1]} "
      f"(Spearman {flat.iloc[-1]:+.3f})")
""")

# ==================================================== B. local neighbourhoods
md(r"""
### Local neighbourhoods, which is a different question

Global geometry and local structure can diverge. Two embeddings can place the
broad cell populations in the same relative arrangement while disagreeing about
which individual cells are neighbours -- and for most downstream single-cell
work, from clustering to label transfer to trajectory inference, it is the
*local* structure that gets used.

`tl.knn_jaccard` measures it per cell: for each cell, the overlap between its
k nearest neighbours in one space and in the other.

One scale warning, because it invites a wrong comparison. Jaccard is
intersection over **union**, not intersection over k, so it is not on the same
scale as `mutual_knn` from the table above and the two should not be read as
interchangeable. `k` is also silently clamped to `n - 1`, which matters only on
tiny objects but matters silently.
""")

code(r"""
K_NEIGHBOURS = 15

rows = []
for i, a in enumerate(keys):
    for b in keys[i + 1:]:
        _, mean_jaccard = tl.knn_jaccard(
            SPACES[a], SPACES[b], k=K_NEIGHBOURS, metric="cosine"
        )
        rows.append({"space_a": a, "space_b": b, "knn_jaccard": mean_jaccard})

JACCARD = pd.DataFrame(rows)
PAIRS = PAIRS.merge(JACCARD, on=["space_a", "space_b"], how="left")

# Global agreement high, local agreement low, is the interesting quadrant:
# the same broad layout, different neighbours.
PAIRS["global_minus_local"] = PAIRS["tsi"] - PAIRS["knn_jaccard"]
display(
    PAIRS.sort_values("global_minus_local", ascending=False)
    [["space_a", "space_b", "tsi", "knn_jaccard", "global_minus_local"]]
    .round(3)
    .head(8)
)
""")

md(r"""
The pairs at the top of that table agree about the overall shape of the data and
disagree about who is next to whom. That is the combination worth knowing about
before you pick an embedding for a neighbour-based task, because the global
metric alone would have told you they were interchangeable.
""")

code(r"""
pl.knn_overlap(prepared, obsm_keys=EMBEDDINGS, k=K_NEIGHBOURS)
plt.show()
""")

# ==================================================== C. the concrete probe
md(r"""
### Two cells, every space

Aggregate metrics are easy to nod along to. A neighbour list is checkable, so
this section takes two specific cells and reports what each space thinks they
sit next to.

The two are chosen from the data rather than named in advance: one from the
**most abundant** label, and one from the **rarest label that still has enough
cells to be meaningful**. Rare populations are where embeddings diverge most --
an abundant cell type is easy for anything to place, while a rare one is where a
model either has learned the population or is folding it into a neighbour.
""")

code(r"""
label_counts = prepared.obs[LABEL_KEY].value_counts()
usable = label_counts[label_counts >= MIN_CELLS_PER_LABEL]
COMMON_LABEL, RARE_LABEL = usable.index[0], usable.index[-1]

# Take the first cell of each, deterministically, so the narrative below
# refers to the same cells on every re-run.
probe_cells = {}
for label in (COMMON_LABEL, RARE_LABEL):
    idx = int(np.flatnonzero((prepared.obs[LABEL_KEY] == label).to_numpy())[0])
    probe_cells[label] = idx

print(f"most abundant label: {COMMON_LABEL!r} ({label_counts[COMMON_LABEL]} cells)")
print(f"rarest usable label: {RARE_LABEL!r} ({label_counts[RARE_LABEL]} cells)")
print(f"probe cells (row index): {probe_cells}")
""")

code(r"""
from sklearn.neighbors import NearestNeighbors

N_SHOW = 5
rows = []
for label, row_idx in probe_cells.items():
    for key in EMBEDDINGS:
        M = SPACES[key]
        nn = NearestNeighbors(n_neighbors=N_SHOW + 1, metric="cosine").fit(M)
        _, neigh = nn.kneighbors(M[row_idx : row_idx + 1])
        # Column 0 is the query itself; drop it.
        neighbours = neigh[0][1:]
        neighbour_labels = prepared.obs[LABEL_KEY].iloc[neighbours].tolist()
        same = sum(1 for lab in neighbour_labels if lab == label)
        rows.append({
            "probe_label": label,
            "space": key,
            f"same_label_of_{N_SHOW}": same,
            "neighbour_labels": ", ".join(str(lab)[:18] for lab in neighbour_labels),
        })

PROBE = pd.DataFrame(rows)
for label in probe_cells:
    print(f"\n--- probe cell from {label!r} ---")
    display(PROBE[PROBE["probe_label"] == label].drop(columns="probe_label")
            .set_index("space"))
""")

md(r"""
The `same_label_of_5` column is the compact version: how many of the five
nearest neighbours share the probe cell's label. For the abundant label, most
spaces should manage 5 of 5 -- if one does not, that is a finding about that
model rather than about the cell. For the rare label, expect the spaces to
diverge, and expect the neighbour lists to name *which* population each model
confuses it with. That confusion is specific and checkable in a way that a
purity score averaged over the whole dataset is not.

This is a single cell per label, so treat it as an illustration rather than a
measurement. [Section 2](#2-scib-does-integration-preserve-the-biology) does the
population-level version with labels, over every cell.
""")

# ==================================================== D. visual
md(r"""
### Looking at them

Two-dimensional projections come last, and with a caveat attached: **a UMAP is
a rendering of an embedding, not the embedding.** The projection makes its own
choices about what to preserve, and two panels that look different may be
closer than they appear, or vice versa. The numbers above are what to conclude
from; these panels are for noticing things worth measuring.
""")

code(r"""
pl.all_embeddings(
    prepared,
    obsm_keys=EMBEDDINGS,
    color=LABEL_KEY,
    method="umap",
)
plt.show()
""")

code(r"""
# The same space, coloured by the biology and by the covariate section 2 will
# ask every model to remove. Seeing both at once frames that question.
BEST_TSI_KEY = (
    PAIRS.groupby("space_a")["tsi"].mean().idxmax()
    if not PAIRS.empty else EMBEDDINGS[0]
)
pl.embedding_color_panel(
    prepared,
    obsm_key=BEST_TSI_KEY,
    method="umap",
    color_keys=[LABEL_KEY, BATCH_KEY],
    ncols=2,
    title=f"{BEST_TSI_KEY}: biology to keep, covariate to remove",
)
plt.show()
""")

md(r"""
**What section 1 established, in one line each.**

* The models do not agree -- the TSI spread across pairs says how much.
* Global and local agreement are different measurements, and the pairs where
  they diverge are the ones to be careful with.
* Four alignment metrics do not rank the pairs identically, so "similarity"
  between two embeddings is not one number.
* None of this used a label, so none of it says which model is *right*. That is
  the next section, and it needs an atlas to answer.
""")

Path(sys.argv[1] if len(sys.argv) > 1 else "cells_part3.json").write_text(json.dumps(CELLS))
print(f"part 3: {len(CELLS)} cells")
