"""Part 3 of docs/notebooks/genes.ipynb -- the full comparison battery."""
from __future__ import annotations
import json, sys
from pathlib import Path

CELLS: list[tuple[str, str]] = []
def md(s: str) -> None: CELLS.append(("markdown", s.strip("\n")))
def code(s: str) -> None: CELLS.append(("code", s.strip("\n")))

# ========================================================== global geometry
md(r"""
## 1. Global geometry

Everything below runs on `dense_embeddings` / `dense_keys` / `dense_space` --
the gap-free subset -- because every metric here is a distance computation, and a
single NaN row does not cost you one cell of a distance matrix, it costs that
gene's whole row and column of it.

The models here fall into two families, and that is the whole point of the
section:

* **sequence** (`DNA_KEYS`) -- read the exon sequence, know nothing about function;
* **knowledge** (`STATIC_KEYS`) -- distilled from text, co-expression and screens,
  know nothing about sequence.

So every pairwise number below belongs to one of three buckets: *within
sequence*, *within knowledge*, or *cross-family*. If the two families really do
encode different things, cross-family pairs should score **below both**
within-family blocks. That is a prediction; the grouped table at the end of this
section is where it gets a number.
""")

code(r"""
# The pl.* helpers address embeddings by .obsm key, so mirror the dense matrices
# into dense_space under one naming convention. A rename, not a recompute.
dense_obsm: dict[str, str] = {}
for key, matrix in dense_embeddings.items():
    dense_space.obsm[f"X_{key}"] = matrix
    dense_obsm[key] = f"X_{key}"

# Order every table and every heatmap sequence-first, so that block structure --
# if there is any -- lands on the diagonal instead of being interleaved away.
dna_set, static_set = set(DNA_KEYS), set(STATIC_KEYS)
dna_dense = [key for key in dense_keys if key in dna_set]
static_dense = [key for key in dense_keys if key in static_set]
unclassified = [key for key in dense_keys if key not in dna_set and key not in static_set]
ordered_keys = dna_dense + static_dense + unclassified

obsm_keys = [dense_obsm[key] for key in ordered_keys]
dims = {key: dense_embeddings[key].shape[1] for key in ordered_keys}

print(f"{dense_space.n_obs} genes x {len(ordered_keys)} models, all gap-free")
print(f"sequence models surviving the dense mask : {dna_dense}")
print(f"knowledge tables surviving the dense mask: {static_dense}")
if unclassified:
    print(f"WARNING: {unclassified} appear in neither roster; their pairs fall "
          "through to cross-family below, which would be a mislabel")
print(f"\ndimensionality spans {min(dims.values())} -> {max(dims.values())}")
print(pd.Series(dims, name="dim").to_string())
""")

md(r"""
**TSI first**, because of that dimensionality spread. TSI takes triplets
`(i, j, k)` and asks how often two spaces agree on whether `j` or `k` is closer
to `i`. It never compares a distance in one space to a distance in the other --
only the *ordering* -- so its null is a fixed **0.5** whatever the width of
either space. A raw correlation of distances has no such fixed reference, and
given the spread printed above that difference is not cosmetic.

Rows and columns come out sequence-first, so a real family split shows up as two
warm blocks on the diagonal with a cool rectangle between them.
""")

code(r"""
tsi_matrix = tl.alignment_matrix(dense_embeddings, metric="tsi", keys=ordered_keys)
display(tsi_matrix.round(3))
""")

md(r"""
One metric is one opinion. The four run below probe different things, and where
they disagree is the diagnostic:

| Metric | What it compares | Null |
| --- | --- | --- |
| `tsi` | triplet distance orderings, anchored on `i` -- local structure | 0.5 |
| `qsi` | quadruplet orderings, `d(i,j)` vs `d(k,l)`, no shared anchor -- global structure | 0.5 |
| `cka` | linear CKA on inner products, not on distances at all | not fixed |
| `mutual_knn` | neighbour *sets* only, ignoring order within them | ~ k/(n-1) |

Two caveats worth carrying into the table. `cka` has no fixed null, so read it as
a ranking within this table and not as an absolute agreement; it is also a ratio
of Frobenius norms, so a few extreme rows can move it a long way while the
ordinal metrics barely notice. And `mutual_knn` called through `alignment_matrix`
uses its own default `k=10`; section 2 repeats it at `k=5` alongside the other
neighbourhood measures.
""")

code(r"""
METRICS = ["tsi", "qsi", "cka", "mutual_knn"]
pairs = [(ordered_keys[i], ordered_keys[j])
         for i in range(len(ordered_keys)) for j in range(i + 1, len(ordered_keys))]

def pair_type(a: str, b: str) -> str:
    if a in dna_set and b in dna_set:
        return "within sequence"
    if a in static_set and b in static_set:
        return "within knowledge"
    return "cross-family"

print(f"{len(pairs)} pairs: "
      + ", ".join(f"{n} {t}" for t, n in
                  pd.Series([pair_type(a, b) for a, b in pairs]).value_counts().items()))
""")

code(r"""
# One matrix per metric, then read off the upper triangle by name -- indexing
# explicitly keeps each pair exactly once, with no half-empty mirror rows.
matrices = {m: tl.alignment_matrix(dense_embeddings, metric=m, keys=ordered_keys)
            for m in METRICS}

summary = pd.DataFrame(
    [{"pair": f"{a} / {b}", "pair_type": pair_type(a, b),
      **{m: matrices[m].loc[a, b] for m in METRICS}}
     for a, b in pairs]
).set_index("pair")

display(summary.round(3).sort_values("tsi", ascending=False))
""")

md(r"""
That table is long enough to lose the argument in. Collapsing it by bucket is the
headline of this section.
""")

code(r"""
headline = summary.groupby("pair_type", observed=True)[METRICS].mean()
headline.insert(0, "n_pairs", summary["pair_type"].value_counts())
display(headline.round(3))

if "cross-family" in headline.index and len(headline) > 1:
    cross = headline.loc["cross-family", "tsi"]
    within_floor = headline.drop(index="cross-family")["tsi"].min()
    print(f"\nmean cross-family TSI : {cross:.3f}")
    print(f"weakest within-family : {within_floor:.3f}")
    if cross < within_floor:
        print("Prediction holds on TSI: the two families agree with themselves "
              "more than with each other.")
    else:
        print("Prediction does NOT hold on TSI: cross-family agreement is at "
              "least as high as within one of the families, so the sequence / "
              "knowledge split is not the dominant axis of variation here.")
else:
    print("\nOnly one bucket is populated -- too few surviving models to split.")
""")

md(r"""
Read the buckets against the 0.5 null rather than against each other only. A
within-family mean near 0.5 says the members of that family share almost no
distance ordering -- entirely possible for the knowledge tables, since GenePT is
distilled from text, Gene2Vec from co-expression and the CRISPR table from
viability screens. "Prior knowledge" is not one thing.
""")

code(r"""
# similarity_correlation works on the pairwise cosine similarities themselves
# rather than on sampled triplets: build both similarity matrices, take the upper
# triangle of each, and correlate. Same question, a different estimator.
corr = pd.concat(
    [tl.similarity_correlation(dense_embeddings[a], matrix_b=dense_embeddings[b],
                               label_a=a, target=b)
     for a, b in pairs],
    ignore_index=True,
)
corr["pair_type"] = [pair_type(a, b) for a, b in pairs]
display(corr.round(3).sort_values("spearman", ascending=False))
""")

md(r"""
Prefer the `spearman` column here. Cosine similarities within a single embedding
space are usually piled into a narrow band with a long tail, and Pearson on such
a distribution is driven by the tail -- a handful of near-duplicate gene pairs can
carry the whole coefficient. Spearman only sees the ranking, which is the same
robustness argument that made TSI the opening metric.

`n_pairs` is identical on every row: it is the number of gene pairs
(`n*(n-1)/2`), not a count of anything model-specific.
""")

# ======================================================= local neighbourhoods
md(r"""
## 2. Local neighbourhoods

Global agreement can hide local disagreement. These ask whether each gene keeps
the *same neighbours* -- which is what you actually act on when you use an
embedding to find "genes like this one".
""")

code(r"""
k = 5
chance = k / (dense_space.n_obs - 1)
print(f"k = {k}, chance level = k/(n-1) = {chance:.3f}\n")

local = []
for a, b in pairs:
    # compute_knn_overlap is the AnnData-facing wrapper around knn_jaccard: same
    # number, but it also writes a per-gene column into dense_space.obs, so you
    # can see *which* genes move rather than only the mean.
    _, overlap = tl.compute_knn_overlap(dense_space, dense_obsm[a], dense_obsm[b], k=k)
    _, jac = tl.knn_jaccard(dense_embeddings[a], dense_embeddings[b], k=k)
    local.append({"pair": f"{a} / {b}", "pair_type": pair_type(a, b),
                  "knn_overlap": overlap, "jaccard": jac,
                  "mutual_knn": tl.mutual_knn(dense_embeddings[a], dense_embeddings[b], k=k)})

local_df = pd.DataFrame(local).set_index("pair").sort_values("mutual_knn", ascending=False)
display(local_df.round(3))
""")

md(r"""
`knn_overlap` and `jaccard` are the same quantity by construction -- the first
call is the AnnData wrapper around the second, at the same `k` and the same
default cosine metric -- so their agreement is a sanity check, not evidence.
`mutual_knn` should sit above both: it divides the intersection by `k` rather
than by the union, which can only push the number up, and it defaults to
Euclidean distance where the Jaccard pair defaults to cosine. The grouped means
below are where to check that.
""")

code(r"""
neighbour_cols = ["knn_overlap", "jaccard", "mutual_knn"]
grouped_local = local_df.groupby("pair_type", observed=True)[neighbour_cols].mean()
grouped_local.insert(0, "n_pairs", local_df["pair_type"].value_counts())
display(grouped_local.round(3))
print(f"chance level for mutual_knn is ~ k/(n-1) = {chance:.3f}; the two Jaccard "
      "columns divide by the union rather than by k, so their null sits lower "
      "still and the three columns are not directly comparable in absolute terms.")
""")

md(r"""
`compare_embedding_matrices` runs a standard battery over every pair in one
call -- the fastest route to a full picture once you know what the columns mean.
It is `similarity_correlation` and `knn_jaccard` bundled, so the numbers should
match the two tables above.
""")

code(r"""
# Passed in family order so the row set and its ordering match the tables above.
battery = tl.compare_embedding_matrices({key: dense_embeddings[key] for key in ordered_keys}, k=k)
display(battery.round(3))
""")

# ================================================================== probes
md(r"""
### Two genes, every space

The tables above are aggregates. This is the same split made concrete, and it is
the cell to read if you only read one:

* **TUBB4B** is a *paralog*. Its true class-mates -- the other tubulins -- are
  duplicates of one ancestral gene, so they share sequence. A DNA model has
  everything it needs to find them.
* **ENO1** is a *pathway member*. Its true class-mates catalyse other steps of
  glycolysis and are unrelated in sequence -- enolase, hexose isomerase and
  aldolase share no fold. Only a knowledge table has been told they belong
  together.

The prediction is that the neighbour lists swap: sequence models should score
well on TUBB4B and near chance on ENO1, knowledge tables the reverse. The `hits`
column counts how many of the top 5 are genuine class-mates. Either probe can be
absent, if a gap in one of the lookup tables put it outside the dense mask; the
cell below then substitutes another member of the same class and says which.
""")

code(r"""
# A gene can be missing here if it was dropped by the dense mask, so fall back to
# another member of the same class rather than raising.
PROBE_CLASSES = {"TUBB4B": "tubulin", "ENO1": "glycolysis"}

probes: dict[str, str] = {}
for wanted, cls in PROBE_CLASSES.items():
    if wanted in dense_space.obs_names:
        probes[wanted] = cls
        continue
    alt = next((g for g in PANEL[cls] if g in dense_space.obs_names), None)
    if alt is None:
        print(f"no {cls} member survived the dense mask -- probe dropped")
    else:
        print(f"{wanted} was dropped by the dense mask; probing {alt} instead")
        probes[alt] = cls

print("probes:", probes)
""")

code(r"""
ids = list(dense_space.obs_names)

for probe, cls in probes.items():
    classmates = [g for g in PANEL[cls] if g in ids and g != probe]
    rows = []
    for key in ordered_keys:
        nn = tl.nearest_neighbors_table(dense_embeddings[key], ids=ids, query=probe, k=5)
        names = list(nn["neighbor_id"])
        rows.append({
            "model": key,
            "family": "sequence" if key in dna_set else
                      ("knowledge" if key in static_set else "unclassified"),
            "hits": sum(1 for n in names if n in classmates),
            **{f"nn{i + 1}": n for i, n in enumerate(names)},
        })

    table = pd.DataFrame(rows).set_index("model").sort_values("hits", ascending=False)
    print(f"\n=== {probe} ({cls}) ===")
    display(table)
    print(f"true class-mates present: {', '.join(classmates)}")
    print(f"expected hits by chance in a top-5 list: "
          f"{5 * len(classmates) / (len(ids) - 1):.2f} of 5")
""")

md(r"""
Two things to check in those two tables. First the `hits` column split by
`family` -- whether the sequence models win on the paralog probe and lose on the
pathway probe, and by how much against the chance figure printed underneath each
one. Second the *identity* of the misses. A neighbour list always comes back
full and ranked whether or not the space knows anything about the query, so a
DNA model asked for ENO1's neighbours returns five confident-looking genes
either way, and nothing in the output marks them as arbitrary. The `hits` column
is the only thing separating a real neighbourhood from a well-formatted one.
""")

# ======================================================= visual comparison
md(r"""
## 3. Visual comparison

The heatmaps are the tables above in a form you can scan. If the family split is
real, both should show two blocks on the diagonal and a cold rectangle
off it.
""")

code(r"""
pl.cross_model_similarity(dense_space, obsm_keys=obsm_keys)
pl.knn_overlap(dense_space, obsm_keys=obsm_keys, k=k)
""")

md(r"""
Scale is worth seeing before you ever concatenate, average, or feed these spaces
to a distance-based method. A mean-pooled DNA embedding and a text-derived lookup
table come out of unrelated procedures and have no reason to land at the same
magnitude; the plot below is where you find out whether they do. Wherever they do
not, a naive concatenation is silently a weighted one -- the larger-norm block
dominates every Euclidean distance downstream. Standardise per block first, or
use a metric that does not care.
""")

code(r"""
pl.embedding_norms(dense_space, obsm_keys=obsm_keys)
pl.embedding_distributions(dense_space, obsm_keys=obsm_keys, n_dims=6)
""")

code(r"""
# Same genes, same colouring, one panel per model -- the qualitative counterpart
# to the tables above. PCA rather than UMAP: on a panel this small a UMAP layout
# is mostly an artefact of its own hyperparameters.
pl.all_embeddings(dense_space, obsm_keys=obsm_keys, method="pca",
                  color="family", ncols=3)
""")

Path(sys.argv[1]).write_text(json.dumps(CELLS))
print(f"part 3: {len(CELLS)} cells")
