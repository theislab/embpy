"""Part 2: the full comparison battery."""
from __future__ import annotations
import json, sys
from pathlib import Path
CELLS: list[tuple[str, str]] = []
def md(s: str) -> None: CELLS.append(("markdown", s.strip("\n")))
def code(s: str) -> None: CELLS.append(("code", s.strip("\n")))

md(r"""
## 1. Global geometry

`alignment_matrix` applies one metric to every pair. **TSI** first: it compares
distance *orderings*, so its null is a fixed 0.5 whatever the dimensionality —
and these spaces range from 320 to 1280 dimensions, so a metric that is
comparable across them matters.
""")

code(r"""
tsi_matrix = tl.alignment_matrix(embeddings, metric="tsi")
display(tsi_matrix.round(3))
""")

md(r"""
One metric is one opinion. Running all four over the same pairs shows where they
disagree, and *that* is the diagnostic — a pair can share global geometry while
disagreeing completely about neighbourhoods.
""")

code(r"""
METRICS = ["tsi", "qsi", "cka", "mutual_knn"]
pairs = [(keys[i], keys[j]) for i in range(len(keys)) for j in range(i + 1, len(keys))]

# One matrix per metric, then read off the upper triangle by name -- indexing
# explicitly keeps each pair exactly once, with no half-empty mirror rows.
matrices = {metric: tl.alignment_matrix(embeddings, metric=metric) for metric in METRICS}
summary = pd.DataFrame(
    [{"pair": f"{a} / {b}", **{m: matrices[m].loc[a, b] for m in METRICS}}
     for a, b in pairs]
).set_index("pair")

display(summary.round(3).sort_values("tsi", ascending=False))
""")

code(r"""
# similarity_correlation asks the question directly: do two spaces rank the same
# pairwise distances the same way?
corr = pd.concat([
    tl.similarity_correlation(embeddings[a], matrix_b=embeddings[b], label_a=f"{a} / {b}")
    for a, b in pairs
])
display(corr.round(3))
""")

md(r"""
## 2. Local neighbourhoods

Global agreement can hide local disagreement. These ask whether each protein
keeps the *same neighbours* — which is what you actually act on when you use an
embedding to find "proteins like this one".
""")

code(r"""
k = 5
chance = k / (protein_space.n_obs - 1)

local = []
for a, b in pairs:
    _, overlap = tl.compute_knn_overlap(protein_space, spaces[a], spaces[b], k=k)
    _, jac = tl.knn_jaccard(embeddings[a], embeddings[b], k=k)
    local.append({"pair": f"{a} / {b}", "knn_overlap": overlap, "jaccard": jac,
                  "mutual_knn": tl.mutual_knn(embeddings[a], embeddings[b], k=k)})

print(f"k = {k}, chance level ~ k/(n-1) = {chance:.3f}\n")
display(pd.DataFrame(local).set_index("pair").round(3).sort_values("knn_overlap", ascending=False))
""")

md(r"""
`compare_embedding_matrices` runs a standard battery over every pair in one
call — the fastest way to a full picture once you know what the columns mean.
""")

code(r"""
display(tl.compare_embedding_matrices(embeddings, k=k).round(3))
""")

md(r"""
The concrete version of the same question: for one protein, who are its
neighbours in each space? `nearest_neighbors_table` is the readout you would
actually use in analysis.
""")

code(r"""
probe = "CDK1"
neighbours = pd.concat([
    tl.nearest_neighbors_table(embeddings[key], ids=list(protein_space.obs_names),
                               query=probe, k=5)
      .assign(model=key)
    for key in keys
])
display(neighbours.set_index(["model", neighbours.index]).round(3))

print(f"\n{probe} is a kinase; its true class-mates are:")
print(" ", ", ".join(PANEL["kinase"]))
""")

md(r"""
## 3. Visual comparison
""")

code(r"""
pl.cross_model_similarity(protein_space, obsm_keys=obsm_keys)
pl.knn_overlap(protein_space, obsm_keys=obsm_keys, k=k)
""")

md(r"""
Scale is worth seeing before you ever concatenate, average, or feed these to a
distance-based method: mean-pooled PLM embeddings sit at very different
magnitudes.
""")

code(r"""
pl.embedding_norms(protein_space, obsm_keys=obsm_keys)
pl.embedding_distributions(protein_space, obsm_keys=obsm_keys, n_dims=6)
""")

code(r"""
# Same proteins, same colouring, one panel per model -- the qualitative
# counterpart to the tables above.
pl.all_embeddings(protein_space, obsm_keys=obsm_keys, method="pca",
                  color="family", ncols=3)
""")

Path(sys.argv[1]).write_text(json.dumps(CELLS))
print(f"part 2: {len(CELLS)} cells")
