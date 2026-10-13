"""Part 4: scoring the central prediction -- does the geometry track gene biology?"""
from __future__ import annotations
import json, sys
from pathlib import Path

CELLS: list[tuple[str, str]] = []
def md(s: str) -> None: CELLS.append(("markdown", s.strip("\n")))
def code(s: str) -> None: CELLS.append(("code", s.strip("\n")))

# ============================================================ the prediction
md(r"""
## Does the geometry track biology?

Everything so far compared the models *to each other*. Agreement is not
correctness: two models can agree because they are both right, or because they
were both distilled from the same abstracts. The `family` labels are the way out
of that -- no model was ever shown them, no table was queried with them, and no
sequence carries them. They are a fair external test, and this is the section
that scores the prediction the panel was built around.

The prediction has two halves, and they point in opposite directions:

* **DNA models should win on the paralog families.** The classes in
  `SEQUENCE_COHERENT` are duplicates of one ancestral gene, so their members are
  literally similar strings. A model that reads DNA and nothing else has
  everything it needs.
* **DNA models should lose on `glycolysis`, `interferon` and `cell_cycle`.**
  Those members share a pathway and nothing else -- different folds, different
  chromosomes, no sequence relationship to find. The knowledge tables, distilled
  from text, co-expression and screens, are organised by exactly that kind of
  functional co-membership, so they should show the mirror-image pattern.

Both halves are predictions, not results. The cells below compute the two
numbers and choose the verdict sentence from the measured values -- including
the sentence for the case where the prediction is wrong.

Two readouts, deliberately different in scope:

* `pl.within_vs_between_similarity` -- are same-class pairs more similar than
  cross-class pairs? Returns both means, the pair counts, and a Mann-Whitney
  **p-value**.
* `pl.knn_label_purity` -- of each gene's `k` nearest neighbours, how many share
  its class? Returns per-class values plus `__overall__` and `__baseline__`, so
  the number is interpretable rather than merely large.
""")

code(r"""
import matplotlib.pyplot as plt

# Score only what actually produced vectors: a backend that refused to load and a
# table that is advertised but absent from the data repo both simply drop out.
DNA_SCORED = [key for key in DNA_KEYS if spaces.get(key) in dense_space.obsm]
STATIC_SCORED = [key for key in STATIC_KEYS if spaces.get(key) in dense_space.obsm]
SCORED = DNA_SCORED + STATIC_SCORED

# Purity has a ceiling set by class size, not by the model. knn_label_purity asks
# for k+1 neighbours and discards the query itself, so a gene in a class of m
# members has only m-1 class-mates left to find and the best attainable value is
# min(k, m-1)/k. Any gene dropped by the dense mask shrinks its own class and no
# other, so the ceiling is worth computing before reading a single purity number.
class_sizes = dense_space.obs["family"].astype(str).value_counts().sort_index()
K_NN = int(min(5, max(2, class_sizes.min() - 1)))
ceiling = (np.minimum(K_NN, class_sizes - 1).clip(lower=0) / K_NN).rename("max_purity")

print(f"models scored : {len(SCORED)}  ({len(DNA_SCORED)} DNA, {len(STATIC_SCORED)} static)")
print(f"panel         : {dense_space.n_obs} genes, k = {K_NN}\n")
print(pd.concat([class_sizes.rename("genes"), ceiling.round(2)], axis=1).to_string())
""")

# ================================================================== ranking
code(r"""
# Both helpers draw as well as return. Sending the per-model figures to one
# throwaway axis pair keeps the ranking table from being buried under 2N plots;
# the models that win get their own plots further down.
scratch = plt.figure(figsize=(9, 3.5))
ax_wb, ax_purity = scratch.add_subplot(121), scratch.add_subplot(122)

scores: list[dict] = []
purities: dict[str, dict] = {}

for key in SCORED:
    obsm = spaces[key]
    X = np.asarray(dense_space.obsm[obsm], dtype=float)
    if not np.isfinite(X).all():
        n_bad = int((~np.isfinite(X)).any(axis=1).sum())
        print(f"skipped {key}: {n_bad} gene(s) still carry NaN, which would poison every mean")
        continue
    try:
        wb = pl.within_vs_between_similarity(dense_space, label_key="family",
                                             obsm_key=obsm, ax=ax_wb)
        purity = pl.knn_label_purity(dense_space, label_key="family",
                                     obsm_key=obsm, k=K_NN, ax=ax_purity)
    except Exception as exc:
        print(f"skipped {key} -- {type(exc).__name__}: {str(exc)[:90]}")
        continue
    purities[key] = purity
    scores.append({
        "model": key,
        "kind": "DNA" if key in DNA_SCORED else "table",
        "within": wb["mean_within"],
        "between": wb["mean_between"],
        "separation": wb["mean_within"] - wb["mean_between"],
        "p_value": wb["p_value"],
        "knn_purity": purity["__overall__"],
        "purity_baseline": purity["__baseline__"],
    })

plt.close(scratch)

# Named columns so an empty sweep still produces a table rather than a KeyError.
ranking = pd.DataFrame(scores, columns=["model", "kind", "within", "between", "separation",
                                        "p_value", "knn_purity", "purity_baseline"])
ranking = ranking.set_index("model")
if len(ranking):
    ranking = ranking.sort_values("separation", ascending=False)
display(ranking.round(4))

best = ranking.index[0] if len(ranking) else None
best_dna = next((m for m in ranking.index if m in DNA_SCORED), None)
best_static = next((m for m in ranking.index if m in STATIC_SCORED), None)
print(f"\nbest class separation : {best}")
print(f"best DNA model        : {best_dna}")
print(f"best knowledge table  : {best_static}")
""")

md(r"""
Read `separation` and `knn_purity` together; they can disagree, and the
disagreement is informative.

**Separation is global.** It averages over every same-class pair and every
cross-class pair at once, so one class parked far away from the rest can carry
the whole number while the remaining classes sit on top of each other.
**Purity is local.** It only ever looks at a gene's `k` nearest neighbours, so it
answers the question you actually act on: if I hand this space a gene and ask for
"genes like this one", do I get its class-mates back? If you are going to cluster
the space, separation is the number to care about; if you are going to query it,
purity is.

The p-value is the least interesting column. With this many pairs the
Mann-Whitney test detects almost any real difference, so a tiny p tells you the
split is not noise -- it says nothing about whether the split is *large*. The two
means and their difference carry that.
""")

# =============================================================== the key cell
md(r"""
### Where each model works, class by class

The overall purity hides the thing the panel was designed to expose. Broken out
per class, and with the attainable ceiling printed next to it, the table below is
the actual test: the `SEQUENCE_COHERENT` rows come first, the pathway rows after,
and the columns are in the ranking order from above.
""")

code(r"""
per_class = pd.DataFrame({
    key: {cls: value for cls, value in purities[key].items() if not cls.startswith("__")}
    for key in ranking.index
})

order = ([c for c in class_sizes.index if c in SEQUENCE_COHERENT]
         + [c for c in class_sizes.index if c not in SEQUENCE_COHERENT])
per_class = per_class.reindex(order)

coherence = pd.Series(
    {c: "sequence" if c in SEQUENCE_COHERENT else "function" for c in order},
    name="coherence",
)
display(pd.concat([coherence, ceiling.reindex(order).round(2), per_class.round(3)], axis=1))
""")

code(r"""
SEQ_CLASSES = [c for c in per_class.index if c in SEQUENCE_COHERENT]
FUN_CLASSES = [c for c in per_class.index if c not in SEQUENCE_COHERENT]


# Mean purity over one block of the table; NaN when the block is empty, so a
# missing model family shows up as NaN instead of a confident-looking zero.
def group_mean(models: list[str], classes: list[str]) -> float:
    cols = [m for m in models if m in per_class.columns]
    if not cols or not classes:
        return float("nan")
    block = per_class.loc[classes, cols].to_numpy(dtype=float)
    return float(np.nanmean(block)) if np.isfinite(block).any() else float("nan")


# family_summary, not summary: section 1 already bound `summary` to the pairwise
# metric table and this notebook runs in one kernel.
family_summary = pd.DataFrame(
    [
        {"paralog families": group_mean(DNA_SCORED, SEQ_CLASSES),
         "pathway classes": group_mean(DNA_SCORED, FUN_CLASSES)},
        {"paralog families": group_mean(STATIC_SCORED, SEQ_CLASSES),
         "pathway classes": group_mean(STATIC_SCORED, FUN_CLASSES)},
    ],
    index=["DNA sequence models", "knowledge tables"],
)
family_summary["advantage"] = (family_summary["paralog families"]
                               - family_summary["pathway classes"])
display(family_summary.round(3))

baseline = purities[ranking.index[0]]["__baseline__"] if len(ranking) else float("nan")
print(f"random baseline (from the class sizes): {baseline:.3f}"
      f"   -- a balanced panel would give 1/{len(PANEL)} = {1 / len(PANEL):.3f}")

seq_ceiling = float(ceiling.reindex(SEQ_CLASSES).mean()) if SEQ_CLASSES else float("nan")
fun_ceiling = float(ceiling.reindex(FUN_CLASSES).mean()) if FUN_CLASSES else float("nan")
if abs(seq_ceiling - fun_ceiling) > 0.02:
    print(f"caution: the two groups do not have the same ceiling "
          f"({seq_ceiling:.2f} vs {fun_ceiling:.2f}), so part of any advantage below "
          "is class size rather than biology.")
""")

code(r"""
dna_adv = float(family_summary.loc["DNA sequence models", "advantage"])
tbl_adv = float(family_summary.loc["knowledge tables", "advantage"])

# The verdict is picked by the measured numbers, not written in advance. Positive
# advantage = that family scores higher on the paralog families than on the
# pathway classes; the prediction is that the two signs are opposite.
if not np.isfinite(dna_adv) or not np.isfinite(tbl_adv):
    PREDICTION_VERDICT = (
        "NOT SCORED: one of the two model families produced no usable space on this "
        "panel, so there is nothing to compare. Re-run the embedding section and "
        "check the skip reasons printed there."
    )
elif dna_adv > 0 > tbl_adv:
    PREDICTION_VERDICT = (
        f"CONFIRMED: on the paralog families the DNA models gain {dna_adv:.3f} purity "
        f"over their own score on the pathway classes, while the knowledge tables lose "
        f"{abs(tbl_adv):.3f} -- opposite signs, which is the prediction. Each family is "
        "better at the classes its own evidence can see; the per-class table above says "
        "how far above the baseline each one still sits on its weak side."
    )
elif dna_adv > tbl_adv:
    leaning = ("both lean towards the paralog families" if dna_adv > 0 and tbl_adv > 0
               else "both lean towards the pathway classes" if dna_adv < 0 and tbl_adv < 0
               else "one of the two is flat")
    PREDICTION_VERDICT = (
        f"PARTLY CONFIRMED: the ordering holds -- the DNA models carry the larger "
        f"sequence-class advantage ({dna_adv:+.3f} against the tables' {tbl_adv:+.3f}) "
        f"-- but the two signs do not oppose each other: {leaning}, so only the size "
        "of the gap separates the two kinds of model. Read the two signs in the "
        "table above before reading the gap."
    )
elif tbl_adv > dna_adv:
    PREDICTION_VERDICT = (
        f"REFUTED: the knowledge tables carry the larger sequence-class advantage "
        f"({tbl_adv:+.3f} against the DNA models' {dna_adv:+.3f}), the opposite way "
        "round from the prediction. One explanation this notebook cannot rule out is "
        "that paralogs are not only sequence neighbours: they are co-expressed, "
        "co-cited and screened together, so a table has its own route to them. "
        "Testing that would mean scoring a table against a paralog set the "
        "literature has not written about, which is not something this panel does."
    )
else:
    PREDICTION_VERDICT = (
        f"INCONCLUSIVE: the two advantages are equal ({dna_adv:+.3f}), so on this "
        "panel the sequence / function split does not separate the two kinds of "
        "model at all. With this few models and this few genes per class an exact "
        "tie is more likely a sign that the purity numbers have saturated -- check "
        "them against the ceiling column above before reading anything into it."
    )

PREDICTION_DELTAS = {"dna_advantage": dna_adv, "table_advantage": tbl_adv,
                     "purity_baseline": baseline, "k": K_NN,
                     "n_dna_models": len([m for m in DNA_SCORED if m in per_class.columns]),
                     "n_table_models": len([m for m in STATIC_SCORED if m in per_class.columns])}
print(PREDICTION_VERDICT)
""")

md(r"""
Whatever the printed verdict, two things about the *shape* of the table are worth
keeping.

The paralog families are the easy case for a sequence model, and for a boring
reason: their members descend from one duplicated ancestral gene, so their coding
sequences are related by descent and "similar embedding" needs no biology at all
-- string similarity is enough. That is a genuine capability and also a low bar.
This notebook never measures percent identity, so treat the mechanism as the
panel's design assumption; what it does measure is the consequence, in the
`hits` column of the TUBB4B probe in the previous section and in the paralog
rows above.

The pathway classes are the hard case for the same boring reason in reverse.
`GAPDH` and `ENO1` catalyse different steps of one route and belong to unrelated
protein families; they are neighbours in a metabolic chart and nowhere else. A
model that recovered them from exon sequence alone would have to have learned
something shared across unrelated loci -- regulatory grammar, codon usage,
GC context -- and distinguishing that from a confound is well beyond what a
40-gene panel can do.

The one place to be careful is the assumption that the tables are *blind* to
paralogy. They are not: tubulin genes co-express, get cited in the same papers,
and drop out together in CRISPR screens, so a co-expression or text-derived table
has its own route to the same answer. That is why the verdict above is stated as
a contrast between two advantages rather than as a DNA-models-only claim.
`PREDICTION_VERDICT` and `PREDICTION_DELTAS` keep the sentence and its numbers in
the kernel, so anything later in the session can quote the measured result
instead of restating it from memory.
""")

# ================================================================== centroids
md(r"""
### Which classes does each family of model confuse?

Centroid similarity answers that directly: each class is reduced to its mean
vector and the heatmap is the cosine between them. The diagonal is 1 by
construction, so only the **off-diagonal** matters -- heat there is a pair the
model cannot tell apart. Best DNA model and best table, side by side, on the same
colour scale.
""")

code(r"""
candidates = ((best_dna, "DNA"), (best_static, "table"))
centroid_panels = [(model, kind) for model, kind in candidates if model is not None]
absent = [kind for model, kind in candidates if model is None]

if not centroid_panels:
    print("neither family produced a usable space, so there is nothing to draw.")
else:
    if absent:
        print(f"no {' or '.join(absent)} model was scored -- only the "
              f"{centroid_panels[0][1]} side is drawn, so this is not a contrast.")
    # category_centroid_similarity fixes vmin/vmax at -1/1, so the two panels are
    # on one colour scale without being told to be.
    fig, axes = plt.subplots(1, len(centroid_panels),
                             figsize=(6.2 * len(centroid_panels), 4.6))
    for ax, (model, kind) in zip(np.atleast_1d(axes), centroid_panels):
        pl.category_centroid_similarity(dense_space, label_key="family",
                                        obsm_key=spaces[model], ax=ax,
                                        title=f"{model} ({kind})")
    fig.tight_layout()
""")

# =============================================================== unsupervised
md(r"""
## Unsupervised structure

Purity and separation both use the labels. Clustering asks the same question
without them: left to itself, does the winning space rediscover the panel?
`cluster_annotation_enrichment` then scores each cluster against `family`, so a
cluster that is one class and nothing else shows up as a high fold-enrichment
row.

Leiden goes through scanpy and needs `leidenalg` and `igraph`; if they are
missing the cell says so and the notebook carries on.
""")

code(r"""
if best is None:
    print("no scored model, so clustering is skipped.")
else:
    try:
        dense_space = tl.cluster_embeddings(
            dense_space, obsm_key=spaces[best], method="leiden",
            resolution=1.0, n_neighbors=8, key_added="cluster",
        )
        print(f"leiden on {best}:")
        print(dense_space.obs["cluster"].value_counts().sort_index().to_string())
        print(f"\n{dense_space.obs['cluster'].nunique()} clusters for "
              f"{len(class_sizes)} true classes\n")
        display(tl.cluster_annotation_enrichment(dense_space, cluster_key="cluster",
                                                 annotation_key="family", top_k=3))
    except ImportError as exc:
        print(f"leiden unavailable -- {exc}\n  pip install leidenalg igraph")
    except Exception as exc:
        print(f"clustering skipped -- {type(exc).__name__}: {str(exc)[:120]}")
""")

md(r"""
The cluster count is worth reading before the enrichment table. Leiden at
resolution 1.0 is told nothing about how many classes the panel has, so compare
the count it printed against the class count next to it: more clusters than
classes means it split a class -- usually the one whose members are least alike
-- and fewer means it merged two, which is the same confusion the off-diagonal
heat above showed, now expressed as a partition.
""")

code(r"""
if best is not None:
    pl.dendrogram(dense_space, obsm_key=spaces[best], metric="cosine",
                  title=f"{best}: hierarchical structure of the panel")
""")

# ================================================================ annotation
md(r"""
## Annotate the genes

`tl.annotate_gene_perturbations` fetches per-gene metadata from MyGene.info
(pathways), STRING and DoRothEA (interactions, regulators) and Open Targets plus
the GWAS Catalog (diseases). Compact counts land in `.obs` as `gene_*`, full
records in `.uns["gene_annotations"]`.

Three practical notes:

* It is one network round-trip per gene per source, so it is slow and it can
  fail. The call is wrapped, and a failure leaves the rest of the notebook
  intact.
* The five `gene_*` columns are written **whatever** you pass as `sources` --
  only the ones whose source was queried carry content. `expression` is not
  requested here, so `gene_top_tissue` should come back empty; the cell after
  measures which columns are actually populated rather than assuming.
* These are real measured properties that no embedding was trained on, which
  makes them usable twice: as colouring here, and as *prediction targets* in the
  next section.
""")

code(r"""
annotated = False
annot_t0 = time.perf_counter()
try:
    dense_space = tl.annotate_gene_perturbations(
        dense_space, column="symbol",
        sources=["pathways", "interactions", "diseases"], copy=True,
    )
    annotated = True
except Exception as exc:
    print(f"annotation skipped -- {type(exc).__name__}: {str(exc)[:140]}")
annot_seconds = time.perf_counter() - annot_t0

records = dense_space.uns.get("gene_annotations", {}) if annotated else {}
gene_cols = [c for c in dense_space.obs.columns if c.startswith("gene_")] if annotated else []
if records:
    print(f"{len(records)} full records in .uns['gene_annotations'], fetched in "
          f"{annot_seconds:.1f}s ({annot_seconds / len(records):.2f}s per gene)")
else:
    print(f"no records in .uns['gene_annotations'] after {annot_seconds:.1f}s")
print(f"gene_* columns written: {gene_cols}\n")

for col in gene_cols:
    values = dense_space.obs[col]
    if pd.api.types.is_numeric_dtype(values):
        filled = int((values > 0).sum())
        print(f"{col:<32} {filled:>3}/{len(values)} non-zero   max {values.max()}")
    else:
        # "" is what the annotator writes for a source it never queried; "nan" and
        # "None" only appear if something upstream wrote a null, and neither counts
        # as content.
        text = values.astype(str).str.strip()
        filled = int((~text.isin(["", "nan", "None", "<NA>"])).sum())
        print(f"{col:<32} {filled:>3}/{len(values)} non-empty")
""")

code(r"""
if gene_cols:
    display(dense_space.obs[["family", *gene_cols]].head(10))
    numeric_cols = [c for c in gene_cols if pd.api.types.is_numeric_dtype(dense_space.obs[c])]
    if numeric_cols:
        by_family = dense_space.obs.groupby("family", observed=True)[numeric_cols].mean()
        display(by_family.round(1))
        # Say which class is richest and which is sparsest rather than leaving it to
        # the reader's eye -- the point of the table is the spread, not the values.
        for col in numeric_cols:
            column = by_family[col].dropna()
            if column.empty or column.max() <= 0:
                print(f"{col:<32} all zero -- the lookup returned nothing for any class")
                continue
            print(f"{col:<32} richest {column.idxmax()} ({column.max():.1f})   "
                  f"sparsest {column.idxmin()} ({column.min():.1f})")
""")

md(r"""
Those per-class means are a sanity check on the *annotations*, not on any
embedding. Well-studied classes should come out richest, and the printed
richest/sparsest lines are where to check that -- interferon and cell-cycle genes
sit in dense interaction neighbourhoods and are heavily cited, so a run that puts
a paralog family top instead is telling you something about coverage rather than
about the panel. A class that comes back all zero means the lookup failed for
those symbols, not that the biology is sparse.
""")

code(r"""
ANNOT_COLOR = next(
    (c for c in ("gene_n_pathways", "gene_n_ppi_partners", "gene_n_disease_assoc")
     if c in dense_space.obs.columns
     and pd.to_numeric(dense_space.obs[c], errors="coerce").fillna(0).max() > 0),
    None,
)

if best is None:
    print("no scored model, so there is nothing to plot.")
elif ANNOT_COLOR is None:
    print("no annotation column landed with a non-zero value, so colouring by one "
          "would draw a single flat colour -- only the label plot below is drawn.")
else:
    pl.plot_embedding_space(dense_space, obsm_key=spaces[best], method="pca",
                            color=ANNOT_COLOR, annotate=False,
                            title=f"{best} coloured by {ANNOT_COLOR}")
""")

code(r"""
if best is not None:
    pl.plot_embedding_space(dense_space, obsm_key=spaces[best], method="pca",
                            color="family", annotate=True, annotate_col="symbol",
                            title=f"{best} coloured by family")
""")

md(r"""
Comparing the two plots is the point, and they go in this order deliberately: the
first is coloured by the annotation count, the second by `family`, and both are
the same PCA of the same space.

`family` is the label the space was scored against. The annotation count is a
measured property nobody optimised for. If the *first* plot shows a gradient
running along the same axis that separates the classes in the *second*, the
embedding has picked up study bias -- well-annotated genes sitting together
because they are well annotated -- and not only biology.

That is an eyeball judgement, which is exactly what it should not stay. The next
section takes one of these `gene_*` columns as a regression target, fits four
plain probes on every embedding in turn and reports R2, so the gradient either
survives as a number or it does not.
""")

Path(sys.argv[1] if len(sys.argv) > 1 else "cells_part4.json").write_text(json.dumps(CELLS))
print(f"part 4: {len(CELLS)} cells")
