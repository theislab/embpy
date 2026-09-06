"""Part 4 of docs/notebooks/small_molecules.ipynb -- does the geometry track biology?"""
from __future__ import annotations
import json, sys
from pathlib import Path

CELLS: list[tuple[str, str]] = []
def md(s: str) -> None: CELLS.append(("markdown", s.strip("\n")))
def code(s: str) -> None: CELLS.append(("code", s.strip("\n")))

# ============================================================ the prediction
md(r"""
## Does the geometry track biology?

The [three comparison sections](#1-global-geometry) asked whether the models
agree with each other. Agreement is not correctness, and for molecules that gap
is wider than for genes or proteins: six of the nine spaces are computed from the
same molecular graph by six closely related enumeration schemes, so they are
*obliged* to agree, and their agreement is a statement about RDKit rather than
about pharmacology.

The `family` column is the way out. No fingerprint was shown it, no tokeniser
encodes it, and it is not derivable from a SMILES string -- it is a hand-built
mechanism label, so it is a fair external test. This is the section that scores
the prediction the panel was designed around.

The prediction has two halves, and they point in opposite directions:

* **The tight series should be easy, for a boring reason.** Every statin carries
  the same dihydroxy-heptanoic acid warhead and every benzodiazepine the same
  fused seven-membered ring, so their members share literal substructures. A
  Morgan fingerprint enumerates substructures. Recovering these families needs no
  chemistry beyond bit counting, and a model that *fails* here is telling you
  something about the model, not about the panel.
* **The scaffold-diverse families should be hard, for the same boring reason in
  reverse.** The NSAIDs share one mechanism -- COX inhibition -- across
  salicylate, propionic acid, coxib and oxicam scaffolds; aspirin and celecoxib
  have almost no substructure in common. The kinase inhibitors share a target
  family while spanning scaffolds, imatinib's phenylaminopyrimidine against
  sorafenib's diaryl urea. A structural method has nothing to find.

The second half carries a rider worth stating before the numbers arrive: a
learned representation *might* do better on the diverse families, because it was
fitted to a corpus rather than to an enumeration rule. That is a prediction, and
the cells below score it rather than assuming it.

The beta blockers are the deliberate middle case and are scored on their own. Most
of them hang an ethanolamine off an aryl ether, so a fingerprint has a partial
series to find, but carvedilol's carbazole and labetalol's salicylamide are not in
it. Neither group would be honest about them.

Two readouts, deliberately different in scope:

* `pl.within_vs_between_similarity` -- are same-family pairs more similar than
  cross-family pairs? Returns both means, the pair counts, and a Mann-Whitney
  **p-value**.
* `pl.knn_label_purity` -- of each compound's `k` nearest neighbours, how many
  share its family? Returns per-family values plus `__overall__` and
  `__baseline__`, so the number is interpretable rather than merely large.
""")

code(r"""
# These three groups restate part 1's design table as code. Writing them down here
# rather than deriving them from any measurement is the point: the verdict cell
# below scores a prediction that was fixed before a single vector existed.
TIGHT_SERIES = ["statin", "benzodiazepine"]
SCAFFOLD_DIVERSE = ["nsaid", "kinase_inhibitor"]
MIDDLE_CASE = [f for f in PANEL if f not in TIGHT_SERIES + SCAFFOLD_DIVERSE]

# Score only what actually produced a matrix. A transformer that refused to load
# and a compound the tokeniser dropped both leave the analysis by this route
# rather than by breaking it.
SCORED = [k for k in SWEEP if f"X_{k}" in dense_space.obsm]
OBSM_OF = {k: f"X_{k}" for k in SCORED}

# The registry suffixes every fingerprint key with `_fp`, so the split between
# enumerated and learned representations is derivable from the roster instead of
# hand-listed -- a key added to ROSTER later lands on the correct side by itself.
FINGERPRINT_KEYS = [k for k in SCORED if k.endswith("_fp")]
LEARNED_KEYS = [k for k in SCORED if not k.endswith("_fp")]

# Purity has a ceiling set by family size, not by the model. knn_label_purity asks
# for k+1 neighbours and discards the query itself, so a compound in a family of m
# members has m-1 family-mates left to find and the best attainable value is
# min(k, m-1)/k. The panel is balanced by construction, but the dense mask in
# part 2 can thin a family, so the ceiling is computed rather than assumed.
family_sizes = dense_space.obs["family"].astype(str).value_counts().sort_index()
K_NN = int(min(5, max(2, family_sizes.min() - 1)))
ceiling = (np.minimum(K_NN, family_sizes - 1).clip(lower=0) / K_NN).rename("max_purity")

print(f"spaces scored : {len(SCORED)}  ({len(FINGERPRINT_KEYS)} fingerprint, "
      f"{len(LEARNED_KEYS)} learned)")
print(f"panel         : {dense_space.n_obs} compounds, k = {K_NN}")
print(f"random baseline for {len(family_sizes)} equal families: "
      f"{1 / len(family_sizes):.3f}\n")
print(pd.concat([family_sizes.rename("compounds"), ceiling.round(2)],
                axis=1).to_string())
""")

md(r"""
Before reading a single purity number, one failure mode specific to fingerprints
needs measuring, because it is invisible in every table that follows.

A fingerprint is a lossy description, and two different molecules can produce the
**same vector**. MACCS is the obvious candidate at 167 bits -- it asks 166 yes/no
questions and two members of a congeneric series can answer all of them
identically -- but a folded 2048-bit space can collide too. Where two rows are
identical, their k-NN ordering is settled by scikit-learn's tie-breaking, which is
index order, so their purity is partly an artefact of how the panel was written
down. The same applies to a zero-norm row: cosine distance is undefined there and
`NearestNeighbors` will happily return something anyway.

Neither is a reason to drop a space. It is a reason to know the number before
quoting a purity to three decimals.
""")

code(r"""
degeneracy = []
for key in SCORED:
    X = np.asarray(dense_space.obsm[OBSM_OF[key]], dtype=float)
    norms = np.linalg.norm(X, axis=1)
    # Rounding before np.unique keeps a float representation difference from being
    # counted as chemistry; fingerprint bits are exact, but the learned spaces are
    # float32 and two rows that differ in the last bit are not a collision.
    _, inverse, counts = np.unique(np.round(X, 6), axis=0,
                                   return_inverse=True, return_counts=True)
    inverse = np.asarray(inverse).ravel()
    degeneracy.append({
        "model": key,
        "dim": int(X.shape[1]),
        "distinct_rows": int(len(counts)),
        "tied_rows": int((counts[inverse] > 1).sum()),
        "zero_norm_rows": int((norms == 0).sum()),
        # Only meaningful for the fingerprints. A dense float space has every
        # coordinate non-zero by construction, so the count there would restate
        # `dim` and invite a comparison that means nothing.
        "mean_bits_set": (float((X != 0).sum(axis=1).mean())
                          if key in FINGERPRINT_KEYS else float("nan")),
    })

# Named columns so an empty sweep still produces a table rather than a KeyError.
DEGENERACY = pd.DataFrame(degeneracy, columns=["model", "dim", "distinct_rows",
                                               "tied_rows", "zero_norm_rows",
                                               "mean_bits_set"]).set_index("model")
display(DEGENERACY.round(3))

tied = DEGENERACY[DEGENERACY["tied_rows"] > 0]
if len(tied):
    print("spaces that cannot tell two panel compounds apart: "
          f"{', '.join(tied.index)} -- their purity rows are partly tie-breaking")
else:
    print(f"every space resolves all {dense_space.n_obs} compounds to distinct "
          "vectors, so no purity number below is decided by tie-breaking")
""")

# ================================================================== ranking
code(r"""
# Both helpers draw as well as return. Sending the per-model figures to one
# throwaway axis pair keeps the ranking table from being buried under 2N plots;
# the spaces that win get their own figures further down.
scratch = plt.figure(figsize=(9, 3.5))
ax_wb, ax_purity = scratch.add_subplot(121), scratch.add_subplot(122)

scores: list[dict] = []
purities: dict[str, dict] = {}

for key in SCORED:
    obsm = OBSM_OF[key]
    X = np.asarray(dense_space.obsm[obsm], dtype=float)
    if not np.isfinite(X).all():
        n_bad = int((~np.isfinite(X)).any(axis=1).sum())
        print(f"skipped {key}: {n_bad} row(s) still carry NaN, which would poison "
              "every mean taken over them")
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
        "kind": "fingerprint" if key in FINGERPRINT_KEYS else "learned",
        "dim": int(X.shape[1]),
        "within": wb["mean_within"],
        "between": wb["mean_between"],
        "separation": wb["mean_within"] - wb["mean_between"],
        "p_value": wb["p_value"],
        "purity": purity["__overall__"],
        "baseline": purity["__baseline__"],
    })

plt.close(scratch)

# Named columns so an empty sweep still produces a table rather than a KeyError.
PURITY = pd.DataFrame(scores, columns=["model", "kind", "dim", "within", "between",
                                       "separation", "p_value", "purity",
                                       "baseline"]).set_index("model")
# Sorted on purity, not separation. Purity is bounded above by the ceiling printed
# earlier and is therefore comparable between a 167-bit space and a 2048-bit one;
# a cosine separation is not, because the attainable cosine range depends on how
# sparse and how wide the space is.
if len(PURITY):
    PURITY = PURITY.sort_values(["purity", "separation"], ascending=False)
display(PURITY.round(4))

BEST_MODEL = PURITY.index[0] if len(PURITY) else None
BEST_FINGERPRINT = next((m for m in PURITY.index if m in FINGERPRINT_KEYS), None)
BEST_LEARNED = next((m for m in PURITY.index if m in LEARNED_KEYS), None)
print(f"\nbest overall     : {BEST_MODEL}")
print(f"best fingerprint : {BEST_FINGERPRINT}")
print(f"best learned     : {BEST_LEARNED}")
""")

md(r"""
Read `separation` and `purity` together; they can disagree, and the disagreement
is informative.

**Separation is global.** It averages over every same-family pair and every
cross-family pair at once, so one family parked far from the rest can carry the
whole number while the other four sit on top of each other. **Purity is local.**
It only ever looks at a compound's `k` nearest neighbours, which is the question
you actually act on: hand the space a compound, ask for "molecules like this one",
and see whether its family-mates come back. If you intend to cluster the space,
separation is the number to care about; if you intend to query it, purity is.

The `within` column has one molecule-specific quirk. Binary fingerprints share
their most common bits with almost everything -- an aromatic ring, a carbonyl,
a nitrogen -- so *every* cosine in a fingerprint space is pushed upwards and the
absolute value of `within` is not comparable to the same column in a ChemBERTa
row. The difference between `within` and `between` inside one row is the part
that survives the comparison, which is why `separation` exists as its own column.

The p-value is the least interesting column. With this many pairs the
Mann-Whitney test detects almost any real difference, so a tiny p says the split
is not noise; it says nothing about whether the split is *large*. The two means
and their difference carry that.
""")

# ================================================================ per family
md(r"""
### Where each model works, family by family

The overall purity hides exactly the thing the panel was built to expose. Broken
out per family, with the attainable ceiling next to it, the table below is the
actual test: the tight series first, the scaffold-diverse families next, the
beta blockers last, and the columns in the ranking order from above.
""")

code(r"""
PURITY_BY_FAMILY = pd.DataFrame({
    key: {fam: value for fam, value in purities[key].items()
          if not fam.startswith("__")}
    for key in PURITY.index
})

# The row order is the argument, so it is imposed rather than sorted: prediction
# groups in the order part 1 stated them, and any family the dense mask emptied
# drops out instead of appearing as a NaN row.
order = [f for f in TIGHT_SERIES + SCAFFOLD_DIVERSE + MIDDLE_CASE
         if f in PURITY_BY_FAMILY.index]
PURITY_BY_FAMILY = PURITY_BY_FAMILY.reindex(order)

expectation = pd.Series(
    {f: ("tight series" if f in TIGHT_SERIES
         else "scaffold-diverse" if f in SCAFFOLD_DIVERSE
         else "middle case") for f in order},
    name="prediction",
)
display(pd.concat([expectation, ceiling.reindex(order).round(2),
                   PURITY_BY_FAMILY.round(3)], axis=1))
""")

code(r"""
TIGHT_PRESENT = [f for f in TIGHT_SERIES if f in PURITY_BY_FAMILY.index]
DIVERSE_PRESENT = [f for f in SCAFFOLD_DIVERSE if f in PURITY_BY_FAMILY.index]
MIDDLE_PRESENT = [f for f in MIDDLE_CASE if f in PURITY_BY_FAMILY.index]


# Mean purity over one block of the table. NaN when the block is empty, so a
# missing group of models reads as NaN rather than as a confident-looking zero.
def block_mean(models: list[str], families: list[str]) -> float:
    cols = [m for m in models if m in PURITY_BY_FAMILY.columns]
    if not cols or not families:
        return float("nan")
    block = PURITY_BY_FAMILY.loc[families, cols].to_numpy(dtype=float)
    return float(np.nanmean(block)) if np.isfinite(block).any() else float("nan")


# family_purity, not summary: section 1 already bound `summary` to the pairwise
# metric table and this notebook runs in one kernel.
family_purity = pd.DataFrame(
    [
        {"tight series": block_mean(FINGERPRINT_KEYS, TIGHT_PRESENT),
         "scaffold-diverse": block_mean(FINGERPRINT_KEYS, DIVERSE_PRESENT),
         "middle case": block_mean(FINGERPRINT_KEYS, MIDDLE_PRESENT)},
        {"tight series": block_mean(LEARNED_KEYS, TIGHT_PRESENT),
         "scaffold-diverse": block_mean(LEARNED_KEYS, DIVERSE_PRESENT),
         "middle case": block_mean(LEARNED_KEYS, MIDDLE_PRESENT)},
    ],
    index=["fingerprints", "learned (SMILES models)"],
)
family_purity["structure advantage"] = (family_purity["tight series"]
                                        - family_purity["scaffold-diverse"])
display(family_purity.round(3))

baseline = PURITY["baseline"].iloc[0] if len(PURITY) else float("nan")
print(f"random baseline from the family sizes: {baseline:.3f}")

tight_ceiling = (float(ceiling.reindex(TIGHT_PRESENT).mean())
                 if TIGHT_PRESENT else float("nan"))
div_ceiling = (float(ceiling.reindex(DIVERSE_PRESENT).mean())
               if DIVERSE_PRESENT else float("nan"))
if np.isfinite(tight_ceiling) and np.isfinite(div_ceiling) \
        and abs(tight_ceiling - div_ceiling) > 0.02:
    print(f"caution: the two groups do not share a ceiling ({tight_ceiling:.2f} vs "
          f"{div_ceiling:.2f}), so part of any advantage below is family size "
          "rather than chemistry.")
""")

code(r"""
fp_gap = float(family_purity.loc["fingerprints", "structure advantage"])
lr_gap = float(family_purity.loc["learned (SMILES models)", "structure advantage"])
fp_hard = float(family_purity.loc["fingerprints", "scaffold-diverse"])
lr_hard = float(family_purity.loc["learned (SMILES models)", "scaffold-diverse"])

# The verdict sentence is chosen by the measured numbers, including a sentence for
# the case where the prediction was wrong. Half one is the sign of the
# fingerprints' structure advantage; half two is whether the learned spaces beat
# the fingerprints on the families a structural method cannot reach.
if not np.isfinite(fp_gap):
    PREDICTION_VERDICT = (
        "NOT SCORED: no fingerprint space survived the sweep on this panel, so the "
        "prediction has nothing to be scored against. Re-run the embedding section "
        "and read the skip reasons printed there."
    )
elif fp_gap > 0 and np.isfinite(lr_hard) and lr_hard > fp_hard:
    PREDICTION_VERDICT = (
        f"CONFIRMED, both halves: the fingerprints score {fp_gap:.3f} higher on the "
        f"tight series than on the scaffold-diverse families, which is the predicted "
        f"sign, and the learned spaces recover the diverse families better than the "
        f"fingerprints do ({lr_hard:.3f} against {fp_hard:.3f}). A representation "
        "fitted to a corpus reaches something an enumeration rule does not. It is a "
        "narrow win, and the next section asks whether it survives being scored "
        "against a therapeutic label instead of a hand-built one."
    )
elif fp_gap > 0 and np.isfinite(lr_hard):
    PREDICTION_VERDICT = (
        f"PARTLY CONFIRMED: half one holds -- the fingerprints gain {fp_gap:.3f} "
        f"purity on the tight series over their own score on the scaffold-diverse "
        f"families. Half two does not: the learned spaces reach {lr_hard:.3f} on the "
        f"diverse families against the fingerprints' {fp_hard:.3f}, so they inherit "
        "the same blind spot rather than fixing it. That is a coherent result, not a "
        "contradiction: ChemBERTa and MoLFormer read SMILES strings, and a SMILES "
        "string is a serialisation of the same molecular graph the fingerprints "
        "enumerate. Neither was ever shown a mechanism."
    )
elif fp_gap > 0:
    PREDICTION_VERDICT = (
        f"HALF SCORED: the fingerprints carry the predicted structure advantage "
        f"({fp_gap:+.3f}), but no learned space survived the sweep, so the second "
        "half of the prediction is untested. Read the sweep's skip column before "
        "reading anything into the first half alone."
    )
elif fp_gap < 0:
    PREDICTION_VERDICT = (
        f"REFUTED: the fingerprints score {abs(fp_gap):.3f} *higher* on the "
        f"scaffold-diverse families than on the tight series, the opposite way round "
        "from the prediction. Two explanations this notebook can distinguish, and "
        "one it cannot. Check the ceiling column first, in case the dense mask "
        "thinned a tight family; check the tied-rows count next, in case a "
        "congeneric series collapsed onto identical vectors and lost its ordering. "
        "What cannot be ruled out here is that the diverse families are diverse in "
        "a way that still leaves them mutually distinguishable -- a family can be "
        "internally heterogeneous and still be the nearest thing to each of its "
        "members if the other four families are further away."
    )
else:
    PREDICTION_VERDICT = (
        f"INCONCLUSIVE: the fingerprints' structure advantage is exactly "
        f"{fp_gap:+.3f}, so on this panel the tight / diverse split does not "
        "separate them at all. With eight compounds per family an exact tie is more "
        "likely to mean the purity numbers have saturated against the ceiling than "
        "to mean the two groups are genuinely equal -- check the per-family table "
        "against the ceiling column before reading anything into it."
    )

PREDICTION_DELTAS = {"fingerprint_gap": fp_gap, "learned_gap": lr_gap,
                     "fingerprint_diverse": fp_hard, "learned_diverse": lr_hard,
                     "purity_baseline": float(baseline), "k": K_NN,
                     "n_fingerprint_spaces": len(FINGERPRINT_KEYS),
                     "n_learned_spaces": len(LEARNED_KEYS)}
print(PREDICTION_VERDICT)
""")

md(r"""
Whatever the printed verdict, two things about the *shape* of that table outlive
it.

**The tight series are a low bar dressed as a capability.** Two statins are near
each other because they contain the same atoms in the same arrangement, and a
method that enumerates substructures finds that without knowing what HMG-CoA
reductase is. This notebook never computes a pairwise Tanimoto against the panel,
so treat shared substructure as the panel's design assumption rather than as
something measured here; what *is* measured is the consequence, in the tight-series
rows above and in the nilotinib probe in
[section 2](#2-local-neighbourhoods).

**The scaffold-diverse families are the case that matters, and the one nothing
here solves.** Aspirin and celecoxib inhibit the same enzyme through a
143-dalton acetylated salicylate and a trifluoromethyl pyrazole sulfonamide
respectively. No enumeration of their subgraphs will place them together, and a
model trained on SMILES strings has been shown the same graphs in a different
notation. Recovering that pair needs evidence from outside the structure --
assay data, target annotations, a clinical record -- which is exactly what
[part 5](#annotate-the-molecules) goes and fetches, and what
[the scaffold-hopping section](#molecule-only-scaffold-hopping) turns into a
single yes-or-no test on celecoxib and rofecoxib.

> **A family label is not ground truth about similarity.** `family` is a
> mechanism label the panel imposed, and a low purity means the space disagrees
> with *that* label, not that the space is wrong. Two beta blockers with different
> receptor subtype selectivity are one family here and two behaviours in a clinic.
> The purity numbers are only as sharp as the labelling, and this labelling is
> deliberately coarse.

`PREDICTION_VERDICT` and `PREDICTION_DELTAS` keep the sentence and the numbers
behind it in the kernel, so the closing section can quote the measured result
instead of restating it from memory.
""")

# ================================================================== centroids
md(r"""
### Which families does each model confuse?

Purity says how often a compound's neighbours are wrong. It does not say *which*
family they came from. Centroid similarity answers that directly: each family is
reduced to its mean vector and the heatmap is the cosine between them. The
diagonal is 1 by construction, so only the **off-diagonal** carries information,
and heat there is a pair the space cannot separate.

Best fingerprint and best learned space, side by side.
`pl.category_centroid_similarity` fixes `vmin`/`vmax` at -1/1, so the two panels
share a colour scale without being told to.
""")

code(r"""
candidates = ((BEST_FINGERPRINT, "fingerprint"), (BEST_LEARNED, "learned"))
centroid_panels = [(model, kind) for model, kind in candidates if model is not None]
absent = [kind for model, kind in candidates if model is None]

if not centroid_panels:
    print("no space survived the sweep, so there is nothing to draw.")
else:
    if absent:
        print(f"no {' or '.join(absent)} space was scored -- only the "
              f"{centroid_panels[0][1]} side is drawn, so this is not a contrast.")
    fig, axes = plt.subplots(1, len(centroid_panels),
                            figsize=(6.4 * len(centroid_panels), 4.8))
    for ax, (model, kind) in zip(np.atleast_1d(axes), centroid_panels):
        pl.category_centroid_similarity(dense_space, label_key="family",
                                        obsm_key=OBSM_OF[model], ax=ax,
                                        title=f"{model} ({kind})")
    fig.tight_layout()
""")

code(r"""
# The heatmaps are for two spaces; the confusion itself is worth having for all of
# them, so every centroid matrix is computed and only the numbers are kept. The
# throwaway axis is the same trick as the ranking cell -- these helpers draw
# whether or not you want the figure.
scratch = plt.figure(figsize=(4, 3))
ax_scratch = scratch.add_subplot(111)

confusions = []
for key in PURITY.index:
    centroids = pl.category_centroid_similarity(
        dense_space, label_key="family", obsm_key=OBSM_OF[key],
        ax=ax_scratch, annot=False,
    )
    values = centroids.to_numpy(dtype=float)
    np.fill_diagonal(values, np.nan)
    flat = np.nanargmax(values)
    i, j = np.unravel_index(flat, values.shape)
    confusions.append({
        "model": key,
        "worst pair": f"{centroids.index[i]} / {centroids.columns[j]}",
        "that cosine": float(values[i, j]),
        "mean off-diagonal": float(np.nanmean(values)),
        "min off-diagonal": float(np.nanmin(values)),
    })

plt.close(scratch)

# Named columns so an empty sweep still produces a table rather than a KeyError.
CONFUSION = pd.DataFrame(confusions, columns=["model", "worst pair", "that cosine",
                                              "mean off-diagonal",
                                              "min off-diagonal"]).set_index("model")
display(CONFUSION.round(3))

pairs = CONFUSION["worst pair"].value_counts()
if len(pairs):
    print(f"most frequently confused pair across {len(CONFUSION)} spaces: "
          f"{pairs.index[0]} ({pairs.iloc[0]} of {len(CONFUSION)})")
""")

md(r"""
The `mean off-diagonal` column is the one to read first, and it is a property of
the *space* rather than of the panel. A binary fingerprint puts every drug-like
molecule in the same corner of the hypercube -- they all have rings, nitrogens and
carbonyls -- so its family centroids sit at a high cosine to each other no matter
how well it separates them. A high mean off-diagonal next to a high purity is not
a contradiction: the neighbourhoods can be clean while the centroids are crowded.
That is why the `min` column is next to it. The spread between mean and min is the
amount of family structure the space actually expresses.

`worst pair` is where the chemistry shows through. Two failure modes are worth
distinguishing when you read it:

* **A shared pharmacophore.** Beta blockers and benzodiazepines are both
  nitrogen-rich and lipophilic; nothing in either family's definition stops them
  being close in a substructure count.
* **A shared size.** Fingerprint cosine is sensitive to how many bits are set,
  and the `mean_bits_set` column above says how much that varies. Two families of
  similarly large molecules -- statins and kinase inhibitors -- can be near each
  other because they are both big, which is a fact about molecular weight
  masquerading as a fact about mechanism.

Neither is diagnosed by this table alone. Both are visible in it, which is the
most a five-family panel can offer.
""")

# =============================================================== unsupervised
md(r"""
## Unsupervised structure

Purity, separation and the centroid heatmaps all use the labels. Clustering asks
the same question without them: left to itself, does the winning space rediscover
the panel? `tl.cluster_annotation_enrichment` then scores each cluster against
`family`, so a cluster that is one family and nothing else appears as a high
fold-enrichment row.

Two implementation facts to know before reading the output.

**The graph is Euclidean, and you cannot change that here.** `tl.leiden` calls
`sc.pp.neighbors` with scanpy's default metric, and exposes no `metric` argument;
`tl.find_nearest_neighbors` does have one, but `tl.leiden` rebuilds the graph
unconditionally, so the two do not compose. For a binary fingerprint this is less
arbitrary than it sounds -- the squared Euclidean distance between two bit vectors
*is* their Hamming distance -- but it is not Tanimoto, which is the coefficient
cheminformatics actually uses, and the difference is exactly the normalisation by
molecule size that the paragraph above was about. The dendrogram at the end of the
section is drawn on cosine distance, so the two views disagree by construction.

**Leiden needs `leidenalg` and `igraph`.** If they are absent the cell says so and
the notebook carries on.
""")

code(r"""
# pl.leiden_overview looks for a column named leiden_{obsm_key} and computes it if
# it is missing. Writing it under that exact name here means the overview figure
# below reuses this partition instead of clustering a second time and drawing a
# picture of a different answer.
LEIDEN_KEY = f"leiden_{OBSM_OF[BEST_MODEL]}" if BEST_MODEL else "leiden_none"
clustered = False

if BEST_MODEL is None:
    print("no scored space, so clustering is skipped.")
else:
    n_neighbors = int(min(8, max(2, dense_space.n_obs - 1)))
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dense_space = tl.leiden(dense_space, obsm_key=OBSM_OF[BEST_MODEL],
                                    resolution=1.0, n_neighbors=n_neighbors,
                                    key_added=LEIDEN_KEY)
        clustered = True
    except ImportError as exc:
        print(f"leiden unavailable -- {exc}\n  pip install leidenalg igraph")
    except Exception as exc:
        print(f"clustering skipped -- {type(exc).__name__}: {str(exc)[:120]}")

if clustered:
    # Both views get the column. dense_space carries it under the name the plotting
    # helper expects, mol_space under the plain name later sections read, and a
    # compound the dense mask dropped is labelled rather than left as a NaN that
    # would silently become its own category in a crosstab.
    labels = pd.Series(dense_space.obs[LEIDEN_KEY].astype(str).values,
                       index=dense_space.obs_names)
    dense_space.obs["leiden"] = labels.values
    mol_space.obs["leiden"] = (labels.reindex(mol_space.obs_names)
                               .fillna("not clustered").values)
    unplaced = int((mol_space.obs["leiden"] == "not clustered").sum())

    print(f"leiden on {BEST_MODEL} (k = {n_neighbors} neighbours, resolution 1.0):")
    print(dense_space.obs["leiden"].value_counts().sort_index().to_string())
    print(f"\n{dense_space.obs['leiden'].nunique()} clusters for "
          f"{len(family_sizes)} designed families")
    if unplaced:
        print(f"{unplaced} compound(s) outside dense_space carry 'not clustered'")
""")

code(r"""
if clustered:
    # The crosstab is the honest version of the enrichment table: it shows the
    # whole partition, including the clusters that are mixtures, which a top_k
    # ranking by fold-enrichment hides.
    display(pd.crosstab(dense_space.obs["leiden"], dense_space.obs["family"]))
    display(tl.cluster_annotation_enrichment(dense_space, cluster_key="leiden",
                                             annotation_key="family",
                                             top_k=3).round(3))
""")

md(r"""
Read the cluster count before the enrichment table. Leiden at resolution 1.0 is
told nothing about how many families the panel has, so compare the two counts
printed next to each other: more clusters than families means it split one --
usually the family whose members are least alike -- and fewer means it merged two,
which is the same confusion the off-diagonal heat showed, now expressed as a
partition of the compounds rather than as a similarity between means.

One caution on the enrichment column. With eight compounds per family and five
families, a cluster of two compounds that happen to share a family scores a
fold-enrichment of 5.0, the maximum, on the strength of two molecules. Read
`in_cluster` alongside `enrichment` and treat the small clusters as unresolved
rather than as discoveries.
""")

code(r"""
if clustered:
    try:
        # color_by adds the metadata UMAP and the composition bar chart, which is
        # the whole reason to call the overview rather than plotting the clusters
        # alone: the cluster panel and the family panel are the same coordinates.
        pl.leiden_overview(dense_space, obsm_key=OBSM_OF[BEST_MODEL],
                           resolution=1.0, color_by="family", figsize=(13, 9))
    except Exception as exc:
        print(f"overview skipped -- {type(exc).__name__}: {str(exc)[:120]}")
""")

code(r"""
if BEST_MODEL is not None:
    # labels= is not optional here. pl.dendrogram falls back to adata.obs_names,
    # and for this notebook those are canonical SMILES, which makes an unreadable
    # axis. The compound column is the human-readable identity.
    pl.dendrogram(dense_space, obsm_key=OBSM_OF[BEST_MODEL], metric="cosine",
                  labels=dense_space.obs["compound"].astype(str).tolist(),
                  leaf_font_size=7,
                  title=f"{BEST_MODEL}: hierarchical structure of the panel")
""")

md(r"""
The dendrogram is the most useful picture in this section, because it shows the
*order* in which the space merges compounds rather than a partition at one
resolution. Three things to look for, in this order:

1. **Do the tight series come out as single early-merging blocks?** They should.
   If a statin joins the benzodiazepines before it joins the other statins, the
   space has a problem that no purity number at k = 5 would have shown you.
2. **Where does the NSAID block break?** The prediction is that it does not exist
   as a block at all -- aspirin near nothing, the coxibs together, the propionic
   acids together. A structural space should shatter this family along scaffold
   lines, and the leaf order says which lines.
3. **Which compound is the outlier?** Every space has one, and it is usually
   informative: the smallest molecule in the panel has the fewest bits set and
   therefore the lowest cosine to everything, so an outlier at the far edge may be
   reporting molecular weight rather than chemistry.

That is as far as structure alone goes. Every label used in this section --
`family` above all -- was written by hand for this notebook, and a hand-built
label can flatter a method that agrees with the same intuition that built it. The
[next part](#annotate-the-molecules) replaces it with the curated clinical record:
ATC class, development phase, mechanism of action and measured target potency, all
fetched from ChEMBL, none of them written by anyone who had seen these embeddings.
Then [the same purity machinery](#does-the-embedding-know-the-clinical-record)
runs again against those labels, and the verdict either survives the swap or it
does not.
""")

Path(sys.argv[1] if len(sys.argv) > 1 else "cells_part4.json").write_text(json.dumps(CELLS))
print(f"part 4: {len(CELLS)} cells")
