"""Part 3 of docs/notebooks/small_molecules.ipynb -- the full comparison battery."""
from __future__ import annotations
import json, sys
from pathlib import Path

CELLS: list[tuple[str, str]] = []
def md(s: str) -> None: CELLS.append(("markdown", s.strip("\n")))
def code(s: str) -> None: CELLS.append(("code", s.strip("\n")))

# ========================================================== global geometry
md(r"""
## 1. Global geometry

Everything below runs on `dense_space` -- the gap-free subset from
[part 2](#one-molecule-space-two-views) -- because every metric here is a
distance computation, and a single NaN row does not cost you one cell of a
distance matrix, it costs that molecule's whole row and column of it.

The spaces fall into two families, and that split is the whole point of the
section:

* **fingerprints** -- six hand-written substructure enumerations. They see the
  molecular graph and nothing else. No training, no corpus, no objective.
* **learned representations** -- two ChemBERTa checkpoints and MoLFormer, all
  three trained on SMILES strings. They see a *tokenised text* of the molecule
  and whatever their pre-training objective rewarded.

So every pairwise number below belongs to one of three buckets: *within
fingerprint*, *within learned*, or *cross-family*. If the two families really do
encode different things, cross-family pairs should score **below both**
within-family blocks. That is a prediction, and the grouped table at the end of
this section is where it gets a number rather than an assertion.

One honest caveat before any of it. Four of the six fingerprints are the same
recipe with a different substructure enumerator: walk the graph, hash what you
find into 2048 buckets, set the bit. A high within-fingerprint block therefore
measures shared *hashing convention* at least as much as it measures shared
chemistry, and it is not evidence that fingerprints are right.
""")

code(r"""
# Derived from dense_space.obsm and SWEEP rather than from a hand-written list, so
# a model that failed in part 2 simply never appears here.
usable, dropped_keys = [], {}
for key in SWEEP:
    obsm_key = f"X_{key}"
    if obsm_key not in dense_space.obsm:
        dropped_keys[key] = "no matrix on dense_space"
        continue
    matrix = np.asarray(dense_space.obsm[obsm_key], dtype=float)
    # alignment_matrix refuses non-finite input outright, so filter here where the
    # reason can be named instead of letting a metric cell raise on row 17.
    if not np.isfinite(matrix).all():
        dropped_keys[key] = "non-finite cells survived the dense mask"
        continue
    usable.append(key)

# Every RDKitWrapper entry in the registry is named with an "_fp" suffix and no
# learned key carries one, so the suffix is the family split. Checked against the
# registry rather than trusted, because the split decides every bucket below.
from embpy.embedder_registry.molecule import MOLECULE_MODELS
from embpy.models.molecule_models import RDKitWrapper

rdkit_keys = {k for k, (wrapper, _) in MOLECULE_MODELS.items()
              if wrapper is RDKitWrapper}
suffix_holds = all(k.endswith("_fp") for k in rdkit_keys) and not any(
    k.endswith("_fp") for k in set(MOLECULE_MODELS) - rdkit_keys
)
print(f"'_fp' suffix identifies the RDKit fingerprints exactly: {suffix_holds}")

FINGERPRINT_KEYS = [k for k in usable if k.endswith("_fp")]
LEARNED_KEYS = [k for k in usable if not k.endswith("_fp")]
MODEL_FAMILY = {**{k: "fingerprint" for k in FINGERPRINT_KEYS},
                **{k: "learned" for k in LEARNED_KEYS}}

# Order fingerprints first everywhere, so block structure -- if there is any --
# lands on the diagonal instead of being interleaved away.
ORDERED_KEYS = FINGERPRINT_KEYS + LEARNED_KEYS
ORDERED_OBSM = [f"X_{k}" for k in ORDERED_KEYS]
dense_matrices = {k: np.asarray(dense_space.obsm[f"X_{k}"], dtype=float)
                  for k in ORDERED_KEYS}

print(f"{dense_space.n_obs} molecules x {len(ORDERED_KEYS)} spaces, all gap-free")
print(f"fingerprints : {FINGERPRINT_KEYS}")
print(f"learned      : {LEARNED_KEYS}")
for key, why in dropped_keys.items():
    print(f"excluded     : {key} -- {why}")
if len(ORDERED_KEYS) < 2:
    print("\nWARNING: fewer than two usable spaces. Every pairwise cell below "
          "reports an empty table and continues; part 2's sweep table is where "
          "the reason lives.")
""")

md(r"""
### Nominal width is not effective width

The widths in play span an order of magnitude: 167 for MACCS, 2048 for the
hashed fingerprints, 384 for the two ChemBERTa checkpoints, 768 for MoLFormer.
That spread is why TSI leads the section -- it compares only the *ordering* of
distances, never a distance in one space against a distance in the other, so its
null is a fixed 0.5 whatever the width of either side. CKA has no such fixed
reference.

But the nominal widths overstate the difference, and the cell below measures by
how much. A 2048-bit Morgan fingerprint is 2048 columns of which forty drug-like
molecules touch a few hundred; the rest are hash buckets no substructure in this
panel lands in. `active_dims` is the honest width. Two things to read off it:

* `active_dims` for `maccs_fp` can be at most **166**. The key set defines 166
  public keys and RDKit returns a 167-long vector with bit 0 unused, so a 167
  that came back as 167 active columns would mean the panel had set a bit that
  is documented not to exist.
* `distinct_values` separates the families more sharply than width does. A
  binary fingerprint has two, a count fingerprint a handful, and a learned
  vector has as many distinct values as it has cells.
""")

code(r"""
occupancy = []
for key in ORDERED_KEYS:
    matrix = dense_matrices[key]
    nonzero = np.abs(matrix) > 0
    occupancy.append({
        "model": key,
        "family": MODEL_FAMILY[key],
        "nominal_dim": matrix.shape[1],
        "active_dims": int(nonzero.any(axis=0).sum()),
        "nonzero_per_mol": float(nonzero.sum(axis=1).mean()),
        "distinct_values": int(np.unique(matrix).size),
        "mean_l2": float(np.linalg.norm(matrix, axis=1).mean()),
    })

OCCUPANCY = pd.DataFrame(occupancy).set_index("model")
display(OCCUPANCY.round(3))
""")

md(r"""
**TSI first**, for the reason just measured. TSI takes triplets `(i, j, k)` and
asks how often two spaces agree on whether `j` or `k` is closer to `i`. Rows and
columns come out fingerprints-first, so a real family split shows up as two warm
blocks on the diagonal with a cool rectangle between them.

One property matters more here than it does for continuous data. Binary
fingerprints produce *tied* distances in bulk -- two pairs of molecules sharing
the same number of bits are exactly equidistant -- and a naive triplet count
would have to decide whether a tie counts as agreement. embpy sidesteps the
question: TSI is recovered from `scipy.stats.kendalltau` (tau-b) together with
the tie counts, so ties are handled exactly rather than approximately
(`src/embpy/tl/alignment.py`). On a 40-molecule panel of sparse bit vectors that
is not a technicality.
""")

code(r"""
if len(ORDERED_KEYS) >= 2:
    tsi_matrix = tl.alignment_matrix(dense_matrices, metric="tsi", keys=ORDERED_KEYS)
    display(tsi_matrix.round(3))
else:
    print("fewer than two spaces -- no TSI matrix to compute")
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

Three caveats, all of which bound how far this table can be pushed:

* `cka` has no fixed null, so read it as a ranking inside this table and not as
  an absolute agreement. It is also a ratio of Frobenius norms, so a few
  large-norm rows can move it while the ordinal metrics barely notice -- and
  `mean_l2` above says which spaces have rows of unequal magnitude.
* `cka` on a binary fingerprint has an unusually literal reading. The Gram
  matrix of a bit vector *is* the count of substructures two molecules share, so
  here CKA asks whether two spaces agree about shared-substructure counts. That
  is closer to a chemist's intuition than CKA usually gets.
* `mutual_knn` called through `alignment_matrix` uses its own default `k=10`.
  [Section 2](#2-local-neighbourhoods) repeats it at `k=5` alongside the other
  neighbourhood measures, so the two numbers are not meant to match.
""")

code(r"""
METRICS = ["tsi", "qsi", "cka", "mutual_knn"]
pairs = [(ORDERED_KEYS[i], ORDERED_KEYS[j])
         for i in range(len(ORDERED_KEYS)) for j in range(i + 1, len(ORDERED_KEYS))]


# Capitalised because later parts read it: PAIR_TYPE is part of the notebook's
# fixed vocabulary, not a local helper. It buckets a pair of model keys, so any
# later pairwise table can be grouped the same way this one is.
def PAIR_TYPE(a: str, b: str) -> str:
    fam_a, fam_b = MODEL_FAMILY.get(a), MODEL_FAMILY.get(b)
    if fam_a != fam_b:
        return "cross-family"
    return f"within {fam_a}" if fam_a else "unclassified"


if pairs:
    counts = pd.Series([PAIR_TYPE(a, b) for a, b in pairs]).value_counts()
    print(f"{len(pairs)} pairs: " + ", ".join(f"{n} {t}" for t, n in counts.items()))
else:
    print("no pairs to compare")

# One matrix per metric, then read off the upper triangle by name -- indexing
# explicitly keeps each pair exactly once, with no half-empty mirror rows.
matrices = ({m: tl.alignment_matrix(dense_matrices, metric=m, keys=ORDERED_KEYS)
             for m in METRICS} if len(ORDERED_KEYS) >= 2 else {})

# Named columns so a degenerate sweep still produces a table rather than a KeyError.
summary = pd.DataFrame(
    [{"pair": f"{a} / {b}", "pair_type": PAIR_TYPE(a, b),
      **{m: matrices[m].loc[a, b] for m in METRICS}}
     for a, b in pairs],
    columns=["pair", "pair_type", *METRICS],
).set_index("pair")

display(summary.round(3).sort_values("tsi", ascending=False))
""")

md(r"""
That table is long enough to lose the argument in. Collapsing it by bucket is the
headline of this section.

The cell also scores one internal control, and it is the more informative half of
the output. `morgan_fp` and `morgan_count_fp` come from the *same* RDKit
generator at the same radius and the same 2048 buckets -- one calls
`GetFingerprintAsNumPy`, the other `GetCountFingerprintAsNumPy`
(`src/embpy/models/molecule_models.py:546`). They enumerate identical
substructures and differ only in whether a bit records presence or multiplicity.
If any pair in the table is going to top it, that one is the obvious candidate.
Naming the expectation first is the point: it is not obviously true, and the
[Tanimoto cell](#1-global-geometry) below is where the discrepancy gets
explained rather than excused.
""")

code(r"""
if not summary.empty:
    headline = summary.groupby("pair_type", observed=True)[METRICS].mean()
    headline.insert(0, "n_pairs", summary["pair_type"].value_counts())
    display(headline.round(3))

    if "cross-family" in headline.index and len(headline) > 1:
        cross = headline.loc["cross-family", "tsi"]
        within_floor = headline.drop(index="cross-family")["tsi"].min()
        print(f"mean cross-family TSI : {cross:.3f}")
        print(f"weakest within-family : {within_floor:.3f}")
        if cross < within_floor:
            print("Prediction holds on TSI: each family agrees with itself more "
                  "than it agrees with the other.")
        else:
            print("Prediction does NOT hold on TSI: cross-family agreement is at "
                  "least as high as within one of the families, so the "
                  "fingerprint / learned split is not the dominant axis here.")
    else:
        print("Only one bucket is populated -- too few surviving spaces to split.")

    CONTROL_PAIR = ("morgan_fp", "morgan_count_fp")
    label = " / ".join(CONTROL_PAIR)
    if label in summary.index:
        ranked = summary["tsi"].sort_values(ascending=False)
        place = list(ranked.index).index(label) + 1
        print(f"\nsame-bits control ({label}): TSI {ranked[label]:.3f}, "
              f"rank {place} of {len(ranked)}")
        if place > 1:
            print("  The binary/count pair is not the closest pair in the table. "
                  "Euclidean distance on a count vector is dominated by the "
                  "high-multiplicity bits, so the two spaces order molecules "
                  "differently despite enumerating the same substructures.")
    else:
        print(f"\n{label} not both present -- the same-bits control is unavailable")
else:
    print("no pairwise numbers to summarise")
""")

md(r"""
Read the buckets against the 0.5 null and not only against each other. A
within-family mean near 0.5 says the members of that family share almost no
distance ordering, and that is a live possibility for the learned block: the two
ChemBERTa checkpoints differ in training objective (multi-task regression versus
masked language modelling) and MoLFormer is a different architecture on a
different corpus. "Learned representation" is not one thing, exactly as "prior
knowledge" is not one thing in [genes](genes.ipynb).

### The distance a chemist would have used

Every number above rests on a choice nothing has justified yet. `alignment_matrix`
defaults to Euclidean distance, and on a binary fingerprint Euclidean distance is
the square root of the Hamming distance -- the count of bits where the two
molecules disagree, with no reference to how many bits either one sets. A
molecule with 60 bits set is far from everything under that metric, large
molecules included.

Cheminformatics does not rank that way. It ranks by **Tanimoto**, which divides
the shared bits by the union, and scipy's `jaccard` metric is exactly `1 -
Tanimoto` for binary input. So the question is whether the section's verdict
survives changing the distance, and that is measurable.

Two things to know before reading the cell:

* scipy's `jaccard` decides membership by **non-zero-ness**, not by value
  (`unequal = bitwise_xor(u != 0, v != 0)`). On a count fingerprint it therefore
  discards the counts. Since `morgan_count_fp` is non-zero in exactly the bits
  `morgan_fp` sets, under Tanimoto the two collapse onto the *same* distance
  matrix, and their TSI must read exactly **1.000**. If it does not, the metric
  is not doing what this paragraph claims.
* Tanimoto is meaningless on a dense learned vector, where every column is
  non-zero and the "union" is the whole width. So this check is
  fingerprint-only, and the cross-family numbers stay on Euclidean.
""")

code(r"""
if len(FINGERPRINT_KEYS) >= 2:
    tsi_tanimoto = tl.alignment_matrix({k: dense_matrices[k] for k in FINGERPRINT_KEYS},
                                       metric="tsi", distance="jaccard",
                                       keys=FINGERPRINT_KEYS)
    display(tsi_tanimoto.round(3))

    fp_pairs = [(a, b) for a, b in pairs
                if MODEL_FAMILY[a] == MODEL_FAMILY[b] == "fingerprint"]
    shift = pd.DataFrame(
        [{"pair": f"{a} / {b}",
          "tsi_euclidean": matrices["tsi"].loc[a, b],
          "tsi_tanimoto": tsi_tanimoto.loc[a, b]} for a, b in fp_pairs]
    ).set_index("pair")
    shift["difference"] = shift["tsi_tanimoto"] - shift["tsi_euclidean"]
    display(shift.round(3).sort_values("difference"))
    print(f"largest shift from changing the distance: "
          f"{shift['difference'].abs().max():.3f}")
else:
    print("fewer than two fingerprint spaces -- the Tanimoto check needs two")

# similarity_correlation works on the pairwise cosine similarities themselves
# rather than on sampled triplets: build both similarity matrices, take the upper
# triangle of each, and correlate. Same question, a different estimator -- and a
# third distance, since it defaults to cosine, which on a binary vector is the
# Ochiai coefficient rather than Tanimoto.
if pairs:
    corr = pd.concat(
        [tl.similarity_correlation(dense_matrices[a], matrix_b=dense_matrices[b],
                                   label_a=a, target=b)
         for a, b in pairs],
        ignore_index=True,
    )
    corr["pair_type"] = [PAIR_TYPE(a, b) for a, b in pairs]
    display(corr.round(3).sort_values("spearman", ascending=False))
else:
    print("no pairs -- nothing to correlate")
""")

md(r"""
Prefer the `spearman` column. Cosine similarities inside one fingerprint space
are piled into a narrow band near zero with a long tail, because most drug pairs
share almost no substructures, and Pearson on that distribution is driven by the
tail -- a handful of congeneric pairs such as the statins can carry the whole
coefficient. Spearman only sees the ranking, which is the same robustness
argument that made TSI the opening metric.

`n_pairs` is identical on every row: it is the number of molecule pairs
(`n*(n-1)/2`), not a count of anything model-specific.
""")

# ======================================================= local neighbourhoods
md(r"""
## 2. Local neighbourhoods

Global agreement can hide local disagreement. These ask whether each molecule
keeps the *same neighbours*, which is what you actually act on when you use an
embedding to find "compounds like this one" -- a similarity search, a nearest-
neighbour read-across, the shortlist you hand a chemist.
""")

code(r"""
# Guarded rather than hard-coded: k must be smaller than the panel, and part 2's
# intersection is allowed to have thinned it.
KNN_K = min(5, max(dense_space.n_obs - 1, 1))
chance = KNN_K / (dense_space.n_obs - 1)
print(f"k = {KNN_K}, chance level = k/(n-1) = {chance:.3f}\n")

local = []
for a, b in pairs:
    # compute_knn_overlap is the AnnData-facing wrapper around knn_jaccard: same
    # number, but it also writes a per-molecule column into dense_space.obs
    # ("knn_jaccard_X_a_X_b"), so you can see *which* molecules move rather than
    # only the mean. One column per pair, so later cells that show .obs should
    # select their columns explicitly.
    _, overlap = tl.compute_knn_overlap(dense_space, f"X_{a}", f"X_{b}", k=KNN_K)
    _, jac = tl.knn_jaccard(dense_matrices[a], dense_matrices[b], k=KNN_K)
    local.append({"pair": f"{a} / {b}", "pair_type": PAIR_TYPE(a, b),
                  "knn_overlap": overlap, "jaccard": jac,
                  "mutual_knn": tl.mutual_knn(dense_matrices[a], dense_matrices[b],
                                              k=KNN_K)})

local_df = pd.DataFrame(local, columns=["pair", "pair_type", "knn_overlap",
                                        "jaccard", "mutual_knn"]).set_index("pair")
display(local_df.round(3).sort_values("mutual_knn", ascending=False))
""")

md(r"""
`knn_overlap` and `jaccard` are the same quantity by construction -- the first
call is the AnnData wrapper around the second, at the same `k` and the same
default cosine metric -- so their agreement is a sanity check, not evidence.
`mutual_knn` should sit above both: it divides the intersection by `k` rather
than by the union, which can only push the number up, and it defaults to
Euclidean where the Jaccard pair defaults to cosine. On binary fingerprints that
default difference is the Hamming-versus-Ochiai contrast from section 1 again,
now decided over neighbour sets instead of orderings.

`compare_embedding_matrices` runs the standard battery over every pair in one
call -- the fastest route to a full picture once you know what the columns mean.
It bundles `similarity_correlation` and `knn_jaccard`, so its numbers should
match the two tables above rather than adding anything.
""")

code(r"""
if not local_df.empty:
    neighbour_cols = ["knn_overlap", "jaccard", "mutual_knn"]
    grouped_local = local_df.groupby("pair_type", observed=True)[neighbour_cols].mean()
    grouped_local.insert(0, "n_pairs", local_df["pair_type"].value_counts())
    display(grouped_local.round(3))
    print(f"chance for mutual_knn is ~ k/(n-1) = {chance:.3f}; the two Jaccard "
          "columns divide by the union rather than by k, so their null sits "
          "lower still and the three columns are not comparable in absolute terms.")

if len(ORDERED_KEYS) >= 2:
    # Passed in family order so the row set and its ordering match the tables above.
    battery = tl.compare_embedding_matrices(
        {key: dense_matrices[key] for key in ORDERED_KEYS}, k=KNN_K)
    display(battery.round(3))
""")

# ================================================================== probes
md(r"""
### Two compounds, every space

The tables above are aggregates. This is the structure-versus-mechanism question
made concrete, and it is the cell to read if you only read one. Two probes, each
with a named partner that the space either finds or does not:

| Probe | Partner | What a hit would mean |
| --- | --- | --- |
| **nilotinib** | imatinib | it recognises a scaffold. Nilotinib was designed from imatinib's phenylaminopyrimidine core, so the two share substructures outright |
| **celecoxib** | rofecoxib | it recognises a *mechanism*. Both are selective COX-2 inhibitors; celecoxib is a pyrazole sulfonamide and rofecoxib a furanone |

The prediction is that the fingerprints ace nilotinib and miss celecoxib, since
the first pair shares a graph and the second shares only a target. If a learned
model finds the coxib pair where the fingerprints do not, that is the first
evidence in this notebook of mechanism-level structure.

The celecoxib probe is not a clean test, and pretending otherwise would be the
easy way to a nice result. Both coxibs carry a para-substituted phenyl bearing a
sulfur(VI) group -- a sulfonamide on celecoxib, a methylsulfonyl on rofecoxib --
so a radius-2 fingerprint has *some* shared signal to work with. The question is
whether it is enough to beat the propionic acids, which share far more of the
graph and nothing of the selectivity.

`partner_rank` is the partner's position over the whole panel rather than only
the top 5, so a near miss is distinguishable from total ignorance. `hits` counts
how many of the top 5 are members of the probe's own hand-built family. Either
probe can be absent if part 2's dense mask dropped it; the cell then substitutes
another member of the same family and says so.
""")

code(r"""
# (probe, family, partner)
PROBE_SPEC = [("nilotinib", "kinase_inhibitor", "imatinib"),
              ("celecoxib", "nsaid", "rofecoxib")]

# Compound names, not obs_names: part 2 embeds against the canonical SMILES column,
# so the index is not guaranteed to be the human-readable name.
PANEL_IDS = list(dense_space.obs["compound"].astype(str))

resolved_probes = []
for probe, fam, partner in PROBE_SPEC:
    if probe not in PANEL_IDS:
        alt = next((c for c in PANEL[fam] if c in PANEL_IDS), None)
        if alt is None:
            print(f"no {fam} member survived the dense mask -- probe dropped")
            continue
        print(f"{probe} was dropped by the dense mask; probing {alt} instead")
        probe = alt
    if partner not in PANEL_IDS:
        print(f"{partner} is outside the dense subset -- partner_rank will be blank "
              f"for the {probe} probe")
        partner = None
    resolved_probes.append((probe, fam, partner))

for probe, fam, partner in resolved_probes:
    classmates = [c for c in PANEL[fam] if c in PANEL_IDS and c != probe]
    rows = []
    for key in ORDERED_KEYS:
        # k = n-1 asks for the full ranking, so the partner's position is readable
        # even when it is nowhere near the top of the list.
        nn = tl.nearest_neighbors_table(dense_matrices[key], ids=PANEL_IDS,
                                        query=probe, k=len(PANEL_IDS) - 1)
        order = list(nn["neighbor_id"])
        hit = nn[nn["neighbor_id"] == partner] if partner else nn.iloc[0:0]
        rows.append({
            "model": key,
            "family": MODEL_FAMILY[key],
            "hits": sum(1 for n in order[:5] if n in classmates),
            "partner_rank": int(hit["rank"].iloc[0]) if len(hit) else None,
            "partner_cos": float(hit["similarity"].iloc[0]) if len(hit) else None,
            **{f"nn{i + 1}": n for i, n in enumerate(order[:5])},
        })

    print(f"\n=== {probe} ({fam}), partner {partner or 'unavailable'} ===")
    display(pd.DataFrame(rows).set_index("model")
            .sort_values(["hits", "partner_rank"], ascending=[False, True]).round(3))
    print(f"family members present: {', '.join(classmates)}")
    print(f"expected hits by chance in a top-5 list: "
          f"{5 * len(classmates) / (len(PANEL_IDS) - 1):.2f} of 5")
""")

md(r"""
Three things to check in those tables, in order of how easy they are to fool
yourself about.

**The `family` split on `hits`.** Whether the fingerprints win on nilotinib and
lose on celecoxib, and by how much against the chance figure printed under each
table. A `hits` value at or below chance is a space that has told you nothing.

**`partner_rank` against 39.** A rank of 2 is a find. A rank of 20 is the
midpoint of the panel, which is what "no information about this pair" looks like,
and it is not visibly different from a rank of 8 unless you go looking.

**The identity of the misses.** A neighbour list always comes back full and
ranked whether or not the space knows anything about the query, so a fingerprint
asked for celecoxib's neighbours returns five confident-looking drugs either way,
and nothing in the output marks them as arbitrary. The `hits` and `partner_rank`
columns are the only things separating a real neighbourhood from a well-formatted
one.

> **A hit here is structural until proven otherwise.** If a space does put
> celecoxib next to rofecoxib, the shared sulfonyl-phenyl fragment is a sufficient
> explanation and no mechanism needs to be invoked.
> [Scaffold hopping](#molecule-only-scaffold-hopping) in part 6 pushes on this
> harder, and the [purity section](#does-the-geometry-track-biology) scores it
> across the whole panel rather than on one pair.
""")

# ======================================================= visual comparison
md(r"""
## 3. Visual comparison

The heatmaps are the tables above in a form you can scan. If the family split is
real, both should show two blocks on the diagonal and a cold rectangle off it.
""")

code(r"""
if len(ORDERED_KEYS) >= 2:
    pl.cross_model_similarity(dense_space, obsm_keys=ORDERED_OBSM)
    pl.knn_overlap(dense_space, obsm_keys=ORDERED_OBSM, k=KNN_K)
""")

md(r"""
Scale is worth seeing before you ever concatenate, average, or feed these spaces
to a distance-based method. A binary fingerprint row has L2 norm equal to the
square root of the number of bits it sets -- roughly 5 to 8 for a drug-sized
molecule, whatever the nominal width -- while a pooled transformer vector lands
wherever its training left it. The `mean_l2` column measured that in section 1;
this is the distribution behind the mean. Wherever the magnitudes differ, a naive
concatenation is silently a weighted one, and the larger-norm block dominates
every Euclidean distance downstream. Standardise per block first, or use a metric
that does not care.

The dimension violins need a warning to be worth anything. The first six columns
of a hashed fingerprint are not six features: they are six arbitrary hash
buckets, and on forty molecules most of them are empty for every row. A flat
violin there is the expected result and says nothing about the model. For the
learned spaces the same panel is meaningful, because dimension 0 of ChemBERTa is
a real coordinate that every molecule has a value for.
""")

code(r"""
if ORDERED_KEYS:
    pl.embedding_norms(dense_space, obsm_keys=ORDERED_OBSM)
    pl.embedding_distributions(dense_space, obsm_keys=ORDERED_OBSM, n_dims=6)
""")

code(r"""
# Same molecules, same colouring, one panel per space -- the qualitative
# counterpart to the tables above. PCA rather than UMAP: on a panel this small a
# UMAP layout is mostly an artefact of its own hyperparameters.
if ORDERED_KEYS:
    pl.all_embeddings(dense_space, obsm_keys=ORDERED_OBSM, method="pca",
                      color="family", ncols=3)
""")

md(r"""
One labelled panel per family, because a five-colour scatter shows you that the
statins clump and hides *which* compound sits in the wrong clump. `annotate_col`
puts the compound name on every point, which is legible at forty rows and would
not be at four hundred.

`plot_embedding_space` caches its 2-D coordinates in `dense_space.obsm` under
`X_pca_<key>`, so the layout is computed once and reused. embpy's key discovery
skips the `X_pca_` prefix by design, so those extra entries do not leak into any
later cell that lets `obsm_keys` default.
""")

code(r"""
SHOWCASE = [k for k in (FINGERPRINT_KEYS[:1] + LEARNED_KEYS[:1])]
for key in SHOWCASE:
    pl.plot_embedding_space(dense_space, obsm_key=f"X_{key}", color="family",
                            method="pca", annotate=True, annotate_col="compound",
                            figsize=(9, 7),
                            title=f"{key} ({MODEL_FAMILY[key]}) -- PCA of "
                                  f"{dense_space.n_obs} molecules")
if not SHOWCASE:
    print("no space survived the sweep -- nothing to plot")
""")

Path(sys.argv[1] if len(sys.argv) > 1 else "cells_part3.json").write_text(json.dumps(CELLS))
print(f"part 3: {len(CELLS)} cells")
