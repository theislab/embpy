"""Part 6 of docs/notebooks/small_molecules.ipynb -- downstream, molecule-only, close."""
from __future__ import annotations
import json, sys
from pathlib import Path

CELLS: list[tuple[str, str]] = []
def md(s: str) -> None: CELLS.append(("markdown", s.strip("\n")))
def code(s: str) -> None: CELLS.append(("code", s.strip("\n")))

# ==================================================== A. downstream benchmark
md(r"""
## Downstream: which model predicts a measured property?

Everything so far compared the spaces against each other, or against labels this
notebook wrote by hand. A benchmark asks a narrower question: given one of these
matrices, can a plain regressor recover a number that came from somewhere else?
`tl.benchmark_embeddings` trains four probes -- linear, ridge, k-NN and random
forest -- on an `.obsm` matrix and scores them against a numeric `.obs` column.

Choosing the target is the whole design, and for molecules it is easy to get
wrong in a way that produces a confident number and no information. Every
fingerprint in the roster is a **deterministic function of the SMILES string**.
So is `mol_qed`. So is `mol_molecular_weight`. Ask a Morgan fingerprint to
predict QED and you are asking whether one function of the SMILES can be
recovered from another function of the same SMILES -- a question about RDKit,
not about chemistry. A high score there is the expected result and means
nothing about whether the embedding is useful.

| Target | Where it came from | Circular? | Prior expectation |
| --- | --- | --- | --- |
| `drug_best_pchembl` | the strongest measured potency across the compound's curated ChEMBL protein targets | no -- it is assay data, and no model in the roster has seen it | **the real test.** Nothing about a bit vector implies a binding constant, so a positive score is evidence and a negative one is honest |
| `mol_qed` | RDKit's QED, computed from the same SMILES the fingerprints were computed from | **yes** | **the fingerprints should win, and it proves nothing.** It is here as a calibration: it shows what "this space encodes the structure faithfully" looks like as a number, so the potency column can be read against it |

Two predictions, then, and the cells below score both. The interesting quantity
is not either score on its own but the **gap** between them. A space that
reproduces its own input and cannot touch the measurement is doing exactly what
a structural descriptor is supposed to do; a space that closes some of that gap
has learned something the structure alone does not say.

R2 is on the standard scale where 0 means "no better than predicting the mean",
and **negative values mean worse than that**. A model scoring below zero on a
target it has no path to is the correct answer, not a broken run.
""")

code(r"""
BENCH_TARGET_COLS = ["drug_best_pchembl", "mol_qed"]

# The annotation section wrote its mol_* and drug_* columns onto whichever object it
# ran on. Prefer the dense subset, because every cross-model number in this notebook
# is computed there and a benchmark on a different row set would not be comparable;
# fall back to the full space when only that one carries the columns.
def _n_targets_present(space) -> int:
    return sum(1 for col in BENCH_TARGET_COLS if col in space.obs.columns)


if _n_targets_present(dense_space) >= _n_targets_present(mol_space):
    bench_space, bench_space_name = dense_space, "dense_space"
else:
    bench_space, bench_space_name = mol_space, "mol_space"

# Read the roster that actually produced matrices rather than iterating .obsm: the
# plots in the visual-comparison section parked 2-d UMAP and PCA coordinates in
# .obsm too, and a projection of a model is not a model. Benchmarking those would
# manufacture rows that look like results.
bench_spaces = {key: f"X_{key}" for key in SWEEP if f"X_{key}" in bench_space.obsm}


def space_kind(key: str) -> str:
    # Every fingerprint registry key ends in _fp; the learned models do not. That is
    # the split the whole notebook is stated in, so it is worth naming once.
    return "fingerprint" if key.endswith("_fp") else "learned"


inventory = pd.DataFrame([
    {"model": key, "obsm_key": obsm, "kind": space_kind(key),
     "dim": int(np.asarray(bench_space.obsm[obsm]).shape[1]),
     "finite_rows": int(np.isfinite(
         np.asarray(bench_space.obsm[obsm], dtype=float)).all(axis=1).sum())}
    for key, obsm in bench_spaces.items()
]).set_index("model")

ignored = sorted(set(bench_space.obsm) - set(bench_spaces.values()))
print(f"benchmarking on {bench_space_name}: {bench_space.n_obs} compounds, "
      f"{len(bench_spaces)} model spaces")
print(f"target columns present: "
      f"{[c for c in BENCH_TARGET_COLS if c in bench_space.obs.columns] or 'none'}")
if ignored:
    print(f"ignored (derived coordinates, not embeddings): {ignored}")
display(inventory)

# A target is only usable if it is numeric, varies, and covers enough compounds to
# split. drug_best_pchembl is the one that will thin out -- a compound with no
# curated activity has no pChEMBL -- so the count is printed rather than assumed.
MIN_ROWS = 16          # under a 0.25 split this is four test points; less is not a score

BENCH_TARGETS: list[tuple[str, str]] = []      # (column, "measured" | "circular")
for col, role in [("drug_best_pchembl", "measured"), ("mol_qed", "circular")]:
    if col not in bench_space.obs.columns:
        print(f"{col}: not in .obs -- the {role} half of the comparison goes unscored")
        continue
    values = pd.to_numeric(bench_space.obs[col], errors="coerce")
    n_ok, n_distinct = int(values.notna().sum()), int(values.nunique(dropna=True))
    if n_ok < MIN_ROWS or n_distinct < 3:
        print(f"{col}: present but not usable ({n_ok} numeric values, "
              f"{n_distinct} distinct) -- skipped")
        continue
    BENCH_TARGETS.append((col, role))
    print(f"{col:<20} {role:<9} {n_ok}/{bench_space.n_obs} compounds, "
          f"{n_distinct} distinct, range "
          f"{values.min():.2f} to {values.max():.2f}")

if not BENCH_TARGETS:
    print("\nno usable target -- the benchmark below will report that and skip")
""")

code(r"""
benchmarks: dict[tuple[str, str], pd.DataFrame] = {}
rows_used: dict[tuple[str, str], int] = {}

for target, _role in BENCH_TARGETS:
    y = pd.to_numeric(bench_space.obs[target], errors="coerce").to_numpy(dtype=float)
    for key, obsm_key in bench_spaces.items():
        matrix = np.asarray(bench_space.obsm[obsm_key], dtype=float)
        # A single NaN reaching a regressor is a crash, not a low score, so the mask
        # is over the target and the whole embedding row together.
        ok = np.isfinite(matrix).all(axis=1) & np.isfinite(y)
        if int(ok.sum()) < MIN_ROWS:
            print(f"{target} / {key}: only {int(ok.sum())} complete compounds -- skipped")
            continue
        sub = bench_space[ok].copy()
        sub.obs[target] = y[ok]                  # force a plain float column
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                benchmarks[(target, key)] = tl.benchmark_embeddings(
                    sub, perturbation_column="compound", perturbation_type="chemical",
                    target=target, obsm_key=obsm_key,
                    models=["linear", "ridge", "knn", "random_forest"],
                    mode="quick", test_size=0.25, random_state=0,
                )
            rows_used[(target, key)] = int(ok.sum())
        except Exception as exc:
            print(f"{target} / {key}: skipped -- {type(exc).__name__}: {str(exc)[:70]}")

best_r2 = pd.DataFrame({
    target: {key: benchmarks[(target, key)]["r2"].max()
             for key in bench_spaces if (target, key) in benchmarks}
    for target, _role in BENCH_TARGETS
}).dropna(how="all")

if best_r2.empty:
    print("\nno benchmark completed")
else:
    best_r2.insert(0, "kind", [space_kind(k) for k in best_r2.index])
    print("\nbest R2 over the four probes (rows = model, columns = target)")
    display(best_r2.round(3))

    counts = pd.DataFrame({
        target: {key: rows_used[(target, key)]
                 for key in bench_spaces if (target, key) in rows_used}
        for target, _role in BENCH_TARGETS
    }).dropna(how="all")
    if not counts.empty:
        spread = counts.stack().unique()
        if len(spread) == 1:
            print(f"every cell used the same {int(spread[0])} compounds")
        else:
            print("compounds used per cell -- these differ, so the columns are not "
                  "strictly comparable:")
            display(counts)
""")

code(r"""
# A positive R2 is not automatically signal. The panel is 40 compounds and a 0.25
# split leaves a single-digit test set, so anything near zero is noise. Tier the
# verdict rather than reporting "beats the mean"; the thresholds are a judgement
# call, written here so they can be argued with.
SIGNAL, NOISE = 0.30, 0.10

BENCH_VERDICT = "the benchmark did not run"
if not best_r2.empty:
    scores = best_r2.drop(columns=["kind"])
    per_target = pd.DataFrame({
        "best_r2": scores.max(),
        "best_model": scores.idxmax(),
        "worst_r2": scores.min(),
        "below_zero": (scores < 0).sum(),
        "n_scored": scores.notna().sum(),
    })
    display(per_target.round(3))

    print("mean best-R2 by model family\n")
    display(scores.groupby(best_r2["kind"]).mean().round(3))

    for target, role in BENCH_TARGETS:
        if target not in scores.columns:
            continue
        r2, model = per_target.loc[target, "best_r2"], per_target.loc[target, "best_model"]
        if r2 >= SIGNAL:
            tier = f"real signal (R2={r2:.3f}, {model})"
        elif r2 >= NOISE:
            tier = f"at the noise floor (R2={r2:.3f}, {model}) -- not a usable predictor"
        else:
            tier = f"nothing usable (best R2={r2:.3f}, {model})"
        print(f"  {target:<20} [{role}] {tier}")

    # The comparison the section exists for: how much of the circular target's score
    # does the measured target recover? Anything close to parity would be surprising.
    circular = [t for t, r in BENCH_TARGETS if r == "circular" and t in scores.columns]
    measured = [t for t, r in BENCH_TARGETS if r == "measured" and t in scores.columns]
    if circular and measured:
        c, m = scores[circular[0]].max(), scores[measured[0]].max()
        BENCH_VERDICT = (
            f"best R2 {c:.3f} on {circular[0]} (a function of the input SMILES) "
            f"against {m:.3f} on {measured[0]} (measured potency): the spaces "
            f"reproduce their own input {'far ' if c - m > 0.4 else ''}better than "
            f"they predict the assay"
        )
        print(f"\n{BENCH_VERDICT}")
    else:
        BENCH_VERDICT = (
            "only one side of the circular / measured contrast was scoreable, so the "
            "gap this section is about could not be measured"
        )
        print(f"\n{BENCH_VERDICT}")

# Full per-probe detail for the single best (target, model) pair. The four rows are
# the point: a random forest beating linear by a wide margin says the relationship
# is non-linear, not that the embedding is good.
if benchmarks and not best_r2.empty:
    _scores = best_r2.drop(columns=["kind"])
    top_target = _scores.max().idxmax()
    top_model = _scores[top_target].idxmax()
    n = rows_used[(top_target, top_model)]
    print(f"all four probes for {top_model} predicting {top_target} "
          f"({n} compounds, {int(round(0.25 * n))} held out):")
    display(benchmarks[(top_target, top_model)].round(3))
    pl.plot_benchmark(benchmarks[(top_target, top_model)],
                      title=f"{top_model} predicting {top_target}")
""")

md(r"""
**Read the baseline first, then the winner.** `benchmark_embeddings` trains a
plain linear probe alongside the rest. If a space's best result is no better than
a linear read of its own vectors, the extra machinery bought nothing for this
task.

**Treat these numbers as a demonstration, not a benchmark result.** Forty
compounds, a 25% split and a single random seed leave a test set you can count on
one hand, so the ordering between two close models is not information. Only the
gap between a model above the signal threshold and one below zero is. For a real
decision use `mode="rigorous"` -- 5-fold cross-validation with a randomised
hyper-parameter search -- and hold out whole **scaffold** groups rather than
random rows, because the two congeneric families in this panel make random
splitting far too easy: a held-out statin has seven near-twins in the training
set.

Three caveats, all of which bound how far this can be pushed:

* **`drug_best_pchembl` is a curation artefact as much as a measurement.** It is
  the best pChEMBL over whatever assays happen to be in ChEMBL for that compound.
  A heavily studied kinase inhibitor has thousands of measurements and a high
  maximum; an old NSAID has a handful. Some of any score here is "how much has
  this compound been assayed", which correlates with family membership and so
  with structure.
* **The circular target is not a straw man.** Reproducing a physicochemical
  property from a fingerprint is genuinely useful when the property is expensive
  to compute or measure. It stops being useful the moment you present it as
  evidence about biology, which is the mistake this section is built to make
  visible.
* **The probes see raw bits.** Nothing here standardises or reduces the 2048-d
  binary spaces, so ridge and k-NN are working in a very sparse geometry.
  `reduce_dim` applies PCA first and would change these numbers; it is left off so
  the input is the same matrix every other section measured.

`tl` also carries `phenotypic_activity` and the perturbation-screen readouts
(`delta_l2`, `deg_overlap`, `phenocopy_score`). Those need a screen -- a control
column, and a measured response per compound -- which a panel of 40 identifiers
does not have, so they live in [cells](cells.ipynb).
""")

# ============================================= B. molecule-only: bit provenance
md(r"""
## Molecule-only: a fingerprint bit is not a feature

This is the one thing a molecular fingerprint has that no other embedding in
embpy does: **every dimension has a name you can draw.** Bit 314 of a Morgan
fingerprint is not an abstract coordinate; it is a hash bucket, and RDKit will
tell you which atom environments landed in it.

That makes a specific kind of debugging possible. When a purity number in the
[interpretation section](#does-the-geometry-track-biology) says a space separates
the statins, you can ask *on what* -- and get an answer that is a substructure
rather than a plausible story.

The cell below re-derives the Morgan matrix with `AdditionalOutput` attached,
because `RDKitWrapper._compute_fingerprint` calls
`GetMorganGenerator(radius=2, fpSize=2048).GetFingerprintAsNumPy(mol)` and throws
the bit-info map away. Re-deriving means the numbers could disagree with the
matrix every other section measured, so the cell checks that they do not instead
of assuming it. Two things could break the agreement: a different generator
configuration, and `embed()` having canonicalised the SMILES differently from
what is sitting in `.obs["smiles"]`.

Bits are then ranked by how sharply they separate one panel family from the rest
-- prevalence inside the family minus prevalence outside it. That is a deliberate
choice over ranking by variance: a high-variance bit is common, and a common
substructure is not an explanation.
""")

code(r"""
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator

RDLogger.DisableLog("rdApp.*")

MORGAN_RADIUS, MORGAN_BITS = 2, 2048      # exactly RDKitWrapper's defaults

# Shared by the three molecule-only sections below: the full 40-compound panel,
# because celecoxib and rofecoxib must both be present for the scaffold test and
# the dense intersection might have dropped one of them.
PANEL_SPACES = {key: f"X_{key}" for key in SWEEP if f"X_{key}" in mol_space.obsm}
COMPOUND_ROW = {str(name): i for i, name in enumerate(mol_space.obs["compound"])}

panel_smiles = [str(s) for s in mol_space.obs["smiles"]]
panel_family = mol_space.obs["family"].astype(str).to_numpy()

_gen = rdFingerprintGenerator.GetMorganGenerator(
    radius=MORGAN_RADIUS, fpSize=MORGAN_BITS)
morgan_bits = np.zeros((len(panel_smiles), MORGAN_BITS), dtype=np.float32)
bit_atoms: list[dict] = []
parsed = 0
for i, smiles in enumerate(panel_smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        bit_atoms.append({})
        continue
    extra = rdFingerprintGenerator.AdditionalOutput()
    extra.AllocateBitInfoMap()
    morgan_bits[i] = _gen.GetFingerprintAsNumPy(mol, additionalOutput=extra)
    bit_atoms.append(dict(extra.GetBitInfoMap()))
    parsed += 1

print(f"re-derived Morgan bits for {parsed}/{len(panel_smiles)} panel SMILES")
if "X_morgan_fp" in mol_space.obsm:
    swept = np.asarray(mol_space.obsm["X_morgan_fp"], dtype=float)
    finite = np.isfinite(swept).all(axis=1)
    agree = bool(np.array_equal(swept[finite], morgan_bits[finite]))
    print(f"identical to the swept X_morgan_fp on its {int(finite.sum())} finite rows: "
          f"{agree}")
    if not agree:
        n_diff = int((swept[finite] != morgan_bits[finite]).any(axis=1).sum())
        print(f"  {n_diff} row(s) differ -- read the bit table below as descriptive of "
              "this re-derivation, not of the swept matrix")
else:
    print("X_morgan_fp is not in .obsm, so there is nothing to check the "
          "re-derivation against")
""")

code(r"""
# Rank bits by family specificity: prevalence inside one family minus prevalence
# outside it. Bits on in almost every compound or almost none are excluded first --
# a substructure shared by 39 of 40 compounds explains no separation.
on_counts = morgan_bits.sum(axis=0)
candidates = np.where((on_counts >= 2) & (on_counts <= len(panel_smiles) - 2))[0]
families = sorted(set(panel_family))

scored = []
for bit in candidates:
    column = morgan_bits[:, bit]
    for family in families:
        inside = panel_family == family
        delta = float(column[inside].mean() - column[~inside].mean())
        scored.append((delta, int(bit), family))
scored.sort(reverse=True)


def bit_substructure(smiles: str, atom_idx: int, radius: int) -> str:
    # Radius 0 is a single atom, and FindAtomEnvironmentOfRadiusN returns an empty
    # bond path for it, so PathToSubmol would hand back an empty molecule. Name the
    # atom directly in that case rather than printing a blank cell.
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return "unparseable"
    if radius == 0:
        atom = mol.GetAtomWithIdx(atom_idx)
        aromatic = " aromatic" if atom.GetIsAromatic() else ""
        return f"single{aromatic} {atom.GetSymbol()}, degree {atom.GetDegree()}"
    env = Chem.FindAtomEnvironmentOfRadiusN(mol, radius, atom_idx)
    submol = Chem.PathToSubmol(mol, env)
    if submol.GetNumAtoms() == 0:
        return "-"
    try:
        return Chem.MolToSmarts(submol)
    except Exception:
        return "unrenderable"


rows, seen_bits = [], set()
for delta, bit, family in scored:
    if bit in seen_bits:
        continue
    seen_bits.add(bit)
    # Any compound carrying the bit will do to decode it; the first one in the
    # marked family keeps the substructure and the label in the same molecule.
    owner = next((i for i in range(len(panel_smiles))
                  if morgan_bits[i, bit] and panel_family[i] == family
                  and bit in bit_atoms[i]), None)
    if owner is None:
        continue
    atom_idx, radius = list(bit_atoms[owner][bit])[0]
    inside = panel_family == family
    rows.append({
        "bit": bit, "marks": family,
        "in_family": float(morgan_bits[inside, bit].mean()),
        "elsewhere": float(morgan_bits[~inside, bit].mean()),
        "delta": delta, "radius": int(radius),
        "decoded_from": str(mol_space.obs["compound"].iloc[owner]),
        "substructure": bit_substructure(panel_smiles[owner], atom_idx, radius),
    })
    if len(rows) == 8:
        break

bit_table = pd.DataFrame(rows).set_index("bit")
print(f"{len(candidates)} of {MORGAN_BITS} bits are on in 2 to "
      f"{len(panel_smiles) - 2} compounds; the eight most family-specific:")
display(bit_table.round(3))
""")

code(r"""
# The contrast. Score the learned spaces' dimensions by exactly the same family
# criterion, then ask the same question of the winner -- which substructure is it?
learned_keys = [k for k in PANEL_SPACES if space_kind(k) == "learned"]
if not learned_keys:
    print("no learned space survived the sweep, so there is nothing to contrast "
          "the decoded bits against")
else:
    learned_key = learned_keys[0]
    matrix = np.asarray(mol_space.obsm[PANEL_SPACES[learned_key]], dtype=float)
    finite = np.isfinite(matrix).all(axis=1)
    matrix, fam = matrix[finite], panel_family[finite]

    # Standardised difference of means, so dimensions on different scales compare.
    best = (0.0, None, None)
    for family in sorted(set(fam)):
        inside = fam == family
        if inside.sum() < 2 or (~inside).sum() < 2:
            continue
        spread = matrix.std(axis=0)
        effect = np.abs(matrix[inside].mean(axis=0) - matrix[~inside].mean(axis=0))
        effect = effect / np.where(spread > 0, spread, np.inf)
        dim = int(np.argmax(effect))
        if effect[dim] > best[0]:
            best = (float(effect[dim]), dim, family)

    effect, dim, family = best
    column = matrix[:, dim]
    wrapper = embedder.get_model(learned_key, load=False)
    print(f"{learned_key}: dimension {dim} separates '{family}' most sharply "
          f"(standardised difference {effect:.2f})")
    print(f"  values run {column.min():.3f} to {column.max():.3f}; "
          f"{int((column != 0).sum())}/{len(column)} compounds are non-zero")
    print(f"  has_attention = {wrapper.has_attention}, declared on "
          f"{type(wrapper).__name__} itself: "
          f"{'has_attention' in type(wrapper).__dict__}")
    print("  bit-info equivalent on the wrapper: "
          f"{[a for a in dir(wrapper) if 'bit' in a.lower()] or 'none'}")
""")

md(r"""
Read the two cells against each other rather than separately.

The bit table names substructures. A row saying bit *b* is on in every statin and
no benzodiazepine, decoded as a SMARTS you can look at, is a *mechanical*
explanation of a purity score: the space separates those compounds because they
share that fragment. Nothing about it is a claim, and nothing about it needed a
narrative.

The learned dimension has the same statistics and no such reading. Every compound
has a value, the value is signed and continuous, and the wrapper exposes no map
from dimension to structure -- because there is not one. Dimension 137 of
ChemBERTa is a direction in a space shaped by a masked-language objective over
millions of SMILES strings; asking which fragment it encodes presumes an
alignment between dimensions and chemistry that training never imposed.

> **`has_attention` here is inherited, not declared.** It defaults to `True` on
> `BaseModelWrapper` and the transformer molecule wrappers do not override it,
> which is why the check above prints `False` for the fact that the class declares
> it. `RDKitWrapper` *does* declare `has_attention = False`, and it is the one
> wrapper in this notebook whose dimensions are interpretable. Do not read the
> flag as a promise about interpretability in either direction; this notebook runs
> no attention extraction, and [06_attention_weights](06_attention_weights.ipynb)
> covers the extract-then-summarise pattern properly.

The practical consequence is a division of labour, not a ranking. When you need
to defend a hit to a chemist, a fingerprint gives you a fragment to point at. When
you need a representation that can put two unrelated scaffolds near each other,
that same literalness is the problem -- which is exactly what the next section
tests.
""")

# ============================================== C. molecule-only: scaffold hop
md(r"""
## Molecule-only: scaffold hopping

Here is the single cleanest structure-versus-mechanism test available in this
panel. Celecoxib and rofecoxib are both selective COX-2 inhibitors -- same
enzyme, same selectivity story, same withdrawal-era literature -- built on
completely different cores: a pyrazole bearing a sulfonamide against a furanone.
A screen that found one and missed the other found half the mechanism.

So: does any space in the roster put them next to each other?

The prediction from [part 1](#a-panel-where-structure-and-mechanism-disagree) is
that the fingerprints will not, because there is very little shared substructure
to hash, and that a learned representation *might*, because it was trained on
SMILES strings in a context where those two molecules keep appearing in the same
company. Two controls keep that honest:

* **imatinib and nilotinib** share the phenylaminopyrimidine core outright.
  Nilotinib was designed as an imatinib analogue. Any method that cannot call
  these two neighbours is broken, so this row calibrates the scale rather than
  testing anything.
* **aspirin and naproxen** are a second hop inside the NSAID family: a salicylate
  ester against a propionic acid, shared mechanism, almost no shared skeleton.

The scaffolds are asserted above and measured below -- Murcko scaffolds and the
Morgan Tanimoto for each pair, so "different scaffold" is a number in the output
rather than a claim in the prose.
""")

code(r"""
from rdkit.Chem.Scaffolds import MurckoScaffold

HOP_PAIRS = [
    ("celecoxib", "rofecoxib", "same target, different scaffold"),
    ("aspirin", "naproxen", "same mechanism, different scaffold"),
    ("imatinib", "nilotinib", "shared core -- the control"),
]


def murcko(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return "unparseable"
    return Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(mol))


scaffold_rows = []
for a, b, note in HOP_PAIRS:
    if a not in COMPOUND_ROW or b not in COMPOUND_ROW:
        print(f"{a}/{b}: one of the pair is not in the panel -- skipped")
        continue
    ia, ib = COMPOUND_ROW[a], COMPOUND_ROW[b]
    va, vb = morgan_bits[ia] > 0, morgan_bits[ib] > 0
    union = int((va | vb).sum())
    scaffold_rows.append({
        "pair": f"{a} / {b}", "note": note,
        "same_family": panel_family[ia] == panel_family[ib],
        "morgan_tanimoto": (int((va & vb).sum()) / union) if union else np.nan,
        "same_murcko": murcko(panel_smiles[ia]) == murcko(panel_smiles[ib]),
        "scaffold_a": murcko(panel_smiles[ia]),
        "scaffold_b": murcko(panel_smiles[ib]),
    })

display(pd.DataFrame(scaffold_rows).set_index("pair").round(3))
""")

code(r"""
def cosine_similarities(matrix: np.ndarray, row: int) -> np.ndarray:
    # Normalise once and take a dot product. A zero-norm row would otherwise divide
    # by zero, and an all-zero fingerprint is possible for a very small molecule.
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    unit = matrix / np.where(norms > 0, norms, 1.0)
    return unit @ unit[row]


hop_rows = []
for key, obsm_key in PANEL_SPACES.items():
    matrix = np.asarray(mol_space.obsm[obsm_key], dtype=float)
    finite = np.isfinite(matrix).all(axis=1)
    sub = matrix[finite]
    # Row indices shift once NaN rows are dropped, so carry the compound names along.
    names = [str(n) for n in mol_space.obs["compound"][finite]]
    local = {name: i for i, name in enumerate(names)}
    for a, b, _note in HOP_PAIRS:
        if a not in local or b not in local:
            continue
        sims = cosine_similarities(sub, local[a])
        order = [names[i] for i in np.argsort(-sims) if names[i] != a]
        hop_rows.append({
            "space": key, "kind": space_kind(key), "pair": f"{a} / {b}",
            "cosine": float(sims[local[b]]),
            "rank": order.index(b) + 1, "of": len(order),
            "nearest": order[0],
        })

hop = pd.DataFrame(hop_rows)
if hop.empty:
    print("no space carried both members of any pair -- nothing to report")
else:
    print("rank of the second compound in the first one's neighbour list "
          "(1 = nearest; lower is better)\n")
    ranks = hop.pivot(index="space", columns="pair", values="rank")
    ranks.insert(0, "kind", [space_kind(k) for k in ranks.index])
    display(ranks)

    print("\nand what each space actually calls the nearest neighbour:")
    display(hop.pivot(index="space", columns="pair", values="nearest"))
""")

md(r"""
Three readings, and the third is the one worth carrying out of this notebook.

**The control row is the sanity check.** Imatinib and nilotinib should be at or
near rank 1 in every space. Where they are not, that space is not measuring
structure in any recognisable sense and its other two columns cannot be
interpreted.

**The hop rows are the result.** A rank near 1 means the space put two molecules
with almost no shared substructure next to each other; a rank in the twenties or
thirties means it did not. Compare the fingerprint rows against the learned rows
directly -- the `kind` column is there for exactly that -- and note that the
learned models are not guaranteed to win. A masked-language model trained on
SMILES is still reading strings, and two different scaffolds are two different
strings.

**A rank is not a mechanism.** Even a space that scores well here has not
discovered that both compounds inhibit COX-2. At best it has learned that
molecules of this kind co-occur in its training corpus, which is a fact about the
literature. The honest way to get mechanism into the geometry is to put it there:
annotate the compounds, as the [clinical record](#the-clinical-record) section
does, and use the annotation as the label rather than hoping the structure
implies it.
""")

# =============================================== D. molecule-only: ChEMBL analogs
md(r"""
## Molecule-only: ChEMBL analogs vs embedding neighbours

`ChEMBLAnnotator.get_analogs` runs a Tanimoto similarity search over the whole of
ChEMBL -- roughly two million distinct compounds -- and returns the closest
matches above a similarity floor. The embedding neighbour lists in the
[local-neighbourhoods section](#2-local-neighbourhoods) rank the 39 other members
of this panel. Both produce a list of similar molecules and they answer completely
different questions:

| | `get_analogs` | embedding neighbours |
| --- | --- | --- |
| Search space | all of ChEMBL | the 40 compounds you chose |
| Similarity | Morgan Tanimoto, computed server-side | cosine in whichever space you ask |
| Answers | *what else in the world looks like this* | *what in my screen looks like this* |
| Fails when | your compound is not in ChEMBL | your panel has no relative of the query |

The comparison below is not a benchmark -- neither list can be wrong relative to
the other. It is here because the two get conflated in practice: a hit list from
a screen gets treated as a claim about chemical space, when it is a claim about
the plate.

The call is a structure search rather than a keyed lookup, which is why
`annotate` excludes `analogs` from `sources="all"`. Three probes, chosen to span
the panel's difficulty: imatinib (a designed series with known relatives),
celecoxib (the scaffold-hop case) and atorvastatin (a tight congeneric family).
Each is wrapped separately so one network failure costs one probe.
""")

code(r"""
from embpy.resources.molecule.chembl import ChEMBLAnnotator

# A fresh instance under its own name so the annotation section's client and cache
# are left alone.
chembl_probe = ChEMBLAnnotator()

ANALOG_PROBES = ["imatinib", "celecoxib", "atorvastatin"]
ANALOG_SIMILARITY, ANALOG_LIMIT, TOP_N = 70, 25, 5

analog_gaps: dict[str, str] = {}
analog_summary = []

# The panel-internal side is read from the Morgan space, so both sides of the
# comparison use the same notion of similarity and only the search space differs.
morgan_matrix = (np.asarray(mol_space.obsm["X_morgan_fp"], dtype=float)
                 if "X_morgan_fp" in mol_space.obsm else None)
panel_names = [str(n) for n in mol_space.obs["compound"]]
panel_lower = {n.lower() for n in panel_names}

for probe in ANALOG_PROBES:
    try:
        analogs = chembl_probe.get_analogs(
            probe, similarity=ANALOG_SIMILARITY, limit=ANALOG_LIMIT)
    except Exception as exc:
        analog_gaps[probe] = f"{type(exc).__name__}: {str(exc)[:70]}"
        print(f"{probe}: analog search failed -- {analog_gaps[probe]}")
        continue
    if not analogs:
        analog_gaps[probe] = "no analog above the similarity floor"
        print(f"{probe}: ChEMBL returned no analog at >= {ANALOG_SIMILARITY}% "
              "similarity")
        continue

    if morgan_matrix is None or probe not in COMPOUND_ROW:
        neighbours = []
    else:
        sims = cosine_similarities(morgan_matrix, COMPOUND_ROW[probe])
        neighbours = [(panel_names[i], float(sims[i]))
                      for i in np.argsort(-sims) if panel_names[i] != probe][:TOP_N]

    side_by_side = pd.DataFrame({
        "chembl_analog": [(a["pref_name"] or a["molecule_chembl_id"] or "?").lower()
                          for a in analogs[:TOP_N]],
        "chembl_tanimoto_pct": [a["similarity"] for a in analogs[:TOP_N]],
        "chembl_phase": [a["development_phase"] or "-" for a in analogs[:TOP_N]],
        "panel_neighbour": [n for n, _ in neighbours] + ["-"] * (TOP_N - len(neighbours)),
        "panel_cosine": [c for _, c in neighbours] + [np.nan] * (TOP_N - len(neighbours)),
    }, index=pd.RangeIndex(1, TOP_N + 1, name="rank"))
    print(f"\n{probe}: {len(analogs)} ChEMBL analog(s) at >= {ANALOG_SIMILARITY}% "
          f"similarity, against {len(neighbours)} panel neighbour(s)")
    display(side_by_side.round(3))

    in_panel = [a for a in analogs
                if (a["pref_name"] or "").lower() in panel_lower]
    approved = sum(1 for a in analogs if (a["max_phase"] or 0) >= 4)
    analog_summary.append({
        "probe": probe, "n_analogs": len(analogs),
        "approved_analogs": approved,
        "also_in_panel": len(in_panel),
        "panel_members": ", ".join(sorted(
            (a["pref_name"] or "").lower() for a in in_panel)) or "-",
        "top_tanimoto_pct": analogs[0]["similarity"],
    })

if analog_summary:
    print("\nhow much of ChEMBL's answer is already on your plate:")
    display(pd.DataFrame(analog_summary).set_index("probe").round(3))
if analog_gaps:
    print(f"\nprobes with no analog list: {analog_gaps}")
""")

# ============================================ E. molecule-only: salt vs parent
md(r"""
## Molecule-only: the same drug, two SMILES

One compound, several registered structures. Imatinib is sold as the mesylate
salt; ChEMBL registers the free base and the salt as separate molecules with
separate identifiers, and a screening plate might be labelled with either. A
representation that put the mesylate somewhere different from the free base would
mean two files describing the same experiment produce two different answers.

`get_molecule_forms` returns the parent and its registered forms, so the question
is answerable rather than hypothetical. The cell below fetches every form of
imatinib, pulls each one's canonical SMILES, and then measures the same pair
twice: once through the raw fingerprint, and once through `embed()`.

That distinction is the whole section, and the two results do not agree.
""")

code(r"""
FORM_PROBE = "imatinib"

form_smiles: dict[str, str] = {}
form_is_parent: dict[str, bool] = {}
try:
    forms = chembl_probe.get_molecule_forms(FORM_PROBE)
    for form in forms:
        cid = form["molecule_chembl_id"]
        record = chembl_probe.get_molecule_record(cid) or {}
        smiles = (record.get("molecule_structures") or {}).get("canonical_smiles")
        if smiles:
            form_smiles[cid] = smiles
            form_is_parent[cid] = bool(form["is_parent"])
    print(f"{FORM_PROBE}: {len(forms)} registered form(s), "
          f"{len(form_smiles)} with a structure")
except Exception as exc:
    print(f"form lookup failed -- {type(exc).__name__}: {str(exc)[:110]}")

if len(form_smiles) < 2:
    print("fewer than two forms with structures -- the comparison below has "
          "nothing to compare and is skipped")
else:
    # Raw fingerprints, straight off the wrapper. get_model returns the same loaded
    # RDKitWrapper the sweep used, and its embed() takes the SMILES string as given
    # -- no canonicalisation, no salt handling.
    fp_wrapper = embedder.get_model("morgan_fp")
    parent_id = next(cid for cid, flag in form_is_parent.items() if flag)
    parent_vec = fp_wrapper.embed(form_smiles[parent_id]) > 0

    form_rows = []
    for cid, smiles in form_smiles.items():
        vec = fp_wrapper.embed(smiles) > 0
        union = int((vec | parent_vec).sum())
        mol = Chem.MolFromSmiles(smiles)
        form_rows.append({
            "chembl_id": cid, "is_parent": form_is_parent[cid],
            "heavy_atoms": mol.GetNumHeavyAtoms() if mol else np.nan,
            "fragments": smiles.count(".") + 1,
            "bits_on": int(vec.sum()),
            "tanimoto_to_parent": (int((vec & parent_vec).sum()) / union)
                                  if union else np.nan,
            "smiles_head": smiles[:38] + ("..." if len(smiles) > 38 else ""),
        })
    display(pd.DataFrame(form_rows).set_index("chembl_id").round(3))
""")

code(r"""
if len(form_smiles) >= 2:
    from embpy.resources.molecule.resolver import DrugResolver

    # What embpy's molecule id scheme does with the same strings. canonicalize_smiles
    # is the single canonicaliser behind entity_type="molecule": it strips isotope
    # labels, removes known salts and solvents, keeps the largest fragment and
    # neutralises charges. Every one of those steps erases a difference between
    # registered forms.
    canon = DrugResolver()
    canonical = {cid: canon.canonicalize_smiles(smiles)
                 for cid, smiles in form_smiles.items()}
    distinct = sorted(set(canonical.values()))
    print(f"{len(form_smiles)} registered structures collapse to "
          f"{len(distinct)} canonical id(s)")
    for cid, value in canonical.items():
        print(f"  {cid:<16} -> {str(value)[:52]}...")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        forms_out = embedder.embed(
            list(form_smiles.values()), entity_type="molecule", id_type="smiles",
            model="morgan_fp", output="anndata", key="X_form_probe",
        )
    print(f"\nembed() on {len(form_smiles)} SMILES returned {forms_out.n_obs} row(s)")
    print(f"surviving id: {list(forms_out.obs_names)[0][:60]}...")
    lost = len(form_smiles) - forms_out.n_obs
    print(f"rows silently dropped as duplicate canonical ids: {lost}")
    if lost:
        print("The only signal was a logging warning from embpy.io._canon "
              "(\"Dropped N rows that collapsed to a duplicate canonical id\"). "
              "No exception, no column recording which rows went.")
""")

md(r"""
> **The canonicaliser protects you, and it does it quietly.** The raw
> fingerprints differ -- the mesylate carries the counterion's bits and a higher
> heavy-atom count, so its Tanimoto to the free base is below 1. Fed those
> strings directly, a screen would place the two forms in different positions.
> `embed()` never asks: `DrugResolver.canonicalize_smiles` reduces every
> registered form to the same string, and `embpy.io._canon.drop_and_dedup` then
> **deletes** the duplicates. You get the right answer and fewer rows than you
> passed in.

Three consequences, in decreasing order of how likely they are to bite:

* **Row counts change without an error.** If your plate has one well for the free
  base and one for the mesylate, `embed()` returns one row. Any `.obs` you were
  planning to join back on position rather than on the returned identifier is now
  misaligned. Compare `n_obs` before and after, every time.
* **Deuterated compounds collapse too.** `IsotopeParentInPlace` runs before
  everything else, so a deuterated analogue -- a real drug class, with its own
  approvals and its own pharmacokinetics -- becomes indistinguishable from its
  parent. If deuteration is the variable you are studying, this canonicaliser is
  the wrong tool and you must embed the strings through the wrapper directly, as
  the previous cell does.
* **Charge state is erased.** `Uncharger` neutralises, so a quaternary ammonium
  and its neutral relative can converge. That is usually what you want for a
  structural comparison and never what you want for a solubility or permeability
  model.

None of this is a bug. Salt stripping is the correct default for structural
comparison, and it is *why* the panel's resolution step in
[part 2](#resolve-the-compounds-and-measure-them-before-trusting-anything)
produced clean one-per-compound rows without anyone having to think about
formulations. The failure mode is trusting it without measuring it.
""")

# ============================================================== F. save
md(r"""
## Save the artifact

The full `mol_space` goes to disk -- all 40 compounds, gaps included, rather than
`dense_space`. The gaps are information: they record which model failed on which
compound, and dropping them would force whoever reloads this file to re-derive
coverage. What is deliberately not preserved is the derived coordinates the plots
produced; a 2-d UMAP of a model is not the model, and it is one line to recompute.

Two things resist `write_h5ad` and both are handled rather than hoped about. The
annotation records in `.uns` are nested dictionaries of lists of dictionaries,
which the h5ad writer cannot type; and any `.obs` column mixing `None` with
numbers is object-dtype, which it also cannot type. ChEMBL produces both
routinely -- `drug_max_phase` is `None` for a preclinical compound and a float
otherwise.

An artifact you cannot read back is not an artifact, so the cell reloads the file
and checks that the `.obsm` keys and the row count survived.
""")

code(r"""
import anndata as ad

OUTPUT_DIR.mkdir(exist_ok=True)
to_write = mol_space.copy()

# Coerce object columns into something h5ad can type. Numbers first, because a
# None/float mix is the common ChEMBL shape and losing it to strings would make the
# column unusable on reload; otherwise strings, with None spelled as empty.
for col in list(to_write.obs.columns):
    if to_write.obs[col].dtype != object:
        continue
    numeric = pd.to_numeric(to_write.obs[col], errors="coerce")
    if numeric.notna().any() and numeric.notna().sum() >= to_write.obs[col].notna().sum():
        to_write.obs[col] = numeric.astype(float)
    else:
        to_write.obs[col] = to_write.obs[col].map(
            lambda v: "" if v is None else str(v)).astype(str)

path = OUTPUT_DIR / "molecule_embeddings.h5ad"
written = False
try:
    to_write.write_h5ad(path)
    written = True
except (TypeError, ValueError) as exc:
    print(f"first write failed ({type(exc).__name__}), serialising .uns: "
          f"{str(exc)[:110]}")
    try:
        to_write.uns = {key: json.dumps(value, default=str)
                        for key, value in to_write.uns.items()}
        to_write.write_h5ad(path)
        written = True
    except Exception as exc2:
        print(f"still not writable -- {type(exc2).__name__}: {str(exc2)[:160]}")

if written:
    print(f"\nwrote {path} ({path.stat().st_size / 1e6:.1f} MB), "
          f"{to_write.n_obs} compounds")
    print(f"  obsm ({len(to_write.obsm)}):")
    for key in to_write.obsm:
        matrix = np.asarray(to_write.obsm[key], dtype=float)
        gaps = int((~np.isfinite(matrix).all(axis=1)).sum())
        print(f"    {key:<28} {matrix.shape[1]:>5}d   "
              f"{'complete' if gaps == 0 else f'{gaps} compound(s) missing'}")
    annotation_cols = [c for c in to_write.obs.columns
                       if c.startswith(("mol_", "drug_"))]
    print(f"  obs: {len(to_write.obs.columns)} columns, "
          f"{len(annotation_cols)} of them annotation")

    reloaded = ad.read_h5ad(path)
    assert reloaded.n_obs == mol_space.n_obs
    assert set(reloaded.obsm) == set(to_write.obsm)
    print(f"  round-trip OK: {reloaded.n_obs} compounds, "
          f"{len(reloaded.obsm)} embedding spaces")
""")

# ============================================================== G. wrap-up
md(r"""
## What we found

**The prediction.** The notebook opened with a panel built so that structural
similarity and mechanism similarity would disagree: two tight congeneric series
that any fingerprint should recover almost perfectly, two families that share a
mechanism across unrelated scaffolds, and a kinase-inhibitor set that shares a
target family while spanning cores. The claim was that fingerprints would win on
the congeneric series and lose on the mechanism-defined ones, and that a learned
representation might narrow the gap. The
[interpretation section](#does-the-geometry-track-biology) scored that with k-NN
label purity and wrote its conclusion into `PREDICTION_VERDICT`; the benchmark at
the top of this section scored the same split a second way, against a number no
model was trained on.

The cell below re-prints those verdicts rather than paraphrasing them. A summary
that restated the figures would stop being true the first time the panel, the
roster or ChEMBL changed underneath it.
""")

code(r"""
scored_verdict = globals().get("PREDICTION_VERDICT")
if scored_verdict is None:
    print("the interpretation section produced no verdict in this run -- re-run it; "
          "everything below still applies.")
else:
    print(scored_verdict)

print(f"\nfrom the benchmark: {BENCH_VERDICT}")

purity = globals().get("PURITY")
if purity is not None and hasattr(purity, "empty") and not purity.empty:
    print("\nk-NN label purity against the hand-built family labels, best first:")
    display(purity.round(3))

if not hop.empty:
    hops = hop[hop["pair"] != "imatinib / nilotinib"]
    best_hop = hops.loc[hops["rank"].idxmin()] if not hops.empty else None
    if best_hop is not None:
        print(f"\nbest scaffold hop anywhere in the roster: "
              f"{best_hop['space']} put {best_hop['pair']} at rank "
              f"{int(best_hop['rank'])} of {int(best_hop['of'])}")

annotated = globals().get("ANNOTATED")
if annotated is not None:
    print(f"\nannotation stage completed: {annotated}")
""")

md(r"""
What does not depend on how those numbers fell: the fingerprints and the learned
models are not interchangeable, they fail on different things, and which one you
want is settled by what your labels mean rather than by a leaderboard. If your
label is a scaffold series, a Morgan fingerprint is very hard to beat and it will
also tell you *which bit* did the work. If your label is a mechanism, a target
class or an ATC code, structure alone is the wrong input and the honest fix is to
annotate the compounds and use the annotation.

**Practical points, each of which cost something to learn here.**

1. **Resolve to SMILES once, embed many.** The canonical molecule id *is* the
   canonical SMILES, so passing compound names to `embed()` re-runs a live
   PubChem and ChEMBL resolution for every model in the roster. Resolve in one
   pass, audit the result, and hand `embed()` structures from then on.
2. **`embed()` strips salts, isotopes and charges, then deletes the duplicates.**
   `DrugResolver.canonicalize_smiles` is the single canonicaliser behind
   `entity_type="molecule"`, and every registered form of imatinib collapses to
   one string. The only signal is a log warning, so compare `n_obs` before and
   after and never join `.obs` back by position.
3. **ChEMBL keys drug-level annotation on the parent molecule.** Filtering
   `mechanism` on `molecule_chembl_id` returns nothing for any drug curated
   against a salt; `parent_molecule_chembl_id` returns the real rows. The
   [mechanism section](#mechanism-and-the-parent-molecule-trap) shows the count
   both ways.
4. **A ChEMBL "target" can be a cell line.** Sort a compound's activities by raw
   potency and a K562 cytotoxicity reading can outrank every protein target it
   has. Filter `target_type` before "most potent" is allowed to mean "primary
   target"; `summarize_selectivity` does that filtering, a hand-rolled `max()`
   does not.
5. **Benchmark against something you did not compute from the input.** `mol_qed`
   and `mol_molecular_weight` are functions of the same SMILES the fingerprints
   are functions of, so a probe recovering them has measured RDKit's
   determinism. Score against assay data, and read the circular target only as a
   calibration.
6. **The catalogue overstates the choice.** `list_available_models("molecule")`
   lists `mole`, which cannot be reached through `BioEmbedder` at all; `minimol`
   and `mhg_gnn` need extras that will not build here. Within what does run,
   `maccs_fp` is 167-wide and ignores `n_bits`, and `RDKitWrapper`'s only pooling
   strategy is `"flat"` and the value is ignored. Probe a key on one compound
   before you build a sweep on it.

**Where to go next.**

* [Comparing embeddings](03_compare_embeddings.ipynb) -- what each metric used in
  the [geometry sections](#1-global-geometry) actually measures, and when they are
  allowed to disagree.
* [Benchmarking models](04_benchmark_models.ipynb) -- `mode="rigorous"`, grouped
  splits, and the machinery this notebook only sampled.
* [Proteins](proteins.ipynb) -- the same programme where the sequence *is* the
  functional unit, so the structure-versus-function tension resolves differently.
* [Genes](genes.ipynb) -- the opposite extreme: a gene has no single canonical
  string, and the notebook spends its first section deciding what to embed.
* [Cells](cells.ipynb) -- where these compounds become perturbations with
  measured responses, and `phenotypic_activity` has something to score.
""")

Path(sys.argv[1] if len(sys.argv) > 1 else "cells_part6.json").write_text(json.dumps(CELLS))
print(f"part 6: {len(CELLS)} cells")
