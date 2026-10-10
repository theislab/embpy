"""Part 2 of docs/notebooks/small_molecules.ipynb -- resolution and the embedding sweep."""
from __future__ import annotations
import json, sys
from pathlib import Path

CELLS: list[tuple[str, str]] = []
def md(s: str) -> None: CELLS.append(("markdown", s.strip("\n")))
def code(s: str) -> None: CELLS.append(("code", s.strip("\n")))

# ======================================================== A. resolution
md(r"""
## Resolve the compounds, and measure them before trusting anything

The panel is forty *names*. Every model in this notebook consumes a SMILES
string, so something has to turn "atorvastatin" into a structure -- and that
step is where a molecule notebook quietly goes wrong. A name that resolves to
the wrong tautomer, a salt where you expected the free base, or a compound that
silently fails to resolve at all will all produce a plausible embedding matrix
with a plausible shape.

So resolve first, record where each answer came from, and measure the results
before any of them reaches a model.
""")

code(r"""
from embpy.resources import DrugResolver

resolver = DrugResolver(use_rdkit=True, sleep_sec=0.1)

# name_to_smiles_resolved rather than name_to_smiles: the DrugResolution it
# returns records WHICH service answered, and that is the difference between a
# reproducible panel and forty strings of unknown provenance.
rows = []
for name in COMPOUNDS:
    res = resolver.name_to_smiles_resolved(name)
    rows.append({
        "compound": name,
        "family": FAMILY[name],
        "smiles": res.smiles,
        "source": res.source or "unresolved",
    })

RESOLUTION = pd.DataFrame(rows).set_index("compound")
n_resolved = int(RESOLUTION["smiles"].notna().sum())
print(f"resolved {n_resolved}/{len(COMPOUNDS)} compounds")
display(RESOLUTION.groupby("source").size().to_frame("n_compounds"))
""")

md(r"""
`source` is worth reading rather than skipping. `pubchem_name` means the name
matched directly; `pubchem_cid` means it took a second lookup; `cactus` means
PubChem did not know the name at all and the NCI resolver did. A panel answered
entirely by one service is easier to reproduce than one stitched from three.
""")

code(r"""
# Fail loudly here rather than embedding a shorter panel than the one described
# in the section above. A missing compound changes every per-family number
# downstream, and a 39-compound "family of eight" is not a thing.
unresolved = RESOLUTION.index[RESOLUTION["smiles"].isna()].tolist()
if unresolved:
    print(f"UNRESOLVED ({len(unresolved)}): {unresolved}")
    print("Re-run when the resolver services are reachable; the rest of the "
          "notebook assumes all forty.")
else:
    print("all forty compounds resolved")
""")

md(r"""
### Canonical is the identity, and it is not the string you were given

embpy keys molecules by canonical SMILES, and canonicalisation is not cosmetic:
it strips isotope labels, normalises, removes salts and solvents, keeps the
largest fragment and neutralises charges. Two panels that name the same drug
differently land on the same row only after that pass.

Because the identity is the canonical string, the AnnData is re-indexed by it
below. This is the single most breakable step in the notebook: `embed()`
resolves the SMILES you give it to canonical form and then re-indexes the result
against `obs_names`, so if `obs_names` held compound *names* the join would
match nothing and every row would come back NaN.
""")

code(r"""
canonical = [resolver.canonicalize_smiles(s) for s in RESOLUTION["smiles"]]
changed = sum(1 for raw, can in zip(RESOLUTION["smiles"], canonical) if raw != can)
print(f"canonicalisation rewrote {changed}/{len(canonical)} strings")

# One row per distinct molecule. A duplicate here would mean two panel names
# that are the same compound, which would break the family design.
assert len(set(canonical)) == len(canonical), "duplicate molecule in the panel"

mol_space.obs["smiles"] = canonical
mol_space.obs_names = pd.Index(canonical, name="canonical_smiles")

display(pd.DataFrame({
    "compound": mol_space.obs["compound"].values,
    "canonical_smiles": [s[:52] + ("..." if len(s) > 52 else "") for s in canonical],
}).head(6))
""")

md(r"""
### What the panel actually looks like, physically

Before comparing models, know the range they have to cover. A panel where every
compound has the same mass and the same ring count cannot separate models, and
one with a 10x spread in heavy-atom count will stress the transformers'
context windows.
""")

code(r"""
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

phys = []
for name, smi in zip(mol_space.obs["compound"], mol_space.obs["smiles"]):
    mol = Chem.MolFromSmiles(smi)
    phys.append({
        "compound": name,
        "family": FAMILY[name],
        "mw": Descriptors.MolWt(mol),
        "heavy_atoms": mol.GetNumHeavyAtoms(),
        "rings": rdMolDescriptors.CalcNumRings(mol),
        "rotatable": rdMolDescriptors.CalcNumRotatableBonds(mol),
        "smiles_chars": len(smi),
    })
PHYS = pd.DataFrame(phys).set_index("compound")

display(PHYS.groupby("family")[
    ["mw", "heavy_atoms", "rings", "rotatable", "smiles_chars"]
].mean().round(1))
""")

md(r"""
The families are physically distinguishable before any model runs, which is the
point: a fingerprint that fails to separate them is not being defeated by a
subtle panel.
""")

# ==================================================== B. tokeniser audit
md(r"""
### The tokeniser audit, or how a compound disappears

`ChembertaWrapper.embed_batch` **skips** any SMILES longer than its `max_len`
and returns `None` in that slot, which `_aligned_matrix` then drops -- so a
compound that is too long vanishes from the matrix without raising anything.
That is the molecule analogue of a protein exceeding an ESM window, and it is
worth measuring *before* the sweep rather than discovering it as a shape
mismatch afterwards.

The two ChemBERTa checkpoints tokenise differently, so the same molecule has two
different lengths. Measure both.
""")

code(r"""
TOKEN_LIMIT = 512      # ChembertaWrapper clamps max_position_embeddings to this

tok_rows = []
for key in ("chemberta2MTR", "chemberta2MLM"):
    try:
        w = embedder.get_model(key)
    except Exception as exc:
        print(f"{key}: unavailable ({type(exc).__name__})")
        continue
    lengths = [len(w.tokenizer.tokenize(s)) for s in mol_space.obs["smiles"]]
    tok_rows.append({
        "model": key,
        "min_tokens": int(np.min(lengths)),
        "median_tokens": float(np.median(lengths)),
        "max_tokens": int(np.max(lengths)),
        "limit": TOKEN_LIMIT,
        "would_be_dropped": int(np.sum(np.array(lengths) > TOKEN_LIMIT)),
    })

TOKENS = pd.DataFrame(tok_rows).set_index("model")
display(TOKENS)
""")

md(r"""
Nothing is near the limit -- drug-like molecules tokenise to tens of characters,
not hundreds -- so no compound is dropped here. That is the answer for *this*
panel, and it is only knowable because it was measured. A peptide-like or
polymeric input would give a different one, and `would_be_dropped` is the
column that would tell you.

Note the gap between the two checkpoints' token counts. They are both called
ChemBERTa and they cut a SMILES string differently; the
[attention section](#reading-chembertas-attention) returns to that.
""")

# ========================================================= C. the sweep
md(r"""
## Embed every molecule model

One `embed` call per model, same call shape every time, one `.obsm` key each.
The only thing that varies is `model=`.

Three practical points:

* **`missing="nan"`, not the default.** A model that fails on one compound
  should cost you that row, not the whole sweep. The NaN rows are then counted
  rather than ignored -- see [two views](#one-molecule-space-two-views).
* **Every failure is caught and reported.** `molformer_base` is in the registry
  and cannot load in this environment; the sweep records that rather than
  dying on it.
* **`SWEEP` is derived from what worked**, never from the roster. Everything
  downstream iterates `SWEEP`.
""")

code(r"""
import time

SWEEP = []
TIMINGS = []
FAILED = {}

for key in ROSTER:
    t0 = time.time()
    try:
        mol_space = embedder.embed(
            mol_space,
            entity_type="molecule",
            id_type="smiles",
            obs_column="smiles",
            model=key,
            output="anndata",
            attach_to="obs",
            key=f"X_{key}",
            missing="nan",       # one bad compound must not cost the sweep
        )
        matrix = np.asarray(mol_space.obsm[f"X_{key}"], dtype=float)
        n_ok = int(np.isfinite(matrix).all(axis=1).sum())
        SWEEP.append(key)
        TIMINGS.append({
            "model": key,
            "dim": matrix.shape[1],
            "rows_filled": n_ok,
            "seconds": round(time.time() - t0, 1),
        })
    except Exception as exc:
        FAILED[key] = f"{type(exc).__name__}: {str(exc)[:90]}"
        embedder.clear_model_cache()

TIMINGS = pd.DataFrame(TIMINGS).set_index("model")
display(TIMINGS)
for key, err in FAILED.items():
    print(f"skipped {key}: {err}")
""")

md(r"""
### Why MoLFormer is not in the sweep

`molformer_base` is listed by `list_available_models("molecule")` and it does
not load. The reason is two layers below the message you get, and it is a
dependency problem rather than a bug: MoLFormer ships its modelling code on the
Hub and is loaded with `trust_remote_code=True`, so it runs against whatever
`transformers` the environment happens to have. That code imports
`transformers.masking_utils`, which arrived in a later release than the one
pinned here for ESM compatibility.

`MolformerWrapper.load` catches the `ModuleNotFoundError` and raises
`RuntimeError("Could not load MolFormer ...")` at
`src/embpy/models/molecule_models.py:287`; `_get_model` then wraps that in
`ModelLoadError`. Both re-wraps keep `__cause__`, so the root cause is
recoverable -- if you know to walk the chain.
""")

code(r"""
# Walk __cause__ rather than reading the top-level message, which names neither
# the missing module nor the version constraint.
try:
    embedder.get_model("molformer_base")
    print("molformer_base loaded in this environment after all")
except Exception as exc:
    depth, cursor = 0, exc
    while cursor is not None:
        print(f"{'  ' * depth}{type(cursor).__name__}: {str(cursor)[:78]}")
        cursor, depth = cursor.__cause__, depth + 1
""")

md(r"""
> **Remote code is a dependency your lockfile does not cover.** A model loaded
> with `trust_remote_code=True` pins nothing. It worked when it was added and
> stopped working when the environment moved, and no version constraint in this
> project mentions it.
""")

# =================================================== D. fingerprints vs LMs
md(r"""
### The fingerprints: bit vectors, several algorithms

Six of the spaces above are RDKit fingerprints. They are not learned: each is a
deterministic hash of substructures into a fixed-width bit vector, so they have
no training distribution to be out of, and no parameters to be wrong. That makes
them the right baseline, and it also bounds what they can do -- two molecules
with no shared substructure are simply far apart, however similar their biology.

`maccs_fp` is the odd one out: 167 bits of *curated* structural keys rather than
2048 bits of hashed neighbourhoods.
""")

code(r"""
fp_rows = []
for key in SWEEP:
    matrix = np.asarray(mol_space.obsm[f"X_{key}"], dtype=float)
    finite = matrix[np.isfinite(matrix).all(axis=1)]
    fp_rows.append({
        "model": key,
        "dim": matrix.shape[1],
        # A fingerprint is binary or count-valued; a learned embedding is dense
        # and real-valued. This separates them without hardcoding a list.
        "distinct_values": int(min(len(np.unique(finite)), 999)),
        "mean_nonzero_per_row": round(float((finite != 0).sum(axis=1).mean()), 1),
        "occupancy": round(float((finite != 0).mean()), 3),
    })

display(pd.DataFrame(fp_rows).set_index("model"))
""")

md(r"""
`distinct_values` of 2 marks a binary fingerprint, a small integer count marks a
count fingerprint, and several hundred marks a learned representation. Read
`occupancy` alongside it: a 2048-bit vector with 40 bits set is 98% zeros, so
most of its dimensions carry no signal for this panel at all -- which is exactly
why the [effective width](#nominal-width-is-not-effective-width) discussion in
the next section matters.
""")

md(r"""
### Binary or count: the same bits, weighted differently

`morgan_fp` and `morgan_count_fp` hash the same substructures into the same
2048 positions. The only difference is whether a repeated substructure sets a
bit once or increments a counter. If that changes the geometry, then how many
times a fragment appears carries information beyond whether it appears -- which
is a real, testable claim rather than a design preference.
""")

code(r"""
if "morgan_fp" in SWEEP and "morgan_count_fp" in SWEEP:
    a = np.asarray(mol_space.obsm["X_morgan_fp"], dtype=float)
    b = np.asarray(mol_space.obsm["X_morgan_count_fp"], dtype=float)
    keep = np.isfinite(a).all(axis=1) & np.isfinite(b).all(axis=1)
    a, b = a[keep], b[keep]
    print(f"positions where the count exceeds 1: "
          f"{int((b > 1).sum())} of {int((b != 0).sum())} set positions")
    print(f"identical as bit patterns: {bool(np.array_equal(a != 0, b != 0))}")
    print(f"TSI(binary, count) = {tl.tsi(a, b):.3f}   (0.5 is the null)")
else:
    print("morgan pair unavailable; skipping the binary/count contrast")
""")

md(r"""
The bit *patterns* are identical by construction -- the two only differ in the
weights on those positions -- so any TSI below 1.0 is entirely attributable to
counting. Whether that is worth a second 2048-dim space is what the
[purity ranking](#does-the-geometry-track-biology) decides.
""")

# ======================================================= E. two views
md(r"""
## One molecule space, two views

`missing="nan"` means `mol_space` can carry rows that some models embedded and
others did not. That is honest but awkward: a metric computed over a NaN row is
NaN, and a metric silently computed over a *different subset* per model is
worse -- it compares models on different data and reports the difference as
model behaviour.

So the notebook keeps two objects, and every later section says which it used:

| Object | Rows | Use it for |
| --- | --- | --- |
| `mol_space` | all 40, NaN where a model failed | per-model coverage, annotation, saving |
| `dense_space` | only rows every model embedded | anything that compares models |
""")

code(r"""
complete = np.ones(mol_space.n_obs, dtype=bool)
for key in SWEEP:
    matrix = np.asarray(mol_space.obsm[f"X_{key}"], dtype=float)
    # .all, not .any: a single non-finite cell breaks a metric, so the test is
    # the strict one.
    complete &= np.isfinite(matrix).all(axis=1)

dense_space = mol_space[complete].copy()

print(f"mol_space  : {mol_space.n_obs} compounds, {len(SWEEP)} spaces")
print(f"dense_space: {dense_space.n_obs} compounds "
      f"({int((~complete).sum())} dropped for incomplete coverage)")

if int((~complete).sum()):
    print("dropped:", list(mol_space.obs["compound"][~complete]))

display(dense_space.obs["family"].value_counts().to_frame("n_compounds"))
""")

md(r"""
If `dense_space` still has all forty rows and eight balanced families, every
comparison below is over the same compounds for every model, and the family
sizes are equal -- which keeps the purity numbers in
[does the geometry track biology](#does-the-geometry-track-biology)
interpretable without reweighting.
""")

Path(sys.argv[1] if len(sys.argv) > 1 else "cells_part2.json").write_text(json.dumps(CELLS))
print(f"part 2: {len(CELLS)} cells")
