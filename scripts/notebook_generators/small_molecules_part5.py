"""Part 5 of docs/notebooks/small_molecules.ipynb -- annotation and the clinical record."""
from __future__ import annotations
import json, sys
from pathlib import Path

CELLS: list[tuple[str, str]] = []
def md(s: str) -> None: CELLS.append(("markdown", s.strip("\n")))
def code(s: str) -> None: CELLS.append(("code", s.strip("\n")))

# ==================================================== A. structural layer
md(r"""
## Annotate the molecules

Everything so far treated a compound as a structure and asked whether the models
agree about it. That is only half of what is known about these forty molecules.
Each one also has a physicochemical profile, a clinical history, a set of
targets, a regulatory record and, for two of them, a reason it is no longer
prescribed.

Two entry points, deliberately separate:

* `tl.annotate_molecules` -- local, RDKit only, no network. Physicochemical
  properties computed from the structure.
* `tl.annotate_drug_perturbations` -- the ChEMBL clinical record. Live API,
  roughly a dozen requests per compound.

Start with the local one, because it cannot fail and it establishes the
`mol_*` column convention.
""")

code(r"""
mol_space = tl.annotate_molecules(
    mol_space, column="smiles", sources=["structural"], copy=True,
)

mol_cols = [c for c in mol_space.obs.columns if c.startswith("mol_")]
print(f"{len(mol_cols)} mol_* columns:", mol_cols)

display(
    mol_space.obs.groupby("family", observed=True)[
        ["mol_molecular_weight", "mol_logp", "mol_tpsa", "mol_qed",
         "mol_lipinski_violations"]
    ].mean().round(2)
)
""")

md(r"""
`mol_qed` is quantitative estimate of drug-likeness, and every family scores
respectably because every compound in this panel *is* a drug. That makes QED a
poor discriminator here, which matters later: predicting it from a fingerprint
is close to circular, since QED is itself a function of the same descriptors a
fingerprint encodes. The
[downstream section](#downstream-which-model-predicts-a-measured-property)
comes back to that.
""")

# ================================================ B. the clinical record
md(r"""
## The clinical record

`tl.annotate_drug_perturbations` reaches ChEMBL for what the structure cannot
tell you: what phase a compound reached, what it is indicated for, whether it
was withdrawn and why, where the WHO files it, what it inhibits, and how it is
cleared.

This is the slow cell in the notebook -- forty compounds against a live API.
Every downstream section reads the `drug_*` columns it writes.
""")

code(r"""
# Annotate mol_space (all forty, and what the save cell writes to disk), then
# copy the columns onto dense_space rather than annotating twice -- the API
# calls are the expensive part and both objects hold the same compounds.
mol_space = tl.annotate_drug_perturbations(
    mol_space, column="compound", sources="all", copy=True,
)

drug_cols = [c for c in mol_space.obs.columns if c.startswith("drug_")]
print(f"{len(drug_cols)} drug_* columns")

matched = int(mol_space.obs["drug_in_chembl"].sum())
print(f"matched in ChEMBL: {matched}/{mol_space.n_obs}")

for col in drug_cols:
    dense_space.obs[col] = mol_space.obs.loc[dense_space.obs_names, col].values
for col in [c for c in mol_space.obs.columns if c.startswith("mol_")]:
    dense_space.obs[col] = mol_space.obs.loc[dense_space.obs_names, col].values

ANNOTATED = f"{matched}/{mol_space.n_obs} compounds matched in ChEMBL"
""")

md(r"""
Note the identifier: `column="compound"`, the lowercase generic names from the
panel, not the SMILES. ChEMBL resolves names, trade names, research codes and
ChEMBL IDs as well as structures, and for a screen whose plate map holds
"imatinib" rather than a SMILES string that is the column you have.
""")

# ==================================================== C. development
md(r"""
### Development status

The first question to ask of a compound list is what kind of compounds are on
it. "Approved drug" and "clinical candidate" and "tool compound" are three very
different claims, and ChEMBL distinguishes them.
""")

code(r"""
display(
    mol_space.obs[[
        "compound", "family", "drug_pref_name", "drug_development_phase",
        "drug_first_approval", "drug_availability", "drug_oral",
    ]].head(10).reset_index(drop=True)
)

display(
    mol_space.obs.groupby("drug_development_phase", observed=True)
    .size().to_frame("n_compounds")
)
""")

md(r"""
> **`None` and `-1` are different claims.** A `max_phase` of `None` means a
> preclinical compound with bioactivity data and no clinical record;
> `-1` means ChEMBL has a clinical candidate but cannot assign a phase to it.
> Folding either into "phase 0" would invent a fact. `drug_development_phase`
> renders them as `"preclinical"` and `"clinical phase unknown"` respectively.

Every compound in this panel is approved, which is a property of how the panel
was chosen rather than a finding. A real screen mixes approved drugs with tool
compounds, and this column is what separates them.
""")

# ==================================================== D. indications
md(r"""
### What each drug is for

Indications come as MeSH headings and EFO terms, each carrying the phase reached
*for that indication* -- which is not the same as the compound's overall maximum
phase. A drug approved for one disease and in phase 2 for another has one
`drug_max_phase` and two very different indication rows.
""")

code(r"""
display(
    mol_space.obs[["compound", "drug_n_indications", "drug_top_indication"]]
    .sort_values("drug_n_indications", ascending=False)
    .head(8).reset_index(drop=True)
)

# The full records live in .uns; obs carries only the summary. Read one out to
# see the per-indication phase, which is the field the summary cannot hold.
ann = mol_space.uns["chembl_annotations"].get("imatinib", {})
for ind in (ann.get("indications") or [])[:5]:
    print(f"  phase {ind['max_phase_for_indication']}  "
          f"{ind['mesh_heading']}  ({ind['efo_term']})")
""")

md(r"""
ChEMBL records one row per drug-indication-reference triple, so the same disease
recurs with different phases; `get_indications` collapses those to one entry per
indication keeping the highest phase, which is why the counts here are smaller
than the raw row counts.

`drug_n_indications` is capped at `DEFAULT_INDICATIONS`, and a compound that
hits the cap gets `drug_n_indications_at_limit` set -- a count from a capped
fetch is not a measurement, and regressing on one without checking that flag is
a silent mistake.
""")

# ======================================================= E. safety
md(r"""
### The safety record

This is where the panel earns its two deliberate inclusions. Rofecoxib was
withdrawn worldwide in 2004; celecoxib, same target and same mechanism, was not.
No structural embedding in this notebook can distinguish those two facts,
because the difference is not in the molecules.
""")

code(r"""
safety = mol_space.obs[[
    "compound", "family", "drug_is_withdrawn", "drug_has_black_box_warning",
    "drug_n_warnings", "drug_withdrawn_reason",
]]
display(safety[safety["drug_n_warnings"] > 0].reset_index(drop=True))

print(f"withdrawn        : {list(safety.loc[safety['drug_is_withdrawn'], 'compound'])}")
print(f"black box warning: {list(safety.loc[safety['drug_has_black_box_warning'], 'compound'])}")
""")

code(r"""
# The full warning records carry the class, country, year and references that
# the obs summary flattens away.
for warning in (mol_space.uns["chembl_annotations"]
                .get("rofecoxib", {}).get("warnings") or [])[:4]:
    print(f"  {warning['warning_type']:20s} {str(warning['warning_class']):18s} "
          f"{str(warning['warning_country']):16s} {warning['warning_year']}")
""")

md(r"""
`cardiotoxicity`, `Worldwide`, `2004`. That is the fact a cheminformatics model
cannot recover from structure, and the reason a compound annotation layer is not
optional decoration on an embedding package.
""")

# ========================================================= F. ATC
md(r"""
### Where the WHO puts them

The ATC hierarchy is a *therapeutic* classification: it files a compound by what
it is used for and where it acts, not by what it looks like. That makes it an
independent label to test the embeddings against, and unlike the hand-built
`family` column it was not chosen by whoever wrote this notebook.
""")

code(r"""
atc = pd.crosstab(mol_space.obs["family"], mol_space.obs["drug_atc_level1"])
display(atc)

print("compounds holding several ATC codes:")
for name in mol_space.obs["compound"]:
    classes = mol_space.uns["chembl_annotations"].get(name, {}).get("atc_classes") or []
    if len(classes) > 1:
        print(f"  {name:16s} {[c['code'] for c in classes]}")
""")

md(r"""
The two classifications disagree, and the disagreements are the informative
part. Aspirin holds five ATC codes spanning antithrombotics and analgesics: it
is one molecule with several therapeutic identities, and any single-label
evaluation has to pick one and throw the rest away.
""")

# ==================================== G. mechanism + the parent trap
md(r"""
### Mechanism, and the parent-molecule trap

Mechanism of action is curated rather than computed, and it is the field most
likely to be silently empty for reasons that have nothing to do with the
compound.

> **Drug-level annotation hangs off the parent molecule.** ChEMBL registers
> mechanisms, indications and warnings against the *parent* form, not the salt
> that was dosed. Imatinib is `CHEMBL941`; the marketed drug is imatinib
> mesylate, `CHEMBL1642`, whose parent is `CHEMBL941`. Query the mechanism table
> by `molecule_chembl_id=CHEMBL941` and it returns **nothing at all** -- not an
> error, an empty list, indistinguishable from a compound with no known
> mechanism.

The cell below asks ChEMBL both ways, so the trap is visible rather than
described.
""")

code(r"""
import requests

CHEMBL_API = "https://www.ebi.ac.uk/chembl/api/data"

both_ways = []
for field in ("molecule_chembl_id", "parent_molecule_chembl_id"):
    try:
        resp = requests.get(
            f"{CHEMBL_API}/mechanism.json",
            params={field: "CHEMBL941", "format": "json"}, timeout=30,
        )
        n = resp.json()["page_meta"]["total_count"]
    except Exception as exc:
        n = f"unreachable ({type(exc).__name__})"
    both_ways.append({"filter": field, "mechanisms_returned": n})

display(pd.DataFrame(both_ways).set_index("filter"))
""")

code(r"""
# embpy resolves the hierarchy first, so the annotator returns the parent's
# mechanisms whichever form you name.
from embpy.resources import ChEMBLAnnotator

chembl = ChEMBLAnnotator(rate_limit_delay=0.1)
for mech in chembl.get_mechanisms("imatinib"):
    print(f"  {mech['action_type']:12s} {mech['mechanism']}")
    print(f"               direct={mech['direct_interaction']} "
          f"efficacy={mech['disease_efficacy']} refs={len(mech['references'])}")
""")

md(r"""
Four mechanisms where the naive query returns zero. `drug_moa` prefers the
mechanism ChEMBL marks as responsible for efficacy rather than simply taking the
first row, which is why the summary column names the ABL inhibition rather than
an off-target.
""")

code(r"""
display(
    mol_space.obs[["compound", "drug_n_mechanisms", "drug_action_type", "drug_moa"]]
    .head(12).reset_index(drop=True)
)
""")

# ======================================================= H. targets
md(r"""
### Targets, as genes rather than accessions

A bare `CHEMBL1862` is not an answer to "what does this drug hit". Resolving
targets through the `target`, `target_component` and `protein_classification`
endpoints turns it into `ABL1` / `P00519` / tyrosine protein kinase.

There is a second trap here. A ChEMBL "target" is not always a protein: cell
lines and tissues are targets too, and for imatinib the most potent single
measurement in the whole table belongs to the K562 *cell line*, which outranks
every protein. Sorting by potency without filtering on `target_type` therefore
reports a cell line as the primary target.
""")

code(r"""
profile = chembl.get_target_profile("imatinib", limit=200)

display(pd.DataFrame([
    {
        "target": (e["target_name"] or "")[:34],
        "type": e.get("target_type"),
        "gene": e.get("gene_symbol"),
        "class": e.get("target_class"),
        "n": e["n_measurements"],
        "best_pchembl": e["best_pchembl"],
        "median_pchembl": e["median_pchembl"],
    }
    for e in profile[:8]
]).set_index("target").round(2))
""")

md(r"""
Read the `n` column before believing `best_pchembl`. A single optimistic assay
outranks a target with dozens of consistent measurements, so
`summarize_selectivity` takes `min_measurements` and `rank_by` for exactly this
reason -- and the honest answer changes depending on how you set them.
""")

code(r"""
for label, kwargs in [
    ("best of any single assay", {}),
    (">= 5 measurements", {"min_measurements": 5}),
    (">= 5, ranked by median", {"min_measurements": 5, "rank_by": "median_pchembl"}),
]:
    s = chembl.summarize_selectivity(profile, **kwargs)
    print(f"  {label:26s} -> {str(s['primary_target_gene']):7s} "
          f"n={s['primary_target_n_measurements']} "
          f"best={s['best_pchembl']} window={s['selectivity_window']}")
""")

md(r"""
Three defensible settings, three different primary targets, and none of them is
ABL1 -- the target imatinib is prescribed against. That is not a bug in the
data; it is what happens when you rank curated measurements from heterogeneous
assays by potency alone. The number to report is whichever your analysis can
defend, and the parameters exist so that choice is explicit rather than
inherited from a default.
""")

# ===================================================== I. selectivity
md(r"""
### Selectivity

Across the panel, the pChEMBL gap between the best and second-best protein
target is a crude selectivity proxy: a large window means the compound is much
more potent on one target than the next, a window near zero means it is not
discriminating between them.
""")

code(r"""
display(
    mol_space.obs[[
        "compound", "family", "drug_n_targets", "drug_n_protein_targets",
        "drug_primary_target_gene", "drug_target_class",
        "drug_best_pchembl", "drug_selectivity_window",
    ]].sort_values("drug_best_pchembl", ascending=False)
    .head(10).reset_index(drop=True).round(2)
)

covered = int(mol_space.obs["drug_best_pchembl"].notna().sum())
print(f"drug_best_pchembl available for {covered}/{mol_space.n_obs} compounds")
""")

# ====================================================== J. metabolism
md(r"""
### How they are cleared

Metabolism records the conversion, the enzyme responsible and the metabolite --
which is where prodrugs and toxic metabolites live.
""")

code(r"""
display(
    mol_space.obs[["compound", "drug_n_metabolites", "drug_is_prodrug"]]
    .sort_values("drug_n_metabolites", ascending=False)
    .head(6).reset_index(drop=True)
)

for met in chembl.get_metabolism("aspirin")[:3]:
    print(f"  {met['substrate_name']} -> {met['metabolite_name']} "
          f"via {met['enzyme_name']}")
    print(f"     {str(met['conversion'])[:76]}")
""")

md(r"""
Aspirin to salicylic acid by ester hydrolysis. The metabolite is
pharmacologically active in its own right, so an experiment dosing aspirin is
partly an experiment on salicylic acid -- a fact no structural embedding of
aspirin encodes.
""")

# ========================================== K. does the embedding know?
md(r"""
## Does the embedding know the clinical record?

The [purity section](#does-the-geometry-track-biology) scored the models against
`family`, a label built by hand for this notebook. ATC class and target class are
labels nobody here chose. Re-running the same measurement against them is the
harder test, and the interesting question is whether the ordering of models
survives the change of label.
""")

code(r"""
LABELS = ["family", "drug_atc_level1", "drug_target_class"]
MIN_LABELLED = 16      # below this a purity score is noise, not a measurement

# knn_label_purity always draws. Only the numbers are wanted here -- part 4
# already plots purity against `family` -- so give it one throwaway axis and
# close it afterwards rather than emitting eight figures.
fig_scratch, ax_scratch = plt.subplots(figsize=(2, 2))

purity_rows = []
for key in SWEEP:
    row = {"model": key}
    for label in LABELS:
        values = dense_space.obs[label].astype("string")
        # A label needs at least two populated classes to be scoreable at all,
        # and ChEMBL leaves target_class empty for anything it never resolved.
        usable = values.notna() & (values != "") & (values != "None")
        if usable.sum() < MIN_LABELLED or values[usable].nunique() < 2:
            row[label] = np.nan
            continue
        sub = dense_space[usable.values].copy()
        sub.obs["_label"] = values[usable].values
        scores = pl.knn_label_purity(
            sub, label_key="_label", obsm_key=f"X_{key}", k=K_NN, ax=ax_scratch,
        )
        row[label] = scores.get("__overall__")
    purity_rows.append(row)

plt.close(fig_scratch)
PURITY_CLINICAL = pd.DataFrame(purity_rows).set_index("model")
display(PURITY_CLINICAL.round(3))
""")

md(r"""
Read the columns against each other rather than the absolute values. `family`
was drawn so that congeneric series fall together, so a structural fingerprint
should score highest there. ATC class is partly confounded with structure -- a
congeneric series usually shares a therapeutic class, so recovering ATC is
partly recovering scaffold -- and target class is the least structural of the
three.

A model that holds its ranking across all three is encoding something more
transferable than one that only wins on the label this notebook invented.
""")

code(r"""
color_key = ("drug_development_phase"
             if dense_space.obs["drug_development_phase"].nunique() > 1
             else "drug_atc_level1")
best_space = PURITY_CLINICAL["family"].idxmax()

pl.plot_embedding_space(
    dense_space, obsm_key=f"X_{best_space}", color=color_key,
    method="umap", title=f"{best_space} coloured by {color_key}",
)
""")

md(r"""
Two honest qualifications on that plot.

* **Forty points is not a manifold.** UMAP on forty compounds is a sketch, and
  its apparent clusters are not stable under reseeding. The
  [numbers](#does-the-embedding-know-the-clinical-record) are the evidence; the
  plot is orientation.
* **The colouring may be constant.** Every compound in this panel is approved,
  so if `drug_development_phase` has one value the cell falls back to ATC class
  rather than drawing a single-colour plot and calling it a result.
""")

Path(sys.argv[1] if len(sys.argv) > 1 else "cells_part5.json").write_text(json.dumps(CELLS))
print(f"part 5: {len(CELLS)} cells")
