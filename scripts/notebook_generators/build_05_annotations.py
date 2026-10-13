"""Build docs/notebooks/05_annotate_entities.ipynb.

A numbered tutorial, not a deep dive: it introduces the annotation surface and
what the annotations are *for*, in the voice of notebooks 01-04 (em dashes,
second person, short cells).
"""
from __future__ import annotations
import json, sys
from pathlib import Path

CELLS: list[tuple[str, str]] = []
def md(s: str) -> None: CELLS.append(("markdown", s.strip("\n")))
def code(s: str) -> None: CELLS.append(("code", s.strip("\n")))

# ------------------------------------------------------------------ intro
md(r"""
# What else does embpy know about your entities?

**What you'll learn.** An embedding tells you where an entity sits relative to
others. It does not tell you what the entity *is*. embpy's annotation layer
fills that in from public databases — physicochemical properties, clinical
history, pathways, targets, disease links — and writes it into the same AnnData
next to the vectors.

That matters for two reasons beyond curiosity:

- **An unlabelled embedding cannot be evaluated.** Annotations give you labels
  nobody in your analysis chose, which is what turns "the clusters look
  sensible" into a number.
- **Some facts are not in the data.** No structural model can tell you a drug
  was withdrawn for cardiotoxicity in 2004. That is a record, not a signal.

| Section | Question |
| --- | --- |
| [Molecules](#molecules-structure-first) | what is this compound, physically? |
| [The clinical record](#the-clinical-record) | is it a drug, what for, and is it safe? |
| [Genes](#genes) | what pathways and diseases is this gene in? |
| [Proteins](#proteins) | what does this protein do, and where? |
| [Putting them to work](#putting-annotations-to-work) | how do annotations score an embedding? |

**Prerequisites.** [Embed with any model](01_embed_any_model.ipynb) and
[Where embeddings live](02_output_contract.ipynb).
""")

md(r"""
## Requirements

The annotation layer is **network-bound, not compute-bound**. There is no model
to download and no GPU involved; every function here queries a public API, so
what it needs is egress and patience.

The core install covers everything in this notebook:

```bash
uv pip install "embpy[cpu]"
```

Two functions reach for optional extras:

| Function | Needs | Install |
| --- | --- | --- |
| `tl.annotate_molecules` | RDKit (already in core) | — |
| `tl.annotate_drug_perturbations` | nothing beyond core | — |
| `tl.annotate_gene_perturbations` | nothing beyond core | — |
| `tl.annotate_proteins` | nothing beyond core | — |
| `tl.annotate_drugs`, `annotate_cell_lines`, `annotate_drug_response` | pertpy | `uv pip install "embpy[pertpy]"` |

Rough cost, so the slow cells are not a surprise: the ChEMBL record is about a
dozen requests per compound, and the gene annotator queries seven separate
services. Both are polite by default (a small delay between calls), so a
forty-compound panel takes minutes rather than seconds.
""")

code(r"""
import os

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import anndata as ad
import numpy as np
import pandas as pd
from IPython.display import display

from embpy import tl
""")

# ------------------------------------------------------------- molecules
md(r"""
## Molecules: structure first

`tl.annotate_molecules` is the local half — RDKit only, no network. It reads
identifiers from `.obs[column]` and writes `mol_*` columns: molecular weight,
LogP, TPSA, hydrogen-bond donors and acceptors, rotatable bonds, QED, Lipinski
violations, Fsp3 and heavy-atom count.

Note the identifiers below are *names*, not SMILES. embpy resolves them.
""")

code(r"""
DRUGS = ["aspirin", "imatinib", "rofecoxib", "atorvastatin", "diazepam"]

mols = ad.AnnData(
    X=np.zeros((len(DRUGS), 1), dtype=np.float32),
    obs=pd.DataFrame({"compound": DRUGS}, index=DRUGS),
)
mols = tl.annotate_molecules(mols, column="compound", sources=["structural"])

mol_cols = [c for c in mols.obs.columns if c.startswith("mol_")]
print(f"{len(mol_cols)} mol_* columns")
display(mols.obs[["mol_molecular_weight", "mol_logp", "mol_tpsa",
                  "mol_qed", "mol_lipinski_violations"]].round(2))
""")

md(r"""
`sources=["structural"]` keeps this cell offline-fast. Dropping it queries ChEBI
roles, KEGG pathways, PubChem cross-references and disease associations too —
useful, but each is a network round trip per compound.
""")

# ------------------------------------------------------- clinical record
md(r"""
## The clinical record

`tl.annotate_drug_perturbations` is the ChEMBL half, and it answers the
questions a structure cannot: what phase did this compound reach, what is it
given for, was it withdrawn and why, where does the WHO file it, what does it
inhibit, and how is it cleared.

It writes 38 `drug_*` columns. The interesting ones are below; the complete
records — every indication, every warning with its references, the full target
profile — land in `.uns["chembl_annotations"]`.
""")

code(r"""
mols = tl.annotate_drug_perturbations(mols, column="compound", sources="all")

drug_cols = [c for c in mols.obs.columns if c.startswith("drug_")]
print(f"{len(drug_cols)} drug_* columns, "
      f"{int(mols.obs['drug_in_chembl'].sum())}/{mols.n_obs} matched in ChEMBL")

display(mols.obs[[
    "drug_pref_name", "drug_development_phase", "drug_first_approval",
    "drug_atc_level1", "drug_moa",
]])
""")

md(r"""
### The part no model can infer

Two of these compounds are chemically unremarkable and clinically very
different. Rofecoxib and celecoxib inhibit the same enzyme; one was withdrawn
worldwide and the other was not. That difference lives in a database, not in a
molecule.
""")

code(r"""
display(mols.obs[[
    "drug_is_withdrawn", "drug_has_black_box_warning",
    "drug_n_warnings", "drug_withdrawn_reason",
]])

# The obs columns are a summary; the full record keeps the class, country,
# year and the references behind each warning.
for warning in (mols.uns["chembl_annotations"]["rofecoxib"]["warnings"] or [])[:3]:
    print(f"  {warning['warning_type']:20s} {str(warning['warning_class']):16s} "
          f"{str(warning['warning_country']):12s} {warning['warning_year']}")
""")

md(r"""
### Targets as genes, not accessions

A `CHEMBL1862` is not an answer to "what does this drug hit". The target
profile resolves each target to a gene symbol, a UniProt accession and a
protein family, and summarises the measurements behind it.
""")

code(r"""
from embpy.resources import ChEMBLAnnotator

chembl = ChEMBLAnnotator(rate_limit_delay=0.1)
profile = chembl.get_target_profile("imatinib", limit=150)

display(pd.DataFrame([
    {"target": (e["target_name"] or "")[:32], "type": e.get("target_type"),
     "gene": e.get("gene_symbol"), "class": e.get("target_class"),
     "n": e["n_measurements"], "best": e["best_pchembl"],
     "median": e["median_pchembl"]}
    for e in profile[:6]
]).set_index("target").round(2))
""")

md(r"""
> **Read `n` before believing `best`.** A ChEMBL "target" can be a cell line
> rather than a protein, and a single optimistic assay outranks a target with
> dozens of consistent measurements. `summarize_selectivity` filters the first
> problem and takes `min_measurements` / `rank_by` for the second — the answer
> genuinely changes with those settings, so pick them deliberately.
""")

code(r"""
for label, kwargs in [
    ("default", {}),
    ("min_measurements=5", {"min_measurements": 5}),
    ("min 5, by median", {"min_measurements": 5, "rank_by": "median_pchembl"}),
]:
    s = chembl.summarize_selectivity(profile, **kwargs)
    print(f"  {label:20s} -> {str(s['primary_target_gene']):8s} "
          f"n={s['primary_target_n_measurements']}")
""")

# ------------------------------------------------------------------ genes
md(r"""
## Genes

`tl.annotate_gene_perturbations` queries seven services — MyGene for pathways,
GTEx for tissue expression, HPA for localisation, STRING for interaction
partners, DoRothEA for transcription factors, Open Targets for disease links and
the GWAS Catalog for associations — and writes `gene_*` summary columns.
""")

code(r"""
GENES = ["TP53", "EGFR", "MYC"]

genes = ad.AnnData(
    X=np.zeros((len(GENES), 1), dtype=np.float32),
    obs=pd.DataFrame({"symbol": GENES}, index=GENES),
)
genes = tl.annotate_gene_perturbations(genes, column="symbol",
                                       sources=["pathways", "interactions"])

display(genes.obs[[c for c in genes.obs.columns if c.startswith("gene_")]])
""")

md(r"""
> **A capped count is not a measurement.** `gene_n_ppi_partners` counts what was
> *fetched*, and the fetch is capped, so for a well-studied gene it reports the
> cap rather than the truth. The `*_at_limit` companion columns say which rows
> are saturated, and `.uns["gene_annotation_limits"]` records the caps.
> Regressing on a capped count without checking the flag is a silent mistake.
""")

code(r"""
limits = genes.uns.get("gene_annotation_limits", {})
print("caps:", dict(limits))
display(genes.obs[[c for c in genes.obs.columns if c.endswith("_at_limit")]])
""")

# --------------------------------------------------------------- proteins
md(r"""
## Proteins

`tl.annotate_proteins` reads UniProt and InterPro: what the protein does, where
it sits, its domains and PTMs, disease involvement, interaction cross-references,
isoforms, and whether the entry is reviewed (Swiss-Prot) or not (TrEMBL).
""")

code(r"""
prots = ad.AnnData(
    X=np.zeros((len(GENES), 1), dtype=np.float32),
    obs=pd.DataFrame({"symbol": GENES}, index=GENES),
)
prots = tl.annotate_proteins(prots, column="symbol",
                             sources=["function", "domains"])

display(prots.obs[[c for c in prots.obs.columns if c.startswith("prot_")]])
""")

md(r"""
`prot_reviewed` is worth a glance before trusting the rest: an unreviewed TrEMBL
entry is a computational prediction, and its domain and PTM counts carry much
less weight than a curated Swiss-Prot record's.
""")

# ------------------------------------------------------ putting to work
md(r"""
## Putting annotations to work

The payoff is not the table — it is that an annotation is a label your analysis
did not choose. Embed the compounds, then score whether the embedding recovers a
label that came from ChEMBL rather than from you.
""")

code(r"""
from embpy import BioEmbedder, pl

embedder = BioEmbedder(device="auto", organism="human")

mols.obs["smiles"] = mols.obs["mol_canonical_smiles"]
mols.obs_names = pd.Index(mols.obs["smiles"], name="canonical_smiles")

mols = embedder.embed(
    mols, entity_type="molecule", id_type="smiles", obs_column="smiles",
    model="morgan_fp", output="anndata", attach_to="obs", key="X_morgan",
)
print("embedding:", mols.obsm["X_morgan"].shape)
""")

code(r"""
# A label with at least two populated classes is the minimum for any purity
# score; with five compounds this is a demonstration of the mechanism, not a
# result to quote.
label = "drug_atc_level1"
usable = mols.obs[label].astype("string")
print(f"{label}: {usable.nunique()} distinct classes over {mols.n_obs} compounds")
display(usable.value_counts().to_frame("n_compounds"))
""")

md(r"""
With a real panel you would hand that column to `pl.knn_label_purity` or
`tl.compute_scib_metrics` and get a score per model. The
[small molecules](small_molecules.ipynb) notebook does exactly that across eight
embedding spaces and three annotation-derived labels — and finds, unsurprisingly,
that a structural fingerprint recovers a therapeutic classification only as far
as therapy correlates with scaffold.

## What is stored where

| Where | What |
| --- | --- |
| `.obs["mol_*"]` | physicochemical scalars |
| `.obs["drug_*"]` | the ChEMBL clinical summary, 38 columns |
| `.obs["gene_*"]`, `.obs["prot_*"]` | gene and protein summaries |
| `.obs["*_at_limit"]` | which counts hit a fetch cap |
| `.uns["molecule_annotations"]` | full per-molecule records |
| `.uns["chembl_annotations"]` | full ChEMBL records — every indication, warning, mechanism, target |
| `.uns["chembl_annotation_limits"]` | the caps themselves |

The summary columns are for filtering and colouring; the `.uns` records are for
reading. Anything the summary flattens away — a warning's references, an
indication's per-indication phase — is in `.uns`.

## Takeaway

- Annotation is **network-bound**: no weights, no GPU, just APIs and patience.
- Summary columns land in `.obs`, full records in `.uns`, and capped counts are
  flagged rather than left to look like measurements.
- The reason to bother is evaluation. An annotation is a label you did not
  choose, and that is the only kind worth scoring an embedding against.

**Next:** [Comparing embeddings](03_compare_embeddings.ipynb) for the metrics,
or [Small molecules](small_molecules.ipynb) for the same annotation layer used in
anger across a forty-compound panel.
""")

out = Path(sys.argv[1] if len(sys.argv) > 1 else "05_annotate_entities.ipynb")
cells = []
for kind, source in CELLS:
    cell = {"id": f"05_annotate_entities-{len(cells):02d}", "cell_type": kind,
            "metadata": {}, "source": source.splitlines(keepends=True)}
    if kind == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    cells.append(cell)

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3 (ipykernel)",
                       "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12.0",
                          "file_extension": ".py", "mimetype": "text/x-python",
                          "nbconvert_exporter": "python",
                          "pygments_lexer": "ipython3"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
out.write_text(json.dumps(nb, indent=1) + "\n")
n_code = sum(1 for c in cells if c["cell_type"] == "code")
print(f"wrote {out}: {len(cells)} cells ({n_code} code, {len(cells) - n_code} markdown)")
