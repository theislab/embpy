# Annotation evidence and missing values

Annotations report provider evidence. A recorded label does not establish that
a perturbation activates a pathway, that an investigated indication is approved,
or that a missing flag is biologically false.

## ChEMBL

`ChEMBLAnnotator.get_development_status()` preserves original curated flags in
`raw_flags`. Route, warning, probe and related curated flags now return `True`
for a recorded positive and `None` otherwise. This deliberately changes earlier
zero-to-`False` decoding. A missing maximum phase is unknown, not evidence of a
preclinical programme; a negative unknown-phase code also leaves approval status
unknown. AnnData summary columns preserve these unknown flags as `None`.
The `natural_product` flag denotes the provider's COCONUT
membership annotation; it is not a validated natural-versus-synthetic label.
The default AnnData summary column is now `drug_coconut_member`, replacing
`drug_is_natural_product`. Existing user-owned columns are not automatically
renamed; callers should update column selections when regenerating annotations.

`get_activities(identifier, limit=None)` retrieves the complete activity table.
Apply species, relation, quality, assay and potency filters to this complete
table before inferring target coverage. With a finite limit,
`activity_fetch_metadata[parent]` reports returned rows, a truncation flag and
the total when known. A truncated fetch has an unknown total and a lower bound.
Both REST and SQLite implement the same contract. `get_target_profile()` and
the target branch of `annotate()` fetch all activities by default.

Target profiles preserve all `gene_symbols`, include the number of distinct
assays, and resolve all targets by default. A singular `gene_symbol` is absent
for multicomponent targets. Unknown target types do not qualify as proteins in
selectivity summaries. Target profile membership alone is an activity
association, not an established mechanism of action.

Parent-family queries reproduce ChEMBL's hierarchy, which may include isotope
variants as well as salts. Call `get_parent_family_audit(identifier)` before
transferring family evidence to an exact compound. It compares neutralized
fragment InChIKeys while retaining isotope and stereochemistry. Incompatible or
unresolved members make `primary_annotation_eligible` false; raw family data
remain available for review.

## Structure matches

ChEBI and KEGG exact matches carry `primary_annotation_eligible=True`.
Connectivity-only matches carry false and remain separately identifiable.
Matching only the connectivity block does not validate stereochemistry.
ChEBI roles are database-asserted `has_role` relationships and can include
investigational or repurposing uses.

PubChem SMILES resolution records the query and returned InChIKeys. A returned
CID can represent a standardized or stereochemically different structure;
only an exact key match is eligible for exact annotation transfer. Failure to
verify the key leaves eligibility false rather than inferring an exact match.

## Cell lines

Cellosaurus `site` is exposed explicitly as `sampling_site` together with
`sampling_site_type`. It no longer populates `tissue`, since a metastatic
sampling site is not tissue of origin. The combined annotation uses DepMap
lineage or Cell Model Passports tissue for that field. Source records remain
available so conflicting classifications can be inspected.

Comparisons within an annotation stratum evaluate performance on its recorded
compound panel. Differences between tissue, disease or cell-line strata may
reflect panel composition and do not establish a causal tissue effect.
