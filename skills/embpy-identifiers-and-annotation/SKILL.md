---
name: embpy-identifiers-and-annotation
description: Convert biological identifiers and annotate entities with the embpy MCP tools - gene symbols to Ensembl ids and back in human or mouse, compound names or SMILES to canonical SMILES, and GO terms, pathways, protein domains, disease associations, interactions or expression for genes, proteins and molecules in an AnnData column. Use when the user needs to map or harmonize gene/compound identifiers, check which identifiers fail to map, or enrich an .h5ad dataset with GO, Reactome/KEGG pathways, InterPro domains, Open Targets diseases, STRING interactions or RDKit properties.
---

# Identifier resolution and annotation with embpy

Use the `embpy` MCP server's `resolve_identifiers` and `annotate_metadata` tools.

## Resolving identifiers

```python
resolve_identifiers(identifiers=["TP53", "BRCA1", "P53"], target_id_type="ensembl", organism="human")
```

```python
resolve_identifiers(path="data/screen.h5ad", var_column="gene_symbol", target_id_type="ensembl", organism="mouse", output_path="data/screen_mapped.h5ad")
```

```python
resolve_identifiers(identifiers=["aspirin", "imatinib"], entity_type="molecule", target_id_type="smiles")
```

**Species.** Pass `organism` whenever you know it; the tool always respects it.
- `organism="human"` or `"mouse"`: symbols are looked up in that species whatever their
  capitalization (`Egfr` with `human` gives human EGFR, and `notes` says it matched
  case-insensitively). An Ensembl id from another species (e.g. `ENSMUSG...` with `human`)
  returns `error_type: "SpeciesMismatchError"`; ask the user which species they meant.
- `organism` omitted: human, except that an Ensembl id's prefix decides its species.
- `organism="auto"`: only when the user has a mixed list and wants the tool to guess. Species
  then come from Ensembl prefixes and symbol case (`TP53` human, `Trp53` mouse); report
  `organism_used` back to the user.

**Targets.** Genes: `target_id_type` is `ensembl` or `symbol` (`auto` converts each id to the
other kind). Molecules: `smiles`. Anything else is rejected. Use `entity_type="molecule"` for
compound names; `auto` only recognizes SMILES strings as molecules.

**Reading the result.**
- `status`: `success` = every entry mapped; `partial_success` = some did; `error` = none did, or
  the request was invalid. Counts are per entry, so duplicates and blanks count (`blank_count`).
- `strict=True` turns any unmapped entry (including blanks) into an error and writes nothing.
  Use it before a step that needs every gene mapped.
- `ambiguous`: the alias names several genes (e.g. `AIM1`); show the candidates and let the
  user choose. Do not pick one.
- `notes`: aliases resolved to the approved symbol (`P53` -> `TP53`) and case-insensitive
  matches. Mention them.
- `invalid`: entries rejected before lookup, with the reason (wrong format, already the target type).
- `warning` / `action_required`: Ensembl or MyGene could not be reached. Pass the install hint on;
  do not claim the unmapped genes do not exist.
- With `path` + `output_path`, the mapped ids are written to `<column>_mapped` in the same rows.

## Annotating entities

```python
annotate_metadata(path="data/screen.h5ad", entity_type="gene", annotation_types=["go", "pathways"], column="target_gene", organism="human")
```

```python
annotate_metadata(path="data/screen.h5ad", entity_type="gene", annotation_types="pathways", sources=["reactome"], column="target_gene")
```

A question about a few entities needs no file: pass them as `identifiers` (and set `entity_type`).
"Which human Reactome pathways include TP53?":

```python
annotate_metadata(identifiers=["TP53"], entity_type="gene", annotation_types="pathways", sources=["reactome"])
```

```python
annotate_metadata(path="data/compounds.h5ad", entity_type="molecule", annotation_types=["properties", "interactions"], column="smiles")
```

Annotation types and their sources per entity type:

| `entity_type` | `annotation_types` (sources) |
|---|---|
| `gene` | `go` (uniprot), `domains` (uniprot, interpro), `pathways` (reactome, kegg, wikipathways), `interactions` (string, dorothea), `diseases` (open_targets, gwas_catalog; human only), `expression` (gtex, hpa; human only) |
| `protein` | `go`, `domains` (uniprot, interpro), `diseases`, `interactions`, `function`, `location` (uniprot) |
| `molecule` | `properties` (rdkit), `interactions` (chembl), `pathways` (kegg), `diseases` (pubchem) |
| `cell_line` | `cell_line` (depmap, needs pertpy) |

- Every requested type runs for every unique entity. `sources` narrows the databases; each
  source must serve a requested type.
- Pass `column` explicitly. The tool refuses to guess when no column name matches the entity type.
- Gene annotations work best on approved symbols or Ensembl ids, so resolve aliases first with
  `resolve_identifiers`.

- Reactome pathways come from Reactome's own release files. The first Reactome request
  downloads them (about 37 MB) into `~/.cache/embpy/reactome` (or `$EMBPY_CACHE/reactome`);
  later requests and sessions reuse them. Nothing has to be downloaded by hand.

**Reading the result.** `per_type` gives, for each type, how many entities were `annotated`,
had `no_match`, or hit a `service_error`. An entity counts as annotated only when that type
returned records. `records_preview` holds, per entity, type and source, the number of records
(`n`) and the first ones (`id`, `name`, ...): answer "which pathways ..." from it.
- `success`: every entity has every requested type. `partial_success`: some do; report `per_type`.
- `error` with `NoAnnotationsFound` (error starts "No pathways found", "No GO terms found", ...):
  the lookups completed and matched nothing. Say so; do not describe it as a failure of the service.
- `error` with `AnnotationServiceError`: the error starts "Couldn't reach <service>." (e.g.
  "Couldn't reach Reactome."). Say that and suggest retrying later; do not claim the genes have
  no annotations.
- `warning` and `unreachable_services` with a success: one source failed while others answered;
  mention which.
- Nothing is written when nothing was annotated, or with `identifiers` unless `output_path` is
  given. Otherwise `.obs` gains `<prefix>_<type>_n` record counts (molecules with `properties`
  also get `mol_<property>` columns) and `.uns["embpy_annotations"]` holds the full records as JSON.
- To add annotation types or sources, call the tool again on its output: earlier records stay.
  Looking up a type and source again replaces its records for those entities (a failed request
  keeps the earlier ones).
