# MCP Tools Reference

The `embpy` MCP server registers nine tools. (Older names of six of them, `inspect_anndata`,
`map_gene_identifiers`, `list_available_models`, `embed_cells`, `evaluate_scib` and
`evaluate_perturbations`, can still be imported from `embpy.mcp` as Python functions, but
they are not MCP tools.) Every tool reads and writes `.h5ad` files on disk and returns
compact JSON with a `status` field:

| `status` | Meaning |
|---|---|
| `success` | Everything that was requested was done. |
| `partial_success` | Some inputs succeeded; the counts in the result say which. |
| `error` | Nothing usable was produced, or the request was invalid. `error_type` and `error` explain why. |

| Tool | Purpose |
|---|---|
| [`inspect_data`](#inspect_data) | Dimensions, columns, layers and count type of a dataset |
| [`resolve_identifiers`](#resolve_identifiers) | Gene symbols <-> Ensembl ids; compound names / SMILES -> canonical SMILES |
| [`preprocess_counts`](#preprocess_counts) | QC filtering, normalization, log1p, HVG selection |
| [`list_model_registry`](#list_model_registry) | Models each embedding modality can run; compute device |
| [`compute_embeddings`](#compute_embeddings) | Embeddings for one modality, from a chosen input column |
| [`annotate_metadata`](#annotate_metadata) | GO, pathways, domains, diseases, interactions, expression, properties |
| [`compute_umap`](#compute_umap) | kNN graph and 2D UMAP on an embedding |
| [`benchmark_scib`](#benchmark_scib) | scIB bio-conservation and batch-correction metrics |
| [`benchmark_perturbations`](#benchmark_perturbations) | cell-eval scoring of predicted vs observed perturbations |

---

## `inspect_data`

`inspect_data(path=None, data=None, max_categories=10)`

Reads an `.h5ad` in backed mode (or inspects a list of strings via `data`) and returns
`n_obs`, `n_vars`, `obs_columns`, `obs_summary`, `layers`, `obsm_keys`, `var_sample`,
`gene_id_type_hint`, `expression_nature` (`detected_nature` is `raw_counts`,
`log_normalized`, `scaled_or_centered` or `continuous_normalized`), and candidate label and
batch columns (`recommended_label_key`, `recommended_batch_key`; picked from column names).

---

## `resolve_identifiers`

```python
resolve_identifiers(
    identifiers=None, genes=None, path=None,
    entity_type="auto",            # "auto" | "gene" | "molecule"
    source_id_type="auto",         # "auto" | "symbol" | "ensembl" | "name" | "smiles"
    target_id_type="auto",         # "auto" | "ensembl" | "symbol" | "smiles"
    organism=None,                 # "human" | "mouse" | "auto" | omitted
    column=None, obs_column=None, var_column=None,
    output_path=None, strict=False,
)
```

**Targets.** Genes convert to `ensembl` or `symbol` (`auto` converts each id to the other kind).
Molecules convert to canonical `smiles` (names are looked up on PubChem/CIR; SMILES are
canonicalized with RDKit). Any other target, or a target that does not fit the entity type, is
rejected. `entity_type="auto"` treats strings with SMILES syntax that RDKit parses as molecules
and everything else as genes; pass `entity_type="molecule"` for compound names.

**Species.**

- `organism="human"` / `"mouse"` is always used. Symbols are looked up in that species
  whatever their capitalization (`Egfr` with `"human"` resolves to human EGFR; `notes` records
  the case-insensitive match). An Ensembl id from another species returns
  `error_type: "SpeciesMismatchError"`.
- Omitted: human, except that an Ensembl id's prefix decides its species (`ENSMUSG...` -> mouse).
- `"auto"`: opt-in detection from Ensembl prefixes and symbol case (`TP53` human, `Trp53` mouse).

`organism_used` reports how many identifiers were resolved in each species.

**Lookups.** Cache -> pyensembl (if installed) -> Ensembl REST -> MyGene.info. MyGene is queried
by exact symbol, then exact alias, for the requested species only. An alias shared by several
genes is reported in `ambiguous` with the candidate symbols instead of being resolved to
whichever gene ranks first; resolved aliases are reported in `notes` (`P53` -> `TP53`).

**Status.** Counted per input entry (duplicates and blanks included): all mapped -> `success`,
some -> `partial_success`, none -> `error`. With `strict=True` any unmapped entry, including a
blank one, makes the call an `error` and nothing is written. Entries rejected before lookup
are listed in `invalid` with the reason.

With `path` and `output_path`, the mapped ids are written to `<column>_mapped` (or
`gene_mapped_id` for `var_names`) row by row, keeping order, duplicates and missing values.

---

## `preprocess_counts`

`preprocess_counts(path, output_path=None, preprocessing="standard", min_genes=200, min_cells=3, target_sum=1e4, n_top_genes=2000, select_hvg=True, log_transform=True, scale=False, auto_adjust=False)`

`standard` applies the thresholds exactly (a small dataset can lose every cell);
`auto` / `auto_adjust=True` lowers them for small datasets and reports `adjustments_applied`;
`none` only copies counts to `.layers["counts"]`. Raw counts stay in `.X` and
`.layers["counts"]`; normalized values go to `.layers["log_normalized"]` (and
`.layers["scaled"]`). Default output: `<input>_preprocessed.h5ad`.

---

## `list_model_registry`

`list_model_registry(entity_type=None)`

Returns, per modality, `recommended_models` (curated canonical keys), `available_models`
(every model that modality's execution path can run) and `execution_paths` (the expected
input and the BioEmbedder backend), plus `compute_device`, `runnable_singlecell_models` and
`optional_extras_status`. With `entity_type`, the lists are limited to that modality.

---

## `compute_embeddings`

```python
compute_embeddings(
    path, models="pca", entity_type="single_cell",
    obs_column=None, var_column=None, input_axis="auto",   # where the inputs are
    id_type=None, organism="human", missing="error",
    batch_size=32, output_key=None, output_path=None,
    preprocessing="auto", device="auto",
)
```

Each modality has one execution path, and a model is accepted only if that path can run it:

| `entity_type` | Input | Backend | Model types accepted |
|---|---|---|---|
| `single_cell` | expression matrix | `BioEmbedder.embed_cells` | single-cell models (`pca`, `scvi`, `scgpt`, `geneformer_v2_12L`, `state`, ...) |
| `gene` | gene symbols / Ensembl ids (`var_names` by default) | `embed(entity_type="gene")` | DNA, protein and text models; static tables (`genept`, `gene2vec`, ...) |
| `perturbation` | each cell's perturbed gene, in `obs_column` | `embed(entity_type="gene", is_perturbation=True)` | as `gene`; one vector per cell |
| `dna` | nucleotide sequences | `embed(entity_type="sequence")` | DNA models |
| `protein` | identifiers or sequences (`id_type`) | `embed(entity_type="protein")` | protein models (ESM-2, ESM-1v, ProtT5, Boltz-2) |
| `molecule` | SMILES (or names with `id_type="name"`) | `embed(entity_type="molecule")` | molecule models (fingerprints, ChemBERTa, MoLFormer) |
| `text` | free text | `embed(entity_type="text")` | text models |
| `morphology` | perturbation labels with HPA/JUMP images | `embed(entity_type="perturbation")` | SubCell models |

A mismatch (e.g. `state` under `perturbation`, `boltz2` under `morphology`) or an unknown name
returns `error_type: "UnsupportedModelError"` naming the modality the model belongs to.
Aliases only map to the same model (`esm2_t33_650m` -> `esm2_650M`); applied aliases are
listed in `model_aliases_applied`.

For every modality except `single_cell` the call is the same as
`BioEmbedder().embed(adata, entity_type=<backend>, model=..., obs_column=..., var_column=..., anndata_axis=..., id_type=..., organism=..., missing=...)`
and gives the same matrix. Results are attached to the axis the inputs came from.
`missing="error"` fails when a row gets no embedding and names it; `missing="nan"` fills
those rows with NaN.

**Single-cell preprocessing.** With `preprocessing="auto"` (the default), an input that already
holds `.layers["log_normalized"]` (`preprocess_counts` output) is used as is: no cells or genes
are filtered and `.var["highly_variable"]` is kept, so PCA uses the genes `preprocess_counts`
selected (`preprocessing_note` says so). Raw counts are filtered, normalized and given HVGs as
the models need, with QC/HVG thresholds lowered for small datasets (`preprocessing_adjustments`).
`standard` and `raw` always run their pipeline; `none` never filters or transforms.
`preprocessing_applied` names the pipeline that ran.

Returns `n_obs_input` / `n_obs_output` and `n_vars_input` / `n_vars_output` (cells and genes
before and after; when preprocessing removed some, `warning` says how many and which
thresholds removed them), `embedding_keys`, `embedding_slots` (`obsm` / `varm`),
`embedding_shapes`, `metadata_keys` (provenance in `.uns`) and `output_path` (default
`<input>_embedded.h5ad`).

---

## `annotate_metadata`

```python
annotate_metadata(
    path=None, entity_type="auto",  # "gene" | "protein" | "molecule" | "cell_line"
    annotation_types=None, annotation_type=None,
    column=None, sources=None, id_type="auto",
    organism="human", output_path=None,
    identifiers=None,               # e.g. ["TP53"]: annotate a list, no file needed
)
```

Give either `path` (an `.h5ad` and its `column`) or `identifiers`. For example, "which human
Reactome pathways include TP53?" is
`annotate_metadata(identifiers=["TP53"], entity_type="gene", annotation_types="pathways", sources=["reactome"])`.

| `entity_type` | `annotation_types` (sources) |
|---|---|
| `gene` | `go` (uniprot), `domains` (uniprot, interpro), `pathways` (reactome, kegg, wikipathways), `interactions` (string, dorothea), `diseases` (open_targets, gwas_catalog; human only), `expression` (gtex, hpa; human only) |
| `protein` | `go`, `domains` (uniprot, interpro), `diseases`, `interactions`, `function`, `location` (uniprot) |
| `molecule` | `properties` (rdkit), `interactions` (chembl), `pathways` (kegg), `diseases` (pubchem) |
| `cell_line` | `cell_line` (depmap, via pertpy) |

Every requested type runs for every unique entity in `column`, in `organism`, against the
requested `sources` (default: all sources of the type). An entity counts as annotated for a
type only if that type returned records; the input name echoed back does not count.
`per_type` reports, per type, how many entities were `annotated`, had `no_match`, or hit a
`service_error` (a request that failed after retries), plus `species_mismatch` when it happened:
for UniProt-backed types the entry's species is checked against `organism`, and an entry of
another species (e.g. human P04637 asked for as mouse) is not annotated; `species_mismatches`
names both species.

- `success`: every entity has every requested type; `partial_success`: some do.
- `error` / `NoAnnotationsFound`: the lookups completed and matched nothing; the error starts
  "No pathways found" (or "No GO terms found", ...).
- `error` / `AnnotationServiceError`: requests failed; the error starts "Couldn't reach
  <service>." (e.g. "Couldn't reach Reactome.") and `service_errors_sample` shows examples.
- `error` / `SpeciesMismatch`: the identifiers lead to UniProt entries of another species; the
  error starts "Species mismatch:" and names both species. With some entities annotated, the
  same text is the `warning`.
- `error` / `ModuleNotFoundError`: a package is missing (e.g. RDKit, used by every molecule
  lookup; pertpy for cell lines); `suggestion` gives the install command.
- A source that failed while others answered is listed in `unreachable_services`, with a
  `warning`.

`records_preview` gives, per entity (first 10), type and source, the number of records and the
first ones (ids and names; up to 50 for a single entity).

Reactome pathways come from Reactome's release files (release 98). The first Reactome request
downloads them (about 37 MB) into `~/.cache/embpy/reactome/98` (`$EMBPY_CACHE` moves it) and
later requests reuse them.

KEGG pathways of molecules are looked up in a local KEGG reference: the folder named by
`$EMBPY_KEGG_REFERENCE`, or else `~/.cache/embpy/kegg` (`$EMBPY_CACHE/kegg`). It is not
downloaded automatically, because KEGG's REST API is for academic use only; build it once with
`embpy.resources.molecule.kegg_index.build_kegg_reference("<folder>")`. The reply's
`kegg_reference` gives the folder and KEGG release used. Without a usable reference the reply is
`error` / `KeggReferenceUnavailable` ("The KEGG search could not run ..."), with the build
command in `suggestion`; "No pathways found" means the search ran and matched nothing.

Nothing is written when nothing was annotated, or for `identifiers` without `output_path`
(`output_path` is then null and `new_columns_added` empty).
Otherwise `.uns["embpy_annotations"]["results_json"]` holds the records as JSON (entity, type,
source), and `.obs` gains `<prefix>_<type>_n` counts of the records stored per row (`gene_`,
`prot_`, `mol_`; molecules with `properties` also get `mol_<property>` columns).

Annotating the output again adds to the stored records: other types and sources stay. A type
and source looked up again replace their records for the entities looked up; where that request
failed, the earlier records stay. `provenance_json` gives the organism and column of each
stored entity type, annotation type and source, and `annotation_types` lists every stored type.

Each identifier column keeps its own counts. `<prefix>_<type>_n` belongs to the first column
annotated for that type, and annotating that column again updates it. When another column is
annotated for a type that already has counts from a different column (e.g. `target` after
`gene`), its counts go to `<prefix>_<type>_n_<column>` (`gene_go_n_target`), listed in
`new_columns_added`. `provenance_json["count_columns"]` maps each count column to its identifier
column, entity type and annotation type.

---

## `compute_umap`

`compute_umap(path, embedding_key="X_pca", n_neighbors=15, min_dist=0.5, output_key=None, output_path=None, random_state=42)`

Stores coordinates in `.obsm["X_umap_<name>"]` and returns `umap_key` and `umap_shape`.

---

## `benchmark_scib`

`benchmark_scib(path, embedding_keys, label_key, batch_key=None, output_csv=None)`

Returns `markdown_table`, `rankings` (by `total`) and per-key `metrics`. Needs `embpy[scib]`.

---

## `benchmark_perturbations`

`benchmark_perturbations(path, control_pert, pert_col="perturbation", split_col=None, pred_label=None, ref_label=None, path_pred=None, embedding_keys=None, output_csv=None, output_dir=None, profile="minimal")`

Either `split_col` with two different categories `pred_label` and `ref_label`, or `path_pred`
pointing to a different predictions file. Identical labels, the same file twice, or two files
with the same cells and values are rejected, since they would score predictions against
themselves. Needs `embpy[eval]` (cell-eval).

Each of `embedding_keys` must be in `.obsm` of both the prediction and the reference data,
with the same number of columns. Otherwise nothing is scored: the error names the missing
key and the files it is missing from, and `available_keys` lists the embeddings both have
(for example "Embedding 'X_DOES_NOT_EXIST' is missing from .obsm of the prediction file
(pred.h5ad) and the reference file (real.h5ad). Available embedding: X_good.").

Two cell-eval metrics score an embedding, the two that `cell-eval run --embed-key` uses:
`discrimination_score_l2` and `discrimination_score_cosine`. They run with `profile="anndata"`
or `"full"`: once on gene expression (the plain columns) and once per requested embedding, in
columns `discrimination_score_l2[<key>]` and `discrimination_score_cosine[<key>]`. Every other
score is computed on gene expression (`.X`). `profile="minimal"` (the default), `"vcc"` and
`"de"` score gene expression only; requested embeddings are then listed in `unscored_keys`,
with a `warning` naming the profiles that score them.

Returns:

- `markdown_table` and `results`: one row per perturbation, one column per score.
- `scored_on`: the data each column was computed on, `"gene expression (.X)"` or
  `"embedding <key> (.obsm['<key>'])"`.
- `evaluated_keys`: the embeddings a metric read. A key is listed only if it was scored.
- `unscored_keys`: requested embeddings the profile does not score.
- `output_csv`: the per-perturbation table, if requested.
- `output_dir` and `report_files`: the reports written to `output_dir`, as `cell-eval run`
  names them: `results.csv` (the table in `results`), `agg_results.csv` (statistics per
  score), and `real_de.csv` and `pred_de.csv` (differential expression of each perturbation
  against `control_pert`). Without `output_dir` nothing is written apart from `output_csv`.

An embedding that cell-eval cannot score (for example one with NaN values) is an error
quoting cell-eval's message; cell-eval itself only logs a failing metric and leaves it out.
