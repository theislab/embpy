# MCP Tools Reference

The `embpy` MCP server registers nine tools (plus older aliases: `inspect_anndata`,
`map_gene_identifiers`, `list_available_models`, `embed_cells`, `evaluate_scib`,
`evaluate_perturbations`). Every tool reads and writes `.h5ad` files on disk and returns
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

Returns `embedding_keys`, `embedding_slots` (`obsm` / `varm`), `embedding_shapes`,
`metadata_keys` (provenance in `.uns`) and `output_path` (default `<input>_embedded.h5ad`).

---

## `annotate_metadata`

```python
annotate_metadata(
    path, entity_type="auto",       # "gene" | "protein" | "molecule" | "cell_line"
    annotation_types=None, annotation_type=None,
    column=None, sources=None, id_type="auto",
    organism="human", output_path=None,
)
```

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
`service_error` (a request that failed after retries).

- `success`: every entity has every requested type; `partial_success`: some do.
- `error` / `NoAnnotationsFound`: the services answered with nothing.
- `error` / `AnnotationServiceError`: requests failed; `service_errors_sample` shows examples.

Nothing is written when nothing was annotated. Otherwise `.obs` gains `<prefix>_<type>_n`
record counts (`gene_`, `prot_`, `mol_`; molecules with `properties` also get `mol_<property>`
columns) and `.uns["embpy_annotations"]` holds the records as JSON.

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
