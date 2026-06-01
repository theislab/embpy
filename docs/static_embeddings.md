# Static Embedding Packages

embpy can package local static gene embeddings into a Hugging Face-friendly
folder layout:

```text
manifest.json
embeddings/<model_key>/
  values.zarr/
  metadata/
    index.parquet
    index.csv
    metadata.json
    uns.json
```

The dense embedding values live in `values.zarr`. Row identifiers, source
provenance, shape, identifier type, and AnnData-like `.uns` metadata live under
`metadata/`. Gene row identifiers are harmonized to Ensembl IDs by default;
source IDs and gene symbols are preserved as extra columns in
`metadata/index.parquet` when they differ from the canonical ID. Each embedding
also records `species`, `taxonomy_id`, and `species_key` metadata; when no
species is provided, embpy defaults to human (`taxonomy_id=9606`).

## Prepare Locally

Start with a dry run:

```bash
python -m embpy.scripts.package_static_embeddings prepare \
  --input data/static_embeddings \
  --output data/static_embedding_package \
  --dry-run
```

Write the package after reviewing the plan:

```bash
python -m embpy.scripts.package_static_embeddings prepare \
  --input data/static_embeddings \
  --output data/static_embedding_package \
  --drop-missing-ids \
  --nan-policy drop-rows
```

By default, existing local package entries are not replaced. Use
`--overwrite` only when you explicitly want to rebuild them.

By default, `prepare` writes gene rows with `id_type="ensembl_id"` using
embpy's `GeneResolver`. This is equivalent to passing
`--target-id-type ensembl_id --on-unresolved-ids drop`; each source uses its
species metadata for resolver calls, falling back to human when species is not
provided.
Unresolved symbols are reported and dropped so the package stays consistently
Ensembl-keyed. Use `--on-unresolved-ids error` for a stricter run, or
`--no-harmonize-ids` only for debugging/source inspection.

Duplicate identifiers, blank identifiers, and NaN/Inf embedding values fail by
default. For sources with known blank rows, use `--drop-missing-ids`; for
duplicate identifiers where the first row should win, use `--duplicates first`.
For numeric missingness, prefer `--nan-policy drop-rows` to discard affected
identifiers without fabricating values. Use `--nan-policy fill-zero` only when a
zero fill is scientifically appropriate for that source.

## Custom Or Messy Files

For a one-off table with an explicit gene column and non-embedding metadata
columns:

```bash
python -m embpy.scripts.package_static_embeddings prepare \
  --source-file my_embedding.csv \
  --source-key my_model \
  --source-id-type symbol \
  --species human \
  --taxonomy-id 9606 \
  --id-column gene_symbol \
  --metadata-columns description source \
  --output data/static_embedding_package \
  --nan-policy drop-rows
```

For reproducible submissions, prefer a JSON source config:

```json
{
  "sources": [
    {
      "key": "my_model",
      "path": "my_embedding.csv",
      "id_column": "gene_symbol",
      "id_type": "symbol",
      "species": "human",
      "taxonomy_id": "9606",
      "sep": ",",
      "metadata_columns": ["description", "source"],
      "embedding_columns": ["dim_0", "dim_1"]
    }
  ]
}
```

Then run:

```bash
python -m embpy.scripts.package_static_embeddings prepare \
  --source-config sources.json \
  --output data/static_embedding_package
```

`species` and `taxonomy_id` may also live inside the source `metadata` object.
This is useful for user-submitted config files because the generated package
will expose a stable `species_key` such as `human_9606` or `mouse_10090` in the
manifest, `metadata/metadata.json`, and `metadata/uns.json`.

Supported inputs are CSV, TSV, TXT, compressed CSV/TSV/TXT, Parquet, and HDF5
files with one 2D embedding dataset plus one aligned row-ID dataset. If a file
has genes as columns instead of rows, pass `--transpose`.
If a text column is accidentally included in the embedding matrix, the parser
reports the first offending column/value and suggests `--metadata-columns` or
`--embedding-columns`.

For HDF5 files such as STRING precomputed embeddings:

```bash
python -m embpy.scripts.package_static_embeddings prepare \
  --source-file precomputed_embeddings_string/node2vec/node2vec/9606.h5 \
  --source-key string_node2vec_9606 \
  --entity-type protein \
  --source-id-type string_protein_id \
  --h5-id-dataset proteins \
  --h5-embedding-dataset embeddings \
  --output data/static_embedding_package
```

The known human STRING files at species `9606` are discovered automatically
when present under `data/static_embeddings/precomputed_embeddings_string/`.
Additional STRING species are never swept in implicitly; request them by NCBI
taxonomy ID:

```bash
python -m embpy.scripts.package_static_embeddings prepare \
  --input data/static_embeddings \
  --output data/static_embedding_package \
  --string-species 10090 10116
```

The STRING `.zip` and per-species `.h5` folders in `data/static_embeddings` are
reported as skipped collections because they contain many species. Package one
chosen species HDF5 file at a time, or rely on the built-in human `9606` specs.

## Validate

```bash
python -m embpy.scripts.package_static_embeddings validate \
  --package data/static_embedding_package
```

Validation opens each `values.zarr`, checks the matrix shape against metadata,
checks that row identifiers are unique, and reads a sample row back from the
store.

## Dataset Card

Generate a Hugging Face dataset card from the prepared package manifest:

```bash
python -m embpy.scripts.package_static_embeddings card \
  --package data/static_embedding_package \
  --repo-id your-org/Embpy_Data
```

This writes `README.md` at the package root, which Hugging Face uses as the
dataset card. Existing cards are not replaced unless `--overwrite` is set. To
preview or copy the markdown without writing a file, add `--print`.

## Upload

Upload is always a separate explicit step. Without `--execute`, the upload
command is a dry run and performs no network upload:

```bash
python -m embpy.scripts.package_static_embeddings upload \
  --package data/static_embedding_package \
  --repo-id your-org/Embpy_Data
```

To upload after local validation:

```bash
python -m embpy.scripts.package_static_embeddings upload \
  --package data/static_embedding_package \
  --repo-id your-org/Embpy_Data \
  --execute
```

Remote paths are not overwritten unless `--allow-overwrite` is set.

## Query Locally

```python
from embpy import load_static_embedding_package

store = load_static_embedding_package(
    "data/static_embedding_package",
    key="genept_scaled",
)

tp53 = store.get("ENSG00000141510")
subset = store.query(["ENSG00000141510", "ENSG00000136997"])

# If the source was symbol-keyed, preserved aliases can be queried explicitly.
tp53_by_symbol = store.get("TP53", id_type="symbol")
```

Missing identifiers raise by default. Use `missing="drop"` or `missing="nan"`
when a partial result is acceptable.

## Query From Hugging Face

The same package layout is understood by `HFHandler`:

```python
from embpy.pp import HFHandler

hf = HFHandler("your-org/Embpy_Data")
table = hf.download_embedding("genept_scaled")
```

For gene lookups through the high-level embedder:

```python
from embpy import BioEmbedder

emb = BioEmbedder(device="cpu")
result = emb.embed(
    ["TP53", "MYC"],
    entity_type="gene",
    model="genept_scaled",
    embedding_source="hf",
    hf_repo_id="your-org/Embpy_Data",
    output="table",
)
```

## Layout Choice

The recommended layout is one Hugging Face dataset repository with one folder
per embedding model under `embeddings/`. This keeps discovery and versioning in
one manifest, avoids repository sprawl, and matches embpy's existing
`HFHandler(repo_id).download_embedding(model_key)` API. Separate repos are best
reserved for unusually large or independently versioned collections.
