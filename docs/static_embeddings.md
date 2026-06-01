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
`metadata/`.

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
  --drop-missing-ids
```

By default, existing local package entries are not replaced. Use
`--overwrite` only when you explicitly want to rebuild them.

Duplicate identifiers and blank identifiers fail by default. For sources with
known blank rows, use `--drop-missing-ids`; for duplicate identifiers where the
first row should win, use `--duplicates first`.

The STRING `.zip`/per-taxon `.h5` artifacts in `data/static_embeddings` are
reported as unsupported by this workflow because they are not single
gene-by-dimension tables. Convert those into a table first, then package the
converted table with the same command.

## Validate

```bash
python -m embpy.scripts.package_static_embeddings validate \
  --package data/static_embedding_package
```

Validation opens each `values.zarr`, checks the matrix shape against metadata,
checks that row identifiers are unique, and reads a sample row back from the
store.

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

tp53 = store.get("TP53")
subset = store.query(["TP53", "MYC"])
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
