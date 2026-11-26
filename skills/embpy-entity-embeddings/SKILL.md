---
name: embpy-entity-embeddings
description: Compute embeddings for genes, perturbation labels, DNA sequences, proteins, small molecules (SMILES or names), free text or microscopy morphology with the embpy MCP tools, reading the inputs from a chosen AnnData column and checking that each model fits the modality. Use when the user wants gene, protein, molecule, DNA, text or perturbation embeddings (e.g. ESM-2, Morgan fingerprints, ChemBERTa, GenePT, HyenaDNA, MiniLM, SubCell) stored in an .h5ad file, or asks which embedding model to use for an entity type.
---

# Entity embeddings with embpy

Use the `embpy` MCP server's `list_model_registry` and `compute_embeddings` tools. For cell
embeddings from an expression matrix, follow the `embpy-single-cell-workflow` skill instead.

## 1. Pick the modality from the input

`entity_type` selects one execution path. Each path accepts only some models:

| `entity_type` | Input | Where the input is | Result |
|---|---|---|---|
| `gene` | gene symbols or Ensembl ids | `var_names` by default, or `obs_column` / `var_column` | one vector per gene (`.varm` for var inputs) |
| `perturbation` | each cell's perturbed gene | `obs_column` (required) | one vector per cell in `.obsm` |
| `dna` | raw nucleotide sequences | `obs_column` or `var_column` | one vector per sequence |
| `protein` | gene/protein ids or amino-acid sequences | `obs_column` / `var_column`; set `id_type` | one vector per protein |
| `molecule` | SMILES, or names with `id_type="name"` | usually `obs_column="smiles"` | one vector per molecule |
| `text` | free text | `obs_column` or `var_column` | one vector per text |
| `morphology` | perturbation labels whose HPA/JUMP images are embedded | `obs_column` | one vector per label |

When the identifiers live in a column, always pass that column (`obs_column` or `var_column`).
Without one, the tool reads `obs_names` / `var_names` (`input_axis` chooses between them), and
those are often sample ids rather than the entities. Run `inspect_data` first if you do not
know the column names.

## 2. Pick a model the path can run

```python
list_model_registry(entity_type="molecule")
```

Choose from `recommended_models` (or `available_models`) for that modality. Do not use model
names from memory: a name the registry does not have is rejected. `execution_paths` shows what
input each modality expects.

Good defaults: `morgan_fp` (molecules, CPU, no download), `esm2_8M` / `esm2_35M` (proteins),
`gene2vec` / `genept` (gene and perturbation lookup tables), `hyenadna_tiny_1k` (short DNA),
`minilm_l6_v2` (text). Larger models such as `esm2_650M`, `prot_t5_xl` or `enformer_human_rough`
need a GPU in practice; check `compute_device` before suggesting them.

## 3. Run it

```python
compute_embeddings(path="data/compounds.h5ad", models="morgan_fp", entity_type="molecule", obs_column="smiles")
```

```python
compute_embeddings(path="data/screen.h5ad", models="gene2vec", entity_type="perturbation", obs_column="target_gene")
```

```python
compute_embeddings(path="data/proteins.h5ad", models="esm2_8M", entity_type="protein", obs_column="gene_symbol", id_type="symbol", organism="human")
```

```python
compute_embeddings(path="data/genes.h5ad", models="gene2vec", entity_type="gene")
```

- `organism` (`human` or `mouse`) is used to look up gene and protein identifiers.
- `output_key` renames the result; only allowed with a single model.
- The result lists `embedding_keys`, `embedding_slots` (`obsm` or `varm`) and `embedding_shapes`.
  `metadata_keys` are provenance records in `.uns`, not embeddings.
- The output equals the Python API call
  `BioEmbedder().embed(adata, entity_type=..., model=..., obs_column=...)`.

## 4. Errors

| `error_type` | What to do |
|---|---|
| `UnsupportedModelError` | The model belongs to another modality or does not exist. The message names the modality it fits and the models this one accepts; pick from those. Do not switch to a different model silently: tell the user. |
| `ValueError` mentioning `obs_column` | `perturbation` needs the column of perturbed genes; `single_cell` does not take a column. |
| `ValueError` about the attach axis | Pass `obs_column` or `var_column` so the tool knows which rows the vectors belong to. |
| `ModelOOMError` | Use a smaller model or `device="cpu"`. |
| `ImportError` | The model's optional dependency is missing; give the user the install command from the message. |

If some rows cannot be embedded (an unknown gene, an invalid SMILES), the call fails with a
`ValueError` that names the first ones. Tell the user which inputs failed. Only if they want to
continue without them, rerun with `missing="nan"`, which fills those rows with NaN.
