# Ready-to-Use Agentic Mega-Prompts

Copy-paste these comprehensive prompts directly into your AI assistant (**Claude Desktop**, **Cursor Composer**, **Windsurf Cascade**, or custom autonomous agents) to trigger the `embpy` MCP server.

---

## 1. The Universal End-to-End Mega-Prompt

Use this prompt to run the complete 4-pillar universal pipeline autonomously from raw data to benchmark victory:

```markdown
You have access to the `embpy` Model Context Protocol (MCP) server. 
I want you to autonomously execute the complete universal embedding pipeline on my dataset at `data/raw_data.h5ad`:

1. **Inspect:** Call `inspect_data` on `data/raw_data.h5ad`. Tell me the cell/gene dimensions, whether counts are raw or log-normalized, and identify the best candidate columns for cell types and technical batches.
2. **Harmonize:** Check the gene symbols in the dataset. If needed, call `resolve_identifiers` to map HGNC gene symbols to Ensembl IDs for foundation model compatibility.
3. **Preprocess:** Run `preprocess_counts` with library size normalization (1e4), log1p transformation, and 2,000 highly variable genes. Save the preprocessed AnnData to `data/preprocessed.h5ad`.
4. **Discover & Embed:** Check `list_model_registry` for runnable single-cell models. Then call `compute_embeddings` to extract representations using `pca` and `scvi` (or `scgpt` if available), saving the embedded object to `data/embedded.h5ad`.
5. **Annotate:** Call `annotate_metadata` to enrich the dataset with biological pathways and GO terms based on the entities present, saving to `data/annotated.h5ad`.
6. **Project & Benchmark:** 
   - Run `compute_umap` on the primary embedding to generate 2D coordinates.
   - Run `benchmark_scib` evaluating the computed embeddings against the ground-truth cell type label and batch column.
7. **Synthesize:** Present the final scIB benchmark Markdown table and declare which model wins in biological conservation versus batch correction.

Adhere strictly to token-saving rules: read and report compact metadata summaries; do not attempt to dump raw matrix values.
```

---

## 2. Unknown Dataset Triage & Quality Control

Use this prompt when an AnnData `.h5ad` file is handed to you without documentation:

```markdown
Please inspect the unknown single-cell dataset at `data/sample_unlabeled.h5ad` using `inspect_data`:
- What are the dimensions ($N$ cells $\times$ $M$ genes)?
- Does `.X` contain raw integer UMI counts or log-normalized expression values?
- What metadata columns are present in `.obs`? Which column is the best candidate for cell type annotations, and which is the best candidate for experimental batch?
- Are there any existing embeddings in `.obsm`?
- Once diagnosed, execute `preprocess_counts` to produce a standardized, log-normalized dataset with highly variable gene flags at `data/sample_cleaned.h5ad`.
```

---

## 3. Foundation Model Latent Space Comparison

Use this prompt to generate and compare multiple foundation models:

```markdown
I have a preprocessed AnnData dataset at `data/cleaned_cells.h5ad`.
Using the `embpy` MCP server:
1. Query `list_model_registry` to see what single-cell and sequence foundation models are supported in our current environment.
2. Call `compute_embeddings` to embed the cells across the available roster: `["pca", "scvi"]`.
3. Call `compute_umap` on each embedding key (`X_pca`, `X_scvi`) to generate 2D coordinates in `.obsm`.
4. Run `benchmark_scib` using `cell_type` as the label key.
5. Provide a comparison table showing NMI, ARI, and ASW scores, and explain which representation best separates distinct cell types.
```

---

## 4. Perturbation Response Differential Scoring

Use this prompt to evaluate CRISPR or chemical perturbation predictions against real experimental observations:

```markdown
I have a perturbation experiment dataset at `data/perturbation_screen.h5ad` containing both real observed profiles and model predictions separated by the `split` column.
Using `embpy`'s `benchmark_perturbations` tool:
- Set `control_pert="ctrl"` and `pert_col="perturbation"`.
- Evaluate prediction fidelity using `profile="minimal"`.
- Export detailed score metrics to `data/pert_evaluation.csv`.
- Return a Markdown summary table highlighting the top 5 best-predicted perturbations (highest $R^2$, lowest $\Delta L_2$, and highest direction agreement).
```

---

## 5. Chemical & Small Molecule Pipeline

Use this prompt for molecular and compound datasets:

```markdown
I have a list of drug candidates: `["aspirin", "imatinib", "metformin", "dexamethasone"]`.
Using the `embpy` MCP server:
1. Call `resolve_identifiers` with `entity_type="molecule"` to resolve these drug names into canonical SMILES strings.
2. Query `list_model_registry` with `entity_type="molecule"` to check available molecular embedding models (e.g. ChemBERTa, Morgan fingerprints).
3. Call `annotate_metadata` to retrieve known mechanism-of-action (MoA) and pharmacological pathways.
4. Summarize the mapping success rate and chemical features for each compound in a clean Markdown table.
```
