# Tested Natural Language Prompts for Agents

Once the `embpy` Model Context Protocol (MCP) server is linked to your AI assistant (Claude Desktop, Cursor, Windsurf, or a custom agent), you can direct complex single-cell bioinformatics tasks using natural language.

Here are tested, high-yield prompt templates categorized by workflow stage.

---

## 1. Dataset Triage & Quality Inspection

### Inspect Unknown `.h5ad` Dataset
> *"Inspect the single-cell dataset located at `data/pbmc3k.h5ad`. What are the observation and variable dimensions? Which metadata columns are present in `obs`, are raw counts stored in `X`, and do we already have precomputed embeddings in `obsm`?"*

**Expected Agent Tool Sequence:**
1. Calls `inspect_anndata(path="data/pbmc3k.h5ad")`.
2. Summarizes cell count, gene count, metadata categories (e.g. `cell_type`, `batch`), count normalization status, and existing embeddings.

---

## 2. Model Discovery & Hardware Check

### Check Foundation Model Capabilities
> *"Check what single-cell foundation models and hardware accelerators are available in my current environment. Recommend whether I should run on CPU or GPU."*

**Expected Agent Tool Sequence:**
1. Calls `list_available_models()`.
2. Reports detected backends (`pca`, `scvi`, `scgpt`, `geneformer`, `state`), detected device (`cuda`, `mps`, `cpu`), and any optional extras that need installation.

---

## 3. Gene Identifier Harmonization

### Reconcile Ensembl IDs with HGNC Symbols
> *"I have a list of Ensembl IDs from our sequencing pipeline: `['ENSG00000141510', 'ENSG00000146648', 'ENSG00000136997', 'ENSG00000171862']`. Convert them to HGNC gene symbols before feeding them into our downstream model, and alert me if any could not be mapped."*

**Expected Agent Tool Sequence:**
1. Calls `map_gene_identifiers(identifiers=[...], source_id_type="ensembl", target_id_type="symbol")`.
2. Formats a clear mapping table (e.g., `ENSG00000141510 -> TP53`, `ENSG00000146648 -> EGFR`, `ENSG00000136997 -> MYC`, `ENSG00000171862 -> PTEN`).

---

## 4. Multi-Model Embedding Execution

### Embed Cells with Classical and Deep Learning Models
> *"Run single-cell embeddings on `data/pbmc3k.h5ad` using both `pca` and `scvi`. Set batch size to 64 and save the updated dataset with embeddings to `data/pbmc3k_embedded.h5ad`."*

**Expected Agent Tool Sequence:**
1. Calls `embed_cells(path="data/pbmc3k.h5ad", models=["pca", "scvi"], batch_size=64, output_path="data/pbmc3k_embedded.h5ad")`.
2. Reports back generated `.obsm` keys (`X_pca`, `X_scvi`), embedding shapes, and confirmation of file creation.

---

## 5. Biological Conservation & Integration Benchmarking

### Benchmark with scIB
> *"Evaluate the embeddings `['X_pca', 'X_scvi']` in `data/pbmc3k_embedded.h5ad` against the ground-truth cell type annotation column `cell_type`. Display a comparison table ranking the models by biological conservation (NMI, ARI, ASW, cLISI)."*

**Expected Agent Tool Sequence:**
1. Calls `evaluate_scib(path="data/pbmc3k_embedded.h5ad", embedding_keys=["X_pca", "X_scvi"], label_key="cell_type")`.
2. Formats and explains the comparative markdown table, highlighting which model best preserves cell-type identity.

---

## 6. Perturbation Analysis & Latent UMAP Projection

### Evaluate Perturbation Predictions
> *"Run perturbation evaluation on `data/perturbation_screen.h5ad` using control label `'control'` and perturbation column `'perturbation'`. Score the latent representations in `X_pca`."*

**Expected Agent Tool Sequence:**
1. Calls `evaluate_perturbations(path="data/perturbation_screen.h5ad", control_pert="control", pert_col="perturbation", embedding_keys=["X_pca"])`.
2. Summarizes perturbation prediction accuracy, MSE, and R2 metrics.

### Project 2D UMAP for Visual Exploration
> *"Compute a 2D UMAP projection on the `X_scvi` embedding of `data/pbmc3k_embedded.h5ad` with 20 nearest neighbors. Save the result to `data/pbmc3k_final.h5ad`."*

**Expected Agent Tool Sequence:**
1. Calls `compute_umap(path="data/pbmc3k_embedded.h5ad", embedding_key="X_scvi", n_neighbors=20, output_path="data/pbmc3k_final.h5ad")`.
2. Verifies coordinates are saved in `.obsm['X_umap_scvi']`.

---

## 7. Autonomous Multi-Step Pipeline Prompt

### Complete End-to-End Analysis Loop
> *"Please perform a complete single-cell evaluation workflow on `data/raw_cells.h5ad`:*
> 1. *Inspect the dataset to check its cell count and identify the cell type annotation column.*
> 2. *Check available foundation models and hardware.*
> 3. *Compute cellular embeddings using PCA and any installed foundation model.*
> 4. *Evaluate cell-type conservation with scIB and tell me which model performs better.*
> 5. *Generate a 2D UMAP on the winning embedding and save the final dataset to `data/analyzed_cells.h5ad`."*

**Expected Autonomous Agent Behavior:**
- The agent sequentially invokes `inspect_anndata`, `list_available_models`, `embed_cells`, `evaluate_scib`, and `compute_umap`.
- Handles intermediate validation autonomously and returns a synthesized executive report summarizing data characteristics, model performance, and downstream visualizations.
