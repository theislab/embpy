"""Generate tutorials/05_mcp_agent_workflows.ipynb and tutorials/mcp_agent_workflows.ipynb."""

import os
import nbformat as nbf

os.makedirs("tutorials", exist_ok=True)

nb = nbf.v4.new_notebook()

cells = []

# Title & Overview
cells.append(nbf.v4.new_markdown_cell("""# 🤖 Model Context Protocol (MCP) Agent Workflows in `embpy`

This tutorial demonstrates how to leverage **`embpy`'s native Model Context Protocol (MCP) server** to expose bioinformatics and single-cell representation capabilities to AI assistants (Claude Desktop, Cursor, Windsurf, or custom autonomous agents).

---

## 1. Introduction & Architecture

The **Model Context Protocol (MCP)** is an open standard that allows Large Language Models (LLMs) and agent frameworks to securely call external tools and retrieve domain context.

In `embpy`, the MCP server operates over standard input/output (`stdio`) using JSON-RPC 2.0 frames:

- **Strict Stdio Isolation:** Model logging, warnings, and library output (e.g., from PyTorch, Scanpy, scIB) are diverted to `sys.stderr` so that `sys.stdout` remains pure, uncorrupted JSON-RPC.
- **Resilient Error Traps:** Tool exceptions (such as out-of-memory or missing observation columns) return actionable JSON error payloads rather than crashing the process.
- **7 Native Bioinformatics Skills:**
  1. `inspect_anndata`: Profile dataset dimensions, counts vs normalized expression, and annotations.
  2. `list_available_models`: Report runnable foundation models, compute hardware (CUDA/MPS/CPU), and optional extras.
  3. `map_gene_identifiers`: Map and validate gene IDs between Ensembl and HGNC symbols.
  4. `embed_cells`: Compute multi-model cell representations via `BioEmbedder`.
  5. `evaluate_scib`: Score biological conservation and batch integration with scIB.
  6. `evaluate_perturbations`: Benchmark perturbation responses with ArcInstitute `cell-eval`.
  7. `compute_umap`: Generate kNN graphs and 2D UMAP coordinates for visual inspection.
"""))

# Imports and Environment Setup
cells.append(nbf.v4.new_code_cell("""import os
import sys

# Ensure local embpy source tree is loaded
src_path = os.path.abspath("src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

import json
import asyncio
import tempfile
import numpy as np
import pandas as pd
from anndata import AnnData

import embpy
from embpy.mcp import (
    create_server,
    inspect_anndata,
    list_available_models,
    map_gene_identifiers,
    embed_cells,
    evaluate_scib,
    evaluate_perturbations,
    compute_umap,
)

print(f"embpy loaded successfully from: {embpy.__file__}")
"""))


# Section 2: Programmatic Tool Inspection
cells.append(nbf.v4.new_markdown_cell("""## 2. Programmatic Tool Inspection

AI assistants discover capabilities by requesting `tools/list` from the MCP server. We can programmatically inspect the registered tools, their parameters, and documentation.
"""))

cells.append(nbf.v4.new_code_cell("""# Instantiate the MCP server instance
server = create_server("embpy-tutorial-server")

async def inspect_server_tools():
    tool_list = await server.list_tools()
    print(f"Discovered {len(tool_list)} registered MCP tools:\\n")
    for tool in tool_list:
        print(f"🔹 Tool: {tool.name}")
        doc_first_line = (tool.description or '').strip().split('\\n')[0]
        print(f"   Description: {doc_first_line}")
        schema = getattr(tool, 'input_schema', getattr(tool, 'inputSchema', None))
        schema_str = json.dumps(schema, indent=2) if isinstance(schema, dict) else str(schema)
        print(f"   Schema: {schema_str[:160]}...\\n")

await inspect_server_tools()
"""))


# Section 3: Hardware & Model Discovery
cells.append(nbf.v4.new_markdown_cell("""## 3. Hardware & Foundation Model Discovery

Agents first query `list_available_models` to determine available GPUs, installed foundation models, and missing extras.
"""))

cells.append(nbf.v4.new_code_cell("""# Query available models and hardware
env_info = list_available_models()
print("Compute Device:", env_info["compute_device"], f"({env_info['device_details']})")
print("Runnable Models:", env_info["runnable_models"])
print("Registered Single-Cell Models Count:", len(env_info["all_singlecell_models"]))

# Audit optional extras
extras_df = pd.DataFrame.from_dict(env_info["optional_extras_status"], orient="index")
extras_df[["installed", "feature", "pip"]]
"""))

# Section 4: Preparing Sample Single-Cell Dataset
cells.append(nbf.v4.new_markdown_cell("""## 4. Preparing a Sample Single-Cell Dataset

We create a self-contained AnnData dataset mimicking a single-cell experiment (cells with cell-type labels, batches, and perturbation conditions).
"""))

cells.append(nbf.v4.new_code_cell("""work_dir = tempfile.mkdtemp(prefix="embpy_mcp_demo_")
dataset_path = os.path.join(work_dir, "pbmc_demo.h5ad")

rng = np.random.RandomState(42)
n_cells, n_genes = 60, 50
counts = rng.poisson(lam=2.5, size=(n_cells, n_genes)).astype(np.float32)

adata = AnnData(X=counts)
adata.obs_names = [f"cell_{i}" for i in range(n_cells)]
adata.var_names = [f"GENE_{i}" for i in range(n_genes)]
# Add biologically realistic gene symbols to test identifier mapping
real_symbols = ["TP53", "GAPDH", "CD4", "CD8A", "MS4A1", "IL7R", "GNLY", "NKG7"]
for idx, sym in enumerate(real_symbols):
    adata.var_names.values[idx] = sym

adata.obs["cell_type"] = (["T_cell", "B_cell", "NK_cell"] * (n_cells // 3))[:n_cells]
adata.obs["batch"] = (["batch_A", "batch_B"] * (n_cells // 2))[:n_cells]
adata.obs["perturbation"] = (["ctrl", "ctrl", "stim_IFN", "stim_IFN"] * 20)[:n_cells]
adata.obs["split"] = (["train", "test"] * 30)[:n_cells]
adata.layers["counts"] = counts.copy()

adata.write_h5ad(dataset_path)
print(f"Sample dataset saved to: {dataset_path} ({n_cells} cells, {n_genes} genes)")
"""))

# Section 5: Simulated Agent Step 1: inspect_anndata
cells.append(nbf.v4.new_markdown_cell("""## 5. Simulated Agent Workflow

### Step 1: Dataset Inspection & Profiling
The agent inspects the file using `inspect_anndata` to verify cell/gene counts, categorical columns, and expression nature (raw counts vs log-normalized).
"""))

cells.append(nbf.v4.new_code_cell("""inspection = inspect_anndata(dataset_path)
print("Status:", inspection["status"])
print("Cell Count (n_obs):", inspection["n_obs"])
print("Gene Count (n_vars):", inspection["n_vars"])
print("Detected Expression Nature:", inspection["expression_nature"]["detected_nature"])
print("Observation Columns:", inspection["obs_columns"])
print("Layers:", inspection["layers"])
"""))

# Section 6: Simulated Agent Step 2: map_gene_identifiers
cells.append(nbf.v4.new_markdown_cell("""### Step 2: Gene Identifier Harmonization
Single-cell foundation models (such as Geneformer and scGPT) require Ensembl IDs. The agent converts symbols using `map_gene_identifiers`.
"""))

cells.append(nbf.v4.new_code_cell("""# Map sample gene symbols to Ensembl IDs
mapping = map_gene_identifiers(
    genes=real_symbols[:5],
    target_type="ensembl",
    organism="human",
)
print("Mapping Rate:", f"{mapping['mapping_rate']}%")
print("Mapped Gene Sample:")
for sym, ens in mapping["mapping_sample"].items():
    print(f"  {sym} -> {ens}")
"""))

# Section 7: Simulated Agent Step 3: embed_cells
cells.append(nbf.v4.new_markdown_cell("""### Step 3: Embedding Generation
The agent computes embeddings on the cells using `embed_cells`. Here we use PCA, and save the updated dataset with embeddings.
"""))

cells.append(nbf.v4.new_code_cell("""embedded_dataset = os.path.join(work_dir, "pbmc_embedded.h5ad")

embed_result = embed_cells(
    path=dataset_path,
    models=["pca"],
    output_path=embedded_dataset,
    preprocessing="none",
    device="cpu",
)
print("Embed Status:", embed_result.get("status"))
if embed_result.get("status") == "success":
    print("Computed Models:", embed_result["models_computed"])
    print("Embedding Keys in .obsm:", embed_result["embedding_keys"])
    print("Shapes:", embed_result["embedding_shapes"])
else:
    print("Embedding Diagnostics:", embed_result.get("error"))
    print("Suggestion:", embed_result.get("suggestion", "Ensure required dependencies are installed."))
"""))

# Section 8: Simulated Agent Step 4: compute_umap
cells.append(nbf.v4.new_markdown_cell("""### Step 4: 2D UMAP Coordinates
The agent constructs a kNN neighborhood graph and calculates 2D UMAP coordinates using `compute_umap`.
"""))

cells.append(nbf.v4.new_code_cell("""umap_dataset = os.path.join(work_dir, "pbmc_umap.h5ad")

umap_result = compute_umap(
    path=embedded_dataset,
    embedding_key="X_pca",
    output_path=umap_dataset,
    n_neighbors=10,
)
print("UMAP Status:", umap_result.get("status"))
if umap_result.get("status") == "success":
    print("UMAP Key in .obsm:", umap_result["umap_key"])
    print("Coordinate Array Shape:", umap_result["umap_shape"])
else:
    print("UMAP Diagnostics:", umap_result.get("error"))
    print("Suggestion:", umap_result.get("suggestion"))
"""))

# Section 9: Simulated Agent Step 5: evaluate_scib
cells.append(nbf.v4.new_markdown_cell("""### Step 5: scIB Quality Benchmarking
The agent evaluates biological conservation and batch integration using `evaluate_scib`.
"""))

cells.append(nbf.v4.new_code_cell("""scib_report_csv = os.path.join(work_dir, "scib_scores.csv")
target_dataset = umap_dataset if os.path.exists(umap_dataset) else embedded_dataset

scib_result = evaluate_scib(
    path=target_dataset,
    embedding_keys=["X_pca"],
    label_key="cell_type",
    batch_key="batch",
    output_csv=scib_report_csv,
)

print("scIB Evaluation Status:", scib_result.get("status"))
if scib_result.get("status") == "success":
    print("Total Composite Score:", scib_result["metrics"]["X_pca"].get("total"))
    from IPython.display import Markdown
    display(Markdown(scib_result["markdown_table"]))
else:
    print("scIB Diagnostics:", scib_result.get("error"))
    print("Suggestion:", scib_result.get("suggestion"))
"""))

# Section 10: Agent Integration Architecture
cells.append(nbf.v4.new_markdown_cell("""## 6. Lightweight Agent Tool-Calling Loop

Here is how an LLM agent dispatcher coordinates tools using a standard tool-dispatch pattern:
"""))

cells.append(nbf.v4.new_code_cell("""TOOL_REGISTRY = {
    "inspect_anndata": inspect_anndata,
    "list_available_models": list_available_models,
    "map_gene_identifiers": map_gene_identifiers,
    "embed_cells": embed_cells,
    "compute_umap": compute_umap,
    "evaluate_scib": evaluate_scib,
}

def execute_agent_tool_call(tool_name: str, arguments: dict):
    \"\"\"Simulate autonomous agent tool execution with structured error traps.\"\"\"
    if tool_name not in TOOL_REGISTRY:
        return {"status": "error", "error": f"Unknown tool: {tool_name}"}
    tool_func = TOOL_REGISTRY[tool_name]
    try:
        return tool_func(**arguments)
    except Exception as exc:
        return {"status": "error", "error": str(exc)}

# Example simulated tool execution
simulated_query = {"path": dataset_path}
response = execute_agent_tool_call("inspect_anndata", simulated_query)
print("Agent Received Response:")
print(f"  Status: {response['status']}, Cells: {response['n_obs']}, Genes: {response['n_vars']}")
"""))

# Clean up
cells.append(nbf.v4.new_code_cell("""# Clean up temporary demonstration files
import shutil
shutil.rmtree(work_dir, ignore_errors=True)
print("Demonstration artifacts cleaned up successfully.")
"""))

nb["cells"] = cells

# Save both filenames to satisfy any references in documentation or prompts
with open("tutorials/05_mcp_agent_workflows.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)

with open("tutorials/mcp_agent_workflows.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print("Notebooks written successfully.")
