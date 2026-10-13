# Hands-On Workflows & Recipes

This guide provides practical, end-to-end recipes for common biological embedding tasks using `embpy`. Each example is designed to be copy-pasteable, self-contained, and cross-platform.

---

## 1. Protein Sequence Embedding & Clustering

**Use Case:** You have a set of protein sequences (e.g. homologous enzymes, antibody variants, or peptides) and want to group them by functional similarity in embedding space.

```python
import numpy as np
from sklearn.decomposition import PCA
from embpy import BioEmbedder
import embpy.tl as tl

# 1. Initialize embedder on the best available hardware
embedder = BioEmbedder(device="auto")

# 2. Example protein sequences (Insulin, Glucagon, and variants)
proteins = [
    "GIVEQCCTSICSLYQLENYCN",                                          # Insulin chain A (Human)
    "FVNQHLCGSHLVEALYLVCGERGFFYTPKT",                                # Insulin chain B (Human)
    "HSQGTFTSDYSKYLDSRRAQDFVQWLMNT",                                 # Glucagon (Human)
    "HAEGTFTSDVSSYLEGQAAKEFIAWLVKGR",                                # GLP-1 (Human)
    "KCNTATCATQRLANFLVHSSNNFGAILSSTNVGSNTY",                         # Amylin (Human)
]

# 3. Generate 1D embedding vectors per sequence with ESM-2
embeddings = embedder.embed(
    proteins,
    entity_type="protein",
    model="esm2_8M",
    output="numpy",
)
print(f"Generated embeddings matrix with shape: {embeddings.shape}")
# Shape: (5, 320)

# 4. Compute pairwise cosine distance matrix
distance_matrix = tl.compute_distance_matrix(embeddings, metric="cosine")
print("Pairwise Cosine Distances:")
print(np.round(distance_matrix, 3))

# 5. Dimensionality reduction (PCA down to 2 components)
pca = PCA(n_components=2)
reduced_coords = pca.fit_transform(embeddings)
print("\n2D PCA Projections:")
for i, (x, y) in enumerate(reduced_coords):
    print(f"  Protein {i+1}: ({x:.3f}, {y:.3f})")
```

??? tip "Choosing the Right ESM-2 Model Size"
    `embpy` supports all ESM-2 checkpoint sizes. For fast local CPU experimentation, use `esm2_8M` or `esm2_35M`. For maximum representation power on GPU, use `esm2_650M` or `esm2_3B`.

---

## 2. Small Molecule Similarity & Compound Screening

**Use Case:** Given a library of SMILES strings, calculate molecular representations and identify the closest analogues to a query drug.

```python
import pandas as pd
from embpy import BioEmbedder
import embpy.tl as tl

embedder = BioEmbedder(device="auto")

# 1. Dictionary of common therapeutic molecules
compounds = {
    "Aspirin": "CC(=O)OC1=CC=CC=C1C(=O)O",
    "Ibuprofen": "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
    "Naproxen": "CC(C1=CC2=C(C=C1)C=C(C=C2)OC)C(=O)O",
    "Paracetamol": "CC(=O)NC1=CC=C(C=C1)O",
    "Caffeine": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
}

# 2. Embed molecules using ChemBERTa transformer or Morgan fingerprints
df_embeddings = embedder.embed(
    list(compounds.values()),
    entity_type="molecule",
    model="chemberta2",
    output="table",
)
df_embeddings.index = list(compounds.keys())
print("Molecule Embeddings Table:")
print(df_embeddings.iloc[:, :5])  # Display first 5 dimensions

# 3. Compute cosine similarity between all pairs
similarity_matrix = tl.compute_similarity(
    df_embeddings.values,
    metric="cosine",
)

sim_df = pd.DataFrame(
    similarity_matrix,
    index=compounds.keys(),
    columns=compounds.keys(),
)
print("\nSimilarity Matrix:")
print(sim_df.round(3))
```

---

## 3. AnnData Integration for Single-Cell Perturbation Screens

**Use Case:** You are analyzing a CRISPR knockout or Perturb-seq dataset in `AnnData` format and need to attach protein foundation model representations for each perturbed gene.

```python
import anndata as ad
import numpy as np
import pandas as pd
from embpy import BioEmbedder

# 1. Simulate an AnnData object containing 10 cells across 3 perturbations
n_cells = 10
adata = ad.AnnData(
    X=np.random.poisson(lam=1.5, size=(n_cells, 50)),
    obs=pd.DataFrame({
        "cell_id": [f"cell_{i}" for i in range(n_cells)],
        "perturbation": ["TP53", "EGFR", "TP53", "BRAF", "EGFR",
                         "TP53", "BRAF", "EGFR", "TP53", "BRAF"],
    }),
)

# 2. Initialize BioEmbedder
embedder = BioEmbedder(device="cpu")

# 3. Embed perturbation genes directly into adata
# Passing `is_perturbation=True` places embeddings in adata.obsm aligned with rows
adata = embedder.embed(
    adata,
    entity_type="gene",
    obs_column="perturbation",
    id_type="symbol",
    model="esm2_8M",
    output="anndata",
    is_perturbation=True,
    key="X_pert_esm2",
)

print("Obsm keys in AnnData:", adata.obsm.keys())
print("Perturbation embedding matrix shape:", adata.obsm["X_pert_esm2"].shape)
```

??? info "Output Contracts"
    - **`output="anndata"`:** Preserves your existing AnnData and appends embeddings to `.obsm` (for observation/cell data) or `.varm` (for gene/protein feature data).
    - **`output="table"`:** Returns a pandas DataFrame with canonical index identifiers.
    - **`output="numpy"`:** Returns a raw NumPy matrix for lightweight computation.

---

## 4. Multi-Modal Cross-Comparison (Sequences vs. Text)

**Use Case:** Compare protein sequence embeddings directly against biomedical text descriptions (e.g. PubMed abstracts or Gene Ontology terms) using shared representation spaces.

```python
from embpy import BioEmbedder

embedder = BioEmbedder(device="auto")

# 1. Biological descriptions
descriptions = [
    "Involved in cellular response to DNA damage and apoptosis.",
    "Receptor tyrosine kinase binding epidermal growth factor.",
    "Serine/threonine-protein kinase involved in MAPK signaling pathway.",
]

# 2. Embed descriptions using PubMedBERT
text_embeddings = embedder.embed(
    descriptions,
    entity_type="text",
    model="pubmedbert",
    output="numpy",
)

print(f"PubMedBERT text embeddings: {text_embeddings.shape}")
```

---

## 5. High-Throughput Batch Processing with GPU Caching

**Use Case:** Efficiently stream tens of thousands of sequences with automatic memory management, adaptive batch sizes, and cache clearing.

```python
import torch
from embpy import BioEmbedder

def process_large_library(fasta_records: list[str]):
    # 1. Use GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedder = BioEmbedder(device=device)

    # 2. Embed with adaptive batching to avoid Out-Of-Memory (OOM)
    results = embedder.embed(
        fasta_records,
        entity_type="protein",
        model="esm2_35M",
        batch_size=64,           # Initial batch size
        output="numpy",
    )

    # 3. Explicitly drop cached model weights and free GPU memory
    embedder.clear_model_cache(which="all")
    
    return results

if __name__ == "__main__":
    # Guard with main block for Windows multiprocessing safety
    sample_records = [
        "MKTIIALSYIFCLVFA",
        "MKLLILTCLVAVALAA",
        "MRLLILTCLLAVALAA",
    ] * 20
    embeddings = process_large_library(sample_records)
    print("Batch processing complete. Embeddings shape:", embeddings.shape)
```

---

## 6. Model Inspection & Direct Wrapper Access

**Use Case:** Inspect model metadata, access raw tokenizers, or extract intermediate layer activations.

```python
from embpy import BioEmbedder

embedder = BioEmbedder(device="cpu")

# 1. Retrieve the underlying model wrapper without triggering a forward pass
wrapper = embedder.get_model("esm2_8M", load=True)

# 2. Inspect wrapper properties
print("Model Name:", wrapper.model_name)
print("Embedding Dimension:", wrapper.EMBEDDING_DIM)

# 3. Tokenize sequences directly
encoded = wrapper.tokenizer("MKTVRQERLKSIVR", return_tensors="pt")
print("Input IDs shape:", encoded["input_ids"].shape)
```
