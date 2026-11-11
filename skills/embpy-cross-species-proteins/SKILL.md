---
name: embpy-cross-species-proteins
description: Compare orthologous proteins across species with embpy - resolve one-to-one orthologs (Ensembl Compara), embed their sequences with a protein language model such as ESM-2, and relate embedding similarity to sequence identity per gene family. Use when the user asks how conserved a protein's representation is between human, mouse, zebrafish or other species, wants cross-species protein embeddings, or wants to check whether a protein model captures evolutionary relationships.
---

# Cross-species protein comparison

## 1. Build the cross-species embedding

```python
import embpy as ep

embedder = ep.BioEmbedder(device="cpu")
xs = ep.tl.build_cross_species_adata(
    embedder,
    genes=["TP53", "GAPDH", "ACTB"],
    species=["human", "mouse", "zebrafish"],
    model="esm2_8M",
)
print(xs.obs[["gene_family", "species", "symbol", "sequence_identity", "protein_length"]])
```

- One row per (gene, species) protein; embeddings in `.obsm["X_emb"]`. `gene_family` is the
  input symbol, `symbol` the species-specific one.
- Orthologs come from Ensembl Compara (`orthology_type="ortholog_one2one"` by default) and
  sequences from UniProt, so this needs internet. Genes without a one-to-one ortholog in a
  species are missing for that species: compare the rows with the request and report gaps.
- `esm2_8M` runs on CPU in seconds. `esm2_650M` and larger give better representations but need
  a GPU in practice.

## 2. Similarity versus sequence identity

```python
pairs = ep.tl.identity_vs_similarity(xs)
print(pairs[["gene_family", "species_a", "species_b", "sequence_identity", "embedding_similarity"]])
print(pairs[["sequence_identity", "embedding_similarity"]].corr(method="spearman"))
```

One row per ortholog pair within a family. A protein model that captures evolution should give
higher `embedding_similarity` to pairs with higher `sequence_identity`. Families where similarity
stays high despite low identity are the interesting cases (conserved fold or function).

## 3. All-against-all and plots

```python
import matplotlib

matplotlib.use("Agg")
matrix = ep.tl.ortholog_similarity_matrix(xs)
fig = ep.pl.plot_identity_vs_similarity(pairs)
fig.savefig("identity_vs_similarity.png", dpi=150, bbox_inches="tight")
fig = ep.pl.plot_species_umap(xs)
fig.savefig("species_umap.png", dpi=150, bbox_inches="tight")
```

`ortholog_similarity_matrix` labels rows and columns by symbol and species. With only a few
genes, prefer the table and the identity plot over a UMAP.

## Reporting

Name the model, species and orthology type, list missing orthologs, and give the correlation
between identity and similarity with the number of pairs it is based on.
