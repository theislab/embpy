---
name: embpy-fasta-embeddings
description: Embed every protein or DNA sequence in a FASTA or FASTQ file (plain or gzipped) with embpy - ESM-2, ProtT5 and other protein language models, or HyenaDNA, Nucleotide Transformer and other DNA models - into an AnnData with one row per sequence, keeping IDs, lengths and header annotations. Use when the user has a .fasta, .fa, .faa, .fna or .fastq file, or sequences rather than gene names, and wants embeddings, similarity, clustering or a classifier on them.
---

# Embedding sequences from a FASTA file

Needs Biopython: `pip install "embpy[bio]"` (the error says so if it is missing).

## 1. Embed

```python
import embpy as ep

embedder = ep.BioEmbedder(device="cpu")
proteins = embedder.embed_fasta("data/proteins.fasta", model="esm2_8M", seq_type="protein")
print(proteins.shape, list(proteins.obsm))
print(proteins.obs[["sequence_id", "sequence_length", "description"]].head())
```

- One row per record. `obs_names` are the record IDs; `.obs` has `sequence_id`,
  `sequence_length`, `description` (the full header line) and `seq_type`.
- The embedding is in `.obsm["X_<model>"]` (here `X_esm2_8M`); `obsm_key=` sets another name.
- `seq_type` is `"protein"` or `"dna"`. Set it explicitly: when omitted, embpy guesses from the
  model, then from the letters in the sequences.
- `.fasta`, `.fa`, `.faa`, `.fna`, `.fastq`, `.fq` and the same with `.gz` are read directly.

DNA works the same way with a DNA model:

```python
promoters = embedder.embed_fasta("data/promoters.fa", model="hyenadna_tiny_1k", seq_type="dna", obsm_key="X_hyena")
print(promoters.obsm["X_hyena"].shape)
```

## 2. Check that every record was embedded

```python
n_records = sum(1 for line in open("data/proteins.fasta") if line.startswith(">"))
print(n_records, proteins.n_obs, proteins.obs["sequence_length"].max())
```

- Report any difference between records and rows to the user.
- Choose a model whose context covers the sequences: ESM-2 was trained on proteins up to 1,022
  residues, `hyenadna_tiny_1k` on 1 kb of DNA. For longer sequences use a long-context model
  (`hyenadna_small_32k`, `caduceus_ph_131k`) or ask the user how to split them.
- On CPU use small models (`esm2_8M`, `esm2_35M`, `hyenadna_tiny_1k`); larger ones need a GPU.

## 3. Keep header annotations as columns

```python
proteins.obs["family"] = proteins.obs["description"].str.extract(r"class=(\S+)")[0]
proteins.write_h5ad("data/proteins_embedded.h5ad")
```

Headers often carry labels (`>P04637 family=p53 organism=human`). Parse them into `.obs`
columns with a regular expression that matches the user's header format.

## Next steps

The saved file works with the other skills: `embpy-similarity-search` (nearest sequences),
`embpy-embedding-clustering`, `embpy-embedding-visualization`, and `embpy-layer-selection`
(which layer to use).
