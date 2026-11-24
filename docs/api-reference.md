# API Reference

This page provides automated API documentation extracted directly from the Python source code using `mkdocstrings`.

---

## Core Embedder

::: embpy.embedder.BioEmbedder

---

## Embedding Fusion

The [fusion guide](fusion.md) covers fitted preprocessing, identifier alignment
and paired benchmarks. Feature outputs use the standard `EmbeddingResult` contract.

::: embpy.tl.fusion

::: embpy.tl.benchmark.benchmark_fusion

::: embpy.tl.benchmark.benchmark_fusion_retrieval

::: embpy.tl.benchmark.FusionBenchmarkResult

---

## Protein–Phenotype Alignment

See the [alignment guide](protein_phenotype_alignment.md) for matched inputs,
control filtering, aggregation and interpretation of the permutation test.

::: embpy.tl.protein_phenotype_alignment

::: embpy.tl.alignment_summary

::: embpy.tl.cross_modal_mantel

---

## Model Wrappers

### Protein Models
::: embpy.models.protein_models

### Text & Description Models
::: embpy.models.text_models

### Molecule Models
::: embpy.models.molecule_models
