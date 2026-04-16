# API Reference

## Core: `BioEmbedder`

```{eval-rst}
.. currentmodule:: embpy

.. autosummary::
    :toctree: generated

    BioEmbedder
```

### Gene and Protein Embedding

```{eval-rst}
.. currentmodule:: embpy

.. autosummary::
    :toctree: generated

    BioEmbedder.embed_gene
    BioEmbedder.embed_genes_batch
    BioEmbedder.embed_protein
    BioEmbedder.embed_proteins_batch
```

### Molecule Embedding

```{eval-rst}
.. currentmodule:: embpy

.. autosummary::
    :toctree: generated

    BioEmbedder.embed_molecule
    BioEmbedder.embed_molecules_batch
```

### Text and Description Embedding

```{eval-rst}
.. currentmodule:: embpy

.. autosummary::
    :toctree: generated

    BioEmbedder.embed_text
    BioEmbedder.embed_texts_batch
    BioEmbedder.embed_description
    BioEmbedder.embed_descriptions_batch
```

### File-based Embedding

```{eval-rst}
.. currentmodule:: embpy

.. autosummary::
    :toctree: generated

    BioEmbedder.embed_fasta
```

### Cell and AnnData Embedding

```{eval-rst}
.. currentmodule:: embpy

.. autosummary::
    :toctree: generated

    BioEmbedder.embed_cells
    BioEmbedder.embed_adata
```

### Morphological Embedding

```{eval-rst}
.. currentmodule:: embpy

.. autosummary::
    :toctree: generated

    BioEmbedder.embed_morphological
    BioEmbedder.embed_morphological_batch
    BioEmbedder.embed_perturbation_morphology
```

### Utilities

```{eval-rst}
.. currentmodule:: embpy

.. autosummary::
    :toctree: generated

    BioEmbedder.list_available_models
```

---

## Resources: `embpy.resources`

Resolvers fetch biological sequences and identifiers; annotators add rich metadata.

### Resolvers

```{eval-rst}
.. currentmodule:: embpy

.. autosummary::
    :toctree: generated

    resources.GeneResolver
    resources.ProteinResolver
    resources.DrugResolver
    resources.TextResolver
    resources.OrthologResolver
```

### Annotators

```{eval-rst}
.. currentmodule:: embpy

.. autosummary::
    :toctree: generated

    resources.GeneAnnotator
    resources.ProteinAnnotator
    resources.MoleculeAnnotator
    resources.CellLineAnnotator
```

### Morphology Data Sources

```{eval-rst}
.. currentmodule:: embpy

.. autosummary::
    :toctree: generated

    resources.fetch_hpa_if_image
    resources.load_hpa_if_image
    resources.get_hpa_antibodies
    resources.build_hpa_subcellular_catalog
    resources.download_hpa_subcellular_images
    resources.fetch_jump_fov
    resources.get_jump_gene_mapper
    resources.get_jump_item_location_metadata
    resources.detect_identifier_type
```

---

## Tools: `embpy.tl`

### Similarity and Distance

```{eval-rst}
.. currentmodule:: embpy

.. autosummary::
    :toctree: generated

    tl.compute_similarity
    tl.compute_distance_matrix
    tl.compute_knn_overlap
    tl.rank_perturbations
    tl.pseudobulk_embeddings
```

### Dimensionality Reduction

```{eval-rst}
.. currentmodule:: embpy

.. autosummary::
    :toctree: generated

    tl.compute_umap
    tl.compute_tsne
```

### Clustering

```{eval-rst}
.. currentmodule:: embpy

.. autosummary::
    :toctree: generated

    tl.leiden
    tl.cluster_embeddings
    tl.find_nearest_neighbors
```

### Phenotypic Activity

```{eval-rst}
.. currentmodule:: embpy

.. autosummary::
    :toctree: generated

    tl.phenotypic_activity
```

### Annotation

```{eval-rst}
.. currentmodule:: embpy

.. autosummary::
    :toctree: generated

    tl.annotate_molecules
    tl.annotate_gene_perturbations
    tl.annotate_proteins
    tl.annotate_genes
    tl.annotate_drugs
    tl.annotate_cell_lines
    tl.annotate_perturbation
    tl.annotate_bulk_rna
    tl.annotate_drug_response
    tl.lookup_cell_lines
    tl.lookup_compounds
    tl.lookup_drug_annotation
    tl.lookup_drug_response
    tl.lookup_moa
    tl.lookup_protein_expression
```

### Benchmarking and Metrics

```{eval-rst}
.. currentmodule:: embpy

.. autosummary::
    :toctree: generated

    tl.benchmark_embeddings
    tl.compute_metrics
    tl.cell_eval
    tl.run_cell_eval
    tl.run_pipeline
    tl.list_embedding_models
    tl.list_use_cases
    tl.deg_overlap
    tl.compare_deg
    tl.deg_direction_agreement
    tl.frac_correct_direction
    tl.get_deg_dataframe
    tl.rank_genes_groups
    tl.gene_r2
    tl.r2
    tl.mse
    tl.delta_l2
    tl.mean_correlation
    tl.phenocopy_score
    tl.phenocopy_score_adata
```

### Weighted Protein Embeddings

```{eval-rst}
.. currentmodule:: embpy

.. autosummary::
    :toctree: generated

    tl.WeightedProteinEmbedder
```

### Cross-Species Analysis

```{eval-rst}
.. currentmodule:: embpy

.. autosummary::
    :toctree: generated

    tl.build_cross_species_adata
    tl.ortholog_similarity_matrix
    tl.identity_vs_similarity
```

### SNP and Variant Embedding

```{eval-rst}
.. currentmodule:: embpy

.. autosummary::
    :toctree: generated

    tl.SNPEmbedder
    tl.SNPContext
    tl.SNPEmbeddingResult
    tl.SequenceProvider
    tl.embed_vcf
    tl.download_hg38_per_chrom
    tl.download_hg38_single_fasta
```

---

## Plotting: `embpy.pl`

### Embedding Space

```{eval-rst}
.. currentmodule:: embpy

.. autosummary::
    :toctree: generated

    pl.plot_embedding_space
    pl.all_embeddings
    pl.umap_feature_panel
    pl.tsne_feature_panel
```

### Heatmaps

```{eval-rst}
.. currentmodule:: embpy

.. autosummary::
    :toctree: generated

    pl.plot_similarity_heatmap
    pl.distance_heatmap
    pl.embedding_clustermap
    pl.correlation_matrix
    pl.cross_embedding_correlation
    pl.cross_model_similarity
    pl.cluster_property_heatmap
    pl.knn_overlap
```

### Distributions

```{eval-rst}
.. currentmodule:: embpy

.. autosummary::
    :toctree: generated

    pl.embedding_distributions
    pl.embedding_norms
    pl.plot_perturbation_ranking
```

### Comparisons

```{eval-rst}
.. currentmodule:: embpy

.. autosummary::
    :toctree: generated

    pl.parallel_coordinates
    pl.radar_chart
    pl.star_coordinates
```

### Clustering

```{eval-rst}
.. currentmodule:: embpy

.. autosummary::
    :toctree: generated

    pl.leiden_overview
    pl.plot_cluster_composition
    pl.dendrogram
```

### Cross-Species

```{eval-rst}
.. currentmodule:: embpy

.. autosummary::
    :toctree: generated

    pl.plot_species_umap
    pl.plot_ortholog_similarity
    pl.plot_identity_vs_similarity
    pl.plot_conservation_barplot
```

### Morphology

```{eval-rst}
.. currentmodule:: embpy

.. autosummary::
    :toctree: generated

    pl.plot_cell_painting
```

### Benchmarking

```{eval-rst}
.. currentmodule:: embpy

.. autosummary::
    :toctree: generated

    pl.plot_benchmark
    pl.plot_benchmark_comparison
```

---

## Preprocessing: `embpy.pp`

### Counts and Embeddings

```{eval-rst}
.. currentmodule:: embpy

.. autosummary::
    :toctree: generated

    pp.preprocess_counts
    pp.reduce_embeddings
    pp.PerturbationProcessor
```

### Data Loading

```{eval-rst}
.. currentmodule:: embpy

.. autosummary::
    :toctree: generated

    pp.load_depmap
    pp.list_depmap_datasets
    pp.depmap_info
    pp.load_lamin
    pp.list_lamin_datasets
    pp.lamin_info
```

### Morphology Preprocessing

```{eval-rst}
.. currentmodule:: embpy

.. autosummary::
    :toctree: generated

    pp.cell_painting_to_subcell
    pp.prepare_subcell_canvas
    pp.composite_cell_painting
    pp.normalize_channels
    pp.load_channels_from_pngs
    pp.save_channels_as_pngs
    pp.max_projection_z
    pp.max_projection_z_multichannel
    pp.crop_spatial
    pp.crop_to_mask
    pp.bbox_from_mask
    pp.resize_to_canvas
    pp.rescale_to_target_nm_per_pixel
```
