# Domain subpackages:
#   tl/protein/   - weighted embeddings, cross-species analysis
#   tl/genomics/  - SNP/variant embedding utilities

__all__ = [
    # alignment (representation-similarity metrics)
    "alignment_matrix",
    "linear_cka",
    "mutual_knn",
    "qsi",
    "sample_size_for",
    "tsi",
    # clustering
    "cluster_annotation_enrichment",
    "cluster_embeddings",
    "find_nearest_neighbors",
    "leiden",
    # dimred
    "compute_pca",
    "compute_tsne",
    "compute_umap",
    # similarity
    "aggregate_embedding_table",
    "compare_embedding_matrices",
    "compute_distance_matrix",
    "compute_knn_overlap",
    "compute_similarity",
    "embedding_similarity_matrix",
    "knn_jaccard",
    "nearest_neighbors",
    "nearest_neighbors_table",
    "pseudobulk_embeddings",
    "rank_perturbations",
    "similarity_correlation",
    # activity
    "phenotypic_activity",
    # basic
    "basic_tool",
    # benchmark
    "benchmark_embeddings",
    # scib
    "compute_scib_metrics",
    # metadata
    "annotate_bulk_rna",
    "annotate_cell_lines",
    "annotate_drug_response",
    "annotate_drugs",
    "annotate_gene_perturbations",
    "annotate_genes",
    "annotate_molecules",
    "annotate_perturbation",
    "annotate_proteins",
    "lookup_cell_lines",
    "lookup_compounds",
    "lookup_drug_annotation",
    "lookup_drug_response",
    "lookup_moa",
    "lookup_protein_expression",
    # metrics
    "cell_eval",
    "compare_deg",
    "compute_metrics",
    "deg_direction_agreement",
    "deg_overlap",
    "delta_l2",
    "frac_correct_direction",
    "gene_r2",
    "get_deg_dataframe",
    "mean_correlation",
    "mse",
    "phenocopy_score",
    "phenocopy_score_adata",
    "r2",
    "rank_genes_groups",
    # pipeline
    "list_embedding_models",
    "list_use_cases",
    "run_cell_eval",
    "run_pipeline",
    # protein
    "WeightedProteinEmbedder",
    "build_cross_species_adata",
    "identity_vs_similarity",
    "ortholog_similarity_matrix",
    # genomics
    "SequenceProvider",
    "SNPContext",
    "SNPEmbedder",
    "SNPEmbeddingResult",
    "download_hg38_per_chrom",
    "download_hg38_single_fasta",
    "embed_vcf",
]

from .activity import phenotypic_activity
from .alignment import (
    alignment_matrix,
    linear_cka,
    mutual_knn,
    qsi,
    sample_size_for,
    tsi,
)
from .basic import basic_tool
from .benchmark import benchmark_embeddings
from .clustering import (
    cluster_annotation_enrichment,
    cluster_embeddings,
    find_nearest_neighbors,
    leiden,
)
from .dimred import compute_pca, compute_tsne, compute_umap
from .genomics.snp_utils import (
    SequenceProvider,
    SNPContext,
    SNPEmbedder,
    SNPEmbeddingResult,
    download_hg38_per_chrom,
    download_hg38_single_fasta,
    embed_vcf,
)
from .metadata import (
    annotate_bulk_rna,
    annotate_cell_lines,
    annotate_drug_response,
    annotate_drugs,
    annotate_gene_perturbations,
    annotate_genes,
    annotate_molecules,
    annotate_perturbation,
    annotate_proteins,
    lookup_cell_lines,
    lookup_compounds,
    lookup_drug_annotation,
    lookup_drug_response,
    lookup_moa,
    lookup_protein_expression,
)
from .metrics import (
    cell_eval,
    compare_deg,
    compute_metrics,
    deg_direction_agreement,
    deg_overlap,
    delta_l2,
    frac_correct_direction,
    gene_r2,
    get_deg_dataframe,
    mean_correlation,
    mse,
    phenocopy_score,
    phenocopy_score_adata,
    r2,
    rank_genes_groups,
)
from .pipeline import list_embedding_models, list_use_cases, run_cell_eval, run_pipeline
from .protein.cross_species import (
    build_cross_species_adata,
    identity_vs_similarity,
    ortholog_similarity_matrix,
)
from .protein.weighted_embedding import WeightedProteinEmbedder
from .scib_metrics import compute_scib_metrics
from .similarity import (
    aggregate_embedding_table,
    compare_embedding_matrices,
    compute_distance_matrix,
    compute_knn_overlap,
    compute_similarity,
    embedding_similarity_matrix,
    knn_jaccard,
    nearest_neighbors,
    nearest_neighbors_table,
    pseudobulk_embeddings,
    rank_perturbations,
    similarity_correlation,
)
