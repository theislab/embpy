# Domain subpackages:
#   pl.protein      -- cross-species protein embedding plots
#   pl.morphology   -- Cell Painting / morphology plots

__all__ = [
    # clustering
    "dendrogram",
    "leiden_overview",
    "plot_cluster_composition",
    # distributions
    "embedding_distributions",
    "embedding_norms",
    "plot_perturbation_ranking",
    # embedding_space
    "all_embeddings",
    "embedding_color_panel",
    "highlight_gene_sets",
    "plot_embedding_space",
    "umap_feature_panel",
    # diagnostics
    "category_centroid_similarity",
    "knn_label_purity",
    "within_vs_between_similarity",
    # heatmaps
    "cluster_property_heatmap",
    "correlation_matrix",
    "cross_embedding_correlation",
    "cross_model_similarity",
    "distance_heatmap",
    "embedding_clustermap",
    "knn_overlap",
    "plot_similarity_heatmap",
    # comparisons
    "parallel_coordinates",
    "radar_chart",
    "star_coordinates",
    "tsne_feature_panel",
    # benchmark
    "plot_benchmark",
    "plot_benchmark_comparison",
    # morphology
    "plot_cell_painting",
    # protein / cross-species
    "plot_conservation_barplot",
    "plot_identity_vs_similarity",
    "plot_ortholog_similarity",
    "plot_species_umap",
]

from .benchmark import plot_benchmark, plot_benchmark_comparison
from .clustering import (
    dendrogram,
    leiden_overview,
    plot_cluster_composition,
)
from .comparisons import (
    parallel_coordinates,
    radar_chart,
    star_coordinates,
    tsne_feature_panel,
)
from .diagnostics import (
    category_centroid_similarity,
    knn_label_purity,
    within_vs_between_similarity,
)
from .distributions import (
    embedding_distributions,
    embedding_norms,
    plot_perturbation_ranking,
)
from .embedding_space import (
    all_embeddings,
    embedding_color_panel,
    highlight_gene_sets,
    plot_embedding_space,
    umap_feature_panel,
)
from .heatmaps import (
    cluster_property_heatmap,
    correlation_matrix,
    cross_embedding_correlation,
    cross_model_similarity,
    distance_heatmap,
    embedding_clustermap,
    knn_overlap,
    plot_similarity_heatmap,
)
from .morphology.cell_painting import plot_cell_painting
from .protein.cross_species import (
    plot_conservation_barplot,
    plot_identity_vs_similarity,
    plot_ortholog_similarity,
    plot_species_umap,
)
