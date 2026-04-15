"""Cross-species protein embedding visualizations."""

from __future__ import annotations

from typing import Any

import numpy as np
from matplotlib.figure import Figure


def plot_ortholog_similarity(
    adata: Any,
    obsm_key: str = "X_emb",
    group_by: str = "gene_family",
    species_col: str = "species_short",
    metric: str = "cosine",
    figsize: tuple[float, float] = (12, 10),
    cmap: str = "RdBu_r",
    title: str | None = None,
) -> Figure:
    """Clustered heatmap of cross-species protein similarity.

    Rows and columns are proteins labeled by symbol + species.
    Row/col colors indicate gene family and species.

    Parameters
    ----------
    adata
        AnnData from :func:`build_cross_species_adata`.
    obsm_key
        Embedding key.
    group_by
        Column for row color grouping (typically ``"gene_family"``).
    species_col
        Column for species color annotation.
    metric
        Similarity metric.
    figsize
        Figure size.
    cmap
        Colormap for the similarity values.
    title
        Plot title.
    """
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns
    from sklearn.metrics.pairwise import cosine_similarity

    X = np.asarray(adata.obsm[obsm_key], dtype=np.float64)
    if metric == "cosine":
        sim = cosine_similarity(X)
    else:
        raise ValueError(f"Unsupported metric: {metric!r}")

    labels = list(adata.obs.index)
    sim_df = pd.DataFrame(sim, index=labels, columns=labels)

    # Color annotations
    families = adata.obs[group_by]
    species = adata.obs[species_col]

    family_palette = dict(zip(
        families.unique(),
        sns.color_palette("husl", n_colors=families.nunique()),
    ))
    species_palette = dict(zip(
        species.unique(),
        sns.color_palette("Set2", n_colors=species.nunique()),
    ))

    row_colors = pd.DataFrame({
        "Gene family": families.map(family_palette),
        "Species": species.map(species_palette),
    }, index=labels)

    cg = sns.clustermap(
        sim_df,
        cmap=cmap,
        vmin=-1, vmax=1,
        figsize=figsize,
        row_colors=row_colors,
        col_colors=row_colors,
        linewidths=0.5,
        dendrogram_ratio=(0.15, 0.15),
        cbar_pos=(0.02, 0.82, 0.03, 0.15),
    )

    if title:
        cg.fig.suptitle(title, y=1.02, fontsize=14, fontweight="bold")

    # Add legend for species
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=c, label=s) for s, c in species_palette.items()
    ]
    cg.ax_heatmap.legend(
        handles=legend_elements, title="Species",
        loc="upper left", bbox_to_anchor=(1.05, 1.0),
        frameon=True, fontsize=8, title_fontsize=9,
    )

    return cg.fig


def plot_identity_vs_similarity(
    identity_df: Any,
    figsize: tuple[float, float] = (8, 6),
    title: str | None = None,
) -> Figure:
    """Scatter plot of sequence identity vs embedding cosine similarity.

    Each point is an ortholog pair. Points are colored by gene family.

    Parameters
    ----------
    identity_df
        DataFrame from :func:`identity_vs_similarity`.
    figsize
        Figure size.
    title
        Plot title.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, ax = plt.subplots(figsize=figsize)

    families = identity_df["gene_family"].unique()
    palette = dict(zip(families, sns.color_palette("husl", len(families))))

    for family in families:
        sub = identity_df[identity_df["gene_family"] == family]
        ax.scatter(
            sub["sequence_identity"],
            sub["embedding_similarity"],
            label=family, s=60, alpha=0.8,
            edgecolors="k", linewidth=0.4,
            color=palette[family],
        )

    # Trend line
    x = identity_df["sequence_identity"].values
    y = identity_df["embedding_similarity"].values
    if len(x) > 2:
        from scipy import stats
        slope, intercept, r_value, p_value, _ = stats.linregress(x, y)
        x_fit = np.linspace(x.min(), x.max(), 100)
        ax.plot(
            x_fit, slope * x_fit + intercept,
            "k--", alpha=0.5, linewidth=1.5,
            label=f"R={r_value:.3f}, p={p_value:.2e}",
        )

    ax.set_xlabel("Sequence identity (%)", fontsize=12)
    ax.set_ylabel("Embedding cosine similarity", fontsize=12)
    ax.set_title(
        title or "Sequence identity vs. embedding similarity",
        fontsize=13, fontweight="bold",
    )
    ax.legend(fontsize=8, loc="lower right", frameon=True)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_species_umap(
    adata: Any,
    obsm_key: str = "X_emb",
    figsize: tuple[float, float] = (16, 6),
    title: str | None = None,
) -> Figure:
    """Side-by-side UMAP colored by species and gene family.

    Parameters
    ----------
    adata
        AnnData from :func:`build_cross_species_adata`.
    obsm_key
        Embedding key for UMAP computation.
    figsize
        Figure size for the two-panel layout.
    title
        Overall figure title.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    from embpy.tl.dimred import compute_umap

    umap_key = f"X_umap_{obsm_key}"
    if umap_key not in adata.obsm:
        n_neighbors = min(5, adata.n_obs - 1)
        compute_umap(adata, obsm_key=obsm_key, n_neighbors=n_neighbors, output_key=umap_key)

    coords = np.asarray(adata.obsm[umap_key])

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    for ax, col, panel_title in [
        (axes[0], "species_short", "Colored by species"),
        (axes[1], "gene_family", "Colored by gene family"),
    ]:
        cats = adata.obs[col].astype("category")
        palette = sns.color_palette("husl", n_colors=cats.cat.categories.size)
        color_map = dict(zip(cats.cat.categories, palette))
        colors = [color_map[c] for c in cats]

        ax.scatter(
            coords[:, 0], coords[:, 1],
            c=colors, s=80, alpha=0.85,
            edgecolors="k", linewidth=0.4,
        )

        # Add gene labels
        for idx in range(adata.n_obs):
            ax.annotate(
                adata.obs["symbol"].iloc[idx],
                (coords[idx, 0], coords[idx, 1]),
                fontsize=6, alpha=0.7,
                textcoords="offset points", xytext=(5, 3),
            )

        # Legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=c, label=cat) for cat, c in color_map.items()
        ]
        ax.legend(
            handles=legend_elements, fontsize=7, loc="best",
            frameon=True, ncol=max(1, len(legend_elements) // 6),
        )
        ax.set_title(panel_title, fontsize=12, fontweight="bold")
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")

    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    return fig


def plot_conservation_barplot(
    adata: Any,
    obsm_key: str = "X_emb",
    reference_species: str = "Homo",
    figsize: tuple[float, float] = (12, 6),
    title: str | None = None,
) -> Figure:
    """Bar plot of embedding similarity to the reference species per gene family.

    For each gene family, shows how similar each species' ortholog
    embedding is to the reference (typically human).

    Parameters
    ----------
    adata
        AnnData from :func:`build_cross_species_adata`.
    obsm_key
        Embedding key.
    reference_species
        The reference species (short name prefix, e.g. ``"Homo"``).
    figsize
        Figure size.
    title
        Plot title.
    """
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns
    from sklearn.metrics.pairwise import cosine_similarity

    X = np.asarray(adata.obsm[obsm_key], dtype=np.float64)
    sim = cosine_similarity(X)
    obs = adata.obs

    rows = []
    for family in obs["gene_family"].unique():
        fam_mask = obs["gene_family"] == family
        ref_mask = fam_mask & obs["species_short"].str.startswith(reference_species)
        if ref_mask.sum() == 0:
            continue
        ref_idx = np.where(ref_mask.values)[0][0]
        for idx in np.where(fam_mask.values)[0]:
            if idx == ref_idx:
                continue
            rows.append({
                "gene_family": family,
                "species": obs.iloc[idx]["species_short"],
                "similarity": float(sim[ref_idx, idx]),
                "identity": float(obs.iloc[idx]["sequence_identity"]),
            })

    if not rows:
        raise ValueError("No ortholog pairs found for the reference species.")

    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=figsize)
    sns.barplot(
        data=df, x="gene_family", y="similarity",
        hue="species", ax=ax, edgecolor="k", linewidth=0.5,
    )
    ax.set_xlabel("Gene family", fontsize=12)
    ax.set_ylabel(f"Cosine similarity to {reference_species}", fontsize=12)
    ax.set_title(
        title or f"Embedding conservation relative to {reference_species}",
        fontsize=13, fontweight="bold",
    )
    ax.legend(title="Species", fontsize=8, title_fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.grid(True, axis="y", alpha=0.3)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    return fig
