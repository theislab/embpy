"""Cross-species protein embedding comparison tools.

Functions for building cross-species AnnData objects, computing
ortholog embedding similarity, and analyzing evolutionary conservation
in embedding space.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def build_cross_species_adata(
    embedder: Any,
    genes: list[str],
    species: list[str],
    model: str,
    source_species: str = "human",
    orthology_type: str = "ortholog_one2one",
    pooling_strategy: str = "mean",
) -> "anndata.AnnData":
    """Embed orthologous proteins across species into a single AnnData.

    Resolves orthologs using :class:`OrthologResolver`, fetches protein
    sequences for each species via :class:`ProteinResolver`, and embeds
    them with the specified protein language model.

    Parameters
    ----------
    embedder
        Initialized ``BioEmbedder`` instance.
    genes
        Gene symbols in the source species.
    species
        Species to compare (including the source). Common names
        accepted (``"human"``, ``"mouse"``, ``"zebrafish"``).
    model
        Protein model key (e.g. ``"esm2_650M"``).
    source_species
        Species of the input gene symbols.
    orthology_type
        Ensembl Compara orthology type filter.
    pooling_strategy
        Pooling strategy for the protein model.

    Returns
    -------
    anndata.AnnData
        Observations are individual proteins (one per gene-species pair).
        ``obs`` columns: ``gene_family`` (source symbol), ``species``,
        ``symbol`` (species-specific symbol), ``ensembl_id``,
        ``sequence_identity``, ``protein_length``.
        ``obsm["X_emb"]`` contains the embedding matrix.
    """
    import anndata as ad
    import pandas as pd

    from embpy.resources.ortholog_resolver import OrthologResolver
    from embpy.resources.protein_resolver import ProteinResolver

    orth = OrthologResolver(source_species=source_species)

    species_norm = [orth._normalize(s) for s in species]
    src_norm = orth._normalize(source_species)
    target_species = [s for s in species_norm if s != src_norm]

    ortho_table = orth.get_ortholog_table(
        genes,
        target_species=target_species,
        orthology_type=orthology_type,
    )

    rows: list[dict[str, Any]] = []
    embeddings: list[np.ndarray] = []

    for gene in genes:
        gene_entries: list[tuple[str, str, str, float]] = []

        gene_entries.append((gene, src_norm, gene, 100.0))

        for tgt in target_species:
            short = tgt.split("_")[0]
            row = ortho_table[ortho_table["source_symbol"] == gene]
            if row.empty:
                continue
            sym = row.iloc[0].get(f"{short}_symbol")
            ident = row.iloc[0].get(f"{short}_identity")
            if sym is None or pd.isna(sym):
                logger.info("No %s ortholog for %s", tgt, gene)
                continue
            gene_entries.append((sym, tgt, gene, float(ident or 0)))

        for symbol, sp, family, identity in gene_entries:
            pr = ProteinResolver(organism=sp)
            try:
                emb = embedder.embed_protein(
                    symbol, model=model, id_type="symbol",
                    organism=sp, pooling_strategy=pooling_strategy,
                )
            except Exception as exc:
                logger.warning(
                    "Failed to embed %s (%s): %s", symbol, sp, exc,
                )
                continue

            seq = pr.get_canonical_sequence(symbol, id_type="symbol")
            prot_len = len(seq) if seq else 0

            rows.append({
                "gene_family": family,
                "species": sp,
                "species_short": sp.split("_")[0].capitalize(),
                "symbol": symbol,
                "sequence_identity": identity,
                "protein_length": prot_len,
            })
            embeddings.append(emb)

    if not embeddings:
        raise RuntimeError("No proteins could be embedded across species.")

    obs_df = pd.DataFrame(rows)
    obs_df.index = [
        f"{r['symbol']}_{r['species_short']}" for _, r in obs_df.iterrows()
    ]

    X = np.stack(embeddings)
    adata = ad.AnnData(X=X, obs=obs_df)
    adata.obsm["X_emb"] = X.copy()

    logger.info(
        "Cross-species AnnData: %d proteins across %d species for %d gene families",
        len(rows), len(set(obs_df["species"])), len(set(obs_df["gene_family"])),
    )
    return adata


def ortholog_similarity_matrix(
    adata: "anndata.AnnData",
    obsm_key: str = "X_emb",
    metric: str = "cosine",
) -> "pandas.DataFrame":
    """Compute pairwise similarity between all proteins in the AnnData.

    Returns a labeled DataFrame with gene symbol + species as index.
    """
    import pandas as pd
    from sklearn.metrics.pairwise import cosine_similarity

    X = np.asarray(adata.obsm[obsm_key], dtype=np.float64)

    if metric == "cosine":
        sim = cosine_similarity(X)
    elif metric == "pearson":
        from scipy.stats import pearsonr
        n = X.shape[0]
        sim = np.ones((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                r, _ = pearsonr(X[i], X[j])
                sim[i, j] = sim[j, i] = r
    else:
        raise ValueError(f"Unknown metric: {metric!r}")

    labels = list(adata.obs.index)
    return pd.DataFrame(sim, index=labels, columns=labels)


def identity_vs_similarity(
    adata: "anndata.AnnData",
    obsm_key: str = "X_emb",
) -> "pandas.DataFrame":
    """Compute per-ortholog-pair sequence identity vs embedding similarity.

    For each pair of proteins from the same gene family but different
    species, returns a row with the sequence identity (from Ensembl
    Compara) and the cosine similarity of their embeddings.

    Returns
    -------
    DataFrame with columns: gene_family, species_a, species_b,
    symbol_a, symbol_b, sequence_identity, embedding_similarity.
    """
    import pandas as pd
    from sklearn.metrics.pairwise import cosine_similarity

    X = np.asarray(adata.obsm[obsm_key], dtype=np.float64)
    sim_matrix = cosine_similarity(X)

    rows = []
    obs = adata.obs
    families = obs["gene_family"].unique()

    for family in families:
        mask = obs["gene_family"] == family
        idxs = np.where(mask.values)[0]
        for i in range(len(idxs)):
            for j in range(i + 1, len(idxs)):
                ii, jj = idxs[i], idxs[j]
                rows.append({
                    "gene_family": family,
                    "species_a": obs.iloc[ii]["species_short"],
                    "species_b": obs.iloc[jj]["species_short"],
                    "symbol_a": obs.iloc[ii]["symbol"],
                    "symbol_b": obs.iloc[jj]["symbol"],
                    "sequence_identity": min(
                        obs.iloc[ii]["sequence_identity"],
                        obs.iloc[jj]["sequence_identity"],
                    ),
                    "embedding_similarity": float(sim_matrix[ii, jj]),
                })

    return pd.DataFrame(rows)
