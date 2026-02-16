from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np
import pandas as pd
from anndata import AnnData
from scipy.sparse import issparse
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from embpy.resources.drug_resolver import DrugResolver
from embpy.resources.gene_resolver import GeneResolver, detect_identifier_type


class PerturbationProcessor:
    """Preprocessor for perturbation experimental data.

    Provides utilities to normalize identifiers (gene symbols, SMILES, etc.),
    resolve them to canonical forms, build embedding matrices from
    BioEmbedder results, and run dimensionality reduction.

    Parameters
    ----------
    embedder : BioEmbedder, optional
        An initialized BioEmbedder instance.  If provided, used for
        resolving identifiers and computing embeddings.
    gene_resolver : GeneResolver, optional
        Explicit GeneResolver instance.  If *embedder* is given this is
        taken from ``embedder.gene_resolver`` automatically.
    """

    def __init__(
        self,
        embedder=None,
        gene_resolver: GeneResolver | None = None,
        drug_resolver: DrugResolver | None = None,
    ):
        self.embedder = embedder
        if gene_resolver is not None:
            self.gene_resolver = gene_resolver
        elif embedder is not None and hasattr(embedder, "gene_resolver"):
            self.gene_resolver = embedder.gene_resolver
        else:
            self.gene_resolver = GeneResolver()
        self.drug_resolver = drug_resolver or DrugResolver()

    # ------------------------------------------------------------------
    # Identifier helpers
    # ------------------------------------------------------------------

    def normalize_gene_names(
        self,
        identifiers: Sequence[str],
        organism: str = "human",
    ) -> dict[str, str | None]:
        """Normalize gene identifiers to canonical Ensembl IDs.

        Parameters
        ----------
        identifiers
            List of gene symbols or Ensembl IDs to normalize.
        organism
            Target organism for resolution.

        Returns
        -------
        dict mapping input identifiers to canonical Ensembl gene IDs,
        or ``None`` where resolution failed.
        """
        result: dict[str, str | None] = {}
        for ident in identifiers:
            kind = detect_identifier_type(ident)
            if kind == "ensembl_id":
                result[ident] = ident.split(".")[0].strip()
            elif kind == "symbol":
                ens = self.gene_resolver.symbol_to_ensembl(ident, organism=organism)
                result[ident] = ens
            else:
                result[ident] = None
        return result

    def resolve_identifiers(
        self,
        identifiers: Sequence[str],
        id_type: str = "auto",
        organism: str = "human",
    ) -> pd.DataFrame:
        """Resolve a mixed list of identifiers to a structured DataFrame.

        Parameters
        ----------
        identifiers
            List of identifiers to resolve.
        id_type
            Type hint: ``"auto"``, ``"gene_symbol"``, ``"ensembl_id"``,
            ``"smiles"``, ``"drug_name"``.
        organism
            Target organism for gene resolution.

        Returns
        -------
        DataFrame with columns ``original_id``, ``canonical_id``,
        ``id_type``, ``resolved``.
        """
        rows: list[dict] = []
        for ident in identifiers:
            detected = detect_identifier_type(ident) if id_type == "auto" else id_type

            canonical: str | None = None
            resolved = False

            if detected == "ensembl_id":
                canonical = ident.split(".")[0].strip()
                resolved = True
            elif detected == "symbol":
                ens = self.gene_resolver.symbol_to_ensembl(ident, organism=organism)
                if ens:
                    canonical = ens
                    resolved = True
            elif detected == "smiles":
                canonical = ident
                resolved = True
            elif detected == "dna_sequence":
                canonical = ident[:40]
                resolved = True
            elif detected == "protein_sequence":
                canonical = ident[:40]
                resolved = True
            else:
                canonical = ident

            rows.append(
                {
                    "original_id": ident,
                    "canonical_id": canonical,
                    "id_type": detected,
                    "resolved": resolved,
                }
            )

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Embedding matrix construction
    # ------------------------------------------------------------------

    def build_embedding_matrix(
        self,
        identifiers: Sequence[str],
        model: str,
        id_type: str = "symbol",
        organism: str = "human",
        pooling_strategy: str = "mean",
        obsm_key: str | None = None,
    ) -> AnnData:
        """Build an AnnData with embeddings stored in ``.obsm``.

        For every identifier the ``BioEmbedder`` is used to compute an
        embedding.  Successful embeddings are stored as a 2-D numpy array
        in ``adata.obsm[obsm_key]``.  A boolean column
        ``adata.obs["embedded"]`` records which perturbations succeeded.

        Parameters
        ----------
        identifiers
            Perturbation identifiers (genes, molecules, etc.).
        model
            Model name (e.g. ``"esm2_650M"``, ``"chemberta2MTR"``).
        id_type
            Identifier type (``"symbol"``, ``"ensembl_id"``, …).
        organism
            Target organism.
        pooling_strategy
            Pooling strategy for the model.
        obsm_key
            Key under which the matrix is stored in ``.obsm``.
            Defaults to ``"X_{model}"``.

        Returns
        -------
        AnnData of shape ``(n_identifiers, 0)`` with the embedding
        matrix in ``adata.obsm[obsm_key]`` and metadata in ``adata.obs``.
        """
        if self.embedder is None:
            raise ValueError("An initialized BioEmbedder is required for build_embedding_matrix.")

        key = obsm_key or f"X_{model}"
        n = len(identifiers)
        id_list = list(identifiers)

        embeddings: list[np.ndarray | None] = []
        for ident in id_list:
            try:
                emb = self.embedder.embed_gene(
                    identifier=ident,
                    model=model,
                    id_type=id_type,
                    organism=organism,
                    pooling_strategy=pooling_strategy,
                )
                embeddings.append(np.asarray(emb, dtype=np.float32).ravel())
            except Exception:  # noqa: BLE001
                logging.warning(f"Embedding failed for '{ident}'; marking as failed.")
                embeddings.append(None)

        embedded_mask = [e is not None for e in embeddings]

        # Determine embedding dimension from the first successful entry
        emb_dim = 0
        for e in embeddings:
            if e is not None:
                emb_dim = e.shape[0]
                break

        matrix = np.zeros((n, emb_dim), dtype=np.float32)
        for i, e in enumerate(embeddings):
            if e is not None:
                matrix[i] = e

        obs = pd.DataFrame(
            {
                "identifier": id_list,
                "id_type": id_type,
                "model": model,
                "pooling": pooling_strategy,
                "embedded": embedded_mask,
            }
        )
        obs.index = obs.index.astype(str)

        adata = AnnData(obs=obs)
        adata.obsm[key] = matrix
        logging.info(
            f"Built embedding matrix: {sum(embedded_mask)}/{n} succeeded, dim={emb_dim}, stored in .obsm['{key}']."
        )
        return adata

    def build_molecule_embedding_matrix(
        self,
        adata: AnnData | None = None,
        identifiers: Sequence[str] | None = None,
        column: str | None = None,
        id_type: str = "auto",
        model: str = "chemberta2MTR",
        pooling_strategy: str = "mean",
        obsm_key: str | None = None,
    ) -> AnnData:
        """Build an AnnData with molecule embeddings stored in ``.obsm``.

        Accepts **either** an existing ``AnnData`` whose ``.obs`` contains a
        column with drug identifiers, **or** a plain list of identifier
        strings.  When an ``AnnData`` is provided, the embeddings are added
        to a *copy* of it; when a list is provided a fresh ``AnnData`` is
        created.

        Drug names are automatically resolved to canonical SMILES via the
        ``DrugResolver`` before embedding.

        Parameters
        ----------
        adata
            An existing ``AnnData`` object.  If given, identifiers are read
            from ``adata.obs[column]``.
        identifiers
            Explicit list of drug identifiers (names or SMILES).  Ignored
            when *adata* is provided.
        column
            Column in ``adata.obs`` that contains the drug identifiers.
            Required when *adata* is given.  When *identifiers* is used
            instead, this column name is stored in the resulting ``.obs``
            as the identifier column (defaults to ``"drug_id"``).
        id_type
            ``"auto"`` (detect per identifier), ``"smiles"``, or
            ``"drug_name"``.
        model
            Model name from the registry (e.g. ``"chemberta2MTR"``,
            ``"molformer_base"``).
        pooling_strategy
            Pooling applied by the model wrapper.
        obsm_key
            Key under which the embedding matrix is stored in ``.obsm``.
            Defaults to ``"X_{model}"``.

        Returns
        -------
        AnnData with the embedding matrix in ``adata.obsm[obsm_key]``,
        a boolean ``adata.obs["embedded"]`` column, and the resolved
        SMILES in ``adata.obs["smiles"]``.
        """
        if self.embedder is None:
            raise ValueError(
                "An initialized BioEmbedder is required for "
                "build_molecule_embedding_matrix."
            )

        # ---- Resolve the identifier list --------------------------------
        if adata is not None:
            if column is None:
                raise ValueError(
                    "When passing an AnnData, 'column' must specify the "
                    ".obs column containing drug identifiers."
                )
            if column not in adata.obs.columns:
                raise KeyError(
                    f"Column '{column}' not found in adata.obs. "
                    f"Available: {list(adata.obs.columns)}"
                )
            id_list: list[str] = adata.obs[column].astype(str).tolist()
            result_adata = adata.copy()
        elif identifiers is not None:
            id_list = list(identifiers)
            col_name = column or "drug_id"
            result_adata = AnnData(
                obs=pd.DataFrame({col_name: id_list}),
            )
            result_adata.obs.index = result_adata.obs.index.astype(str)
        else:
            raise ValueError(
                "Provide either 'adata' (+ column) or 'identifiers'."
            )

        key = obsm_key or f"X_{model}"
        n = len(id_list)

        # ---- Resolve identifiers to SMILES ------------------------------
        smiles_list: list[str | None] = []
        for ident in id_list:
            ident = ident.strip()
            if id_type == "smiles":
                smiles_list.append(ident)
            elif id_type == "drug_name":
                smiles_list.append(self.drug_resolver.name_to_smiles(ident))
            else:
                # auto-detect
                detected = detect_identifier_type(ident)
                if detected == "smiles":
                    smiles_list.append(ident)
                else:
                    # Treat as a drug name and resolve
                    smi = self.drug_resolver.name_to_smiles(ident)
                    smiles_list.append(smi)

        # ---- Embed each resolved SMILES ---------------------------------
        embeddings: list[np.ndarray | None] = []
        for smi in smiles_list:
            if smi is None:
                embeddings.append(None)
                continue
            try:
                emb = self.embedder.embed_molecule(
                    identifier=smi,
                    model=model,
                    pooling_strategy=pooling_strategy,
                )
                embeddings.append(
                    np.asarray(emb, dtype=np.float32).ravel()
                )
            except Exception:  # noqa: BLE001
                logging.warning(
                    "Embedding failed for SMILES '%s'; marking as failed.",
                    smi,
                )
                embeddings.append(None)

        embedded_mask = [e is not None for e in embeddings]

        # Determine embedding dimension
        emb_dim = 0
        for e in embeddings:
            if e is not None:
                emb_dim = e.shape[0]
                break

        matrix = np.zeros((n, emb_dim), dtype=np.float32)
        for i, e in enumerate(embeddings):
            if e is not None:
                matrix[i] = e

        # ---- Store results in AnnData -----------------------------------
        result_adata.obs["smiles"] = [s or "" for s in smiles_list]
        result_adata.obs["embedded"] = embedded_mask
        result_adata.obs["model"] = model
        result_adata.obs["pooling"] = pooling_strategy
        result_adata.obsm[key] = matrix

        logging.info(
            "Built molecule embedding matrix: %d/%d succeeded, dim=%d, "
            "stored in .obsm['%s'].",
            sum(embedded_mask),
            n,
            emb_dim,
            key,
        )
        return result_adata

    # ------------------------------------------------------------------
    # Filtering & combining
    # ------------------------------------------------------------------

    def filter_failed_embeddings(
        self,
        adata: AnnData,
        obsm_key: str | None = None,
    ) -> AnnData:
        """Remove perturbations that failed to embed.

        Parameters
        ----------
        adata
            AnnData returned by :meth:`build_embedding_matrix`.
        obsm_key
            The ``.obsm`` key to filter on.  If ``None`` the first key
            found in ``.obsm`` is used.

        Returns
        -------
        Filtered AnnData with only the successfully embedded rows.
        """
        if "embedded" not in adata.obs.columns:
            raise ValueError("adata.obs must contain an 'embedded' column (from build_embedding_matrix).")

        mask = adata.obs["embedded"].values.astype(bool)
        return adata[mask].copy()

    def combine_perturbation_spaces(
        self,
        *adatas: AnnData,
        labels: Sequence[str] | None = None,
        obsm_key: str | None = None,
    ) -> AnnData:
        """Combine multiple perturbation embedding AnnData objects.

        Concatenates rows and stores the unioned ``.obsm`` matrices.
        A ``perturbation_type`` column is added to ``obs``.

        Parameters
        ----------
        *adatas
            One or more AnnData objects (e.g. genetic, molecular).
        labels
            Per-AnnData labels written to ``obs["perturbation_type"]``.
            Defaults to ``["set_0", "set_1", …]``.
        obsm_key
            If given, only this key is kept in ``.obsm`` across all
            inputs (they must share the same key and dimensionality).

        Returns
        -------
        Combined AnnData.
        """
        import anndata as ad

        if not adatas:
            raise ValueError("At least one AnnData must be provided.")

        if labels is None:
            labels = [f"set_{i}" for i in range(len(adatas))]
        if len(labels) != len(adatas):
            raise ValueError("Number of labels must match number of AnnData objects.")

        for a, label in zip(adatas, labels, strict=True):
            a.obs["perturbation_type"] = label

        combined = ad.concat(list(adatas), join="outer")
        logging.info(f"Combined {len(adatas)} perturbation spaces → {combined.n_obs} observations.")
        return combined

    # ------------------------------------------------------------------
    # Dimensionality reduction
    # ------------------------------------------------------------------

    @staticmethod
    def reduce_embeddings(
        adata: AnnData,
        obsm_key: str,
        n_components: int = 50,
        scale: bool = True,
        output_key: str | None = None,
    ) -> AnnData:
        """Reduce embedding dimensionality via PCA.

        The pipeline is:

        1. Optionally standardise features with ``StandardScaler``
           (zero-mean, unit-variance).
        2. Fit PCA and transform.
        3. Store the reduced matrix back in ``.obsm``.

        Parameters
        ----------
        adata
            AnnData whose ``.obsm[obsm_key]`` contains the embedding
            matrix to reduce.
        obsm_key
            Key in ``.obsm`` holding the full-dimensional embeddings.
        n_components
            Number of principal components to keep.
        scale
            If ``True`` (default), apply ``StandardScaler`` before PCA.
            This is recommended when combining embeddings from different
            models, as their scales can differ significantly.
        output_key
            Key under which the reduced matrix is stored.
            Defaults to ``"{obsm_key}_pca"``.

        Returns
        -------
        The same AnnData (modified **in-place**) with the reduced
        matrix in ``.obsm[output_key]`` and the fitted ``PCA`` and
        ``StandardScaler`` objects stored in ``.uns``.
        """
        if obsm_key not in adata.obsm:
            raise KeyError(f"'{obsm_key}' not found in adata.obsm. Available: {list(adata.obsm.keys())}")

        X_raw = adata.obsm[obsm_key]
        if issparse(X_raw):
            X = np.asarray(X_raw.todense(), dtype=np.float64)
        else:
            X = np.asarray(X_raw, dtype=np.float64)

        n_samples, n_features = X.shape
        actual_components = min(n_components, n_samples, n_features)

        if scale:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
        else:
            scaler = None
            X_scaled = X

        pca = PCA(n_components=actual_components, random_state=0)
        X_reduced = pca.fit_transform(X_scaled).astype(np.float32)

        out_key = output_key or f"{obsm_key}_pca"
        adata.obsm[out_key] = X_reduced

        uns_key = f"{out_key}_params"
        adata.uns[uns_key] = {
            "n_components": actual_components,
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "total_variance_explained": float(pca.explained_variance_ratio_.sum()),
            "scaled": scale,
        }

        logging.info(
            f"PCA: {n_features} → {actual_components} components "
            f"({pca.explained_variance_ratio_.sum():.1%} variance explained), "
            f"stored in .obsm['{out_key}']."
        )
        return adata


def reduce_embeddings(
    adata: AnnData,
    obsm_key: str,
    n_components: int = 50,
    scale: bool = True,
    output_key: str | None = None,
) -> AnnData:
    """Convenience wrapper around :meth:`PerturbationProcessor.reduce_embeddings`.

    See :meth:`PerturbationProcessor.reduce_embeddings` for full docs.
    """
    return PerturbationProcessor.reduce_embeddings(
        adata=adata,
        obsm_key=obsm_key,
        n_components=n_components,
        scale=scale,
        output_key=output_key,
    )
