"""Weighted protein embedding strategies for perturbation modeling.

Combines protein language model embeddings with expression data and
functional annotations to produce biologically-informed gene
representations.

Three strategies are available:

1. **TPM-weighted isoform average**: embed each isoform separately,
   then compute a weighted average using expression-level (TPM) values
   per isoform.

2. **Annotation-weighted pooling**: use UniProt functional site
   positions (active sites, binding sites, domains) to create a
   per-residue importance mask that upweights functionally important
   residues during pooling.

3. **Expression-context embedding**: concatenate the protein embedding
   with a tissue/cell-type expression vector to capture the expression
   context of the perturbation.
"""

from __future__ import annotations

import logging
from typing import Any, Literal

import numpy as np

logger = logging.getLogger(__name__)


class WeightedProteinEmbedder:
    """Produce biologically-weighted protein embeddings.

    Parameters
    ----------
    embedder
        An initialized ``BioEmbedder`` instance.
    organism : str
        Organism for sequence/annotation resolution.
    """

    def __init__(self, embedder: Any, organism: str = "human") -> None:
        self.embedder = embedder
        self.organism = organism

    # ==================================================================
    # 1. TPM-weighted isoform average
    # ==================================================================

    def tpm_weighted_embedding(
        self,
        gene: str,
        model: str,
        tpm_values: dict[str, float] | None = None,
        adata: Any = None,
        tpm_layer: str | None = None,
        pooling_strategy: str = "mean",
        id_type: str = "symbol",
    ) -> np.ndarray:
        """Compute a TPM-weighted average of isoform embeddings.

        For each isoform of the gene, embed the protein sequence and
        weight the embedding by the isoform's expression level (TPM).

        Parameters
        ----------
        gene
            Gene symbol or Ensembl ID.
        model
            Protein model name (e.g. ``"esm2_650M"``).
        tpm_values
            Dict mapping isoform/transcript IDs to TPM values.
            If ``None``, all isoforms are weighted equally.
        adata
            Optional AnnData with expression data. If provided and
            ``tpm_values`` is None, TPM values are extracted from
            the expression matrix.
        tpm_layer
            Layer in ``adata`` containing TPM values (default ``.X``).
        pooling_strategy
            Pooling for the per-isoform embedding.
        id_type
            Identifier type for the gene.

        Returns
        -------
        np.ndarray
            1-D weighted average embedding.
        """
        isoform_embs = self.embedder.embed_protein(
            identifier=gene,
            model=model,
            id_type=id_type,
            organism=self.organism,
            pooling_strategy=pooling_strategy,
            isoform="all",
        )

        if not isinstance(isoform_embs, dict) or not isoform_embs:
            logger.warning(
                "No isoform embeddings for %s; falling back to canonical",
                gene,
            )
            return self.embedder.embed_protein(
                identifier=gene, model=model, id_type=id_type,
                organism=self.organism, pooling_strategy=pooling_strategy,
                isoform="canonical",
            )

        iso_ids = list(isoform_embs.keys())
        emb_matrix = np.stack([isoform_embs[k] for k in iso_ids])

        if tpm_values is not None:
            weights = np.array([
                tpm_values.get(iso_id, 0.0) for iso_id in iso_ids
            ], dtype=np.float64)
        elif adata is not None:
            weights = self._extract_tpm_weights(
                adata, gene, iso_ids, tpm_layer,
            )
        else:
            weights = np.ones(len(iso_ids), dtype=np.float64)

        total = weights.sum()
        if total <= 0:
            weights = np.ones(len(iso_ids), dtype=np.float64)
            total = weights.sum()

        weights = weights / total
        weighted_emb = (emb_matrix * weights[:, np.newaxis]).sum(axis=0)

        logger.info(
            "TPM-weighted embedding for %s: %d isoforms, weights=%s",
            gene, len(iso_ids),
            {k: round(w, 3) for k, w in zip(iso_ids, weights)},
        )
        return weighted_emb.astype(np.float32)

    def _extract_tpm_weights(
        self,
        adata: Any,
        gene: str,
        iso_ids: list[str],
        tpm_layer: str | None,
    ) -> np.ndarray:
        """Extract TPM weights from an AnnData for isoform IDs."""
        import scipy.sparse as sp

        if tpm_layer and tpm_layer in adata.layers:
            X = adata.layers[tpm_layer]
        else:
            X = adata.X

        weights = np.zeros(len(iso_ids), dtype=np.float64)

        var_names = list(adata.var_names)
        for i, iso_id in enumerate(iso_ids):
            base_id = iso_id.split("-")[0]
            for candidate in (iso_id, base_id, gene):
                if candidate in var_names:
                    col_idx = var_names.index(candidate)
                    if sp.issparse(X):
                        val = X[:, col_idx].toarray().mean()
                    else:
                        val = X[:, col_idx].mean()
                    weights[i] = float(val)
                    break

        return weights

    # ==================================================================
    # 2. Annotation-weighted pooling
    # ==================================================================

    def annotation_weighted_embedding(
        self,
        gene: str,
        model: str,
        weight_sites: list[str] | None = None,
        site_boost: float = 3.0,
        pooling_strategy: str = "mean",
        id_type: str = "symbol",
    ) -> np.ndarray:
        """Embed a protein with upweighted functionally important residues.

        Uses UniProt functional site annotations to create a per-residue
        importance mask. Residues at active sites, binding sites, or
        domain boundaries get higher weight during pooling.

        Parameters
        ----------
        gene
            Gene symbol or Ensembl ID.
        model
            Protein model name.
        weight_sites
            Which site types to upweight. Default:
            ``["active_sites", "binding_sites", "motifs"]``.
        site_boost
            Multiplicative factor for residues at functional sites.
            E.g. ``3.0`` means those residues count 3x more than
            background residues.
        pooling_strategy
            Base pooling (only ``"mean"`` supports annotation weighting;
            others fall back to standard pooling).
        id_type
            Identifier type.

        Returns
        -------
        np.ndarray
            1-D embedding with annotation-weighted pooling.
        """
        if weight_sites is None:
            weight_sites = ["active_sites", "binding_sites", "motifs"]

        from ..models.protein_models import ESM2Wrapper

        seq = self.embedder.protein_resolver.get_canonical_sequence(
            gene, id_type=id_type, organism=self.organism,
        )
        if not seq:
            raise ValueError(f"Could not resolve protein sequence for {gene}")

        inst = self.embedder._get_model(model)

        if not hasattr(inst, "tokenizer") or inst.tokenizer is None:
            logger.info(
                "Model %s does not expose tokenizer; "
                "falling back to standard pooling", model,
            )
            return inst.embed(input=seq, pooling_strategy=pooling_strategy)

        import torch

        tokenized = inst.tokenizer(seq, return_tensors="pt", truncation=True)
        input_ids = tokenized["input_ids"].to(inst.device)
        attention_mask = tokenized.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(inst.device)

        with torch.no_grad():
            outputs = inst.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            hidden = outputs.last_hidden_state

        if hidden.dim() == 3 and hidden.shape[0] == 1:
            hidden = hidden.squeeze(0)

        seq_len = hidden.shape[0]
        importance = torch.ones(seq_len, device=hidden.device)

        from embpy.resources.protein_annotator import ProteinAnnotator
        pa = ProteinAnnotator(organism=self.organism)
        accession = self.embedder.protein_resolver.resolve_uniprot_id(
            gene, id_type=id_type, organism=self.organism,
        )
        if accession:
            entry = pa._fetch_uniprot_entry(accession)
            if entry:
                sites = pa.get_functional_sites(entry)
                for site_type in weight_sites:
                    for site in sites.get(site_type, []):
                        start = site.get("start")
                        end_val = site.get("end")
                        if start is not None and end_val is not None:
                            s = max(0, int(start) - 1)
                            e = min(seq_len, int(end_val))
                            importance[s:e] = site_boost

        importance = importance.unsqueeze(-1)
        if attention_mask is not None:
            mask = attention_mask.squeeze(0).unsqueeze(-1).float()
            importance = importance * mask

        weighted = (hidden * importance).sum(dim=0) / importance.sum(dim=0).clamp(min=1)
        result = weighted.float().cpu().numpy()

        n_boosted = int((importance.squeeze(-1) > 1.0).sum().item())
        logger.info(
            "Annotation-weighted embedding for %s: %d/%d residues boosted (%.1fx)",
            gene, n_boosted, seq_len, site_boost,
        )
        return result

    # ==================================================================
    # 3. Expression-context embedding
    # ==================================================================

    def expression_context_embedding(
        self,
        gene: str,
        model: str,
        expression_vector: np.ndarray | None = None,
        adata: Any = None,
        use_gtex: bool = False,
        pooling_strategy: str = "mean",
        id_type: str = "symbol",
    ) -> np.ndarray:
        """Concatenate protein embedding with expression context.

        The final embedding is ``[protein_emb || expression_vector]``,
        giving the model both sequence-level and expression-level
        information about the perturbation target.

        Parameters
        ----------
        gene
            Gene symbol or Ensembl ID.
        model
            Protein model name.
        expression_vector
            Pre-computed expression vector to concatenate. If ``None``,
            extracted from ``adata`` or GTEx.
        adata
            AnnData to extract per-gene expression profile from.
            Uses mean expression across cells as the context vector.
        use_gtex
            If ``True`` and no other expression data is provided,
            fetch the GTEx tissue expression profile (54-dim vector).
        pooling_strategy
            Pooling for the protein embedding.
        id_type
            Identifier type.

        Returns
        -------
        np.ndarray
            1-D concatenated embedding ``[protein_emb, expression_ctx]``.
        """
        protein_emb = self.embedder.embed_protein(
            identifier=gene, model=model, id_type=id_type,
            organism=self.organism, pooling_strategy=pooling_strategy,
            isoform="canonical",
        )

        if expression_vector is not None:
            ctx = np.asarray(expression_vector, dtype=np.float32)
        elif adata is not None:
            ctx = self._extract_expression_context(adata, gene)
        elif use_gtex:
            ctx = self._fetch_gtex_context(gene)
        else:
            logger.warning(
                "No expression context provided for %s; "
                "returning protein embedding only", gene,
            )
            return protein_emb

        combined = np.concatenate([protein_emb, ctx])
        logger.info(
            "Expression-context embedding for %s: protein=%d + context=%d = %d",
            gene, len(protein_emb), len(ctx), len(combined),
        )
        return combined

    def _extract_expression_context(
        self, adata: Any, gene: str,
    ) -> np.ndarray:
        """Extract mean expression profile for a gene from AnnData."""
        import scipy.sparse as sp

        if gene in adata.var_names:
            col_idx = list(adata.var_names).index(gene)
            if sp.issparse(adata.X):
                profile = np.asarray(
                    adata.X[:, col_idx].toarray(), dtype=np.float32,
                ).ravel()
            else:
                profile = np.asarray(
                    adata.X[:, col_idx], dtype=np.float32,
                ).ravel()
            return profile

        logger.warning("Gene %s not in adata.var_names", gene)
        return np.zeros(1, dtype=np.float32)

    def _fetch_gtex_context(self, gene: str) -> np.ndarray:
        """Fetch GTEx tissue expression as a context vector."""
        from embpy.resources.gene_annotator import GeneAnnotator
        ga = GeneAnnotator(organism=self.organism)
        tissues = ga.get_tissue_expression(gene)

        if not tissues:
            return np.zeros(54, dtype=np.float32)

        tpm_values = [t.get("median_tpm", 0) for t in tissues]
        ctx = np.array(tpm_values, dtype=np.float32)

        total = ctx.sum()
        if total > 0:
            ctx = ctx / total

        return ctx

    # ==================================================================
    # Convenience: combined embedding for perturbation
    # ==================================================================

    def embed_perturbation(
        self,
        gene: str,
        model: str,
        strategy: Literal[
            "canonical", "tpm_weighted", "annotation_weighted",
            "expression_context", "full",
        ] = "canonical",
        tpm_values: dict[str, float] | None = None,
        adata: Any = None,
        tpm_layer: str | None = None,
        weight_sites: list[str] | None = None,
        site_boost: float = 3.0,
        expression_vector: np.ndarray | None = None,
        use_gtex: bool = False,
        pooling_strategy: str = "mean",
        id_type: str = "symbol",
    ) -> np.ndarray | dict[str, np.ndarray]:
        """Unified entry point for weighted perturbation embeddings.

        Parameters
        ----------
        gene
            Gene symbol or Ensembl ID.
        model
            Protein model name.
        strategy
            Embedding strategy:

            - ``"canonical"`` -- standard canonical protein embedding
            - ``"tpm_weighted"`` -- TPM-weighted isoform average
            - ``"annotation_weighted"`` -- functional-site-weighted
              residue pooling
            - ``"expression_context"`` -- protein emb + expression
              context concatenation
            - ``"full"`` -- returns a dict with all strategies

        Returns
        -------
        np.ndarray
            When strategy is not ``"full"``.
        dict[str, np.ndarray]
            When ``strategy="full"`` -- keys are strategy names.
        """
        if strategy == "canonical":
            return self.embedder.embed_protein(
                identifier=gene, model=model, id_type=id_type,
                organism=self.organism, pooling_strategy=pooling_strategy,
                isoform="canonical",
            )

        if strategy == "tpm_weighted":
            return self.tpm_weighted_embedding(
                gene, model, tpm_values=tpm_values, adata=adata,
                tpm_layer=tpm_layer, pooling_strategy=pooling_strategy,
                id_type=id_type,
            )

        if strategy == "annotation_weighted":
            return self.annotation_weighted_embedding(
                gene, model, weight_sites=weight_sites,
                site_boost=site_boost,
                pooling_strategy=pooling_strategy, id_type=id_type,
            )

        if strategy == "expression_context":
            return self.expression_context_embedding(
                gene, model, expression_vector=expression_vector,
                adata=adata, use_gtex=use_gtex,
                pooling_strategy=pooling_strategy, id_type=id_type,
            )

        if strategy == "full":
            results: dict[str, np.ndarray] = {}
            results["canonical"] = self.embedder.embed_protein(
                identifier=gene, model=model, id_type=id_type,
                organism=self.organism, pooling_strategy=pooling_strategy,
                isoform="canonical",
            )
            try:
                results["tpm_weighted"] = self.tpm_weighted_embedding(
                    gene, model, tpm_values=tpm_values, adata=adata,
                    tpm_layer=tpm_layer, pooling_strategy=pooling_strategy,
                    id_type=id_type,
                )
            except Exception as e:  # noqa: BLE001
                logger.warning("TPM-weighted failed for %s: %s", gene, e)

            try:
                results["annotation_weighted"] = self.annotation_weighted_embedding(
                    gene, model, weight_sites=weight_sites,
                    site_boost=site_boost,
                    pooling_strategy=pooling_strategy, id_type=id_type,
                )
            except Exception as e:  # noqa: BLE001
                logger.warning("Annotation-weighted failed for %s: %s", gene, e)

            try:
                results["expression_context"] = self.expression_context_embedding(
                    gene, model, expression_vector=expression_vector,
                    adata=adata, use_gtex=use_gtex,
                    pooling_strategy=pooling_strategy, id_type=id_type,
                )
            except Exception as e:  # noqa: BLE001
                logger.warning("Expression-context failed for %s: %s", gene, e)

            return results

        raise ValueError(f"Unknown strategy '{strategy}'")
