"""Precomputed PPI (Protein-Protein Interaction) embeddings from STRING.

Loads precomputed cross-species functional network embeddings (SPACE,
512-dim) and node2vec embeddings (128-dim) from HDF5 files produced by
the `SPACE <https://doi.org/10.5281/zenodo.15600639>`_ project for
1,322 eukaryotic species in STRING 12.0.

Gene-name resolution is performed via the STRING REST API at load time,
so users can look up embeddings by familiar gene symbols (e.g. ``"TP53"``)
rather than raw STRING protein accessions.

Requires the optional ``h5py`` dependency::

    pip install h5py
"""

from __future__ import annotations

import io
import logging
import os
from collections.abc import Sequence
from typing import Any, Literal

import numpy as np
import pandas as pd
import requests
import torch

from .base import BaseModelWrapper

logger = logging.getLogger(__name__)

_STRING_API = "https://version-12-0.string-db.org/api"

_H5_PATHS: dict[str, str] = {
    "functional": "functional_embeddings/functional_emb/{species}.h5",
    "node2vec": "node2vec/node2vec/{species}.h5",
}


def _require_h5py():  # type: ignore[no-untyped-def]
    """Lazily import h5py, raising a clear error if not installed."""
    try:
        import h5py

        return h5py
    except ImportError as exc:
        raise ImportError(
            "h5py is required for PrecomputedPPIWrapper. "
            "Install with: pip install h5py"
        ) from exc


def _fetch_gene_names(
    accessions: Sequence[str],
    species: int,
    batch_size: int = 500,
) -> dict[str, str]:
    """Map STRING protein accessions to preferred gene names via the REST API.

    Parameters
    ----------
    accessions
        Protein accessions (without the taxonomy prefix).
    species
        NCBI taxonomy ID.
    batch_size
        Number of accessions per API call.

    Returns
    -------
    ``{accession: preferred_gene_name}`` dict.
    """
    gene_map: dict[str, str] = {}
    for i in range(0, len(accessions), batch_size):
        chunk = accessions[i : i + batch_size]
        resp = requests.post(
            f"{_STRING_API}/tsv/get_string_ids",
            data={
                "identifiers": "\r".join(chunk),
                "species": species,
                "echo_query": 1,
                "caller_identity": "embpy",
            },
            timeout=120,
        )
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text), sep="\t")
        for _, row in df.iterrows():
            gene_map[str(row["queryItem"])] = str(row["preferredName"])
        logger.info(
            "Gene-name mapping: %d / %d accessions resolved",
            min(i + batch_size, len(accessions)),
            len(accessions),
        )
    return gene_map


class PrecomputedPPIWrapper(BaseModelWrapper):
    """Precomputed PPI embedding loader for STRING network embeddings.

    Loads SPACE functional (512-dim) or node2vec (128-dim) precomputed
    embeddings from HDF5 files and maps STRING protein accessions to
    gene names via the STRING REST API.

    Typical workflow::

        wrapper = PrecomputedPPIWrapper(
            data_dir="/path/to/precomputed_embeddings_string",
            species=9606,
            embedding_type="functional",
        )
        wrapper.load(torch.device("cpu"))

        emb = wrapper.embed("TP53")         # np.ndarray (512,)
        embs = wrapper.embed_batch(["TP53", "BRCA1"])

    Parameters
    ----------
    data_dir
        Root directory containing ``functional_embeddings/`` and
        ``node2vec/`` sub-directories with per-species ``.h5`` files.
    species
        NCBI taxonomy ID (default 9606 = human).
    embedding_type
        ``"functional"`` for SPACE 512-dim embeddings or ``"node2vec"``
        for 128-dim node2vec embeddings.
    batch_size
        Number of protein accessions per STRING API call when building
        the gene-name index at load time.
    """

    model_type: Literal["dna", "protein", "molecule", "text", "ppi", "unknown"] = "ppi"  # type: ignore[assignment]
    available_pooling_strategies: list[str] = ["none"]

    def __init__(
        self,
        data_dir: str | None = None,
        species: int = 9606,
        embedding_type: Literal["functional", "node2vec"] = "functional",
        batch_size: int = 500,
        **kwargs: Any,
    ) -> None:
        super().__init__(model_path_or_name=data_dir, **kwargs)
        self.data_dir = data_dir
        self.species = species
        self.embedding_type = embedding_type
        self.batch_size = batch_size

        if embedding_type not in _H5_PATHS:
            raise ValueError(
                f"Unknown embedding_type '{embedding_type}'. "
                f"Choose from {list(_H5_PATHS)}."
            )

        self._embeddings: np.ndarray | None = None
        self._protein_ids: list[str] = []
        self._protein_to_idx: dict[str, int] = {}
        self._gene_to_idx: dict[str, int] = {}
        self._gene_map: dict[str, str] = {}

    # ------------------------------------------------------------------
    # BaseModelWrapper interface
    # ------------------------------------------------------------------

    def load(self, device: torch.device) -> None:
        """Load precomputed embeddings from H5 and build the gene-name index.

        Parameters
        ----------
        device
            Stored but not used (embeddings are NumPy arrays on CPU).
        """
        self.device = device

        if self.data_dir is None:
            raise ValueError(
                "data_dir must be provided to PrecomputedPPIWrapper."
            )

        h5py = _require_h5py()
        h5_rel = _H5_PATHS[self.embedding_type].format(species=self.species)
        h5_path = os.path.join(self.data_dir, h5_rel)

        if not os.path.isfile(h5_path):
            raise FileNotFoundError(
                f"HDF5 file not found: {h5_path}. "
                f"Ensure the precomputed embeddings for species "
                f"{self.species} are downloaded."
            )

        with h5py.File(h5_path, "r") as f:
            self._embeddings = np.array(f["embeddings"], dtype=np.float32)
            raw_ids = np.array(f["proteins"])
            self._protein_ids = [
                pid.decode() if isinstance(pid, bytes) else str(pid)
                for pid in raw_ids
            ]

        self._protein_to_idx = {
            pid: i for i, pid in enumerate(self._protein_ids)
        }

        logger.info(
            "Loaded %s embeddings for species %d: %d proteins, dim=%d",
            self.embedding_type,
            self.species,
            len(self._protein_ids),
            self._embeddings.shape[1],
        )

        accessions = [
            pid.split(".", 1)[1] for pid in self._protein_ids
        ]
        self._gene_map = _fetch_gene_names(
            accessions, species=self.species, batch_size=self.batch_size
        )

        self._gene_to_idx = {}
        for pid in self._protein_ids:
            accession = pid.split(".", 1)[1]
            gene_name = self._gene_map.get(accession)
            if gene_name and gene_name not in self._gene_to_idx:
                self._gene_to_idx[gene_name] = self._protein_to_idx[pid]

        logger.info(
            "Gene-name index built: %d unique gene names mapped",
            len(self._gene_to_idx),
        )

    def embed(
        self,
        input: str,
        pooling_strategy: str = "none",
        **kwargs: Any,
    ) -> np.ndarray:
        """Return the precomputed embedding for a gene or protein identifier.

        Parameters
        ----------
        input
            Gene symbol (e.g. ``"TP53"``) or full STRING protein ID
            (e.g. ``"9606.ENSP00000269305"``).
        pooling_strategy
            Ignored (each protein has a single embedding vector).

        Returns
        -------
        1-D ``np.ndarray`` of shape ``(embedding_dim,)``.
        """
        if self._embeddings is None:
            raise RuntimeError(
                "Embeddings not loaded. Call load(device) first."
            )

        idx = self._gene_to_idx.get(input)
        if idx is None:
            idx = self._protein_to_idx.get(input)
        if idx is None:
            raise ValueError(
                f"'{input}' not found in gene names or protein IDs "
                f"({self.num_proteins} proteins loaded). "
                "Check available_genes or available_proteins."
            )
        return self._embeddings[idx]

    def embed_batch(
        self,
        inputs: Sequence[str],
        pooling_strategy: str = "none",
        **kwargs: Any,
    ) -> list[np.ndarray]:
        """Embed multiple genes / proteins.

        Identifiers not found are returned as zero vectors with a warning.
        """
        if self._embeddings is None:
            raise RuntimeError(
                "Embeddings not loaded. Call load(device) first."
            )

        results: list[np.ndarray] = []
        dim = self._embeddings.shape[1]
        for name in inputs:
            idx = self._gene_to_idx.get(name)
            if idx is None:
                idx = self._protein_to_idx.get(name)
            if idx is not None:
                results.append(self._embeddings[idx])
            else:
                logger.warning(
                    "'%s' not found; returning zero vector.", name
                )
                results.append(np.zeros(dim, dtype=np.float32))
        return results

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def available_genes(self) -> list[str]:
        """Gene symbols that can be used with :meth:`embed`."""
        return list(self._gene_to_idx.keys())

    @property
    def available_proteins(self) -> list[str]:
        """Full STRING protein IDs that can be used with :meth:`embed`."""
        return list(self._protein_ids)

    @property
    def embedding_dim(self) -> int:
        """Dimensionality of the loaded embeddings (0 if not loaded)."""
        if self._embeddings is None:
            return 0
        return int(self._embeddings.shape[1])

    @property
    def num_proteins(self) -> int:
        """Number of proteins in the loaded embedding matrix."""
        return len(self._protein_ids)
