"""Shared dataset infrastructure for perturbation transcriptomics.

The world model trains on *sequences* of (state, action) tuples. We do
not have natural temporal trajectories in perturb-seq data, so each
sequence is a synthetic chain of perturbations: at each step ``t`` we
sample a perturbation label, draw ``stack_size`` cells with that label
(or controls when the label is the control), and emit a tuple
``(obs_stack_t, action_t, next_obs_stack_t)``.

This mirrors the way the reference paper turns Atari trajectories into
training tuples: the world model only ever needs ``(s_t, a_t, s_{t+1})``
triples, regardless of whether the data is temporally ordered or
re-sampled.

Concretely, every ``__getitem__`` call returns a dict with:

* ``obs_stack``       -- ``(T, K, G)`` past gene-expression stack
* ``next_obs_stack``  -- ``(T, K, G)`` next-step stack (post-perturbation)
* ``action_indices``  -- ``(T, n_pert)`` long indices into the gene table
* ``next_expression`` -- ``(T, G)`` per-step decoder target (mean of stack)
* ``perturbations``   -- list[str] of length T (raw labels, for debugging)
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Gene-symbol -> action-index lookup
# ----------------------------------------------------------------------


@dataclass
class GeneIndexer:
    """Bidirectional ``gene_symbol <-> row_index`` lookup.

    Index 0 is reserved for the "control / non-targeting / padding"
    action. Real genes start at index 1.
    """

    symbol_to_index: dict[str, int]
    index_to_symbol: list[str]

    @classmethod
    def from_symbols(cls, symbols: Sequence[str]) -> GeneIndexer:
        cleaned = [s for s in symbols if isinstance(s, str) and s]
        symbol_to_index = {"<control>": 0}
        index_to_symbol: list[str] = ["<control>"]
        for sym in cleaned:
            if sym not in symbol_to_index:
                symbol_to_index[sym] = len(index_to_symbol)
                index_to_symbol.append(sym)
        return cls(symbol_to_index=symbol_to_index, index_to_symbol=index_to_symbol)

    def __len__(self) -> int:
        return len(self.index_to_symbol)

    def encode(self, perturbation: str | None, control_label: str) -> list[int]:
        """Encode a perturbation string into a list of ``int`` indices.

        Multi-gene perturbations are assumed to be encoded as
        ``"GENE_A+GENE_B"`` or ``"GENE_A,GENE_B"``. Genes not in the
        embedding table are silently dropped (and a warning is logged
        on first occurrence).
        """
        if perturbation is None or perturbation == control_label:
            return [0]
        # Tolerant splitting for the two most common conventions.
        parts = perturbation.replace(",", "+").split("+")
        out: list[int] = []
        for p in parts:
            p = p.strip()
            if not p or p == control_label:
                continue
            idx = self.symbol_to_index.get(p)
            if idx is not None:
                out.append(idx)
        return out or [0]


# ----------------------------------------------------------------------
# Gene embedding loader
# ----------------------------------------------------------------------


def load_gene_embedding_table(
    path: str | Path,
    symbols: Sequence[str],
) -> tuple[np.ndarray, GeneIndexer]:
    """Build an ``(n_rows + 1, embedding_dim)`` table aligned to ``symbols``.

    Row 0 is the control / padding row (zeros). Rows ``1..n+1`` follow
    the order of ``symbols``. Symbols missing from the embedding source
    receive a zero row and are *kept* in the indexer so downstream
    code does not need to handle missing keys.

    Supports CSV files where the first column is the gene symbol and
    the remaining columns are the embedding entries, plus ``.npz``
    archives with ``"symbols"`` and ``"embeddings"`` arrays.
    """
    import pandas as pd  # noqa: PLC0415

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Gene embedding file not found: {path}")

    if path.suffix == ".npz":
        archive = np.load(path, allow_pickle=True)
        if "symbols" not in archive or "embeddings" not in archive:
            raise KeyError(f"Expected 'symbols' and 'embeddings' arrays in {path}")
        ext_symbols = list(archive["symbols"])
        ext_embeddings = np.asarray(archive["embeddings"], dtype=np.float32)
    else:
        df = pd.read_csv(path, index_col=0)
        ext_symbols = [str(s) for s in df.index.tolist()]
        ext_embeddings = df.to_numpy(dtype=np.float32)

    if ext_embeddings.ndim != 2:
        raise ValueError(f"Embedding table must be 2D, got shape {ext_embeddings.shape}")
    embedding_dim = ext_embeddings.shape[1]
    sym_to_row = {sym: i for i, sym in enumerate(ext_symbols)}

    indexer = GeneIndexer.from_symbols(symbols)
    table = np.zeros((len(indexer), embedding_dim), dtype=np.float32)

    n_missing = 0
    for sym, idx in indexer.symbol_to_index.items():
        if idx == 0:
            continue
        row = sym_to_row.get(sym)
        if row is None:
            n_missing += 1
            continue
        table[idx] = ext_embeddings[row]

    logger.info(
        "Built gene embedding table: %d rows, dim=%d, missing=%d",
        table.shape[0], embedding_dim, n_missing,
    )
    return table, indexer


# ----------------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------------


@dataclass
class SequenceSample:
    """Plain container for the tensors produced per sample."""

    obs_stack: Any        # (T, K, G)
    next_obs_stack: Any   # (T, K, G)
    action_indices: Any   # (T, n_pert)
    next_expression: Any  # (T, G)
    perturbations: list[str]


class PerturbationSequenceDataset:
    """Map-style dataset that emits ``(state, action, next_state)`` sequences.

    The class implements the :class:`torch.utils.data.Dataset` protocol
    (``__len__`` + ``__getitem__``) without inheriting from it, so the
    module can be imported in environments without torch.

    Parameters
    ----------
    expression
        ``(n_cells, n_genes)`` log-normalised gene-expression matrix.
    perturbation_labels
        ``(n_cells,)`` string array of perturbation labels.
    indexer
        Pre-built :class:`GeneIndexer` aligning labels to action indices.
    sequence_length
        T -- length of each emitted sequence.
    stack_size
        K -- number of past frames stacked per timestep.
    n_pert
        Maximum number of genes per perturbation. Multi-gene
        perturbations longer than this are truncated; shorter ones are
        right-padded with index 0.
    control_label
        Label that flags control cells.
    rng
        Optional :class:`numpy.random.Generator` for reproducible sampling.
    n_sequences_per_epoch
        Number of sequences materialised per epoch. The default scales
        with the number of perturbed cells.
    """

    def __init__(
        self,
        expression: np.ndarray,
        perturbation_labels: np.ndarray,
        indexer: GeneIndexer,
        sequence_length: int = 8,
        stack_size: int = 4,
        n_pert: int = 2,
        control_label: str = "non-targeting",
        rng: np.random.Generator | None = None,
        n_sequences_per_epoch: int | None = None,
    ) -> None:
        if expression.ndim != 2:
            raise ValueError(f"expression must be 2D, got shape {expression.shape}")
        if perturbation_labels.shape[0] != expression.shape[0]:
            raise ValueError(
                "expression and perturbation_labels must have matching number of rows"
            )

        self.expression = np.ascontiguousarray(expression, dtype=np.float32)
        self.perturbation_labels = np.asarray(perturbation_labels)
        self.indexer = indexer
        self.sequence_length = int(sequence_length)
        self.stack_size = int(stack_size)
        self.n_pert = int(n_pert)
        self.control_label = control_label
        self.rng = rng if rng is not None else np.random.default_rng()
        self.n_genes = int(expression.shape[1])

        is_control = self.perturbation_labels == control_label
        self.control_idx = np.flatnonzero(is_control)
        self.perturbed_idx = np.flatnonzero(~is_control)
        if self.control_idx.size == 0:
            raise ValueError(f"No control cells (label={control_label!r}) found.")
        if self.perturbed_idx.size == 0:
            raise ValueError("No perturbed cells found.")

        # Bucket perturbed cells by label so the K-frame stack draws come
        # from the same condition. This is the key inductive bias: the
        # encoder sees a clean within-condition stack rather than a mix
        # of conditions.
        self._cells_by_label: dict[str, np.ndarray] = {}
        for label in np.unique(self.perturbation_labels):
            self._cells_by_label[str(label)] = np.flatnonzero(self.perturbation_labels == label)

        # Pool of "next-step" labels we can sample at each step. We keep
        # all labels (control + perturbed) so the model also learns the
        # identity action.
        self._sampleable_labels = sorted(self._cells_by_label.keys())

        if n_sequences_per_epoch is None:
            n_sequences_per_epoch = int(self.perturbed_idx.size)
        self.n_sequences_per_epoch = int(n_sequences_per_epoch)

        logger.info(
            "PerturbationSequenceDataset: cells=%d, genes=%d, perts=%d, controls=%d, "
            "T=%d, K=%d, n_pert=%d, sequences/epoch=%d",
            self.expression.shape[0], self.n_genes, len(self._sampleable_labels) - 1,
            self.control_idx.size, self.sequence_length, self.stack_size, self.n_pert,
            self.n_sequences_per_epoch,
        )

    # ------------------------------------------------------------------
    # PyTorch protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self.n_sequences_per_epoch

    def __getitem__(self, idx: int) -> dict[str, Any]:
        import torch  # noqa: PLC0415

        rng = self.rng
        T = self.sequence_length
        K = self.stack_size
        G = self.n_genes
        n_pert = self.n_pert

        # We deterministically seed a per-sample RNG so the sample is
        # reproducible given (idx, dataset rng state).
        seed = int(rng.integers(2**31)) ^ idx
        local = np.random.default_rng(seed)

        obs_stack = np.empty((T, K, G), dtype=np.float32)
        next_obs_stack = np.empty((T, K, G), dtype=np.float32)
        action_indices = np.zeros((T, n_pert), dtype=np.int64)
        next_expression = np.empty((T, G), dtype=np.float32)
        labels: list[str] = []

        # First step starts from a control stack -- the agent always
        # begins in an "unperturbed" state, mirroring perturb-seq design.
        prev_label = self.control_label
        for t in range(T):
            cur_cells = self._cells_by_label[prev_label]
            prev_idx = local.choice(cur_cells, size=K, replace=cur_cells.size < K)
            obs_stack[t] = self.expression[prev_idx]

            next_label = str(local.choice(self._sampleable_labels))
            next_cells = self._cells_by_label[next_label]
            next_idx = local.choice(next_cells, size=K, replace=next_cells.size < K)
            next_obs_stack[t] = self.expression[next_idx]
            next_expression[t] = next_obs_stack[t].mean(axis=0)
            labels.append(next_label)

            encoded = self.indexer.encode(next_label, self.control_label)
            for j, gid in enumerate(encoded[:n_pert]):
                action_indices[t, j] = gid

            prev_label = next_label

        return {
            "obs_stack": torch.from_numpy(obs_stack),
            "next_obs_stack": torch.from_numpy(next_obs_stack),
            "action_indices": torch.from_numpy(action_indices),
            "next_expression": torch.from_numpy(next_expression),
            "perturbations": labels,
        }


__all__ = [
    "GeneIndexer",
    "PerturbationSequenceDataset",
    "SequenceSample",
    "load_gene_embedding_table",
]
