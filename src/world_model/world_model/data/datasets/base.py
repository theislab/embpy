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

Optional context-bucketing
--------------------------

Set ``cell_buckets`` to a per-cell integer array (e.g. assay batch,
gem-group, cell-cycle bin) to constrain every emitted sequence to a
single bucket. The intuition: by holding the biological / technical
substrate fixed across the T timesteps and varying only the
perturbation, the transformer must learn how that substrate responds
to different actions -- the invariances and equivariances of a cell.
Without bucketing, ``S_{t+1}`` is independent of ``S_t`` in the data and
the model collapses to predicting the per-action mean.

Concretely, every ``__getitem__`` call returns a dict with:

* ``obs_stack``       -- ``(T, K, G)`` past gene-expression stack
* ``next_obs_stack``  -- ``(T, K, G)`` next-step stack (post-perturbation)
* ``action_indices``  -- ``(T, n_pert)`` long indices into the gene table
* ``next_expression`` -- ``(T, G)`` per-step decoder target (mean of stack)
* ``perturbations``   -- list[str] of length T (raw labels, for debugging)
* ``bucket_id``       -- int, the bucket the whole sequence was drawn
                         from. ``-1`` if bucketing is disabled.
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
    cell_buckets
        Optional ``(n_cells,)`` integer array assigning each cell to a
        bucket id (e.g. assay batch, gem-group). When provided, every
        emitted sequence is drawn from a single bucket so the
        transformer sees a coherent biological/technical context with
        only the perturbation varying across timesteps. ``None``
        disables bucketing (the legacy global-pool sampler is used).
    bucket_value_map
        Optional ``{bucket_id: human_readable_label}`` map kept around
        for diagnostic logging only. Has no effect on sampling.
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
        allowed_cell_indices: np.ndarray | None = None,
        cell_buckets: np.ndarray | None = None,
        bucket_value_map: dict[int, str] | None = None,
        context_mode: str = "trajectory",
        incontext_support_size: int = 16,
    ) -> None:
        if expression.ndim != 2:
            raise ValueError(f"expression must be 2D, got shape {expression.shape}")
        if perturbation_labels.shape[0] != expression.shape[0]:
            raise ValueError(
                "expression and perturbation_labels must have matching number of rows"
            )

        self.expression = np.ascontiguousarray(expression, dtype=np.float32)
        # Always-gene-space view of the observation matrix. Survives the
        # foreign-backbone swap in build_dataloaders' pre-encode step
        # (which overwrites .expression with cell embeddings) so that
        # evaluation can compare predictions to gene-space ground truth
        # via backbone.decode(). For local-backbone runs this aliases
        # .expression -- no extra memory cost.
        self.raw_expression = self.expression
        self.perturbation_labels = np.asarray(perturbation_labels)
        self.indexer = indexer
        self.sequence_length = int(sequence_length)
        self.stack_size = int(stack_size)
        self.n_pert = int(n_pert)
        self.control_label = control_label
        self.rng = rng if rng is not None else np.random.default_rng()
        self.n_genes = int(expression.shape[1])
        self.context_mode = str(context_mode)
        self.incontext_support_size = int(incontext_support_size)

        if allowed_cell_indices is None:
            allowed_mask = np.ones(self.expression.shape[0], dtype=bool)
        else:
            allowed_mask = np.zeros(self.expression.shape[0], dtype=bool)
            allowed_mask[np.asarray(allowed_cell_indices, dtype=np.int64)] = True
        self._allowed_mask = allowed_mask

        is_control = self.perturbation_labels == control_label
        self.control_idx = np.flatnonzero(is_control & allowed_mask)
        self.perturbed_idx = np.flatnonzero((~is_control) & allowed_mask)
        if self.control_idx.size == 0:
            raise ValueError(f"No control cells (label={control_label!r}) in allowed subset.")
        if self.perturbed_idx.size == 0:
            raise ValueError("No perturbed cells in allowed subset.")

        # Bucket perturbed cells by label so the K-frame stack draws come
        # from the same condition (key inductive bias).
        self._cells_by_label: dict[str, np.ndarray] = {}
        for label in np.unique(self.perturbation_labels[allowed_mask]):
            mask = (self.perturbation_labels == label) & allowed_mask
            self._cells_by_label[str(label)] = np.flatnonzero(mask)

        # Sampleable labels include the control: the model also learns the identity action.
        self._sampleable_labels = sorted(self._cells_by_label.keys())

        # --------------------------------------------------------------
        # Optional context-bucketing: a per-cell bucket id (e.g. assay
        # batch, gem-group, cell-cycle bin). When provided, every
        # emitted sequence is drawn from a single bucket so the
        # transformer sees a coherent context with only the
        # perturbation varying across timesteps.
        # --------------------------------------------------------------
        self._cell_buckets: np.ndarray | None = None
        self._bucket_value_map: dict[int, str] | None = None
        # Map: bucket_id -> { label -> ndarray of allowed cell indices }
        self._cells_by_bucket_label: dict[int, dict[str, np.ndarray]] = {}
        # Map: bucket_id -> list of labels with >= 1 allowed cell (incl. control).
        self._labels_by_bucket: dict[int, list[str]] = {}
        # Buckets that have at least one control cell AND at least one
        # non-control perturbation. Sampleable as sequence anchors.
        self._sampleable_buckets: list[int] = []

        if cell_buckets is not None:
            buckets = np.asarray(cell_buckets)
            if buckets.shape[0] != self.expression.shape[0]:
                raise ValueError(
                    f"cell_buckets has {buckets.shape[0]} rows but expression "
                    f"has {self.expression.shape[0]}; they must match."
                )
            if buckets.dtype.kind not in {"i", "u"}:
                buckets = buckets.astype(np.int64)
            self._cell_buckets = buckets
            self._bucket_value_map = dict(bucket_value_map) if bucket_value_map else None

            allowed_bucket_ids = np.unique(buckets[allowed_mask])
            for bid in allowed_bucket_ids:
                bid_int = int(bid)
                in_bucket = (buckets == bid) & allowed_mask
                labels_here: dict[str, np.ndarray] = {}
                for label in np.unique(self.perturbation_labels[in_bucket]):
                    cells = np.flatnonzero(in_bucket & (self.perturbation_labels == label))
                    if cells.size > 0:
                        labels_here[str(label)] = cells
                if not labels_here:
                    continue
                has_control = control_label in labels_here
                has_pert = any(k != control_label for k in labels_here)
                if not (has_control and has_pert):
                    # Skip buckets that cannot anchor a sequence (need a
                    # control to start from and at least one perturbation
                    # to draw actions from).
                    continue
                self._cells_by_bucket_label[bid_int] = labels_here
                self._labels_by_bucket[bid_int] = sorted(labels_here.keys())
                self._sampleable_buckets.append(bid_int)
            self._sampleable_buckets.sort()

            if not self._sampleable_buckets:
                raise ValueError(
                    "cell_buckets was provided but no bucket has both a "
                    "control cell and at least one non-control perturbation "
                    "in the allowed subset; cannot anchor any sequence."
                )

        if n_sequences_per_epoch is None:
            n_sequences_per_epoch = int(self.perturbed_idx.size)
        self.n_sequences_per_epoch = int(n_sequences_per_epoch)

        logger.info(
            "PerturbationSequenceDataset: cells_used=%d/%d, genes=%d, perts=%d, controls=%d, "
            "T=%d, K=%d, n_pert=%d, sequences/epoch=%d",
            int(allowed_mask.sum()), self.expression.shape[0], self.n_genes,
            len(self._sampleable_labels) - 1,
            self.control_idx.size, self.sequence_length, self.stack_size, self.n_pert,
            self.n_sequences_per_epoch,
        )
        if self._cell_buckets is not None:
            n_buckets = len(self._sampleable_buckets)
            sizes = np.array([
                sum(arr.size for arr in self._cells_by_bucket_label[b].values())
                for b in self._sampleable_buckets
            ])
            label_counts = np.array([
                len(self._labels_by_bucket[b]) for b in self._sampleable_buckets
            ])
            logger.info(
                "PerturbationSequenceDataset: bucketing ON -- buckets=%d "
                "(cells/bucket min/med/max=%d/%d/%d, labels/bucket min/med/max=%d/%d/%d)",
                n_buckets,
                int(sizes.min()), int(np.median(sizes)), int(sizes.max()),
                int(label_counts.min()), int(np.median(label_counts)), int(label_counts.max()),
            )

    # ------------------------------------------------------------------
    # PyTorch protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self.n_sequences_per_epoch

    def __getitem__(self, idx: int) -> dict[str, Any]:
        import torch  # noqa: PLC0415

        if self.context_mode == "incontext_set":
            return self._getitem_incontext(idx)

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

        # Pick the context bucket ONCE per sequence. Every (current,
        # next) cell drawn below is constrained to this bucket so the
        # transformer sees a coherent biological/technical substrate
        # with only the perturbation varying across timesteps.
        if self._cell_buckets is not None:
            bucket_id = int(local.choice(self._sampleable_buckets))
            cells_by_label = self._cells_by_bucket_label[bucket_id]
            sampleable_labels = self._labels_by_bucket[bucket_id]
        else:
            bucket_id = -1
            cells_by_label = self._cells_by_label
            sampleable_labels = self._sampleable_labels

        # First step starts from a control stack -- the agent always
        # begins in an "unperturbed" state, mirroring perturb-seq design.
        prev_label = self.control_label
        for t in range(T):
            cur_cells = cells_by_label[prev_label]
            prev_idx = local.choice(cur_cells, size=K, replace=cur_cells.size < K)
            obs_stack[t] = self.expression[prev_idx]

            next_label = str(local.choice(sampleable_labels))
            next_cells = cells_by_label[next_label]
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
            "bucket_id": bucket_id,
        }

    def _getitem_incontext(self, idx: int) -> dict[str, Any]:
        """One in-context task: a SET of support triplets + a query.

        A triplet is ``(s_control, action, s_perturbed)`` where the
        "before" state is ALWAYS a fresh control stack (control-anchored;
        no fabricated chaining) and the "after" state is a stack of
        cells carrying that perturbation. ``M = incontext_support_size``
        support triplets show the model how perturbations behave in this
        substrate; the model must then predict the perturbed state of
        ONE held-out query triplet. The set has no order -- the
        bidirectional dynamics treats it permutation-invariantly.

        Emitted keys (all torch tensors unless noted):

        * ``support_obs``      ``(M, K, G)``   control stacks
        * ``support_act``      ``(M, n_pert)`` support action indices
        * ``support_next``     ``(M, K, G)``   perturbed stacks
        * ``query_obs``        ``(K, G)``      control stack
        * ``query_act``        ``(n_pert,)``   query action indices
        * ``query_next``       ``(K, G)``      target perturbed stack
        * ``query_next_expression`` ``(G,)``   decoder target (mean)
        * ``obs_stack``        ``(K, G)``      alias of ``query_obs`` so
          the trainer's batch-size bookkeeping is unchanged.
        * ``perturbations``    list[str]       [*support, query]
        * ``bucket_id``        int
        """
        import torch  # noqa: PLC0415

        K = self.stack_size
        G = self.n_genes
        n_pert = self.n_pert
        M = self.incontext_support_size

        seed = int(self.rng.integers(2**31)) ^ idx
        local = np.random.default_rng(seed)

        # One substrate per task (same intent as trajectory bucketing).
        if self._cell_buckets is not None:
            bucket_id = int(local.choice(self._sampleable_buckets))
            cells_by_label = self._cells_by_bucket_label[bucket_id]
            label_pool = [
                lbl for lbl in self._labels_by_bucket[bucket_id]
                if lbl != self.control_label
            ]
        else:
            bucket_id = -1
            cells_by_label = self._cells_by_label
            label_pool = [
                lbl for lbl in self._sampleable_labels
                if lbl != self.control_label
            ]

        # M support + 1 query distinct perturbations where possible.
        n_draw = M + 1
        replace = len(label_pool) < n_draw
        chosen = list(
            local.choice(np.asarray(label_pool, dtype=object),
                         size=n_draw, replace=replace)
        )
        support_labels = [str(x) for x in chosen[:M]]
        query_label = str(chosen[M])

        control_cells = cells_by_label[self.control_label]

        def _control_stack() -> np.ndarray:
            idx_ = local.choice(
                control_cells, size=K, replace=control_cells.size < K,
            )
            return self.expression[idx_]

        def _pert_stack(lbl: str) -> np.ndarray:
            cells = cells_by_label[lbl]
            idx_ = local.choice(cells, size=K, replace=cells.size < K)
            return self.expression[idx_]

        def _action(lbl: str) -> np.ndarray:
            a = np.zeros((n_pert,), dtype=np.int64)
            for j, gid in enumerate(
                self.indexer.encode(lbl, self.control_label)[:n_pert]
            ):
                a[j] = gid
            return a

        support_obs = np.empty((M, K, G), dtype=np.float32)
        support_next = np.empty((M, K, G), dtype=np.float32)
        support_act = np.zeros((M, n_pert), dtype=np.int64)
        for m, lbl in enumerate(support_labels):
            support_obs[m] = _control_stack()
            support_next[m] = _pert_stack(lbl)
            support_act[m] = _action(lbl)

        query_obs = _control_stack()
        query_next = _pert_stack(query_label)
        query_act = _action(query_label)

        return {
            "support_obs": torch.from_numpy(support_obs),
            "support_act": torch.from_numpy(support_act),
            "support_next": torch.from_numpy(support_next),
            "query_obs": torch.from_numpy(query_obs),
            "query_act": torch.from_numpy(query_act),
            "query_next": torch.from_numpy(query_next),
            "query_next_expression": torch.from_numpy(
                query_next.mean(axis=0).astype(np.float32)
            ),
            # Alias so WorldModelTrainer's `batch["obs_stack"].size(0)`
            # batch-size accounting keeps working untouched.
            "obs_stack": torch.from_numpy(query_obs),
            "perturbations": [*support_labels, query_label],
            "bucket_id": bucket_id,
        }

    # ------------------------------------------------------------------
    # Subset / view helpers
    # ------------------------------------------------------------------

    def subset(
        self,
        cell_indices: np.ndarray,
        *,
        n_sequences_per_epoch: int | None = None,
        rng: np.random.Generator | None = None,
    ) -> PerturbationSequenceDataset:
        """Return a new dataset that draws sequences only from ``cell_indices``.

        The underlying expression matrix and indexer are *shared* (not
        copied); only the sampling pool changes. Used to enforce
        train/test splits without duplicating arrays in memory.
        """
        return type(self)(
            expression=self.expression,
            perturbation_labels=self.perturbation_labels,
            indexer=self.indexer,
            sequence_length=self.sequence_length,
            stack_size=self.stack_size,
            n_pert=self.n_pert,
            control_label=self.control_label,
            rng=rng if rng is not None else self.rng,
            n_sequences_per_epoch=n_sequences_per_epoch,
            allowed_cell_indices=np.asarray(cell_indices, dtype=np.int64),
            cell_buckets=self._cell_buckets,
            bucket_value_map=self._bucket_value_map,
            context_mode=self.context_mode,
            incontext_support_size=self.incontext_support_size,
        )

    def available_perturbations(self) -> list[str]:
        """Sorted list of perturbation labels present in the allowed subset."""
        return [lbl for lbl in self._sampleable_labels if lbl != self.control_label]

    def bucket_summary(self, bucket_id: int) -> dict[str, Any]:
        """Return a small summary for ``bucket_id`` (diagnostic only).

        Empty / missing buckets return ``{}``. Used by the startup
        context-inspector hook to log what each sequence is anchored
        to without dumping multi-megabyte arrays.
        """
        if self._cell_buckets is None or bucket_id < 0:
            return {}
        if bucket_id not in self._cells_by_bucket_label:
            return {}
        per_label = self._cells_by_bucket_label[bucket_id]
        n_cells = int(sum(arr.size for arr in per_label.values()))
        n_labels = len(per_label)
        n_control = int(per_label.get(self.control_label, np.array([])).size)
        value = (self._bucket_value_map or {}).get(bucket_id)
        return {
            "bucket_id": int(bucket_id),
            "bucket_value": value,
            "n_cells": n_cells,
            "n_labels": n_labels,
            "n_control_cells": n_control,
        }

    def expression_view(self, cell_indices: np.ndarray) -> np.ndarray:
        """Return ``(len(cell_indices), n_genes)`` slice of the expression matrix."""
        return self.expression[np.asarray(cell_indices, dtype=np.int64)]


__all__ = [
    "GeneIndexer",
    "PerturbationSequenceDataset",
    "SequenceSample",
    "load_gene_embedding_table",
]
