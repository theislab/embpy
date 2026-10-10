"""Reduce attention tensors to 2-D summaries that fit embpy's output contract.

Why summaries rather than raw tensors
-------------------------------------
An attention tensor is ``(batch, n_heads, seq_len, seq_len)`` -- rank 4.
:class:`embpy.io.result.EmbeddingResult` requires a ``(n_entities, n_dims)`` float32
matrix, so a raw attention tensor has no container, no exporter and no provenance
path in this package. Rather than introduce a parallel result type, these helpers
reduce attention to ``(n_entities, n_dims)`` *at extraction time*, which inherits
every existing exporter (``to_table``, ``to_anndata``, npz/zarr) and the provenance
record for free -- ``EmbeddingProvenance`` already carries ``layer`` and ``pooling``.

If you genuinely need the rank-4 tensor, keep it out of ``EmbeddingResult`` and write
it to ``.npz``/``.zarr`` yourself.

In single-cell models the tokens are genes, which makes these summaries directly
interpretable: per-head entropy says whether a head reads broadly or focuses, and
received mass says *which genes* the model reads.

What attention does and does not tell you
-----------------------------------------
Treat these as **structural** readouts -- what attends to what -- not as importance
scores. Jain & Wallace (NAACL 2019, *Attention is not Explanation*) showed attention
distributions can be substantially altered while leaving model predictions intact, so
"this gene got high attention" does not establish that it drove the output. For
"which genes drive this embedding", a gradient-based attribution on the frozen model
is better founded.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

__all__ = [
    "attention_entropy",
    "attention_to_gene_set",
    "head_uniformity",
    "received_attention",
    "summarize_attention",
]


def _as_attention(attn: Any) -> np.ndarray:
    """Coerce to ``(batch, heads, seq, seq)`` float64, accepting rank-3 input."""
    arr = np.asarray(attn.detach().cpu() if hasattr(attn, "detach") else attn, dtype=np.float64)
    if arr.ndim == 3:  # (batch, seq, seq) -> single head
        arr = arr[:, None, :, :]
    if arr.ndim != 4:
        raise ValueError(f"Attention must be (batch, heads, seq, seq) or (batch, seq, seq), got shape {arr.shape}.")
    if arr.shape[-1] != arr.shape[-2]:
        raise ValueError(f"Attention must be square in its last two axes, got {arr.shape}.")
    return arr


def attention_entropy(attn: Any) -> np.ndarray:
    """Mean Shannon entropy of each head's attention distribution, per entity.

    Low entropy means the head concentrates on a few tokens; high means it spreads
    attention broadly. Averaged over query positions.

    Returns
    -------
    numpy.ndarray
        ``(batch, n_heads)`` -- one column per head, so it drops straight into
        ``.obsm``.
    """
    arr = _as_attention(attn)
    # out= must be supplied alongside where=, otherwise the masked-out entries are
    # uninitialised memory rather than the 0 that 0*log(0) requires.
    logs = np.log(arr, out=np.zeros_like(arr), where=arr > 0)
    ent = -(arr * logs).sum(axis=-1)  # (batch, heads, seq)
    return ent.mean(axis=-1).astype(np.float32)


def received_attention(attn: Any, *, normalize: bool = True) -> np.ndarray:
    """Total attention mass each token *receives*, per entity (head-averaged).

    With gene tokens this reads as "which genes does the model look at?".

    Parameters
    ----------
    normalize
        Divide by the number of query positions so values sum to 1 across tokens,
        making entities with different sequence lengths comparable.

    Returns
    -------
    numpy.ndarray
        ``(batch, seq_len)`` -- one column per token.
    """
    arr = _as_attention(attn)
    mass = arr.sum(axis=-2).mean(axis=1)  # sum over queries, mean over heads
    if normalize:
        mass = mass / arr.shape[-2]
    return mass.astype(np.float32)


def head_uniformity(attn: Any) -> np.ndarray:
    """Per-head distance from the uniform distribution, per entity.

    0 means the head attends perfectly uniformly (contributing no selectivity); larger
    values mean a more specialised head. Total-variation distance to uniform, averaged
    over query positions and scaled to ``[0, 1]``.

    Returns
    -------
    numpy.ndarray
        ``(batch, n_heads)``.
    """
    arr = _as_attention(attn)
    seq = arr.shape[-1]
    uniform = 1.0 / seq
    tv = 0.5 * np.abs(arr - uniform).sum(axis=-1)  # (batch, heads, seq)
    scale = 1.0 - uniform  # maximum achievable TV distance
    return (tv.mean(axis=-1) / scale).astype(np.float32)


def attention_to_gene_set(attn: Any, indices: Any) -> np.ndarray:
    """Attention mass each head directs at a designated token subset, per entity.

    With gene tokens this is "how much does each head attend to this pathway?".

    Parameters
    ----------
    indices
        Token positions forming the set (e.g. the genes of a pathway).

    Returns
    -------
    numpy.ndarray
        ``(batch, n_heads)`` -- fraction of attention mass landing on the set.
    """
    arr = _as_attention(attn)
    idx = np.asarray(indices, dtype=int).ravel()
    if idx.size == 0:
        raise ValueError("indices is empty; pass at least one token position.")
    seq = arr.shape[-1]
    if idx.min() < 0 or idx.max() >= seq:
        raise IndexError(f"indices must lie in [0, {seq - 1}], got [{idx.min()}, {idx.max()}].")
    selected = arr[..., idx].sum(axis=-1)  # (batch, heads, seq_queries)
    return selected.mean(axis=-1).astype(np.float32)


def summarize_attention(
    attn: Any,
    kind: Literal["entropy", "received", "uniformity"] = "entropy",
    **kwargs: Any,
) -> np.ndarray:
    """Dispatch to one of the 2-D attention summaries.

    Convenience for pipelines that carry the choice as a string.

    Returns
    -------
    numpy.ndarray
        ``(n_entities, n_dims)`` float32, ready for ``.obsm`` or any embpy exporter.
    """
    table = {
        "entropy": attention_entropy,
        "received": received_attention,
        "uniformity": head_uniformity,
    }
    if kind not in table:
        raise ValueError(f"kind must be one of {sorted(table)}, got {kind!r}.")
    return table[kind](attn, **kwargs)
