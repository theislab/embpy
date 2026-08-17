# Attention extraction: what is possible, per model

Attention weights are something these models already compute. Where they can be
observed, embpy exposes them; where they cannot, it says so instead of returning
something meaningless. This page records **which models support extraction and why**,
based on reading the installed packages rather than on assumption.

## How to get attention

```python
attn = wrapper.extract_attention(input_ids, layers=[-1])
# {layer_index: Tensor(batch, n_heads, seq_len, seq_len)}
```

Two paths are tried in order:

1. **HuggingFace models** — `output_attentions=True`, the one uniform interface.
2. **Non-HF models** — forward hooks on the layer modules found by
   `_get_layer_modules()`, mirroring the hidden-state fallback. Where a module can
   return weights on request (`torch.nn.MultiheadAttention` takes `need_weights`,
   which `nn.TransformerEncoderLayer` hardcodes to `False`), a forward *pre*-hook
   flips that back on without modifying the model.

If neither yields weights, you get a `NotImplementedError` explaining the cause.

## The hard limit: fused kernels

A hook can only observe tensors a module actually produces. Fused attention kernels —
`F.scaled_dot_product_attention`, FlashAttention, Triton — compute the softmax inside
the kernel and return only the output. **The attention matrix never exists as a
tensor, so no hook can recover it.** Extracting it would require replacing the
attention call with an explicit `softmax(QKᵀ/√d)`, which is a model-architecture
change and out of scope.

Note this is not merely a performance flag: even forcing the math backend still
returns only the output, never the weights.

## Per-model findings

Verified by reading the installed packages (`arc-gpu` and `helical-gpu` environments).

| Model | Attention implementation | Extractable | Evidence |
|---|---|---|---|
| **Geneformer** | HuggingFace BERT | ✅ yes, native | `output_attentions` supported |
| **TranscriptFormer** | explicit `F.softmax` in a module | ✅ yes, via hook | `transcriptformer/model_dir/layers.py:252` |
| **UCE** | `torch.nn.TransformerEncoderLayer` | ✅ yes, via pre-hook | `uce/uce_model.py:74-75` |
| **Tahoe** | `attn_impl` switch; torch path is eager | ⚠️ likely, untested on GPU | LLM-Foundry vendored code; the fast CUDA path is not taken under `attn_impl="torch"` |
| **scGPT** | `FlashMHA`, unconditionally | ❌ no | `scgpt/model_dir/model.py:625` constructs `FlashMHA` with no torch fallback |
| **STATE** | `F.scaled_dot_product_attention` | ❌ no | `state/emb/nn/flash_transformer.py:67` |
| HyenaDNA, Caduceus | attention-free (long convolution / SSM) | ❌ n/a | `has_attention = False` |
| MiniMol, MHG-GNN | message-passing GNNs | ❌ n/a | `has_attention = False` |

**scGPT** deserves a note: its `transformer.py` contains a `need_weights=True` path,
but the encoder layer actually used by `model.py` builds `FlashMHA` in `__init__` with
no conditional, so the eager path is unreachable without patching the package.

### A structural caveat

The single-cell wrappers (`ScGPTWrapper`, `GeneformerWrapper`, `UCEWrapper`,
`TranscriptFormerWrapper`, `TahoeWrapper`, `StateEmbeddingWrapper`) inherit
`SingleCellWrapper`, which is a **separate hierarchy from `BaseModelWrapper`** and
exposes only `load` / `embed_cells` / `decode_cells`. It has no `self.model`
convention and no layer-introspection methods, so `extract_attention` is not
available on them today even where the underlying architecture would allow it. The
table above therefore describes *architectural* feasibility; wiring it into the
single-cell hierarchy is separate work.

## Getting attention into an AnnData

An attention tensor is rank 4 — `(batch, heads, seq, seq)` — while
`EmbeddingResult` requires a 2-D `(n_entities, n_dims)` float32 matrix. Rather than
add a parallel result type, reduce attention to 2-D summaries at extraction time;
they then inherit every existing exporter and the provenance record:

```python
from embpy import tl

tl.attention_entropy(attn)          # (n_cells, n_heads)  diffuse vs focused
tl.received_attention(attn)         # (n_cells, n_genes)  which genes are read
tl.head_uniformity(attn)            # (n_cells, n_heads)  head specialisation
tl.attention_to_gene_set(attn, idx) # (n_cells, n_heads)  attention on a pathway
```

In single-cell models the tokens are genes, which is what makes `received_attention`
directly interpretable.

If you truly need the rank-4 tensor, write it to `.npz`/`.zarr` yourself — do not
force it through `EmbeddingResult`.

## What attention does not tell you

Treat these as **structural** readouts — what attends to what — not importance
scores. Jain & Wallace (NAACL 2019, *Attention is not Explanation*) showed attention
distributions can be altered substantially while leaving predictions intact, so "this
gene received high attention" does not establish that it drove the output.

For "which genes drive this embedding", gradient-based attribution on the frozen
model is better founded and needs no architectural change.
