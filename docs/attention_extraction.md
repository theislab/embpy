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
   Recent `transformers` releases default most architectures to the **SDPA**
   kernel, which returns no weights, so `output_attentions=True` on its own comes
   back empty. `extract_attention` therefore switches the model to eager attention
   for the duration of the extraction forward pass and restores the original
   setting afterwards — your `embed()` calls keep the faster kernel, and
   extraction works on a default-loaded model with no flags to remember.
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

Two cases look identical at the call site but are not:

* **Fused by default** — the architecture supports eager attention but ships with
  SDPA selected (every HuggingFace model, since `transformers` 4.48). Recoverable:
  embpy flips the implementation for the extraction pass, as described above.
* **Fused by construction** — the architecture calls the fused kernel
  unconditionally, with no eager path to select (scGPT's `FlashMHA`, STATE, and the
  ESM SDK's shared attention layer). Not recoverable without patching the package,
  so those wrappers declare `has_attention = False` and fail fast with the reason.

`has_attention` is a class attribute, so it is answerable without downloading
weights: `embedder.get_model(key, load=False).has_attention`.

## Per-model findings

Verified by reading the installed packages (`arc-gpu` and `helical-gpu` environments).

| Model | Attention implementation | Extractable | Evidence |
|---|---|---|---|
| **ESM-2, ESM-1b, ESM-1v** | HuggingFace `EsmModel` | ✅ yes, via eager switch | SDPA by default; eager path exists and is used for extraction |
| **ProtT5** | HuggingFace T5 encoder | ✅ yes, via eager switch | same as above |
| **ESM-C, ESM3** | ESM SDK, fused unconditionally | ❌ no | `esm/layers/attention.py:70,76` — `F.scaled_dot_product_attention` in both the masked and unmasked branch |
| **Geneformer** | HuggingFace BERT | ✅ yes, native | `output_attentions` supported |
| **TranscriptFormer** | explicit `F.softmax` in a module | ✅ yes, via hook | `transcriptformer/model_dir/layers.py:252` |
| **UCE** | `torch.nn.TransformerEncoderLayer` | ✅ yes, via pre-hook | `uce/uce_model.py:74-75` |
| **Tahoe** | `attn_impl="torch"` + `needs_weights` | ✅ **yes, verified on cluster** | 12 eager `GroupedQueryAttention`; captured `(4, 8, 1606, 1606)` from a real `embed_cells` run |
| **scGPT** | `FlashMHA`, unconditionally | ❌ no | `scgpt/model_dir/model.py:625` constructs `FlashMHA` with no torch fallback |
| **STATE** | `F.scaled_dot_product_attention` | ❌ no | `state/emb/nn/flash_transformer.py:67` |
| HyenaDNA, Caduceus | attention-free (long convolution / SSM) | ❌ n/a | `has_attention = False` |
| MiniMol, MHG-GNN | message-passing GNNs | ❌ n/a | `has_attention = False` |

**Tahoe** was verified end to end on the cluster rather than inferred. The run also
corrected the earlier guess: `attn_impl="torch"` is **necessary but not sufficient**.
helical's `GroupedQueryAttention.forward` takes `needs_weights` -- note the *s*, not
torch's `need_weights` -- and defaults it to `False`, so
`scaled_multihead_dot_product_attention` computes `attn_weight` and then returns
`None` for it. A forward pass with plain hooks therefore captures nothing. embpy's
pre-hook now inspects each module's signature and sets whichever flag it accepts, so
the weights come back: a 4-cell `embed_cells` yielded `(4, 8, 1606, 1606)`.

Note the capture came from the model's `nn.TransformerEncoder` self-attention rather
than from all 12 `GroupedQueryAttention` blocks, so coverage within Tahoe is partial;
treat "Tahoe attention is available" as established and "which blocks" as
model-specific detail worth checking for your use.

**scGPT** deserves a note: its `transformer.py` contains a `need_weights=True` path,
but the encoder layer actually used by `model.py` builds `FlashMHA` in `__init__` with
no conditional, so the eager path is unreachable without patching the package.

### Single-cell wrappers

`SingleCellWrapper` is a **separate hierarchy from `BaseModelWrapper`** and
originally exposed only `load` / `embed_cells` / `decode_cells`, so none of these
models could reach `extract_attention` regardless of architecture. They now can:
`SingleCellWrapper.extract_attention` / `.extract_hidden_states` resolve the
underlying `torch.nn.Module` via `torch_module()` — which handles both the direct
case (STATE) and the helical convention of nesting it at `._model.model` — and
delegate to `BaseModelWrapper`'s extractors through a thin adapter, so the
extraction logic lives in one tested place.

`ScGPTWrapper` and `StateEmbeddingWrapper` declare `has_attention = False` for the
reasons in the table, so they fail fast with an explanation rather than running a
forward pass that cannot produce weights.

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
