"""Part 5 of docs/notebooks/genes.ipynb -- attention on a DNA model."""
from __future__ import annotations
import json, sys
from pathlib import Path

CELLS: list[tuple[str, str]] = []
def md(s: str) -> None: CELLS.append(("markdown", s.strip("\n")))
def code(s: str) -> None: CELLS.append(("code", s.strip("\n")))

# ================================================================== framing
md(r"""
## Reading a DNA model's attention

Everything so far used a pooled vector: one number per dimension for a whole
locus, with the positions averaged away. Attention is the opposite -- one number
per token pair, before any pooling -- so it says which parts of the input the
model routed information between.

For proteins that is unusually easy to read, because the tokens *are* amino acids
at known positions. For DNA it is not. Nucleotide Transformer v2 tokenises
**6-mers**, GENA-LM uses **BPE units of variable length**, and neither is a base,
a codon, or an exon. So a token index is not a coordinate, and every positional
statement below is approximate by construction. Where a mapping from tokens to
base pairs is needed, the section derives the bases-per-token factor from the
tokeniser's own output rather than assuming 6.

> **Structural, not causal.** Attention says what attends to what. Jain & Wallace
> (NAACL 2019, *Attention is not Explanation*) showed attention can be altered
> substantially without changing a model's predictions, so a high-attention token
> is not thereby shown to have driven the embedding. What follows is a structural
> readout worth checking against known biology -- not an explanation of the
> vector.

`get_model()` returns the wrapper rather than an array, and the wrapper exposes
the internals.
""")

# ================================================================ gene pick
md(r"""
### Choosing a locus

Attention is quadratic: one block holds `heads x tokens x tokens` floats, and
`layers=None` asks for every block at once, so doubling the input quadruples every
one of those matrices. A multi-kilobase locus across a few dozen head-blocks runs
into gigabytes of tensors that exist only to be reduced to a couple of summary
statistics. The sensible input is therefore the *shortest* locus already resolved
by the [resolution section](#resolve-the-loci-and-measure-them-before-trusting-anything)
-- no new Ensembl calls and no new download.

The pick comes from that section's in-memory `exon_cache`, **not** from re-reading
`gene_exons.json`. The difference matters here more than anywhere else in the
notebook: the in-memory dict holds only genes whose exon count matched Ensembl's
canonical transcript, and a silently truncated gene is short. Picking the shortest
sequence out of the raw file is therefore the one selection rule most likely to
land on exactly the corrupted entry that section was written to catch.

One extra requirement: the readout further down compares the flanking exons
against the interior ones, which needs at least three exons to mean anything. So
the pick is the shortest validated sequence that has three or more exons, falling
back to the shortest sequence of any kind if none qualifies.
""")

code(r"""
ATTN_MODEL = "nt_v2_100m"
MAX_ATTN_BP = 2000       # hard ceiling, so a large cache entry cannot blow up memory

# exon_cache and dna_genes come from the resolution section. Reusing them rather
# than re-reading gene_exons.json is deliberate: that dict was filtered to entries
# whose exon count matched Ensembl, and it is restricted to this panel.
usable = {g: exon_cache[g] for g in dna_genes if exon_cache[g].get("sequence")}
multi_exon = {g: rec for g, rec in usable.items() if int(rec.get("n_exons", 0)) >= 3}
pool = multi_exon or usable

ATTN_GENE = min(pool, key=lambda g: len(pool[g]["sequence"])) if pool else None
attn_sequence, exon_lengths = "", []

if ATTN_GENE is None:
    print("no panel gene carries a validated exon sequence -- the resolution section "
          "printed why; this whole attention section is skipped.")
else:
    record = pool[ATTN_GENE]
    attn_sequence = record["sequence"][:MAX_ATTN_BP]
    # Clip the exon lengths to whatever survived the truncation, so the boundaries
    # below still describe the string actually fed to the model.
    remaining = len(attn_sequence)
    for length in record.get("exon_lengths", []):
        if remaining <= 0:
            break
        take = min(int(length), remaining)
        exon_lengths.append(take)
        remaining -= take

    cached_lengths = list(record.get("exon_lengths", []))
    print(f"chose {ATTN_GENE}: shortest of {len(pool)} resolved "
          f"{'multi-exon ' if multi_exon else ''}loci, so the attention tensors stay small")
    print(f"  chromosome      : {record.get('chrom', '?')}")
    print(f"  exons           : {record.get('n_exons', len(cached_lengths))} "
          f"({', '.join(str(v) for v in cached_lengths)} bp)")
    print(f"  spliced length  : {len(record['sequence']):,} bp"
          + (f" -- truncated to {len(attn_sequence):,} bp at the {MAX_ATTN_BP:,} bp ceiling, "
             f"leaving {len(exon_lengths)} exon(s)"
             if len(attn_sequence) < len(record['sequence']) else ""))
    print(f"  genomic span    : {record.get('span', 0):,} bp "
          f"(the introns are not in the input; see the caveat below)")
""")

# ================================================================ extraction
code(r"""
attn: dict = {}
tokens = None

if ATTN_GENE is not None:
    try:
        wrapper = embedder.get_model(ATTN_MODEL)
        tokens = wrapper.tokenizer(attn_sequence, return_tensors="pt")
        attn = wrapper.extract_attention(
            tokens["input_ids"],
            attention_mask=tokens["attention_mask"],
            layers=None,          # every block; pass e.g. [-1] for just the last
        )
    except Exception as exc:
        print(f"{ATTN_MODEL}: attention unavailable -- {type(exc).__name__}: {str(exc)[:160]}")

n_tokens = int(tokens["input_ids"].shape[1]) if attn else 0
last = max(attn) if attn else -1

if attn:
    print(f"{ATTN_GENE}: {len(attn_sequence):,} bp -> {n_tokens} tokens")
    print(f"implied bases per token : {len(attn_sequence) / n_tokens:.2f}  "
          "(6-mers, plus the special tokens counted in the total)")
    print(f"blocks returned         : {len(attn)}  (indices {min(attn)}..{last})")
    print(f"tensor shape            : {tuple(attn[last].shape)} = (batch, heads, seq, seq)")

    row_sums = attn[last].sum(-1).cpu().numpy()    # tensors come back on the model's device
    print(f"rows are distributions  : {bool(np.allclose(row_sums, 1.0, atol=1e-4))}")
""")

md(r"""
Every query row sums to 1: each token distributes a fixed budget of attention
over the sequence, so "receives a lot of attention" is always relative to the
other tokens of the same locus and never comparable in absolute terms across
inputs of different length.

> **Layer indexing.** `extract_attention` returns **one entry per transformer
> block** -- there is no embedding-layer entry, unlike `extract_hidden_states`
> where index 0 *is* the embedding layer. Mixing the two conventions silently
> shifts you one block; `tl.block_to_attention_index` and
> `tl.block_to_hidden_state_index` convert between them so the call site names
> the convention instead of assuming it.

### How focused is each head, and does that change with depth?

Entropy of a head's attention distribution: low means it concentrates on a few
tokens, high means it spreads out. The ceiling is `log(n_tokens)`, which is what
a head attending uniformly would score -- and which moves with sequence length,
so it is printed rather than assumed.
""")

code(r"""
if not attn:
    depth = pd.DataFrame()
    print("no attention tensors -- depth profile skipped")
else:
    print(f"maximum possible entropy (uniform over {n_tokens} tokens): "
          f"{np.log(n_tokens):.2f}\n")

    profile = []
    for layer in sorted(attn):
        entropy = tl.attention_entropy(attn[layer])          # (batch, n_heads)
        profile.append({"block": layer,
                        "mean_entropy": float(entropy.mean()),
                        "most_focused": float(entropy.min()),
                        "most_diffuse": float(entropy.max()),
                        "specialisation": float(tl.head_uniformity(attn[layer]).mean())})

    depth = pd.DataFrame(profile).set_index("block")
    step = max(1, len(depth) // 8)                            # keep the table readable
    display(depth.round(3).iloc[::step])
""")

code(r"""
if len(depth):
    ax = depth[["mean_entropy", "most_focused", "most_diffuse"]].plot(
        figsize=(9, 4), title=f"{ATTN_MODEL} on {ATTN_GENE}: attention entropy with depth")
    ax.set_xlabel("transformer block")
    ax.set_ylabel("entropy (nats)")
    ax.axhline(np.log(n_tokens), ls="--", c="grey", lw=1)
    ax.annotate("uniform ceiling", xy=(0.5, np.log(n_tokens)),
                xytext=(2, -12), textcoords="offset points", fontsize=8, color="grey")
""")

md(r"""
`head_uniformity` measures the same property on a fixed 0-1 scale (total-variation
distance to uniform), which is what makes it comparable across models and
sequence lengths -- entropy is not, because its ceiling is `log(n_tokens)` and
two tokenisers need not cut the same locus into the same number of tokens. The
GENA-LM comparison below leans on exactly that.

### Which parts of the locus does the model read?

`received_attention` sums the attention each token *receives*, head-averaged and
normalised so the values sum to 1 across tokens.
""")

code(r"""
mass = None
if attn:
    mass = tl.received_attention(attn[last])[0]              # (n_tokens,)
    # .tolist() first: convert_ids_to_tokens wants plain ints, not a tensor.
    token_strings = wrapper.tokenizer.convert_ids_to_tokens(tokens["input_ids"][0].tolist())

    top = np.argsort(mass)[::-1][:12]
    print(f"{'token':>6}  {'k-mer':>10}  {'attention':>10}   (uniform would be "
          f"{1.0 / n_tokens:.5f})")
    for t in top:
        print(f"{int(t):>6}  {token_strings[int(t)]:>10}  {mass[int(t)]:>10.5f}")
""")

md(r"""
Read the k-mer column before reading anything into the ranking, and check where
the special tokens landed. The marker the tokeniser prepends is a convenient sink
for attention a head does not want to spend, and in transformer language models
it often draws a disproportionate share; where it does, that is a property of the
training objective rather than of this gene. The rest of the ranking is a list of
6-mers, which is as far as this model's tokens go -- they are not codons, they do
not respect reading frame, and consecutive tokens do not overlap.

### Mapping tokens back onto exons

The input was built by concatenating the exons of one gene, and the cache records
each exon's length, so the cumulative lengths give the exon boundaries in base
pairs. Converting a token index to a base-pair position needs a scale factor, and
the honest way to get it is from the tokeniser's own output: the number of
non-special tokens divided into the sequence length. Nucleotide Transformer
tokenises 6-mers, so the factor should come out near 6 -- the cell prints it, and
that printed value is where you check the assumption rather than make it.

Three caveats, all of which bound how far this can be pushed:

* **Token boundaries do not align with exon boundaries.** A 6-mer or a BPE unit
  straddles a junction whenever the exon length is not a multiple of the token
  length, so an exon's token set is right to within a token at each end.
* **The splice structure is an artefact of the input.** Concatenated exons contain
  no introns, so the junctions the model sees are not junctions in the genome --
  they are places where the sequence jumps by however many kilobases the intron
  spanned. A model trained on genomic sequence has never seen this input.
* **Position effects are confounded with the ends of the string.** The first and
  last exons are also the first and last tokens, and transformers attend
  differently at sequence boundaries regardless of content.
""")

code(r"""
exon_table = pd.DataFrame()
if mass is not None and len(exon_lengths) >= 2:
    special = set(getattr(wrapper.tokenizer, "all_special_ids", []) or [])
    ids = tokens["input_ids"][0].tolist()
    seq_tokens = [i for i, t in enumerate(ids) if t not in special]
    bases_per_token = len(attn_sequence) / max(len(seq_tokens), 1)

    # Centre of each non-special token, in base pairs along the spliced sequence.
    token_bp = (np.arange(len(seq_tokens)) + 0.5) * bases_per_token
    ends = np.cumsum(exon_lengths)
    starts = np.concatenate([[0], ends[:-1]])

    print(f"{len(seq_tokens)} sequence tokens ({n_tokens - len(seq_tokens)} special), "
          f"{bases_per_token:.2f} bp per token")

    rows = []
    for i, (start, end) in enumerate(zip(starts, ends), start=1):
        picked = [seq_tokens[j] for j in np.flatnonzero((token_bp >= start) & (token_bp < end))]
        if not picked:
            continue
        share = float(mass[picked].sum())
        rows.append({"exon": i, "bp": int(end - start), "tokens": len(picked),
                     "attention_share": share, "per_token": share / len(picked)})

    # Named columns, so an exon list that mapped to no tokens at all still yields a
    # table rather than a KeyError on the index.
    exon_table = pd.DataFrame(rows, columns=["exon", "bp", "tokens",
                                             "attention_share", "per_token"])
    exon_table = exon_table.set_index("exon")
    exon_table["vs_uniform"] = exon_table["per_token"] * n_tokens
    display(exon_table.round(5))
    print("vs_uniform > 1 means the exon draws more attention per token than a flat "
          "distribution over all tokens -- specials included -- would give it.")
elif mass is not None:
    print(f"{len(exon_lengths)} exon(s) in the input -- the boundary mapping needs at "
          "least two, so it is skipped")
""")

code(r"""
if len(exon_table) >= 3:
    flanks = exon_table["per_token"].iloc[[0, -1]]
    interior = exon_table["per_token"].iloc[1:-1]
    ratio = float(flanks.mean() / interior.mean()) if interior.mean() else float("nan")
    print(f"first + last exon : {flanks.mean():.5f} per token")
    print(f"interior exons    : {interior.mean():.5f} per token  "
          f"({len(interior)} exon{'s' if len(interior) != 1 else ''})")
    print(f"ratio             : {ratio:.2f}x")
    print("\nA ratio far from 1 is as easily a sequence-boundary effect as a biological "
          "one -- the flanking exons are also the ends of the string. One gene cannot "
          "separate those; the same table over the whole panel could.")
elif len(exon_table):
    print(f"{len(exon_table)} exon(s) mapped -- too few to compare flanks against an interior.")
""")

# ================================================================== GENA-LM
md(r"""
### The same locus through a different tokeniser

GENA-LM is a BERT trained by a different group on a different corpus, and it
tokenises with BPE rather than fixed 6-mers. Running both on the same sequence
separates what is a property of the locus from what is a property of the
tokeniser.

The comparison has to be made carefully. Two tokenisers need not cut the same DNA
into the same number of tokens, and mean entropy is bounded by `log(n_tokens)`,
so a model with more tokens can score higher entropy while being *no less*
focused. `head_uniformity` is on a fixed 0-1 scale and is the column that stays
comparable whatever the token counts do; `entropy_vs_ceiling` divides the entropy
by its own ceiling for the same reason. The `tokens` and `entropy_ceiling`
columns are there so you can see how far apart the two scales actually are.
""")

code(r"""
# Load a model, extract every block's attention, reduce it to scalars, then drop
# both the tensors and the checkpoint -- one model resident at a time, as in the
# embedding sweep.
def attention_summary(key: str, sequence: str) -> dict | None:
    try:
        w = embedder.get_model(key)
        tk = w.tokenizer(sequence, return_tensors="pt")
        a = w.extract_attention(tk["input_ids"], attention_mask=tk["attention_mask"],
                                layers=None)
    except Exception as exc:
        print(f"{key}: skipped -- {type(exc).__name__}: {str(exc)[:140]}")
        embedder.clear_model_cache()
        return None

    n = int(tk["input_ids"].shape[1])
    entropies = np.array([tl.attention_entropy(a[b]).mean() for b in sorted(a)])
    uniformity = np.array([tl.head_uniformity(a[b]).mean() for b in sorted(a)])
    summary = {
        "tokens": n,
        "bases_per_token": len(sequence) / n,
        "blocks": len(a),
        "heads": int(a[max(a)].shape[1]),
        "entropy_ceiling": float(np.log(n)),
        "mean_entropy": float(entropies.mean()),
        "entropy_vs_ceiling": float(entropies.mean() / np.log(n)),
        "mean_uniformity": float(uniformity.mean()),
        "uniformity_last_block": float(uniformity[-1]),
    }
    del a                      # a rank-4 tensor per block adds up; drop it early
    embedder.clear_model_cache()
    return summary


COMPARE_MODELS = [ATTN_MODEL, "gena_lm_bert_base"]
tokeniser_table = pd.DataFrame()

if ATTN_GENE is not None:
    summaries = {key: attention_summary(key, attn_sequence) for key in COMPARE_MODELS}
    summaries = {key: value for key, value in summaries.items() if value is not None}
    if summaries:
        tokeniser_table = pd.DataFrame(summaries).T
        display(tokeniser_table.round(3))
""")

md(r"""
Read the `tokens` column before the entropy columns. Where the two token counts
are close, the ceilings are close and `mean_entropy` can be compared directly;
where they are not, it cannot, and only `entropy_vs_ceiling` and
`mean_uniformity` survive the comparison. The block and head counts differ too,
so a difference in `mean_uniformity` is a difference between two whole
architectures and corpora, not an isolated statement about attention.

Whether the two models agree about how selective their heads are is a question
about attention as a measurement rather than about this gene: agreement would
suggest the selectivity is a property of DNA language modelling in general,
disagreement that it belongs to the architecture or the training corpus. One gene
cannot settle that either way -- the loop above runs over any list of sequences,
which is how you would.
""")

# ========================================================= attention-free wall
md(r"""
### The attention-free wall

Two of the five DNA architectures in the roster cannot answer any of the above,
and not because embpy has not got round to them. **HyenaDNA** replaces attention
with implicit long convolutions, and **Caduceus** is a bi-directional state-space
model. Neither builds a token-by-token attention matrix at any point in its
forward pass, so there is nothing to extract -- a hook cannot capture a tensor
that was never materialised.

They declare `has_attention = False`, and `extract_attention` fails immediately
with the reason instead of running a forward pass that cannot produce weights.
The check happens before the load guard, so it costs nothing: no weights are
downloaded to be told no.
""")

code(r"""
for key in ["hyenadna_small_32k", "caduceus_ph_131k"]:
    try:
        w = embedder.get_model(key, load=False)          # no weights downloaded
        print(f"{key}: has_attention = {w.has_attention}")
        w.extract_attention(np.zeros((1, 8), dtype="int64"), layers=[-1])
        print(f"{key}: unexpectedly returned attention")
    except NotImplementedError as exc:
        print(f"{key} -> NotImplementedError: {exc}\n")
    except Exception as exc:
        print(f"{key} -> {type(exc).__name__}: {str(exc)[:160]}\n")
""")

md(r"""
That message is the honest answer to "why can I not have attention here", and it
names the alternative: per-layer activations exist in every architecture,
attention-free or not, so `extract_hidden_states` is the readout to reach for
instead -- for HyenaDNA now, and for Caduceus once `mamba_ssm` is installed,
without which it cannot be loaded at all. The per-model reasoning is recorded in
[the attention-extraction reference](../attention_extraction.md).

### Keeping `has_attention` honest

`has_attention` is a class attribute, so it can be read without downloading
anything -- which is exactly what makes it useful as a capability check, and
exactly what makes a wrong value expensive.

The trap is the default. `BaseModelWrapper.has_attention` is `True`
(`src/embpy/models/base.py:51`), so a wrapper inherits the promise unless it opts
out. Four DNA wrappers used to inherit it while being unable to keep it:
`EnformerWrapper`, `BorzoiWrapper`, `EvoWrapper` and `Evo2Wrapper` are not
HuggingFace models, so `extract_attention` falls through to the forward-hook
path, which needs `_get_layer_modules()` -- and none of them overrides it. The
flag said `True`; the call raised. All four now declare `False` explicitly, with
the reason recorded next to the declaration:

* Enformer exposes no `blocks`/`layers` `ModuleList` for the hooks to find.
* Borzoi computes its attention eagerly but never *returns* it, and a forward
  hook can only observe what a module returns.
* Evo and Evo2 call FlashAttention, which computes the softmax inside the kernel,
  so the matrix never exists as a tensor.

The table below re-derives that from the classes themselves rather than repeating
it: for each wrapper it reads the flag and, beside it, whether the class declared
the flag itself and whether it overrides the hook entry point. All three are
class-level facts, so `load=False` answers them without a download.
""")

code(r"""
FLAG_ROSTER = ["hyenadna_small_32k", "caduceus_ph_131k", "nt_v2_100m",
               "gena_lm_bert_base", "enformer_human_rough", "borzoi_v0"]

# Is `name` set anywhere below BaseModelWrapper, or only inherited from it?
def declared_by_subclass(cls, name: str) -> bool:
    return any(name in klass.__dict__ for klass in cls.__mro__
               if klass.__name__ != "BaseModelWrapper")

rows = []
for key in FLAG_ROSTER:
    try:
        w = embedder.get_model(key, load=False)
        cls = type(w)
        rows.append({"model": key, "wrapper": cls.__name__,
                     "has_attention": w.has_attention,
                     "flag_declared": declared_by_subclass(cls, "has_attention"),
                     "overrides_layer_modules": declared_by_subclass(cls, "_get_layer_modules")})
    except Exception as exc:
        rows.append({"model": key, "wrapper": "-", "has_attention": None,
                     "flag_declared": None, "overrides_layer_modules": None,
                     "wrapper_error": type(exc).__name__})

display(pd.DataFrame(rows).set_index("model"))
# The invariant: claiming attention requires a way to reach it. A row with
# has_attention=True, flag_declared=False and overrides_layer_modules=False is a
# wrapper that inherited the promise without anyone reviewing it -- the state all
# four non-HF DNA wrappers were in.
suspect = [r["model"] for r in rows
           if r.get("has_attention") and not r.get("overrides_layer_modules")
           and not r.get("flag_declared")]
print(f"\nwrappers claiming attention by inherited default: {suspect or 'none'}")
print("The cells above are the live test for Nucleotide Transformer and GENA-LM; "
      "the next cell is the live test for Enformer, whose declared False should now "
      "agree with what the call actually does.")
""")

md(r"""
The table is a claim about the classes; the next cell checks it against
behaviour. Enformer now declares `False`, so asking it for attention should fail
-- and a declared capability is only worth anything if the failure it predicts is
the failure you get. Its input is
not tokens but a one-hot tensor, built by the wrapper's own `_preprocess_sequence`
-- which also shows the other reason Enformer is the wrong model for this
notebook's inputs: it pads every input to 196,608 bp, so a spliced-exon sequence
of a couple of kilobases is almost entirely padding. Enformer wants a genomic
*window*, which is how [Variant effects](variant_effects.ipynb) uses it.
""")

code(r"""
probe_sequence = attn_sequence or "ACGT" * 500

try:
    enformer = embedder.get_model("enformer_human_rough")
    one_hot = enformer._preprocess_sequence(probe_sequence)
    padding = 1 - len(probe_sequence) / enformer.SEQUENCE_LENGTH
    print(f"declared has_attention : {enformer.has_attention}")
    print(f"one-hot shape          : {tuple(one_hot.shape)}")
    print(f"input is {len(probe_sequence):,} bp of {enformer.SEQUENCE_LENGTH:,} "
          f"-> {padding:.2%} padding\n")
    enformer.extract_attention(one_hot, layers=[-1])
    print("extract_attention returned -- the flag was right after all")
except NotImplementedError as exc:
    print(f"NotImplementedError: {exc}")
except Exception as exc:
    print(f"{type(exc).__name__}: {str(exc)[:220]}")
""")

md(r"""
The failure is explanatory -- it names the method a subclass would have to
override -- but note *when* it arrives: after the weights have been downloaded and
loaded, which is the cost `has_attention` exists to let you avoid. That is the
argument for the flag being correct rather than merely present, and it is why the
four inherited `True`s were worth changing.

Two habits survive the fix. Read `has_attention` before loading, since it is now
a reviewed claim for every DNA wrapper rather than a default for four of them.
And still wrap the extraction in a `try` for anything you have not run before: the
flag is a statement about the architecture, not a guarantee about your particular
checkpoint, tokeniser or `transformers` version. Every cell in this section does
both. The per-model reasoning is tracked in
[the attention-extraction reference](../attention_extraction.md).
""")

code(r"""
# Attention tensors are the largest objects this notebook creates; drop them and
# the model cache before moving on.
if attn:
    del attn
embedder.clear_model_cache()
""")

Path(sys.argv[1] if len(sys.argv) > 1 else "cells_part5.json").write_text(json.dumps(CELLS))
print(f"part 5: {len(CELLS)} cells")
