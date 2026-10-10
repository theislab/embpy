"""Part 4: attention."""
from __future__ import annotations
import json, sys
from pathlib import Path
CELLS: list[tuple[str, str]] = []
def md(s: str) -> None: CELLS.append(("markdown", s.strip("\n")))
def code(s: str) -> None: CELLS.append(("code", s.strip("\n")))

md(r"""
## Reading ESM-2's attention

A pooled vector is one number per dimension for a whole protein. Attention is
per-residue, and for proteins that is unusually interpretable: the tokens *are*
amino acids at known positions, so "what does the model read" can be checked
against annotated domains.

`get_model()` hands back the wrapper; the wrapper exposes the internals.

> **Structural, not causal.** Attention says what attends to what. Jain & Wallace
> (NAACL 2019, *Attention is not Explanation*) showed attention can be altered
> substantially without changing predictions, so a high-attention residue is not
> thereby shown to have driven the embedding. For "what drove this vector",
> gradient attribution on the frozen model is the better-founded tool.
""")

code(r"""
ATTN_MODEL = "esm2_650M"
ATTN_PROTEIN = "TP53"

wrapper = embedder.get_model(ATTN_MODEL)
sequence = sequences[ATTN_PROTEIN]
tokens = wrapper.tokenizer(sequence, return_tensors="pt")

attn = wrapper.extract_attention(
    tokens["input_ids"],
    attention_mask=tokens["attention_mask"],
    layers=None,          # every block; pass e.g. [-1] for just the last
)

last = max(attn)
print(f"{ATTN_PROTEIN}: {len(sequence)} residues -> {tokens['input_ids'].shape[1]} tokens")
print(f"blocks returned : {len(attn)}  (indices {min(attn)}..{last})")
print(f"tensor shape    : {tuple(attn[last].shape)} = (batch, heads, seq, seq)")

row_sums = attn[last].sum(-1).cpu().numpy()   # tensors come back on the model's device
print(f"rows are distributions: {bool(np.allclose(row_sums, 1.0, atol=1e-4))}")
""")

md(r"""
Every query row sums to 1: each residue distributes a fixed budget of attention
over the sequence.

> **Layer indexing.** `extract_attention` returns **one entry per transformer
> block** — there is no embedding-layer entry, unlike `extract_hidden_states`
> where index 0 *is* the embedding layer. `tl.block_to_attention_index` and
> `tl.block_to_hidden_state_index` convert between the two conventions.

### How focused is each head?

Entropy of a head's attention distribution: low means it concentrates on a few
residues, high means it spreads out. The ceiling is `log(seq_len)`.
""")

code(r"""
print(f"maximum possible entropy (uniform over {tokens['input_ids'].shape[1]} tokens): "
      f"{np.log(tokens['input_ids'].shape[1]):.2f}\n")

profile = []
for layer in sorted(attn):
    entropy = tl.attention_entropy(attn[layer])        # (batch, n_heads)
    profile.append({"block": layer, "mean_entropy": float(entropy.mean()),
                    "most_focused": float(entropy.min()),
                    "most_diffuse": float(entropy.max()),
                    "specialisation": float(tl.head_uniformity(attn[layer]).mean())})

depth = pd.DataFrame(profile).set_index("block")
display(depth.round(3).iloc[::4])   # every 4th block, to keep the table readable
""")

code(r"""
ax = depth[["mean_entropy", "most_focused", "most_diffuse"]].plot(
    figsize=(9, 4), title=f"{ATTN_MODEL} on {ATTN_PROTEIN}: attention entropy with depth")
ax.set_xlabel("transformer block")
ax.set_ylabel("entropy (nats)")
ax.axhline(np.log(tokens["input_ids"].shape[1]), ls="--", c="grey", lw=1)
ax.annotate("uniform ceiling", xy=(0.5, np.log(tokens["input_ids"].shape[1])),
            xytext=(2, -12), textcoords="offset points", fontsize=8, color="grey")
""")

md(r"""
`head_uniformity` measures the same thing on a fixed 0–1 scale, which is what
lets you compare across models and sequence lengths rather than only across
blocks of one model.

### Which residues does the model read?

`received_attention` sums the attention each token *receives*. For a protein this
is directly checkable: TP53's domain boundaries are annotated in UniProt
(P04637), so we can ask whether the high-attention residues fall in the
structured DNA-binding core or in the disordered tails.
""")

code(r"""
mass = tl.received_attention(attn[last])[0]      # (seq_len,)

# ESM prepends <cls>, so residue i (1-based) is token i.
top = np.argsort(mass)[::-1][:12]
print(f"{'token':>6}  {'residue':>9}  {'attention':>10}")
for t in top:
    aa = sequence[t - 1] if 1 <= t <= len(sequence) else "<special>"
    label = f"{aa}{t}" if aa != "<special>" else f"tok{t}"
    print(f"{t:>6}  {label:>9}  {mass[t]:>10.4f}")
""")

code(r"""
# UniProt P04637 domain architecture.
DOMAINS = {
    "transactivation (1-42, disordered)": (1, 42),
    "proline-rich (63-97)": (63, 97),
    "DNA-binding core (102-292)": (102, 292),
    "tetramerisation (323-356)": (323, 356),
    "C-terminal regulatory (363-393, disordered)": (363, 393),
}

rows = []
for name, (start, end) in DOMAINS.items():
    positions = [p for p in range(start, end + 1) if p <= len(sequence)]
    share = float(mass[positions].sum())
    rows.append({"domain": name, "residues": len(positions),
                 "attention_share": share,
                 "per_residue": share / max(len(positions), 1)})

domain_table = pd.DataFrame(rows).set_index("domain")
domain_table["vs_uniform"] = (domain_table["per_residue"]
                              / (1.0 / tokens["input_ids"].shape[1]))
display(domain_table.round(4))
print("\nvs_uniform > 1 means the domain draws more attention per residue than "
      "a flat distribution would give it.")
""")

code(r"""
# Is the top of that list a coincidence? p53's core binds a structural Zn(2+) ion
# through four ligands (C176, H179, C238, C242) and grips DNA through R248/R273,
# so the ranking can be checked against annotation instead of eyeballed.
ZINC_LIGANDS = {176, 179, 238, 242}
DNA_CONTACTS = {248, 273, 280, 283}
CORE = set(range(102, 293))

top_n = 12
ranked = [int(t) for t in np.argsort(mass)[::-1] if 1 <= t <= len(sequence)][:top_n]
residues = [(t, sequence[t - 1]) for t in ranked]

print("top attended residues: " + ", ".join(f"{aa}{pos}" for pos, aa in residues))
print(f"  in the DNA-binding core  : {sum(p in CORE for p, _ in residues)}/{top_n}"
      f"   (core is {len(CORE) / len(sequence):.0%} of the sequence)")
print(f"  cysteines                : {sum(aa == 'C' for _, aa in residues)}/{top_n}"
      f"   (Cys is {sequence.count('C') / len(sequence):.0%} of the sequence)")
print("  annotated Zn(2+) ligands : "
      + (", ".join(f"{aa}{p}" for p, aa in residues if p in ZINC_LIGANDS) or "none"))
print("  annotated DNA contacts   : "
      + (", ".join(f"{aa}{p}" for p, aa in residues if p in DNA_CONTACTS) or "none"))
""")

md(r"""
This is the payoff of looking inside the model. Nothing told ESM-2 where p53's zinc
site or DNA interface is — it saw sequences, never structures or annotations — yet
its attention concentrates on the cysteines of the DNA-binding core, including
ligands of the structural Zn²⁺ ion, and on the arginines that contact DNA. Those
are the residues whose mutation actually destroys p53 function in tumours.

Two honest qualifications. Cysteine is a rare residue, and rare tokens routinely
attract attention in language models for reasons unrelated to function, so some of
this may be a frequency effect rather than a structural one. And per the caveat at
the top, concentration is not causation. What this gives you is a structural
readout that lines up with known biology — a good reason to look further, not a
finished result.
""")

md(r"""
`attention_to_gene_set` does this per head rather than pooled, so you can see
whether a *few specialised heads* carry a domain's signal or whether all of them
attend to it evenly. (The name is from the single-cell case, where tokens are
genes and the set is a pathway; here the set is a domain.)
""")

code(r"""
core = [p for p in range(102, 293) if p <= len(sequence)]
tail = [p for p in range(363, 394) if p <= len(sequence)]

comparison = []
for layer in sorted(attn):
    core_heads = tl.attention_to_gene_set(attn[layer], core)
    tail_heads = tl.attention_to_gene_set(attn[layer], tail)
    comparison.append({
        "block": layer,
        "core_mean": float(core_heads.mean()),
        "core_max_head": float(core_heads.max()),
        "tail_mean": float(tail_heads.mean()),
        "tail_max_head": float(tail_heads.max()),
    })

domains_by_depth = pd.DataFrame(comparison).set_index("block")
display(domains_by_depth.round(3).iloc[::4])

print(f"\nDNA-binding core is {len(core)}/{len(sequence)} = "
      f"{len(core)/len(sequence):.0%} of the sequence, so a model attending "
      "uniformly would put that fraction of its mass there.")
""")

md(r"""
### Attention summaries satisfy the output contract

The reductions are already `(n_entities, n_dims)`, so they store in `.obsm` and
inherit every exporter and the provenance record — no parallel result type.
""")

code(r"""
attention_space = ad.AnnData(
    X=np.zeros((1, 1), dtype=np.float32),
    obs=pd.DataFrame({"protein": [ATTN_PROTEIN]}, index=[ATTN_PROTEIN]),
)
attention_space.obsm["X_attn_entropy"] = tl.attention_entropy(attn[last])
attention_space.obsm["X_attn_received"] = tl.received_attention(attn[last])
attention_space.obsm["X_attn_uniformity"] = tl.head_uniformity(attn[last])

print({key: value.shape for key, value in attention_space.obsm.items()})
del attn                      # a rank-4 tensor per block adds up; drop it early
embedder.clear_model_cache()
""")

md(r"""
### What you cannot get, and why

Not every protein model can do this, and embpy says so instead of returning
something meaningless. ESM-C and ESM3 share an attention layer that calls
`F.scaled_dot_product_attention` unconditionally
(`esm/layers/attention.py:70,76`): the fused kernel computes the softmax inside
the kernel and returns only the output, so the per-head matrix never exists as a
tensor and no forward hook can recover it.

They therefore declare `has_attention = False` and fail fast with the reason,
rather than running a forward pass that cannot produce weights.
""")

code(r"""
for key in ["esm2_650M", "esm1b", "prot_t5_xl_half", "esmc_300m", "esm3_small"]:
    try:
        w = embedder.get_model(key, load=False)      # no weights downloaded
        verdict = "extractable" if w.has_attention else "declared unavailable"
        print(f"  {key:18s} has_attention={str(w.has_attention):5s}  {verdict}")
    except Exception as exc:
        print(f"  {key:18s} unavailable here ({type(exc).__name__})")
""")

code(r"""
# The failure is explanatory rather than a bare exception.
try:
    dummy = embedder.get_model("esmc_300m", load=False)
    dummy.extract_attention(np.zeros((1, 8), dtype="int64"), layers=[-1])
except NotImplementedError as exc:
    print(f"NotImplementedError: {exc}")
except Exception as exc:
    print(f"{type(exc).__name__}: {exc}")
""")

md(r"""
One implementation detail worth knowing, because it changes results silently:
recent `transformers` releases default these architectures to the **SDPA** kernel,
which also cannot return weights. embpy switches the model to eager attention for
the duration of the extraction forward pass and restores the original setting
afterwards, so `extract_attention` works on a default-loaded model and your
embedding calls keep the faster kernel.

The full per-model matrix — including the single-cell models — is in
[the attention-extraction reference](../attention_extraction.md).
""")

Path(sys.argv[1]).write_text(json.dumps(CELLS))
print(f"part 4: {len(CELLS)} cells")
