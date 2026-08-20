"""Part 6 of docs/notebooks/small_molecules.ipynb -- reading ChemBERTa's attention."""
from __future__ import annotations
import json, sys
from pathlib import Path

CELLS: list[tuple[str, str]] = []
def md(s: str) -> None: CELLS.append(("markdown", s.strip("\n")))
def code(s: str) -> None: CELLS.append(("code", s.strip("\n")))

# ==================================================== A. reading attention
md(r"""
## Reading ChemBERTa's attention

The fingerprints in this notebook have no inside to look at. A Morgan bit is
set or it is not, and the mapping from substructure to bit is a hash -- there
is no intermediate representation to interrogate. The two ChemBERTa
checkpoints are transformers, so there is: one attention matrix per block, and
each row of it is a distribution over the tokens of the molecule.

That makes a specific question askable. *Which parts of a molecule does the
model look at when it builds its representation?* The cells below extract the
matrices, measure how focused each head is, and map the answer back onto the
SMILES string.

> **Structural, not causal.** Attention says what attends to what. It does not
> say what the model used to reach its output. Jain & Wallace (NAACL 2019,
> *Attention is not Explanation*) showed that attention weights can be
> substantially altered without changing predictions, so read what follows as
> a description of the architecture's routing, not as an explanation of its
> chemistry.
""")

code(r"""
ATTN_MODEL = "chemberta2MTR"
ATTN_COMPOUND = "aspirin"

# Aspirin rather than one of the kinase inhibitors: 23 tokens fit on a plot
# axis and every one of them is legible as a substructure, which is the whole
# point of mapping attention back onto the string.
attn_smiles = mol_space.obs.loc[
    mol_space.obs["compound"] == ATTN_COMPOUND, "smiles"
].iloc[0]

wrapper = embedder.get_model(ATTN_MODEL)
tokens = wrapper.tokenizer(attn_smiles, return_tensors="pt")

# extract_attention takes tokenised `input_ids`, NOT the SMILES string. Passing
# the string raises `AttributeError: 'str' object has no attribute 'to'` from
# deep inside base.py, which is an unhelpful way to learn the signature.
attn = wrapper.extract_attention(
    tokens["input_ids"],
    attention_mask=tokens["attention_mask"],
    layers=None,          # every block; pass e.g. [-1] for just the last
)

token_labels = wrapper.tokenizer.convert_ids_to_tokens(tokens["input_ids"][0])
last = max(attn)

print(f"{ATTN_COMPOUND}: {attn_smiles}")
print(f"blocks returned: {sorted(attn)}")
print(f"block {last} shape: {tuple(attn[last].shape)}  (batch, heads, seq, seq)")
print(f"tokens ({len(token_labels)}): {token_labels}")

row_sums = attn[last].sum(-1).cpu().numpy()   # tensors come back on the model's device
print(f"rows are distributions: {bool(np.allclose(row_sums, 1.0, atol=1e-4))}")
""")

md(r"""
Every query row sums to 1: each token distributes a fixed budget of attention
over the molecule.

> **Layer indexing.** `extract_attention` returns **one entry per transformer
> block** -- there is no embedding-layer entry, unlike `extract_hidden_states`
> where index 0 *is* the embedding layer. `tl.block_to_attention_index` and
> `tl.block_to_hidden_state_index` convert between the two conventions, and
> the [layer sweep](proteins.ipynb#which-layer-should-you-take) in the sibling
> [proteins](proteins.ipynb) notebook shows why the distinction matters.

Note how short the model is. `chemberta2MTR` has three blocks, against ESM-2's
thirty-three. A three-block model cannot build much hierarchy, and the entropy
curve below is correspondingly flat -- that is a property of this checkpoint,
not of chemical language models in general.
""")

# ======================================================== B. head focus
md(r"""
### How focused is each head, and does that change with depth?

Attention entropy is the obvious summary: a head attending uniformly over
`n` tokens has entropy `log(n)`, and a head attending to exactly one token has
entropy 0. The interesting quantity is not the absolute value but whether it
*falls* with depth, which is what specialisation looks like.

The prediction is that it barely moves here, because there are only three
blocks to move across. That is a prediction, and the cell below scores it
rather than assuming it.
""")

code(r"""
uniform_entropy = float(np.log(len(token_labels)))

rows = []
for block in sorted(attn):
    entropy = tl.attention_entropy(attn[block])          # (batch, heads)
    uniformity = tl.head_uniformity(attn[block])         # (batch, heads)
    flat = np.ravel(entropy)
    rows.append({
        "block": block,
        "mean_entropy": float(flat.mean()),
        "min_entropy": float(flat.min()),
        "max_entropy": float(flat.max()),
        # Normalised against log(n_tokens) so the number is comparable across
        # molecules of different length, which raw entropy is not.
        "mean_frac_of_uniform": float(flat.mean() / uniform_entropy),
        "mean_uniformity": float(np.ravel(uniformity).mean()),
    })

entropy_by_block = pd.DataFrame(rows).set_index("block")
print(f"uniform-attention entropy for {len(token_labels)} tokens: "
      f"{uniform_entropy:.3f}")
display(entropy_by_block.round(3))
""")

code(r"""
fig, ax = plt.subplots(figsize=(6, 3.5))
for block in sorted(attn):
    per_head = np.ravel(tl.attention_entropy(attn[block]))
    ax.scatter([block] * len(per_head), per_head, s=18, alpha=0.7,
               label=f"block {block}" if block == 0 else None)
ax.axhline(uniform_entropy, ls="--", lw=1, color="grey")
ax.text(0.02, uniform_entropy, " uniform", va="bottom", fontsize=8,
        color="grey", transform=ax.get_yaxis_transform())
ax.set_xlabel("transformer block")
ax.set_ylabel("attention entropy (nats)")
ax.set_xticks(sorted(attn))
ax.set_title(f"{ATTN_MODEL}: per-head entropy on {ATTN_COMPOUND}")
fig.tight_layout()
plt.show()
""")

md(r"""
Each dot is one head. Read the spread rather than the mean: a block where every
head sits at the uniform line is doing no routing at all, and a block with a
wide spread has heads that disagree about what matters.
""")

# ==================================================== C. which atoms
md(r"""
### Which parts of the molecule does the model read?

Entropy says how concentrated a head is. It does not say *where*. Summing
attention over query positions gives received attention per token -- how much
of the molecule's total attention budget each token attracts.
""")

code(r"""
received = np.ravel(tl.received_attention(attn[last]))    # (batch, seq) -> (seq,)

received_table = pd.DataFrame({
    "position": np.arange(len(token_labels)),
    "token": token_labels,
    "received": received,
}).sort_values("received", ascending=False)

print(f"block {last}, top tokens by received attention:")
display(received_table.head(8).round(3))
""")

code(r"""
fig, ax = plt.subplots(figsize=(9, 3))
colours = ["tab:grey" if t.startswith("[") else "tab:blue" for t in token_labels]
ax.bar(np.arange(len(token_labels)), received, color=colours)
ax.set_xticks(np.arange(len(token_labels)))
ax.set_xticklabels(token_labels, fontsize=8)
ax.set_xlabel(f"token ({ATTN_COMPOUND}: {attn_smiles})")
ax.set_ylabel("received attention")
ax.set_title(f"{ATTN_MODEL}, block {last} -- grey bars are special tokens")
fig.tight_layout()
plt.show()
""")

md(r"""
Two things to check before reading chemistry into this plot.

* **Special tokens absorb attention.** `[CLS]` and `[SEP]` are grey above. In
  most transformers they act as a sink, and a large `[CLS]` bar says more about
  the architecture than about the molecule. Since `"cls"` pooling is what
  produces the embedding used everywhere else in this notebook, that sink is
  not incidental -- it *is* the representation.
* **A token is not an atom.** For this character-level tokeniser they are
  nearly aligned, which is why aspirin is a readable example. The next
  subsection shows what happens when they are not.
""")

# ================================================= D. two tokenisers
md(r"""
### The same molecule through two tokenisers

`chemberta2MTR` and `chemberta2MLM` are both called ChemBERTa and both sit in
the same registry family, which makes it easy to assume they are the same
architecture trained two ways. They are not. They differ in depth, in width,
and -- most consequentially for anything token-level -- in how they cut a SMILES
string into tokens.
""")

code(r"""
TOKENISER_PROBE = ["chemberta2MTR", "chemberta2MLM"]

rows = []
token_lists = {}
for key in TOKENISER_PROBE:
    w = embedder.get_model(key)
    pieces = w.tokenizer.tokenize(attn_smiles)
    token_lists[key] = pieces
    n_blocks = len(w.extract_attention(
        w.tokenizer(attn_smiles, return_tensors="pt")["input_ids"], layers=None,
    ))
    rows.append({
        "model": key,
        "n_tokens": len(pieces),
        "n_blocks": n_blocks,
        "embedding_dim": int(mol_space.obsm[f"X_{key}"].shape[1]),
        "first_8": " ".join(pieces[:8]),
    })

display(pd.DataFrame(rows).set_index("model"))
for key, pieces in token_lists.items():
    print(f"{key}: {pieces}")
""")

md(r"""
The MTR checkpoint splits aspirin one character at a time; the MLM checkpoint
uses a BPE vocabulary and merges `CC`, `(=`, `Oc` and `ccccc` into single
tokens. So a received-attention bar for the MLM model covers a *group* of
atoms, and the neat token-to-substructure reading of the previous plot does not
carry over.

They also differ in width -- 384 against 768 -- which is worth stating plainly
because "the same model, two objectives" is the natural assumption and it is
wrong. The two columns above are the evidence.

> **What this costs you.** Any attention analysis that maps back onto atoms is
> tokeniser-specific. Comparing received attention between two chemical
> language models is only meaningful if you first agree on what a position
> *is*, and for these two checkpoints there is no shared answer.
""")

# ================================================ E. molformer wall
md(r"""
### Where MoLFormer would have gone

MoLFormer-XL uses a third scheme again -- an atom-level regex -- which would
have made a three-way tokeniser comparison. It is in the registry as
`molformer_base` and `list_available_models("molecule")` lists it, but it does
not load in this environment, and the sweep above already recorded that.

The reason is worth surfacing rather than hiding, because the error the user
sees is two layers away from the cause.
""")

code(r"""
# The wrapper wraps the real ImportError in a RuntimeError, and `_get_model`
# wraps that in a ModelLoadError, so the message you get names neither the
# missing module nor the version constraint. Walk the __cause__ chain.
try:
    embedder.get_model("molformer_base")
    print("molformer_base loaded -- the three-way comparison is available")
except Exception as exc:
    chain = []
    cursor = exc
    while cursor is not None:
        chain.append(f"{type(cursor).__name__}: {str(cursor)[:80]}")
        cursor = cursor.__cause__
    for depth, entry in enumerate(chain):
        print(f"{'  ' * depth}{entry}")
""")

md(r"""
The root is `ModuleNotFoundError: No module named
'transformers.masking_utils'`. MoLFormer ships its modelling code on the Hub
and is loaded with `trust_remote_code=True`, so it runs against whatever
`transformers` the environment has; that module arrived in a later release than
the 4.48.1 pinned here for ESM compatibility. Remote code is a moving
dependency that your lockfile does not cover.

`MolformerWrapper.load` catches the failure and raises
`RuntimeError("Could not load MolFormer ...")` at
`src/embpy/models/molecule_models.py:287`, and `_get_model` turns that into
`ModelLoadError`. Both re-wraps preserve `__cause__`, which is why the chain
above is readable -- but only if you know to walk it.
""")

# ============================================== F. attention-free wall
md(r"""
### The attention-free wall

Six of the nine models in the sweep are fingerprints. Asking them for attention
is not a matter of a missing feature; there is no computation to inspect. embpy
declares that up front rather than failing at the tensor level.
""")

code(r"""
print(f"RDKitWrapper.has_attention = "
      f"{embedder.get_model('morgan_fp', load=False).has_attention}")

# The failure is explanatory rather than a bare exception, and it arrives
# without loading weights -- has_attention is a class attribute, so the check
# happens before the load guard.
try:
    embedder.get_model("morgan_fp", load=False).extract_attention(
        np.zeros((1, 8), dtype="int64"), layers=[-1],
    )
except NotImplementedError as exc:
    print(f"\nNotImplementedError: {exc}")
""")

# ============================================ G. keeping the flag honest
md(r"""
### Keeping `has_attention` honest

`has_attention` is only useful if a wrapper that claims attention can actually
produce it. `BaseModelWrapper` defaults it to `True`, so a subclass that never
overrides it inherits the claim whether or not the claim holds. The audit below
separates the two cases: which molecule wrappers *declare* the flag, and which
merely inherit it.
""")

code(r"""
FLAG_ROSTER = ["morgan_fp", "maccs_fp", "chemberta2MTR", "chemberta2MLM",
               "molformer_base", "minimol", "mhg_gnn", "mole"]

# Is `has_attention` set anywhere below BaseModelWrapper, or only inherited?
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
                     "is_hf_model": cls.__name__ in {"ChembertaWrapper",
                                                     "MolformerWrapper"}})
    except Exception as exc:
        rows.append({"model": key, "wrapper": "-", "has_attention": None,
                     "flag_declared": None, "is_hf_model": None,
                     "wrapper_error": type(exc).__name__})

display(pd.DataFrame(rows).set_index("model"))
""")

md(r"""
The fingerprint and graph wrappers declare `False`, which is honest. The two
ChemBERTa wrappers inherit `True` and earn it -- the extraction above worked.

`mole` is the uncomfortable row. `MolEWrapper` inherits `has_attention = True`
without declaring it, and it is not a HuggingFace model, so
`extract_attention` would fall through to the forward-hook path rather than
`output_attentions=True`. The question is academic here only because the
wrapper cannot be constructed at all through `BioEmbedder` -- see the
[catalogue section](#which-molecule-models-does-embpy-have). An inherited
`True` on a non-HF wrapper is a claim nothing checks, and this table is how you
find those.
""")

Path(sys.argv[1] if len(sys.argv) > 1 else "cells_part6.json").write_text(json.dumps(CELLS))
print(f"part 6: {len(CELLS)} cells")
