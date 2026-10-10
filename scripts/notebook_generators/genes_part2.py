"""Part 2 of docs/notebooks/genes.ipynb -- locus resolution and the embedding sweep."""
from __future__ import annotations
import json, sys
from pathlib import Path

CELLS: list[tuple[str, str]] = []
def md(s: str) -> None: CELLS.append(("markdown", s.strip("\n")))
def code(s: str) -> None: CELLS.append(("code", s.strip("\n")))

# ============================================================ A. resolution
md(r"""
## Resolve the loci, and measure them before trusting anything

A protein is a sequence. A gene is not -- not until you say *which* sequence you
mean. Three candidates exist for every gene in the panel and they are not close
to interchangeable:

* the **full locus**, promoter to terminator, introns included;
* the **spliced exons**, concatenated -- what the transcript is made of;
* the **introns** alone, which is what you would use to ask about regulation.

The choice matters because DNA models have hard context limits and loci are
mostly intron. `GeneResolver.get_gene_regions(..., region="exons")` exists for
exactly this reason, and the geometry cell below measures the ratio that
justifies it rather than quoting it.

The fetch is the expensive part. `get_gene_regions` makes **one Ensembl REST call
per exon**, so a gene with thirty exons is thirty round trips, and Ensembl
intermittently returns a 500 under load. Two consequences shape the code below:

1. **Cache to disk, incrementally.** Write the cache back after *every* gene, so
   a failure at gene 31 does not discard the first thirty.
2. **Take the coordinates from the same call as the sequence.** Each returned
   region carries `start`/`end` alongside `sequence`, so the locus span is free.
   Fetching it separately would be a second round trip for data already in hand.

There is a third consequence, and it used to be the one that bit hardest. **A
timed-out exon was dropped silently.** `_fetch_region_sequence` catches the
request exception, logs a warning and returns `None`; `get_gene_regions` then
appended only the regions that came back truthy. So a transient network blip did
not raise -- it returned a *shorter gene*. Observed while writing this notebook:

```
WARNING:root:Failed to fetch sequence 6:30722918-30723028: Read timed out
WARNING:root:Failed to fetch sequence 6:30723340-30725422: Read timed out
TUBB   2 exons    321 bp
```

TUBB has four exons and about 2,500 bp. Two of them vanished, the call returned
successfully, and the only signal was a log line that nothing reads. An embedding
computed from that string looks entirely normal -- right dimensionality, sensible
norm, plausible neighbours -- and is wrong.

`get_gene_regions` now refuses to do that: if any exon fetch fails it logs an
error naming the exon and returns `None`, because a caller can retry a `None` but
cannot detect a short sequence. The cell below keeps its own check anyway, and the
belt-and-braces is deliberate -- it asks Ensembl how many exons the canonical
transcript *has* before fetching, so a gene where the annotation itself disagrees
with the sequence server is distinguishable from a network failure, and the cache
on disk carries a `complete` flag rather than being trusted because it exists.
One extra lookup per gene against dozens of per-exon calls is a good trade.
""")

code(r"""
import requests

from embpy.resources.gene.resolver import GeneResolver

# species=, not organism= -- the constructor and the per-call argument disagree
# on spelling, and passing organism= here is a TypeError.
resolver = GeneResolver(species="human", auto_download=False)

CACHE_PATH = OUTPUT_DIR / "gene_exons.json"
try:
    _raw_cache = json.loads(CACHE_PATH.read_text()) if CACHE_PATH.exists() else {}
except Exception as exc:            # a half-written cache is a nuisance, not a stop
    print(f"cache unreadable ({type(exc).__name__}), starting empty: {CACHE_PATH}")
    _raw_cache = {}

# Only entries that passed the completeness check below are reused. An entry written
# by an older run has no "complete" flag, so it cannot be distinguished from a
# truncated fetch and is re-fetched rather than trusted -- which is why the count of
# rejected entries is printed rather than hidden.
exon_cache: dict[str, dict] = {k: v for k, v in _raw_cache.items() if v.get("complete")}
unvalidated = [s for s in symbols if s in _raw_cache and s not in exon_cache]
cached_at_start = sum(1 for s in symbols if s in exon_cache)
print(f"exon cache: {cached_at_start}/{len(symbols)} panel genes already on disk "
      f"({CACHE_PATH})")
if unvalidated:
    print(f"  {len(unvalidated)} cached entr(ies) carry no completeness flag and will "
          f"be re-fetched: {', '.join(unvalidated[:8])}")


def canonical_transcript(symbol: str) -> dict:
    # The gene record that says how many exons the answer *should* have. This is the
    # same lookup get_gene_regions makes internally; asking for it up front is what
    # lets the caller tell a complete fetch from a truncated one.
    resp = requests.get(
        f"https://rest.ensembl.org/lookup/symbol/homo_sapiens/{symbol}?expand=1",
        headers={"Content-Type": "application/json"}, timeout=60,
    )
    resp.raise_for_status()
    transcripts = resp.json().get("Transcript", [])
    if not transcripts:
        raise RuntimeError("Ensembl returned no transcripts")
    return next((t for t in transcripts if t.get("is_canonical") == 1), transcripts[0])


def fetch_exons(symbol: str, attempts: int = 3) -> dict:
    # One resolver call gives both the exon sequences and their coordinates, so the
    # locus span costs nothing extra. The expected count is one extra lookup, and it
    # is the whole reason this function can be trusted -- see the note above.
    tx = canonical_transcript(symbol)
    n_expected = len(tx.get("Exon", []))
    got = 0
    for attempt in range(attempts):
        regions = resolver.get_gene_regions(symbol, id_type="symbol",
                                            organism="human", region="exons")
        if regions and len(regions) == n_expected and all(str(r["sequence"]) for r in regions):
            starts = [int(r["start"]) for r in regions]
            ends = [int(r["end"]) for r in regions]
            return {
                "chrom": str(regions[0]["seq_region_name"]),
                "strand": int(regions[0]["strand"]),
                "transcript": tx.get("id", ""),
                "n_exons": len(regions),
                "exon_lengths": [len(str(r["sequence"])) for r in regions],
                "span": max(ends) - min(starts) + 1,
                "sequence": "".join(str(r["sequence"]) for r in regions),
                "complete": True,
            }
        got = len(regions) if regions else 0
        print(f"  {symbol}: got {got}/{n_expected} exons "
              f"(attempt {attempt + 1}/{attempts}) -- retrying")
    raise RuntimeError(f"only {got}/{n_expected} exons after {attempts} attempts")


t0 = time.perf_counter()
fetched, failures = [], {}
for symbol in symbols:
    if symbol in exon_cache:
        continue
    t_gene = time.perf_counter()
    try:
        record = fetch_exons(symbol)
        exon_cache[symbol] = record
        fetched.append(symbol)
        # Write after each gene: a mid-run Ensembl 500 then costs one gene, not all.
        CACHE_PATH.write_text(json.dumps(exon_cache))
        print(f"  {symbol:<8} {record['n_exons']:>3} exons  "
              f"{len(record['sequence']):>6,} bp  "
              f"{time.perf_counter() - t_gene:5.1f}s")
    except Exception as exc:
        failures[symbol] = f"{type(exc).__name__}: {str(exc)[:80]}"
        print(f"  {symbol:<8} failed after {time.perf_counter() - t_gene:.1f}s -- "
              f"{failures[symbol]}")

elapsed = time.perf_counter() - t0
resolved = [s for s in symbols if s in exon_cache]
print(f"\n{len(resolved)}/{len(symbols)} resolved  "
      f"({cached_at_start} from cache, {len(fetched)} from the network) "
      f"in {elapsed:.1f}s")
if failures:
    print(f"{len(failures)} gene(s) unresolved: {sorted(failures)}")
""")

md(r"""
Two things about that cache are worth stating plainly rather than discovering
later.

**It is a cache of one particular reading of each gene.** `get_gene_regions`
picks the *canonical* transcript -- the cache records which one -- and fetches
each exon on the gene's own strand, but concatenates the exons in ascending
genomic-coordinate order. For a minus-strand gene that is the reverse of
transcription order, so the string is not the mature mRNA. The next cell counts
how many panel genes that applies to. Every gene is treated identically, so the
cross-model comparison stays fair -- but do not read these vectors as transcript
embeddings.

**A gene that fails to resolve is not dropped from the notebook.** The static
lookup tables need only a symbol, so an unresolved gene still gets a row in every
static space. It only leaves the DNA half. The `has_sequence` column below is
the switch, and every DNA cell filters on it.
""")

code(r"""
gene_space.obs["has_sequence"] = [s in exon_cache for s in symbols]
gene_space.obs["exon_len"] = [
    len(exon_cache[s]["sequence"]) if s in exon_cache else np.nan for s in symbols
]
gene_space.obs["n_exons"] = [
    exon_cache[s]["n_exons"] if s in exon_cache else np.nan for s in symbols
]
gene_space.obs["locus_span"] = [
    exon_cache[s]["span"] if s in exon_cache else np.nan for s in symbols
]
gene_space.obs["span_ratio"] = gene_space.obs["locus_span"] / gene_space.obs["exon_len"]

geom = gene_space.obs.loc[gene_space.obs["has_sequence"]]
if geom.empty:
    print("no gene resolved -- the geometry summary and the DNA sweep below have "
          "nothing to run on, and every DNA cell will report that and continue")

print(f"exon length   : min {geom['exon_len'].min():,.0f}  "
      f"median {geom['exon_len'].median():,.0f}  max {geom['exon_len'].max():,.0f} bp")
print(f"locus span    : min {geom['locus_span'].min():,.0f}  "
      f"median {geom['locus_span'].median():,.0f}  max {geom['locus_span'].max():,.0f} bp")
print(f"span / exon   : median {geom['span_ratio'].median():.1f}x  "
      f"max {geom['span_ratio'].max():.1f}x")

n_minus = sum(1 for s in resolved if exon_cache[s].get("strand") == -1)
print(f"minus strand  : {n_minus}/{len(resolved)} genes -- exons concatenated in "
      f"genomic order, so reversed relative to transcription\n")

print("five widest loci:")
display(geom.nlargest(5, "locus_span")[
    ["family", "n_exons", "exon_len", "locus_span", "span_ratio"]].round(1))
""")

md(r"""
The `span_ratio` column is the whole argument for `region="exons"` in one number.
A gene whose locus is an order of magnitude longer than its exons spends that
order of magnitude on introns -- and feeding the full locus to a model with a
fixed context window means most of the window is intron, or the sequence does not
fit at all.

The panel's own spans decide how much of a problem that is. Take HyenaDNA's
32,768 bp context as the yardstick. It is the longest context among the DNA
*language models* the sweep below is expected to get through -- Caduceus is
nominally longer at 131 kb, but it needs a dependency this environment may not
have, and the sweep reports what actually happened instead of assuming it. So
this is the friendly case. (Enformer's window is six times longer again, but it
is not a language model and not a fair comparison; the
[subsection below](#why-enformer-is-not-in-the-sweep) measures why.)
""")

code(r"""
HYENADNA_CONTEXT = 32_768   # the '32k' in hyenadna_small_32k

over_locus = geom[geom["locus_span"] > HYENADNA_CONTEXT]
over_exons = geom[geom["exon_len"] > HYENADNA_CONTEXT]

print(f"of {len(geom)} resolved genes:")
print(f"  {len(over_locus):>2} have a full locus longer than {HYENADNA_CONTEXT:,} bp")
print(f"  {len(over_exons):>2} have exons longer than {HYENADNA_CONTEXT:,} bp")

if len(over_locus):
    print("\nwidest loci that overflow a 32 kb context:")
    display(over_locus.nlargest(min(8, len(over_locus)), "locus_span")[
        ["family", "exon_len", "locus_span", "span_ratio"]].round(1))

n_fit = len(geom) - len(over_exons)
print(f"\nSo: exons for the sweep. {n_fit}/{len(geom)} spliced sequences fit inside "
      f"{HYENADNA_CONTEXT:,} bp.")
if len(over_exons):
    print(f"The other {len(over_exons)} exceed it. Nothing is discarded -- the batch path "
          "splits an over-long input into context-sized chunks and mean-pools the chunk "
          "vectors -- but those rows are an average over chunks, not one forward pass.")
""")

# ================================================================== B. embed
md(r"""
## Embed every gene-capable model

Two families, and they are not variations on a theme.

### The static tables: eight lookups, six distinct sources

Every one of these is a **precomputed table**, keyed by gene identifier and
downloaded from `theislab/Embpy_Data`. That is the important property. A lookup
cannot generalise: a gene absent from the table has no vector, and no amount of
compute will produce one. This is why `missing="nan"` is not optional below --
with the default `missing="error"` a single unlisted gene aborts the call.

The evidence each one is distilled from decides what it can possibly know:

| Key | Distilled from | Keyed by | Therefore knows | Therefore cannot know |
| --- | --- | --- | --- | --- |
| `genept` | GPT-3.5 embeddings of NCBI gene summary text | Ensembl ID | what has been *written* about the gene | anything not yet curated into a summary |
| `genept_scaled` | the same text embeddings, z-scored | symbol | the same, on a comparable scale | the same blind spot |
| `gene2vec` | co-expression across GEO | Ensembl ID | which genes move together across experiments | causality, or anything about un-profiled genes |
| `crispr_gene_effect_1178` | DepMap CRISPR knockout fitness, 1178-d | symbol | which genes are needed by which cell lines | genes never screened, and non-fitness function |
| `crispr_gene_effect_205` | the same screen, 205-d | symbol | the same signal, compressed | the same |
| `omics` | a 256-d omics feature table -- the source spec says no more than that | Ensembl ID | whatever those features measure | it is the one row here you cannot reason about from provenance |
| `pops` | PoPS polygenic priority features, 256-d | Ensembl ID | GWAS-informed trait relevance | anything about genes without association data |
| `wikicrow` | LLM-written gene review articles | symbol | a longer-form written account than GenePT's | the same text-derived limits, one generation removed |

Two of these are *text about the gene*, two are the *same* CRISPR screen at two
widths, and the rest are separate measurements. That mixture is deliberate: the
[comparison section](#1-global-geometry) should show text tables agreeing with
each other more than they agree with the screens, and the key column above is
the reason to expect different coverage gaps for `genept` and `genept_scaled`
even though the numbers behind them are the same.
""")

md(r"""
### What the catalogue promises, and what it can deliver

`model_catalog("static")` used to advertise eleven gene keys, three of which
could not be loaded at all: `ccle` and `ccle_ensembl` had no source specification
behind them, and `crispr_gene_effect` had one pointing at a file the public data
repository does not ship. `embed()` got as far as the download and raised
`FileNotFoundError`.

The cause was three lists with no agreement between them --
`DEFAULT_STATIC_EMBEDDING_MODELS` (a hand-written `frozenset` in
`embpy/embedder.py`), the source specification table (`_DEFAULT_SOURCE_SPECS` in
`embpy/pp/static_embeddings.py`) and the contents of the repository itself. The
roster is now *derived* from the specification table, so the first two cannot
drift apart again; the cell below checks that invariant rather than trusting it.

What derivation cannot fix is the third list. A key is advertised when embpy knows
which file to ask for, which is not the same as the repository serving it:
`crispr_gene_effect` wants the raw DepMap matrix, and the public repository only
carries the two reduced versions. That failure is still real, and still worth
seeing.

The STRING tables in the same specification file are a separate matter. They
declare `entity_type="protein"` and are keyed by STRING protein identifiers, so
they are deliberately *not* in the gene roster -- offering them here would make
the loader try to resolve some nineteen thousand `9606.ENSP...` identifiers into
gene symbols one request at a time.
One cell to record it, then on with the sweep.
""")

code(r"""
from embpy.embedder import DEFAULT_STATIC_EMBEDDING_MODELS
from embpy.pp.static_embeddings import static_embedding_keys

advertised = set(embedder.model_catalog("static")["model"])
print(f"advertised gene lookups : {len(advertised)}")
print(f"derived from the specs  : {advertised == static_embedding_keys('gene')}")
print(f"retired phantom keys    : {sorted({'ccle', 'ccle_ensembl'} & advertised) or 'none'}")
print(f"protein-keyed tables    : {sorted(static_embedding_keys('protein'))}"
      "  <- correctly absent from the gene roster\n")

# Two instructive failures: a key whose file the public repo does not carry, and a
# protein table asked for as though it were a gene table.
for key in ["crispr_gene_effect", "string_functional_9606"]:
    try:
        embedder.embed(gene_space.copy(), entity_type="gene", id_type="symbol",
                       obs_column="symbol", model=key, output="anndata",
                       is_perturbation=True, key="X_probe", missing="nan")
        print(f"{key:<24} unexpectedly worked")
    except Exception as exc:
        print(f"{key:<24} {type(exc).__name__}: {str(exc)[:110]}")
""")

md(r"""
### The static sweep

One `embed` call per working table. Each writes its own `.obsm` key, and because
these are lookups rather than inference the cost is a download plus an index
join -- seconds, not minutes, and no model cache to clear.

`is_perturbation=True` is what sends the matrix to `.obsm` instead of `.varm`.
embpy treats gene embeddings as *feature* embeddings by default, which is right
when genes are the columns of an expression matrix and wrong here, where each
gene is a row in its own right.
""")

code(r"""
STATIC_ROSTER = ["genept", "genept_scaled", "gene2vec",
                 "crispr_gene_effect_1178", "crispr_gene_effect_205",
                 "omics", "pops", "wikicrow"]

spaces: dict[str, str] = {}
static_rows = []

for key in STATIC_ROSTER:
    obsm_key = f"X_{key}"
    t0 = time.perf_counter()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gene_space = embedder.embed(
                gene_space, entity_type="gene", id_type="symbol",
                obs_column="symbol", model=key, output="anndata",
                is_perturbation=True, key=obsm_key, missing="nan",
            )
        matrix = np.asarray(gene_space.obsm[obsm_key], dtype=float)
        gaps = [s for s, bad in zip(gene_space.obs_names, np.isnan(matrix).all(axis=1)) if bad]
        # A table that covers none of the panel is not a usable space: registering it
        # would make the all-spaces intersection below empty for everyone else.
        status = "ok" if len(gaps) < matrix.shape[0] else "skipped -- covers no panel gene"
        if len(gaps) < matrix.shape[0]:
            spaces[key] = obsm_key
        static_rows.append({"model": key, "dim": matrix.shape[1],
                            "seconds": round(time.perf_counter() - t0, 1),
                            "n_missing": len(gaps),
                            "missing": ", ".join(gaps) if gaps else "-",
                            "status": status})
    except Exception as exc:
        static_rows.append({"model": key, "dim": None,
                            "seconds": round(time.perf_counter() - t0, 1),
                            "n_missing": None, "missing": "-",
                            "status": f"skipped -- {type(exc).__name__}: {str(exc)[:50]}"})

display(pd.DataFrame(static_rows).set_index("model"))
""")

md(r"""
Read the `n_missing` column as a property of the *table*, not of the gene. A
`NaN` row means the table was built without that gene, which is a coverage
statement about the underlying resource -- which cell lines DepMap screened,
which genes have a curated NCBI summary, which have GWAS signal. It is not a
model failing to compute an answer; there was never an answer to look up.

That matters downstream because several of the comparison metrics take a
distance over rows, and a `NaN` row poisons every distance it touches. The
[end of this section](#one-gene-space-two-views) deals with it explicitly rather
than letting it propagate.
""")

md(r"""
### The DNA sweep

The DNA models take a sequence, so the exons resolved above are the input. Two
practical decisions:

* **Resolve once, embed many.** Passing gene symbols to `embed` would make the
  resolver run again for every model, multiplying an already slow fetch by the
  number of models. The cached exon strings go in directly via
  `entity_type="sequence"`.
* **`clear_model_cache()` between models.** embpy caches wrappers, which is right
  when you re-embed and wrong when you sweep -- three checkpoints resident at
  once for no benefit. Dropping each after use keeps the peak at one model.

One wrinkle in the sequence path is worth knowing before it surprises you. When
`entity_type="sequence"`, the entity *identity* is the sequence string itself, so
embpy cannot re-index the result onto rows named by gene symbol -- attaching to
`gene_space` directly raises `Cannot auto-resolve attach axis`. The call below
therefore embeds a plain list and gets a standalone AnnData back, whose
`.obs["original_sequence"]` column is used to put the rows where they belong.

`caduceus_ph_131k` is in the loop on purpose. Its state-space backbone needs
`mamba_ssm`, which is not part of this environment's install, so it is the
expected skip case -- the loop prints whatever the failure actually is rather
than asserting it in advance, and carries on to the next model.
""")

code(r"""
DNA_SWEEP = ["hyenadna_small_32k", "nt_v2_100m", "gena_lm_bert_base", "caduceus_ph_131k"]

dna_genes = [s for s in symbols if s in exon_cache]
dna_sequences = [exon_cache[s]["sequence"] for s in dna_genes]
row_of = {s: i for i, s in enumerate(gene_space.obs_names)}
if dna_genes:
    print(f"embedding {len(dna_genes)} genes "
          f"({sum(len(q) for q in dna_sequences):,} bp of spliced sequence in total)\n")
else:
    print("no resolved sequences -- the DNA sweep has nothing to embed\n")

dna_rows = []
for key in (DNA_SWEEP if dna_genes else []):
    obsm_key = f"X_{key}"
    t0 = time.perf_counter()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = embedder.embed(
                dna_sequences, entity_type="sequence", id_type="sequence",
                model=key, output="anndata", key=obsm_key, pooling_strategy="mean",
            )
        matrix = np.asarray(out.obsm[obsm_key], dtype=float)

        # Re-order defensively. The returned rows have followed input order every
        # time here, but the check costs nothing and a silent mis-alignment would be
        # invisible in every metric downstream. Anything the check cannot repair is
        # raised, so the model is reported as skipped rather than quietly scrambled.
        returned = [str(s) for s in out.obs.get("original_sequence", [])]
        if returned and returned != dna_sequences:
            if sorted(returned) != sorted(dna_sequences):
                raise RuntimeError(
                    f"returned {len(returned)} rows for {len(dna_sequences)} sequences"
                )
            pos = {seq: i for i, seq in enumerate(returned)}
            matrix = matrix[[pos[seq] for seq in dna_sequences]]
        elif matrix.shape[0] != len(dna_sequences):
            raise RuntimeError(
                f"returned {matrix.shape[0]} rows for {len(dna_sequences)} sequences"
            )

        full = np.full((gene_space.n_obs, matrix.shape[1]), np.nan, dtype=np.float32)
        for symbol, row in zip(dna_genes, matrix):
            full[row_of[symbol]] = row
        gene_space.obsm[obsm_key] = full

        spaces[key] = obsm_key
        dna_rows.append({"model": key, "dim": matrix.shape[1],
                         "seconds": round(time.perf_counter() - t0, 1),
                         "n_missing": gene_space.n_obs - len(dna_genes),
                         "status": "ok"})
    except Exception as exc:
        dna_rows.append({"model": key, "dim": None,
                         "seconds": round(time.perf_counter() - t0, 1),
                         "n_missing": None,
                         "status": f"skipped -- {type(exc).__name__}: {str(exc)[:70]}"})
    finally:
        embedder.clear_model_cache()

# Named columns so an empty sweep still produces a table rather than a KeyError.
display(pd.DataFrame(dna_rows, columns=["model", "dim", "seconds", "n_missing",
                                        "status"]).set_index("model"))
""")

md(r"""
The `seconds` column is not a throughput benchmark and should not be read as one.
It mixes a one-off checkpoint download on a model's first ever run with the
inference itself, and the three models tokenise the same DNA into very different
numbers of tokens -- HyenaDNA is single-nucleotide, NT v2 emits roughly one token
per 6 bp, GENA-LM uses BPE -- so a shared bp count is not a shared token count.
The [attention section](#reading-a-dna-models-attention) measures the token
counts directly on one sequence.

One shared mechanism does apply to all three. `_hf_batched_embed` tokenises each
input *without* truncation, splits it into context-sized chunks, embeds every
chunk and mean-pools the chunk vectors, so an over-long sequence costs more
forward passes rather than losing its tail. That is worth knowing for reasons
beyond timing: a gene whose exons overflow the context gets a vector averaged
over several partial views of itself, which is not quite the same object as a
short gene's single-pass vector.

`n_missing` is identical across DNA rows by construction: it is the count of
panel genes that never resolved, and they are the same genes for every DNA model
because they all read the same cache.
""")

# ---------------------------------------------------------------- Enformer
md(r"""
### Why Enformer is not in the sweep

Enformer is on the roster in [part 1](#which-gene-models-does-embpy-have) and it
is deliberately *not* in the loop above. It is not a question of cost. Enformer
is a supervised track predictor with a **fixed 196,608 bp input window**, and
`EnformerWrapper._preprocess_sequence` centre-pads or centre-crops whatever you
give it to exactly that length. Hand it a few kilobases of spliced exons and
almost the entire window is padding.

That is measurable, so measure it rather than asserting it. `_preprocess_sequence`
is private and reaching into it is the point here -- it is the step that would
otherwise happen invisibly inside `embed`.
""")

code(r"""
if dna_genes:
    probe_symbol = dna_genes[0]
    probe_seq = exon_cache[probe_symbol]["sequence"]
    try:
        enformer = embedder.get_model("enformer_human_rough")
        one_hot = enformer._preprocess_sequence(probe_seq)   # (1, window, 4)
        window = enformer.SEQUENCE_LENGTH
        informative = int((one_hot.sum(dim=-1) > 0).sum().item())

        print(f"one-hot shape           : {tuple(one_hot.shape)}")
        print(f"Enformer input window   : {window:,} bp")
        print(f"{probe_symbol} spliced exons".ljust(24) + f": {len(probe_seq):,} bp")
        print(f"rows carrying a base    : {informative:,}")
        print(f"blank rows              : {(window - informative) / window:.2%} of the window")
    except Exception as exc:
        print(f"Enformer probe skipped -- {type(exc).__name__}: {str(exc)[:110]}")
    finally:
        embedder.clear_model_cache()
else:
    print("no resolved sequences, skipping the Enformer probe")
""")

md(r"""
Padding is encoded as an all-zero one-hot row, and so is any `N` inside the
sequence itself -- neither maps to a base -- so the count above is "positions
carrying a base" and the fraction is the share of Enformer's receptive field
that carries no information. Whatever comes out of the trunk is dominated by the
model's response to blank input, and comparing that against a HyenaDNA embedding
of the same exons would be comparing two different questions.

The fix is not a longer sequence, it is a *different* sequence. Enformer wants a
genomic **window** -- a real stretch of chromosome centred on a position, introns
and regulatory context included -- and it is built to score what a variant does
to that window. That is a different notebook:
[Variant effects](variant_effects.ipynb) covers the window-based path, Borzoi,
`SNPEmbedder` and variant-effect profiling in full.
""")

# ------------------------------------------------------- downstream contract
md(r"""
## One gene space, two views

Everything after this section reads from a small, fixed set of names, so a model
that failed above simply drops out of the analysis instead of breaking it.

`STATIC_KEYS` and `DNA_KEYS` are the split that the notebook's central prediction
is stated in: the DNA models should recover the paralog families and struggle on
the pathway classes, and the prior-knowledge tables should do the reverse. Every
later section reports along that split.
""")

code(r"""
embeddings = {key: np.asarray(gene_space.obsm[obsm]) for key, obsm in spaces.items()}
keys = list(embeddings)
obsm_keys = [spaces[k] for k in keys]

STATIC_KEYS = [k for k in keys if k in set(STATIC_ROSTER)]
DNA_KEYS = [k for k in keys if k in set(DNA_SWEEP)]

print(f"{len(keys)} usable spaces: {len(STATIC_KEYS)} static, {len(DNA_KEYS)} DNA\n")
print("static:", ", ".join(f"{k}({embeddings[k].shape[1]}d)" for k in STATIC_KEYS))
print("dna   :", ", ".join(f"{k}({embeddings[k].shape[1]}d)" for k in DNA_KEYS))
""")

md(r"""
Now the part that is easy to hide and expensive to hide. Two independent things
put `NaN` rows into these matrices: a gene missing from a static table, and a
gene whose exons never resolved. Neither is survivable for the metrics that
follow -- CKA, the k-NN overlaps, anything that takes a distance over rows will
either refuse the input outright or carry the `nan` into every score computed
from it, and one bad row is enough.

So the notebook keeps **both** views and says which one each section uses:

* `gene_space` -- all 40 genes, `NaN` included. Right for per-model coverage
  questions and for any plot that colours by family.
* `dense_space` -- only the genes present in *every* surviving space. Right for
  the cross-model metrics, because it is the only subset on which they are all
  comparable.

Restricting to the intersection is a real cost, and not a random one: a table's
gaps are where nobody screened, curated or profiled the gene, so the surviving
subset leans towards well-studied genes. The table below names every gene that
goes and which space dropped it, which is as close as you get to reading the bias
off directly.
""")

code(r"""
# .any, not .all: the sweep counted a gene as missing when its whole row was NaN,
# but a metric breaks on a single NaN cell, so the mask is the stricter test.
complete = np.ones(gene_space.n_obs, dtype=bool)
for key in keys:
    complete &= ~np.isnan(embeddings[key]).any(axis=1)
gene_space.obs["complete"] = complete
if not keys:
    print("no space survived the sweep -- every section below will say so and skip\n")

row_index = {s: i for i, s in enumerate(gene_space.obs_names)}
dropped = gene_space.obs_names[~complete].tolist()
print(f"complete in all {len(keys)} spaces: {int(complete.sum())}/{gene_space.n_obs} genes")
if dropped:
    reasons = pd.DataFrame(
        {key: [bool(np.isnan(embeddings[key][row_index[s]]).any()) for s in dropped]
         for key in keys},
        index=pd.Index(dropped, name="gene"),
    )
    reasons.insert(0, "family", gene_space.obs.loc[dropped, "family"].astype(str).values)
    print("\ndropped genes, and which spaces are missing them (True = absent):")
    display(reasons)

dense_space = gene_space[complete].copy()
dense_embeddings = {k: np.asarray(dense_space.obsm[spaces[k]], dtype=float) for k in keys}
dense_keys = list(dense_embeddings)

print(f"\ndense_space: {dense_space.n_obs} genes x {len(dense_keys)} spaces")
kept = dense_space.obs.groupby(["coherence", "family"], observed=True).size()
print(kept.to_string())

# State the damage instead of leaving it to be eyeballed: a class that has lost
# members has a lower purity ceiling, and a class that has emptied out silently
# changes what "within-family similarity" means in every later section.
full_counts = gene_space.obs["family"].value_counts()
shrunk = {fam: (int(full_counts[fam]), int((dense_space.obs["family"] == fam).sum()))
          for fam in full_counts.index}
lost = {f: (a, b) for f, (a, b) in shrunk.items() if b < a}
if lost:
    print("\nclasses thinned by the intersection (before -> after):")
    for fam, (before, after) in sorted(lost.items(), key=lambda kv: kv[1][1]):
        note = "  <-- EMPTY, family-level metrics cannot use it" if after == 0 else ""
        print(f"  {fam:<12} {before} -> {after}{note}")
if dense_space.n_obs < 10:
    print(f"\nWARNING: only {dense_space.n_obs} genes survive in every space. The "
          "cross-model metrics below are computed on that subset and are weakly "
          "determined; read them as indicative, not as measurements of the panel.")
""")

md(r"""
The "thinned by the intersection" lines are the ones to carry forward. Every gene
the mask drops shrinks exactly one class and no other, so a within-versus-between
comparison against `family` is run on a slightly different panel from the one
part 1 designed -- and if a class has emptied out entirely, on a materially
different one. The [interpretation section](#does-the-geometry-track-biology)
computes the purity ceiling implied by the surviving class sizes before it reads
a single purity number.
""")

Path(sys.argv[1] if len(sys.argv) > 1 else "cells_part2.json").write_text(json.dumps(CELLS))
print(f"part 2: {len(CELLS)} cells")
