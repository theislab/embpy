"""Part 6: downstream benchmarks, annotate_genes, variants, save, wrap-up."""
from __future__ import annotations
import json, sys
from pathlib import Path
CELLS: list[tuple[str, str]] = []
def md(s: str) -> None: CELLS.append(("markdown", s.strip("\n")))
def code(s: str) -> None: CELLS.append(("code", s.strip("\n")))

# ================================================== downstream benchmarking
md(r"""
## Downstream: which model predicts a measured property?

Everything so far compared the models against each other, or against the panel
labels. A benchmark asks a different question: given this embedding, can a plain
regressor recover a number we measured independently? `tl.benchmark_embeddings`
trains four probes -- linear, ridge, k-NN, random forest -- on an embedding and
scores them against a target column in `.obs`.

The targets are chosen so that they should behave *differently*. A benchmark you
cannot fail is not a benchmark, and the interesting result here is the
**crossover**: the two model families should win on opposite targets.

| Target | Where it came from | Prior expectation |
| --- | --- | --- |
| `exon_len` | total exonic length, from the exon cache | **DNA models should win** -- these vectors are mean-pooled over tokens, and mean pooling is known to leak how many tokens there were |
| `n_exons` | exon count at the locus | same story but weaker -- exon count correlates with length rather than being readable off it |
| `gene_n_pathways` | pathway memberships from `annotate_gene_perturbations` | **knowledge tables should win** -- a lookup row distilled from literature, co-expression and screens is a record of what the gene *does*; a DNA model has never seen a pathway database |

Those are predictions, and the cells below score them. Whichever of the three
columns actually landed in `.obs` is used; a missing one is skipped rather than
faked.

Two things to hold on to while reading the numbers. R2 is on the standard scale
where 0 means "no better than predicting the mean", and **negative values mean
worse than that** -- a model scoring below zero on a target it should not be able
to reach is the correct answer, not a broken run. And the static tables should
land near zero on `exon_len` for a specific reason: a lookup row is keyed by gene
identity and carries nothing about the locus. If one of them does score, the
likely cause is a confound rather than sequence knowledge -- well-studied genes
tend to be both long and richly annotated.

A warning about the knowledge counts before they are used as a target. Only
`gene_n_pathways` is a genuine count. `gene_n_ppi_partners` and
`gene_n_disease_assoc` count what was *fetched*, and the fetch is capped -- the
STRING call passes `limit=DEFAULT_PPI_PARTNERS` and the Open Targets query
`size=DEFAULT_DISEASE_ASSOCIATIONS` -- so for any well-studied gene they report
the cap rather than a measurement. Measured across eight genes,
`gene_n_ppi_partners` was exactly 10 every time.

`annotate_gene_perturbations` now flags this rather than leaving you to notice it:
each capped column gets a companion `*_at_limit` boolean, and the caps themselves
are recorded in `.uns["gene_annotation_limits"]`. The selector below uses both --
it skips any candidate that is saturated for most genes, and any candidate with
fewer than three distinct values, and says which and why.
""")

code(r"""
# dense_space, and not only because its rows are gap-free: the gene_* targets were
# written by the annotation cell in the previous section, which ran on dense_space.
# gene_space never received them.
bench_space = dense_space
pert_col = "symbol"

# Read the model -> obsm-key map built by the embedding sweep rather than iterating
# .obsm. That matters here: the PCA scatter plots earlier parked their 2-d
# coordinates in .obsm as X_pca_<key>, and a projection of a model is not a model.
# Benchmarking those would invent rows that look like results.
bench_spaces = {name: obsm for name, obsm in spaces.items() if obsm in bench_space.obsm}

# The crossover prediction is stated per family, so it has to be scored per family.
# DNA_KEYS and STATIC_KEYS are the rosters that actually produced vectors.
_dna, _static = set(DNA_KEYS), set(STATIC_KEYS)


def space_kind(name: str) -> str:
    if name in _dna:
        return "dna"
    if name in _static:
        return "static"
    return "other"


inventory = pd.DataFrame([
    {"model": name, "obsm_key": obsm, "kind": space_kind(name),
     "dim": int(np.asarray(bench_space.obsm[obsm]).shape[1]),
     "finite_rows": int(np.isfinite(
         np.asarray(bench_space.obsm[obsm], dtype=float)).all(axis=1).sum())}
    for name, obsm in bench_spaces.items()
]).set_index("model")

ignored = sorted(set(bench_space.obsm) - set(bench_spaces.values()))
print(f"benchmarking on dense_space: {bench_space.n_obs} genes, "
      f"{len(bench_spaces)} model spaces")
if ignored:
    print(f"ignored (derived coordinates, not embeddings): {ignored}")
display(inventory)
""")

code(r"""
# Pick one target from each group, whichever candidate landed in .obs. The two
# sequence targets come from the exon cache; the knowledge target is whichever
# gene_* count the annotator managed to populate, since that call can partly fail.
TARGET_GROUPS = [
    ("length",    "dna",    ["exon_len"]),
    ("count",     "dna",    ["n_exons"]),
    ("knowledge", "static", ["gene_n_pathways", "gene_n_ppi_partners",
                             "gene_n_disease_assoc", "gene_n_transcription_factors"]),
]

TARGETS: list[tuple[str, str]] = []          # (column, which family should win)
for group, expected, candidates in TARGET_GROUPS:
    for col in candidates:
        if col not in bench_space.obs:
            continue
        values = pd.to_numeric(bench_space.obs[col], errors="coerce")
        if values.notna().sum() < 16 or values.nunique(dropna=True) < 3:
            print(f"{col}: present but not usable as a target "
                  f"({int(values.notna().sum())} numeric values, "
                  f"{int(values.nunique(dropna=True))} distinct) -- skipped")
            continue
        # A capped column is a lower bound, not a measurement. The annotator now
        # says so directly, so use that rather than inferring it from the spread.
        limit_col = f"{col}_at_limit"
        if limit_col in bench_space.obs:
            saturated = bench_space.obs[limit_col].astype(bool).mean()
            if saturated > 0.5:
                cap = bench_space.uns.get("gene_annotation_limits", {}).get(col)
                print(f"{col}: {saturated:.0%} of genes are at the fetch limit"
                      + (f" ({cap})" if cap else "")
                      + " -- a lower bound, not a count; skipped")
                continue
        TARGETS.append((col, expected))
        break
    else:
        print(f"no {group} target found (tried {', '.join(candidates)}) -- "
              "that section of the prediction goes unscored")

print("\ntargets:", ", ".join(f"{col} (expect {exp} ahead)" for col, exp in TARGETS) or "none")
""")

code(r"""
# dense_space is already gap-free in every embedding, so the mask below normally
# drops nothing. It is not decoration: it also drops genes whose *target* is NaN,
# and if every DNA model had failed to load, dense_space would be the intersection
# of the lookup tables alone and would still contain unresolved genes with no
# exon_len. A NaN reaching a regressor is a crash, not a low score.
MIN_ROWS = 16          # under a 0.25 split this is four test points; less is not a score

benchmarks: dict[tuple[str, str], pd.DataFrame] = {}
rows_used: dict[tuple[str, str], int] = {}

for target, _expected in TARGETS:
    y = pd.to_numeric(bench_space.obs[target], errors="coerce").to_numpy(dtype=float)
    for name, obsm_key in bench_spaces.items():
        matrix = np.asarray(bench_space.obsm[obsm_key], dtype=float)
        ok = np.isfinite(matrix).all(axis=1) & np.isfinite(y)
        if int(ok.sum()) < MIN_ROWS:
            print(f"{target} / {name}: only {int(ok.sum())} complete genes -- skipped")
            continue
        sub = bench_space[ok].copy()
        sub.obs[target] = y[ok]                      # force a plain float column
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                benchmarks[(target, name)] = tl.benchmark_embeddings(
                    sub, perturbation_column=pert_col, perturbation_type="genetic",
                    target=target, obsm_key=obsm_key,
                    models=["linear", "ridge", "knn", "random_forest"],
                    mode="quick", test_size=0.25, random_state=0,
                )
            rows_used[(target, name)] = int(ok.sum())
        except Exception as exc:
            print(f"{target} / {name}: skipped -- {type(exc).__name__}: {str(exc)[:70]}")

best_r2 = pd.DataFrame({
    target: {name: benchmarks[(target, name)]["r2"].max()
             for name in bench_spaces if (target, name) in benchmarks}
    for target, _expected in TARGETS
}).dropna(how="all")

print("\nbest R2 over the four probes (rows = model, columns = target)")
display(best_r2.round(3))

counts = pd.DataFrame({
    target: {name: rows_used[(target, name)]
             for name in bench_spaces if (target, name) in rows_used}
    for target, _expected in TARGETS
}).dropna(how="all")
if not counts.empty and counts.min().min() != counts.max().max():
    print("genes used per cell -- these differ, so columns are not strictly comparable:")
    display(counts)
elif not counts.empty:
    print(f"every cell used the same {int(counts.max().max())} genes")
""")

code(r"""
# A positive R2 is not automatically signal. The panel is 40 genes, dense_space is
# whatever survived the gap mask, and a 0.25 split leaves a single-digit test set --
# so anything near zero is noise. Tier the verdict rather than reporting "beats the
# mean"; the thresholds are a judgement call, stated here so they can be argued with.
SIGNAL, NOISE = 0.30, 0.10

if best_r2.empty:
    print("no benchmark completed -- nothing to score")
else:
    per_target = pd.DataFrame({
        "best_r2": best_r2.max(),
        "best_model": best_r2.idxmax(),
        "worst_r2": best_r2.min(),
        "worst_model": best_r2.idxmin(),
    })
    display(per_target.round(3))

    for target, row in per_target.iterrows():
        r2, model = row["best_r2"], row["best_model"]
        below_mean = int((best_r2[target] < 0).sum())
        n_scored = int(best_r2[target].notna().sum())
        if r2 >= SIGNAL:
            verdict = f"real signal (R2={r2:.3f}, {model})"
        elif r2 >= NOISE:
            verdict = f"at the noise floor (R2={r2:.3f}, {model}) -- not a usable predictor"
        else:
            verdict = f"nothing usable (best R2={r2:.3f}, {model})"
        print(f"  {target:24s} {verdict}; "
              f"{below_mean}/{n_scored} models score below zero")
""")

code(r"""
# Score the crossover prediction directly: per target, which family leads?
if not best_r2.empty:
    kinds = pd.Series({name: space_kind(name) for name in best_r2.index})
    by_kind = best_r2.groupby(kinds).mean()
    print("mean best-R2 by model family\n")
    display(by_kind.round(3))

    for target, expected in TARGETS:
        if target not in by_kind.columns:
            continue
        available = by_kind[target].dropna()
        if len(available) < 2:
            print(f"  {target:24s} only {list(available.index)} present -- not a contrast")
            continue
        leader = available.idxmax()
        held = "as predicted" if leader == expected else "CONTRARY to the prediction"
        margin = available.max() - available.min()
        print(f"  {target:24s} expected {expected:6s} ahead; {leader} leads "
              f"by {margin:.3f} -- {held}")
""")

code(r"""
# Full per-probe detail for the single best (target, model) pair, plus the plot.
if benchmarks and not best_r2.empty:
    top_target = best_r2.max().idxmax()
    top_model = best_r2[top_target].idxmax()
    print(f"all four probes for {top_model} predicting {top_target} "
          f"({rows_used[(top_target, top_model)]} genes, "
          f"{int(round(0.25 * rows_used[(top_target, top_model)]))} held out):")
    display(benchmarks[(top_target, top_model)].round(3))
    pl.plot_benchmark(benchmarks[(top_target, top_model)],
                      title=f"{top_model} predicting {top_target}")
""")

md(r"""
**Read the baseline, not the winner.** `benchmark_embeddings` trains a plain
linear probe alongside the rest, and that probe is the number to check first: a
space whose best result is no better than a linear read of its own vectors bought
you nothing for this task. A random forest beating linear by a wide margin is
usually telling you the relationship is non-linear, not that the embedding is
good.

**Treat these particular numbers as a demonstration, not a benchmark result.** The
cell above printed how many genes went in and how many were held out; at a panel of
40 and a gap-free subset smaller than that, a 25% split leaves a single-digit test
set. That is far too few for a stable R2, so the ordering between two close models
here is not information -- only the gap between a model near the signal threshold
and one below zero is. For a real decision use `mode="rigorous"` (5-fold CV with a
hyper-parameter search) and hold out whole families rather than random rows, so the
score measures generalisation instead of memorising which gene is which.

The length result, wherever it landed, is worth pausing on for a reason that
outlives this panel. Mean pooling averages over tokens, so a mean-pooled DNA
embedding carries a trace of how many tokens there were. If your real target
correlates with locus or transcript length -- expression level and mappability
both do -- a probe can score well by reading off length and learning nothing
about your biology. Regress length out, or stratify by it, before believing the
score.

`tl.__all__` also carries `phenotypic_activity`, the perturbation-screen readouts
(`delta_l2`, `deg_overlap`, `phenocopy_score`) and `compute_scib_metrics`. Those
need perturbation or single-cell data -- a control column in `.obs`, or batch and
cell-type labels -- which a panel of 40 gene identifiers does not have, so they
live in [cells](cells.ipynb).
""")

# =================================================== annotate_genes vs the rest
md(r"""
## A calling-convention trap: `annotate_genes` is not `annotate_gene_perturbations`

Two functions in `tl` start with `annotate` and look interchangeable. They are
not, and the differences are the kind that fail quietly:

| | `annotate_gene_perturbations` | `annotate_genes` |
| --- | --- | --- |
| Reads identifiers from | `.obs[column]` -- you name the column | `.var_names`, unless you pass `query_ids` |
| Identifier type | inferred (symbols or Ensembl IDs) | fixed by `reference_id` |
| Returns | a modified **copy** by default (`copy=True`) | **`None`** |
| Writes | `gene_*` columns in `.obs`, full records in `.uns` | see below -- the cell measures it |
| Backed by | MyGene, GTEx, HPA, STRING, DoRothEA, Open Targets, GWAS | pertpy's `CellLine` LookUp |

The trap is the return value. `adata = tl.annotate_genes(adata, ...)` silently
replaces your AnnData with `None`, because this one mutates in place -- or does
not write at all. And the default axis is wrong for this notebook: our genes are
**rows**, so `.var_names` holds a placeholder and `query_ids` has to be passed
explicitly or the overlap check runs against nothing.

Rather than assert what it writes, the cell below diffs the object before and
after and prints whatever actually landed.
""")

code(r"""
# Run on a copy: whatever this does, it should not touch the object the rest of
# the notebook is using. `annot_probe` rather than `annotated`, which the previous
# section already used for a flag.
annot_probe = bench_space.copy()
before_obs = set(annot_probe.obs.columns)
before_uns = set(annot_probe.uns)

try:
    returned = tl.annotate_genes(
        annot_probe,
        reference_id="hgnc_symbol",
        query_ids=list(annot_probe.obs[pert_col]),
    )
    new_obs = sorted(set(annot_probe.obs.columns) - before_obs)
    new_uns = sorted(set(annot_probe.uns) - before_uns)
    print(f"\nreturn value : {returned!r}   <- not an AnnData")
    print(f"new .obs cols: {new_obs or 'none'}")
    print(f"new .uns keys: {new_uns or 'none'}")
    if new_obs:
        display(annot_probe.obs[new_obs].head())
except Exception as exc:
    print(f"annotate_genes unavailable: {type(exc).__name__}: {str(exc)[:200]}")
    print("It goes through pertpy's CellLine LookUp, which downloads its metadata "
          "on first use -- an offline kernel fails here and loses nothing else.")
""")

md(r"""
Read the two "new ... cols/keys" lines against the source. `annotate_genes` calls
pertpy's `LookUp.available_genes_annotation`, which *prints* a coverage report;
it does not assign to `.obs` or `.uns`, which is why the return value is `None`
and why the diff above is what it is. Treat it as a coverage check to run
**before** committing to an identifier type, and use
`annotate_gene_perturbations` -- which produced the `gene_*` columns benchmarked
above -- when you want annotations as data.

### The same class of trap, already sprung inside embpy

The resolution section flagged that `GeneResolver` is constructed with `species=`
while its own `get_gene_regions` takes `organism=`. That inconsistency is not
hypothetical. `embpy.embedder._resolve_gene_jump_fallback` -- the path that
canonicalises a gene symbol when a JUMP lookup misses -- constructs
`GeneResolver(organism="human")`, and the whole body sits inside a bare
`except Exception: pass`. If that call is a `TypeError`, the fallback returns
nothing and the MyGene.info alias lookup written below it never runs either.

Nothing in this notebook depends on that path. The cell is here because a
swallowed exception is invisible by design: the only way to find out is to call
the function and read what comes back against what the constructor accepts.
""")

code(r"""
import inspect

from embpy.embedder import _resolve_gene_jump_fallback
from embpy.resources.gene.resolver import GeneResolver as _GeneResolver

try:
    jump_rows, jump_source = _resolve_gene_jump_fallback("TP53", return_source=True)
    accepted = [p for p in inspect.signature(_GeneResolver.__init__).parameters
                if p != "self"]
    body = inspect.getsource(_resolve_gene_jump_fallback)

    print(f"_resolve_gene_jump_fallback('TP53') -> {len(jump_rows)} rows, "
          f"source={jump_source!r}")
    print(f"GeneResolver.__init__ accepts             : {accepted}")
    print(f"fallback calls GeneResolver(organism=...) : "
          f"{'GeneResolver(organism=' in body}")
    print(f"...inside a bare 'except Exception'       : {'except Exception' in body}")
except Exception as exc:
    print(f"probe skipped -- {type(exc).__name__}: {str(exc)[:160]}")
""")

md(r"""
Read the four lines together rather than any one of them. A row count of zero with
no source, from a function whose only job is to produce rows, next to a keyword the
constructor does not list, is a fallback that cannot fire. Dead code that returns a
valid-looking empty answer is worse than dead code that raises, because every caller
reads the empty answer as "no match found" -- which is also what a genuine miss
looks like.
""")

# ================================================================== variants
md(r"""
## Variants, briefly

A gene embedding answers "what is this gene like". A variant asks something
narrower: what changes when one base changes. That is a different enough problem
to have [its own notebook](variant_effects.ipynb), which covers Borzoi's track
predictions, `SNPEmbedder` over real chromosomes, and profile scoring in depth.
This section only shows what exists and runs the mechanism once, so you know what
you are reaching for.

`embpy.tl.genomics` holds the variant utilities. Some are re-exported on `tl`
itself and some are not, which is worth checking rather than guessing -- the cell
below does exactly that.

| Name | What it is for |
| --- | --- |
| `SNPContext` | describes one variant: position, ref, alts, context window, strand |
| `SNPEmbedder` | embeds ref and alt contexts with any loaded DNA wrapper, returns deltas |
| `SNPEmbeddingResult` | the container -- ref/alt/delta vectors, `delta_norms`, cosine similarities |
| `SequenceProvider` | serves chromosome sequence from a per-chromosome FASTA directory, one indexed multi-record FASTA, or the Ensembl REST API -- in that preference order |
| `embed_vcf` | the batch path: a VCF path, a loaded wrapper, and a `chromosome_sequences` dict keyed by the VCF's CHROM column |
| `profile_variant_effect_score` | log2 fold-change over predicted coverage bins (the Borzoi statistic) |
| `genomic_to_bin_indices` | maps a genomic interval onto a track model's output bins |
| `download_hg38_per_chrom`, `download_hg38_single_fasta` | fetch the reference (~900 MB -- not run here) |
""")

code(r"""
import embpy.tl.genomics as genomics

GENOMICS_API = ["SNPContext", "SNPEmbedder", "SNPEmbeddingResult", "SequenceProvider",
                "VariantEffectResult", "embed_vcf", "profile_variant_effect_score",
                "genomic_to_bin_indices", "download_hg38_per_chrom",
                "download_hg38_single_fasta"]

genomics_api = pd.DataFrame([
    {"name": name,
     "on embpy.tl.genomics": hasattr(genomics, name),
     "re-exported on tl": hasattr(tl, name)}
    for name in GENOMICS_API
]).set_index("name")
display(genomics_api)

submodule_only = list(genomics_api.index[~genomics_api["re-exported on tl"]])
print(f"reachable only by importing the submodule: {submodule_only or 'none'}")
""")

md(r"""
Whatever that last line names is the practical point: `tl.__all__` re-exports part
of this API and not the rest, so a call written against `tl.` alone will work for
some of these names and raise `AttributeError` for others. Import
`embpy.tl.genomics` and you have all of it -- which is what the worked example
below does, so it does not depend on which names happen to be re-exported.

### One worked example, with no genome download

`SNPEmbedder.embed_snp` takes a plain sequence string and a 1-based position into
it. That string is normally a chromosome, but nothing requires it to be: the exon
sequence already cached in this notebook works as a **synthetic contig**, which
makes the mechanism runnable without the ~900 MB hg38 fetch.

The position and reference base are hard-coded and asserted against the cached
sequence. If the cache is ever rebuilt differently the assertion fires, the demo
is skipped with the mismatch printed, and nothing downstream reports a number --
rather than the cell quietly mutating whichever base happens to sit there and
presenting the result as a variant.

Be clear about what this is: a **mechanism demo on a synthetic contig**, not a
variant-effect result. The contig is spliced exons, so the variant has no
promoter, no splice sites and no intronic context around it -- exactly the
information a variant-effect model needs. Real work belongs in
[variant_effects.ipynb](variant_effects.ipynb).
""")

code(r"""
# Read the exon cache from disk under its own name, so the resolution section's
# in-memory copy is left alone.
cache_path = OUTPUT_DIR / "gene_exons.json"
cached_exons = json.loads(cache_path.read_text()) if cache_path.exists() else {}

CONTIG_GENE, SNP_POS, SNP_REF, SNP_ALT = "TUBA1A", 800, "A", "G"

snp_result = None
try:
    contig = cached_exons[CONTIG_GENE]["sequence"].upper()
    # embed_vcf takes exactly this shape, keyed by the VCF's CHROM column.
    chromosome_sequences = {f"synthetic_{CONTIG_GENE}": contig}

    observed = contig[SNP_POS - 1]
    assert observed == SNP_REF, (
        f"{CONTIG_GENE} position {SNP_POS} is '{observed}', not '{SNP_REF}' -- "
        f"the cache changed; pick a new position"
    )

    hyena_wrapper = embedder.get_model("hyenadna_small_32k")
    snp = genomics.SNPContext(
        position=SNP_POS, ref_allele=SNP_REF, alt_alleles=[SNP_ALT],
        context_window=512, chrom=f"synthetic_{CONTIG_GENE}",
        variant_id=f"{CONTIG_GENE}:{SNP_POS}{SNP_REF}>{SNP_ALT}",
    )
    snp_embedder = genomics.SNPEmbedder(hyena_wrapper, pooling_strategy="mean")
    snp_result = snp_embedder.embed_snp(snp, chromosome_sequences[snp.chrom])

    print(f"contig: {len(contig)} bp of spliced {CONTIG_GENE} exons")
    print(f"window: {len(snp_result.ref_sequence)} bp centred on position {SNP_POS}")
    # to_dict() returns a list of rows, one per alt allele -- not a dict.
    display(pd.DataFrame(snp_result.to_dict()))
except Exception as exc:
    print(f"SNP demo skipped -- {type(exc).__name__}: {str(exc)[:300]}")
""")

code(r"""
# A distance with no scale means nothing. Compare the one-base shift against the
# distances this same model puts between different genes.
if snp_result is not None:
    # From bench_spaces, not from .obsm directly: the PCA scatter plots left 2-d
    # coordinate arrays under X_pca_X_hyenadna_..., and a distance measured in
    # those would be a distance in a plot, not in the model's space.
    hyena_key = next((obsm for name, obsm in bench_spaces.items() if "hyenadna" in name), None)
    if hyena_key is None:
        print("no HyenaDNA panel space in .obsm -- the shift has no scale to be read against")
    else:
        from scipy.spatial.distance import pdist

        panel = np.asarray(bench_space.obsm[hyena_key], dtype=float)
        panel = panel[np.isfinite(panel).all(axis=1)]
        between = pdist(panel, metric="euclidean")
        shift = snp_result.delta_norms[0]

        print(f"one-base shift (ref -> alt)   : {shift:.4f}")
        print(f"closest two distinct genes    : {between.min():.4f}")
        print(f"median gene-gene L2           : {np.median(between):.4f}")
        print(f"ratio (shift / median gene-gene): {shift / np.median(between):.4f}")
        print(f"\ncosine(ref, alt) = {snp_result.cosine_similarities[0]:.6f}")
        print("Caveat: the panel vectors were pooled over the whole exon sequence and "
              "these two over a 512 bp window, so the ratio is indicative, not exact.")
embedder.clear_model_cache()
""")

md(r"""
Expect that ratio to be small, and the reason matters more than the number:
averaging over hundreds of token vectors dilutes a one-token change almost to
nothing. It is the same behaviour [proteins.ipynb](proteins.ipynb) measures for
point mutations, and it is a property of the pooling rather than evidence the
model is blind to the base -- the per-position representation at that site does
change, the mean just does not carry it.

So do not build a variant-effect score by diffing mean-pooled embeddings. Use a
model that predicts something measurable at the locus and score the *prediction*,
which is what `profile_variant_effect_score` does for Borzoi's coverage tracks,
and what [variant_effects.ipynb](variant_effects.ipynb) works through properly --
with real genomic context, on real chromosomes.
""")

# =============================================================== save artifact
md(r"""
## Save the artifact

The full `gene_space` is written -- all 40 genes, NaNs and all, rather than
`dense_space`. The gaps are information: they record which genes each lookup table
has no row for, and dropping them here would force whoever reloads this file to
re-derive coverage. It also means the saved obsm keys are the raw model spaces
only; the PCA coordinates the plots produced live on `dense_space` and are not
worth persisting.

Sequence strings must not go in. Each is a few kilobases, they would dominate the
file, and they already exist in `gene_exons.json`. This notebook kept them in that
cache rather than in `.obs`, so the guard below usually finds nothing -- it is
there because parking the sequence in a column is the obvious thing to do when
adapting this notebook, and the resulting h5ad grows by an order of magnitude
without anyone noticing. What it keeps is the length, which is what downstream
analysis reads anyway.
""")

code(r"""
OUTPUT_DIR.mkdir(exist_ok=True)
to_write = gene_space.copy()

# Any object-dtype column averaging over 200 characters is a sequence, not
# metadata. Keep the length, drop the string.
LONG_STRING = 200
for col in list(to_write.obs.columns):
    values = to_write.obs[col]
    if values.dtype != object:
        continue
    lengths = values.astype(str).str.len()
    if lengths.mean() > LONG_STRING:
        to_write.obs[f"{col}_bp"] = lengths.astype(int).to_numpy()
        to_write.obs = to_write.obs.drop(columns=[col])
        print(f"replaced obs['{col}'] with obs['{col}_bp'] "
              f"(max {int(lengths.max())} characters)")

path = OUTPUT_DIR / "gene_embeddings.h5ad"
written = False
try:
    to_write.write_h5ad(path)
    written = True
except (TypeError, ValueError) as exc:
    # h5ad needs every .uns leaf to be a writable type; nested annotation records
    # are the usual offender, so serialise them rather than silently drop them.
    print(f"first write failed ({type(exc).__name__}), serialising .uns: {str(exc)[:100]}")
    try:
        to_write.uns = {key: json.dumps(str(value)) for key, value in to_write.uns.items()}
        to_write.write_h5ad(path)
        written = True
    except Exception as exc2:
        print(f"still not writable -- {type(exc2).__name__}: {str(exc2)[:160]}")

if written:
    print(f"\nwrote {path} ({path.stat().st_size / 1e6:.1f} MB), {to_write.n_obs} genes")
    print(f"  obsm ({len(to_write.obsm)}): {', '.join(to_write.obsm)}")
    for key in to_write.obsm:
        matrix = np.asarray(to_write.obsm[key], dtype=float)
        gaps = int((~np.isfinite(matrix).all(axis=1)).sum())
        print(f"    {key:34s} {matrix.shape[1]:5d}d   "
              f"{'complete' if gaps == 0 else f'{gaps} gene(s) missing'}")
    print(f"  obs: {list(to_write.obs.columns)}")

    # An artifact you cannot read back is not an artifact.
    reloaded = ad.read_h5ad(path)
    assert reloaded.n_obs == gene_space.n_obs
    assert set(reloaded.obsm) == set(to_write.obsm)
    print(f"  round-trip OK: {reloaded.n_obs} genes, {len(reloaded.obsm)} embedding spaces")
""")

# ================================================================== wrap-up
md(r"""
## What we found

**The prediction.** The notebook opened with a claim: DNA sequence models should
recover the paralog families, because those families *are* shared sequence, and
should struggle on the pathway classes, because a pathway is a fact about function
that leaves no common signature in the DNA -- with the prior-knowledge tables doing
the reverse. The [interpretation section](#does-the-geometry-track-biology) scored
it with k-NN label purity against labels no model saw, and wrote its conclusion
into `PREDICTION_VERDICT`; the benchmark at the top of this section scored the same
split a second way, against numeric targets neither family was built for.

The cell below re-prints that verdict rather than paraphrasing it. A summary that
restated the number would stop being true the first time the panel, the roster or
Ensembl changed underneath it.
""")

code(r"""
scored_verdict = globals().get("PREDICTION_VERDICT")
scored_deltas = globals().get("PREDICTION_DELTAS", {})

if scored_verdict is None:
    print("the interpretation section did not produce a verdict in this run -- "
          "re-run it; everything below still applies.")
else:
    print(scored_verdict)
    if scored_deltas:
        print("\nthe numbers behind it: " + "  ".join(
            f"{name}={value:.3f}" if isinstance(value, float) else f"{name}={value}"
            for name, value in scored_deltas.items()))

if not best_r2.empty:
    print(f"\nand from the benchmark above, best R2 per target:\n"
          f"{best_r2.max().round(3).to_string()}")
""")

md(r"""
What does not depend on how those numbers fell: the two model families are not
interchangeable, they fail on different things, and which one you want is settled
by your labels rather than by a leaderboard.

**Practical points, each of which cost something to learn here.**

1. **Fetch exons, not the locus.** `region="exons"` was not a stylistic choice.
   A gene's intronic span runs many times its exonic length -- the geometry table
   earlier measured the ratio for these genes -- so a whole-locus fetch is mostly
   intron, and at those lengths you are no longer comparing models, you are
   comparing how they truncate.
2. **Enformer is not a drop-in sequence encoder.** It centre-pads every input to
   196,608 bp, so a few kilobases of spliced exons arrive as overwhelmingly
   padding. It wants a genomic window; give it one, in
   [variant_effects.ipynb](variant_effects.ipynb).
3. **`missing="nan"` for lookup tables.** Coverage is not uniform across the
   static tables and the default is to fail on the first gene a table does not
   have. `missing="nan"` turns a fatal lookup into a gap you can see, count and
   decide about -- which is why the coverage table exists.
4. **The static catalogue and the data repository disagree in both directions.**
   `model_catalog("static")` lists `ccle`, `ccle_ensembl` and
   `crispr_gene_effect`, which the repository does not carry -- the probe cell in
   the embed section ran two of them and got `FileNotFoundError`. In the other
   direction, `string_functional_9606` and `string_node2vec_9606` are real files
   with no registry entry, so `embed()` raises `ModelNotFoundError` before it
   looks. The catalogue is generated from a source specification, not from the
   repository contents, and nothing checks that they agree. Probe a key on one
   gene before building a sweep on it.
5. **`clear_model_cache()` between models in a sweep.** embpy keeps loaded
   wrappers, which is right when you re-embed and wrong when you sweep: holding
   several DNA checkpoints resident at once buys nothing and is the easiest way
   to turn a sweep that would have fitted into an out-of-memory error.
6. **`has_attention` is trustworthy only for the HuggingFace-backed wrappers.**
   It is a class attribute defaulting to `True` on `BaseModelWrapper`, and only
   HyenaDNA and Caduceus override it to `False`. Enformer, Borzoi and the Evo
   wrappers inherit `True` while overriding no layer introspection, so they
   advertise a capability they do not have; the attention section demonstrates
   Enformer raising `NotImplementedError` after the weights have already loaded.
   Where it matters, the reliable check is to attempt the extraction and catch
   that exception.

**Where to go next.**

* [Comparing embeddings](03_compare_embeddings.ipynb) -- what each metric used
  here actually measures, and when they are allowed to disagree.
* [Reading a model's attention](06_attention_weights.ipynb) -- the
  extract-then-summarise pattern applied one step at a time, on smaller inputs.
* [Variant effects](variant_effects.ipynb) -- the proper version of the section
  above: Borzoi, real genomic context, and profile-based scoring.
* [Proteins](proteins.ipynb) -- the same programme for protein language models,
  where the sequence-versus-function tension resolves differently because the
  sequence *is* the functional unit.
""")

Path(sys.argv[1] if len(sys.argv) > 1 else "cells_part6.json").write_text(json.dumps(CELLS))
print(f"part 6: {len(CELLS)} cells")
