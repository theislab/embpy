"""Part 5: downstream tasks, protein-only metrics, mutations, weighting, wrap-up."""
from __future__ import annotations
import json, sys
from pathlib import Path
CELLS: list[tuple[str, str]] = []
def md(s: str) -> None: CELLS.append(("markdown", s.strip("\n")))
def code(s: str) -> None: CELLS.append(("code", s.strip("\n")))

md(r"""
## Downstream: which model predicts a measured property?

Structural comparison says the models differ; a benchmark says which difference
*pays*. `tl.benchmark_embeddings` trains probes on an embedding and scores them
against a target.

Three targets, chosen because they should behave *differently* — a benchmark you
cannot fail is not a benchmark:

| Target | What it is | Prior expectation |
| --- | --- | --- |
| `length` | residue count | **easy** — mean pooling is known to leak length |
| `prot_n_ptms` | curated modification sites | plausible — PTM motifs are sequence patterns |
| `prot_n_interactions` | curated interaction partners | **should fail** — hub-ness is a network property, not a sequence one |

R² is on the standard scale where 0 means "no better than predicting the mean" and
**negative values mean worse than that**. A model scoring below zero on
interaction count is the correct answer, not a broken run.
""")

code(r"""
TARGETS = [t for t in ["length", "prot_n_ptms", "prot_n_interactions"]
           if t in protein_space.obs]

# benchmark_embeddings returns one row per regressor, indexed by regressor name.
benchmarks: dict[tuple[str, str], pd.DataFrame] = {}
for target in TARGETS:
    for key in keys:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                benchmarks[(target, key)] = tl.benchmark_embeddings(
                    protein_space, perturbation_column="symbol",
                    perturbation_type="genetic", target=target,
                    obsm_key=spaces[key],
                    models=["linear", "ridge", "knn", "random_forest"],
                    mode="quick", test_size=0.25, random_state=0,
                )
        except Exception as exc:
            print(f"{target} / {key}: skipped -- {type(exc).__name__}: {str(exc)[:70]}")

best_r2 = pd.DataFrame(
    {target: {key: benchmarks[(target, key)]["r2"].max()
              for key in keys if (target, key) in benchmarks}
     for target in TARGETS}
)
print("best R2 over four probes (rows = model, columns = target)")
display(best_r2.round(3))
""")

code(r"""
# Spread across targets is the diagnostic: a target that every model fails tells
# you about the target, not about the models.
per_target = pd.DataFrame({
    "best_r2": best_r2.max(),
    "best_model": best_r2.idxmax(),
    "worst_r2": best_r2.min(),
})
display(per_target.round(3))

# A positive R2 is not automatically signal: with ten test points anything near
# zero is noise, so tier the verdict instead of reporting "beats the mean".
SIGNAL, NOISE = 0.30, 0.10
for target, row in per_target.iterrows():
    r2, model = row["best_r2"], row["best_model"]
    if r2 >= SIGNAL:
        verdict = f"real signal (R2={r2:.3f}, {model})"
    elif r2 >= NOISE:
        verdict = f"at the noise floor (R2={r2:.3f}) -- not a usable predictor"
    else:
        verdict = f"nothing usable (best R2={r2:.3f}); most models do worse than the mean"
    print(f"  {target:22s} {verdict}")
""")

code(r"""
# Full per-probe detail for the best (target, model) pair, plus the standard plot.
if benchmarks:
    top_target = per_target["best_r2"].idxmax()
    top_model = per_target.loc[top_target, "best_model"]
    print(f"all four probes for {top_model} predicting {top_target}:")
    display(benchmarks[(top_target, top_model)].round(3))
    pl.plot_benchmark(benchmarks[(top_target, top_model)],
                      title=f"{top_model} predicting {top_target}")
""")

md(r"""
**Read the baseline, not the winner.** `benchmark_embeddings` trains a plain
linear probe alongside the rest; a foundation model that cannot beat linear-on-its
own-embedding bought you nothing for this task.

**And treat these particular numbers as a demonstration, not a result.** Forty
proteins with a single 25% split is ten test points — far too few for a stable
R², so the ranking between close models here should not be trusted. For a real
decision use `mode="rigorous"` (5-fold CV with hyper-parameter search) and hold
out whole functional classes rather than random rows, so the score reflects
generalisation instead of memorising which protein is which.

That `length` is the easiest target is itself worth pausing on: it confirms
mean-pooled embeddings encode how long the sequence was. If your real target
correlates with protein length, a probe can score well by reading off length and
learning nothing about your biology — so regress length out, or stratify by it.

`tl.__all__` also carries `phenotypic_activity`, the perturbation-screen
readouts (`delta_l2`, `deg_overlap`, `phenocopy_score`), and
`compute_scib_metrics`. Those want perturbation or single-cell data — an
`.obs` control column, or batch and cell-type labels — which a panel of 40
protein sequences does not have, so they belong in
[cells](cells.ipynb) rather than here.
""")

md(r"""
## Protein-only: does embedding distance track evolutionary distance?

Everything so far applies to any modality. The next two sections do not: they
need orthology and residue positions, so they exist only for proteins.

`build_cross_species_adata` resolves orthologs through Ensembl Compara, fetches
each species' protein sequence, and embeds them into one AnnData with
`sequence_identity` recorded per protein. `identity_vs_similarity` then pairs
sequence identity against embedding similarity for every ortholog pair.

Ensembl's REST API is rate-limited and intermittently times out, so this is
wrapped: a partial table is still informative, and a failed lookup drops one
gene rather than the section.
""")

code(r"""
CROSS_SPECIES_GENES = ["TP53", "CDK1", "MAPK1", "GAPDH", "CASP3", "SRC", "JUN", "LDHA"]
SPECIES = ["human", "mouse", "zebrafish"]

cross_species = None
try:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cross_species = tl.build_cross_species_adata(
            embedder, genes=CROSS_SPECIES_GENES, species=SPECIES,
            model="esm2_8M", source_species="human",
        )
    print(f"resolved {cross_species.n_obs} proteins "
          f"across {cross_species.obs['species_short'].nunique()} species")
    display(cross_species.obs.groupby("species_short", observed=True)
            .agg(n=("symbol", "size"), mean_identity=("sequence_identity", "mean")).round(1))
except Exception as exc:
    print(f"cross-species build unavailable: {type(exc).__name__}: {str(exc)[:160]}")
finally:
    embedder.clear_model_cache()
""")

code(r"""
identity_table = None
if cross_species is not None and cross_species.n_obs > 2:
    identity_table = tl.identity_vs_similarity(cross_species, obsm_key="X_emb")
    display(identity_table.round(3))

    if len(identity_table) > 2:
        rho = identity_table[["sequence_identity", "embedding_similarity"]].corr(
            method="spearman").iloc[0, 1]
        print(f"\nSpearman(sequence identity, embedding similarity) = {rho:.3f} "
              f"over {len(identity_table)} ortholog pairs")
""")

code(r"""
if cross_species is not None and cross_species.n_obs > 2:
    display(tl.ortholog_similarity_matrix(cross_species, obsm_key="X_emb").round(3))
""")

code(r"""
# The four protein/cross-species plots in embpy.pl.
if identity_table is not None and len(identity_table):
    pl.plot_identity_vs_similarity(identity_table,
                                   title="sequence identity vs embedding similarity")
if cross_species is not None and cross_species.n_obs > 2:
    pl.plot_ortholog_similarity(cross_species, obsm_key="X_emb")
    pl.plot_conservation_barplot(cross_species, obsm_key="X_emb",
                                 reference_species="Homo")
""")

code(r"""
# plot_species_umap needs enough rows for a spectral UMAP initialisation.
if cross_species is not None and cross_species.n_obs >= 15:
    pl.plot_species_umap(cross_species, obsm_key="X_emb")
elif cross_species is not None:
    print(f"only {cross_species.n_obs} proteins resolved -- UMAP needs ~15+ rows "
          "for a spectral initialisation, so this plot is skipped.")
""")

md(r"""
## Protein-only: how far does a point mutation move the embedding?

`id_type="sequence"` embeds a raw amino-acid string, so you can compare a wild
type against a variant you construct yourself. TP53's hotspot mutations are among
the best-characterised in cancer genetics, which makes them a fair test — and the
result is a useful negative one.

Note that `embed()`'s AnnData path deliberately accepts only resolvable
identifiers (`symbol`, `ensembl_id`, `uniprot_id`); raw sequences go through
`embed_protein`.
""")

code(r"""
# Position/residue pairs are asserted against the resolved sequence rather than
# trusted, so a UniProt isoform change would fail loudly instead of silently
# mutating the wrong residue.
VARIANTS = [("R175H", 175, "R", "H", "hotspot, structural DNA-binding"),
            ("R248Q", 248, "R", "Q", "hotspot, DNA contact"),
            ("R273H", 273, "R", "H", "hotspot, DNA contact"),
            ("R282W", 282, "R", "W", "hotspot, structural"),
            ("P72R",   72, "P", "R", "common benign polymorphism")]

wt = sequences["TP53"]
wrapper = embedder.get_model("esm2_650M")

wt_vector = np.asarray(wrapper.embed(wt, pooling_strategy="mean"))
rows = []
for name, position, expected, replacement, note in VARIANTS:
    observed = wt[position - 1]
    if observed != expected:
        print(f"{name}: expected {expected}{position}, found {observed}{position} -- skipped")
        continue
    mutant = wt[:position - 1] + replacement + wt[position:]
    vector = np.asarray(wrapper.embed(mutant, pooling_strategy="mean"))
    cosine = float(vector @ wt_vector /
                   (np.linalg.norm(vector) * np.linalg.norm(wt_vector)))
    rows.append({"variant": name, "cosine_to_wt": cosine,
                 "l2_shift": float(np.linalg.norm(vector - wt_vector)),
                 "note": note})

variant_table = pd.DataFrame(rows).set_index("variant")
display(variant_table.round(6))
""")

code(r"""
# How big is that shift compared with the distances the panel actually spans?
panel_matrix = embeddings["esm2_650M"] if "esm2_650M" in embeddings else None
if panel_matrix is not None and len(variant_table):
    from scipy.spatial.distance import pdist
    between_proteins = pdist(panel_matrix, metric="euclidean")
    biggest_variant = variant_table["l2_shift"].max()
    print(f"largest single-residue shift : {biggest_variant:.4f}")
    print(f"closest two distinct proteins: {between_proteins.min():.4f}")
    print(f"median protein-protein L2    : {np.median(between_proteins):.4f}")
    print(f"\nratio (variant shift / median protein distance): "
          f"{biggest_variant / np.median(between_proteins):.4f}")
""")

md(r"""
The shift from a hotspot mutation is orders of magnitude smaller than the
distance between two different proteins — one substitution in ~400 residues
barely moves a **mean-pooled** vector, because pooling averages the change away.
That is a property of the pooling, not evidence that the model is blind to the
mutation: the per-residue representation at that position does change.

So for variant effect prediction, do not diff mean-pooled embeddings. Use the
per-residue output (`pooling_strategy="none"`), or a model trained for the task —
`esm1v_*` exists precisely for this, and ships as a 5-seed ensemble because a
single seed is noisy. embpy's variant-effect tooling
([variant_effects.ipynb](variant_effects.ipynb)) works on DNA rather than
protein sequence.
""")

md(r"""
## Protein-only: one gene, several proteins

Point mutations are one axis of sequence variation; **isoforms** are the other,
and a much larger one. Alternative splicing means a single gene yields several
distinct proteins, and UniProt curates them. `embed_protein(..., isoform="all")`
returns one vector per isoform instead of collapsing the gene to its canonical
sequence.

TP53 is a good demonstration: UniProt records nine isoforms for it, several
truncated at the N- or C-terminus, and some (Δ133p53, p53β/γ) have their own
documented biology.
""")

code(r"""
isoform_vectors = embedder.embed_protein(
    "TP53", model="esm2_650M", id_type="symbol", isoform="all",
)
isoform_sequences = resolver.get_isoforms("TP53", id_type="symbol")

canonical_id = min(isoform_vectors, key=len)      # the displayed sequence has no -N suffix
canonical_vec = np.asarray(isoform_vectors[canonical_id])

rows = []
for accession, vector in sorted(isoform_vectors.items()):
    vector = np.asarray(vector)
    rows.append({
        "accession": accession,
        "residues": len(isoform_sequences.get(accession, "")),
        "cosine_to_canonical": float(
            vector @ canonical_vec
            / (np.linalg.norm(vector) * np.linalg.norm(canonical_vec))
        ),
    })

isoform_table = pd.DataFrame(rows).set_index("accession")
isoform_table["is_canonical"] = isoform_table.index == canonical_id
display(isoform_table.round(4))

print(f"\n{len(isoform_vectors)} isoforms embedded; "
      f"canonical is {canonical_id} at {isoform_table.loc[canonical_id, 'residues']} residues")
""")

code(r"""
# Compare the spread of isoform-vs-canonical distances against the point
# mutations above -- same gene, same model, two kinds of sequence change.
iso_shift = 1 - isoform_table.loc[~isoform_table["is_canonical"], "cosine_to_canonical"]
if len(variant_table):
    var_shift = 1 - variant_table["cosine_to_wt"]
    print(f"{'':22s}{'median':>12}{'max':>12}")
    print(f"{'single-residue variant':22s}{var_shift.median():>12.6f}{var_shift.max():>12.6f}")
    print(f"{'isoform':22s}{iso_shift.median():>12.6f}{iso_shift.max():>12.6f}")
    print(f"\nisoform / variant ratio (median): "
          f"{iso_shift.median() / max(var_shift.median(), 1e-12):.0f}x")
""")

md(r"""
Isoforms move the embedding one to two orders of magnitude further than a point
mutation does, and the shift tracks how much sequence the isoform loses — the
shortest isoforms sit furthest from the canonical vector. That is the same
mean-pooling behaviour seen in the length benchmark above, now working in your
favour: pooling is insensitive to single substitutions but very sensitive to
losing a domain.

The practical consequence is that **which isoform you embed is a real modelling
decision**, not an implementation detail. Defaulting to canonical is defensible
and is what `embed()` does, but if your assay measures a specific transcript,
embed that isoform instead.

> A caveat on counting: `prot_n_isoforms` from `annotate_proteins` counts the
> isoforms UniProt *describes*, while `isoform="all"` returns those it serves a
> distinct sequence for. The two agree for TP53 but need not in general.
""")

md(r"""
## Protein-only: weighting residues before pooling

If mean pooling dilutes the residues you care about, weight them.
`tl.WeightedProteinEmbedder` pools with per-residue weights drawn from
annotation — boosting UniProt's active sites, binding sites and motifs — so
functional positions count more than the average residue.
""")

code(r"""
weighted = tl.WeightedProteinEmbedder(embedder, organism="human")

rows = []
for gene in ["TP53", "CDK1", "SRC", "CASP3"]:
    try:
        plain = np.asarray(embedder.embed_protein(gene, model="esm2_650M", id_type="symbol"))
        boosted = np.asarray(weighted.annotation_weighted_embedding(
            gene, model="esm2_650M", site_boost=3.0))
        rows.append({"protein": gene,
                     "cosine_plain_vs_weighted": float(
                         boosted @ plain / (np.linalg.norm(boosted) * np.linalg.norm(plain))),
                     "l2_shift": float(np.linalg.norm(boosted - plain))})
    except Exception as exc:
        print(f"{gene}: {type(exc).__name__}: {str(exc)[:80]}")

if rows:
    display(pd.DataFrame(rows).set_index("protein").round(5))
    print("\nsite_boost=3.0 gives annotated functional residues 3x the weight of "
          "an average residue. Larger boosts move the vector further; whether "
          "that helps is a question for benchmark_embeddings on your task.")
embedder.clear_model_cache()
""")

md(r"""
`embed_perturbation` exposes the other weighting strategies from the same class
(`tpm_weighted`, `expression_context`, `annotation_weighted`, `full`) for when a
protein's embedding should reflect the expression context it acts in.
""")

# ============================================================ save + wrap-up
md(r"""
## Save the artifact
""")

code(r"""
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# h5ad needs every .uns leaf to be a writable type. The UniProt records are
# deeply nested dicts, so serialise them; anything else that resists writing is
# serialised the same way rather than silently dropped.
to_write = protein_space.copy()
if "protein_annotations" in to_write.uns:
    to_write.uns["protein_annotations"] = json.dumps(
        {key: str(value) for key, value in to_write.uns["protein_annotations"].items()}
    )

path = OUTPUT_DIR / "protein_embeddings.h5ad"
try:
    to_write.write_h5ad(path)
except (TypeError, ValueError) as exc:
    print(f"first write attempt failed ({type(exc).__name__}), serialising .uns: {str(exc)[:100]}")
    to_write.uns = {key: json.dumps(str(value)) for key, value in to_write.uns.items()}
    to_write.write_h5ad(path)

print(f"wrote {path} ({path.stat().st_size / 1e6:.1f} MB)")
print(f"  obsm: {list(to_write.obsm)}")
print(f"  obs : {list(to_write.obs.columns)}")

# Round-trip it, because an artifact you cannot read back is not an artifact.
reloaded = ad.read_h5ad(path)
assert reloaded.n_obs == protein_space.n_obs
assert set(reloaded.obsm) == set(to_write.obsm)
print(f"  round-trip OK: {reloaded.n_obs} proteins, {len(reloaded.obsm)} embedding spaces")
""")

md(r"""
## Takeaway

**What ran here.** Five distinct protein models — ESM-1b, ESM-1v, ESM-2, ESM-C,
ProtT5 — over 40 proteins in five functional classes, compared with the global
metrics (`tsi`, `qsi`, `cka`, `similarity_correlation`), the local ones
(`compute_knn_overlap`, `knn_jaccard`, `mutual_knn`, `compare_embedding_matrices`),
and scored against labels no model ever saw.

**What to carry away.**

1. **Pick a model with a label, not a leaderboard.** The metric tables say the
   models differ; only `within_vs_between_similarity`, `knn_label_purity` and
   `benchmark_embeddings` — all scored against external labels — say which
   difference is worth anything for your task.
2. **The last layer is a choice, not a default.** `rank_layers` plus a
   class-separation sweep shows where the signal actually peaks.
3. **Mean pooling is lossy in a specific way.** It dilutes single-residue changes
   almost to nothing. Use per-residue output, a purpose-trained model, or
   `WeightedProteinEmbedder` when positions matter.
4. **Attention is available where the architecture materialises it** — ESM-2 and
   ProtT5 yes, ESM-C and ESM3 no, for a reason embpy can state. Read it as
   structural evidence, never as importance.

**Protein-specific tooling used here:** `identity_vs_similarity`,
`ortholog_similarity_matrix`, `build_cross_species_adata`,
`WeightedProteinEmbedder`, `annotate_proteins`, and the four `pl.plot_*` helpers
for cross-species comparison.

**Next.** [genes](genes.ipynb), [cells](cells.ipynb) and
[small molecules](small_molecules.ipynb) run the same programme for their
modalities. [Comparing embeddings](03_compare_embeddings.ipynb) is the reference
for choosing between the metrics used above.
""")

Path(sys.argv[1]).write_text(json.dumps(CELLS))
print(f"part 5: {len(CELLS)} cells")
