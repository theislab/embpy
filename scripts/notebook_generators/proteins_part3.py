"""Part 3: biology-tracking, clustering, annotation, layers."""
from __future__ import annotations
import json, sys
from pathlib import Path
CELLS: list[tuple[str, str]] = []
def md(s: str) -> None: CELLS.append(("markdown", s.strip("\n")))
def code(s: str) -> None: CELLS.append(("code", s.strip("\n")))

md(r"""
## Does the geometry track biology?

Everything so far compared models *to each other*. None of it says whether any
of them is right. The `family` labels were never shown to any model, so they are
a fair external test — and this is the section that actually picks a model.

Two complementary readouts:

* `within_vs_between_similarity` — are same-class pairs more similar than
  cross-class pairs? Returns the two means, the counts, and a **p-value**.
* `knn_label_purity` — of each protein's `k` neighbours, how many share its
  class? Reports `__baseline__` (the class-prior floor) alongside, so the number
  is interpretable rather than merely large.
""")

code(r"""
scores = []
for key in keys:
    wb = pl.within_vs_between_similarity(protein_space, label_key="family",
                                         obsm_key=spaces[key])
    purity = pl.knn_label_purity(protein_space, label_key="family",
                                 obsm_key=spaces[key], k=k)
    scores.append({
        "model": key,
        "within": wb["mean_within"],
        "between": wb["mean_between"],
        "separation": wb["mean_within"] - wb["mean_between"],
        "p_value": wb["p_value"],
        "knn_purity": purity["__overall__"],
        "purity_baseline": purity["__baseline__"],
    })

ranking = pd.DataFrame(scores).set_index("model").sort_values("separation", ascending=False)
display(ranking.round(4))

best = ranking.index[0]
print(f"\nbest class separation: {best}")
""")

md(r"""
Read `separation` and `knn_purity` together. Separation is a global statement
(are the classes pulled apart on average) while purity is local (is *this*
protein among its own kind). A model can win one and lose the other, and which
you care about depends on whether you will use the space for grouping or for
lookup.

Per-class purity shows *where* the geometry works. The prediction from the panel
design was that kinases are easy and secreted signalling is hard:
""")

code(r"""
per_class = pd.DataFrame({
    key: pl.knn_label_purity(protein_space, label_key="family",
                             obsm_key=spaces[key], k=k)
    for key in keys
})
per_class = per_class.drop(index=[i for i in ("__baseline__", "__overall__") if i in per_class.index])
per_class = per_class.loc[per_class.mean(axis=1).sort_values().index]
display(per_class.round(3))

print(f"\nbaseline for every row: {1 / len(PANEL):.2f}\n")
easiest, hardest = per_class.index[-1], per_class.index[0]
print(f"easiest class across models : {easiest} "
      f"(mean purity {per_class.loc[easiest].mean():.2f})")
print(f"hardest class across models : {hardest} "
      f"(mean purity {per_class.loc[hardest].mean():.2f})")
""")

md(r"""
The ordering is the interesting part, and it follows structure rather than
function. The classes held together by a **shared fold** — glycolytic enzymes,
kinases — are recovered well by every model. The class defined purely by *what the
protein does* is recovered worst: "protease" groups together catalytic machinery
that is genuinely unrelated in sequence (cysteine proteases, matrix
metalloproteinases, and the pro-apoptotic BCL-2 members BAX and BID, which are not
proteases at all but sit in the same pathway).

That is the honest reading of a protein language model: it encodes sequence and
therefore fold, and it recovers functional categories only to the extent that they
happen to be structural ones. If your labels are pathway or mechanism labels
rather than fold labels, expect exactly this.
""")

code(r"""
# Which classes does the winning model confuse? Centroid similarity says so
# directly -- off-diagonal heat is a pair the model cannot tell apart.
pl.category_centroid_similarity(protein_space, label_key="family",
                                obsm_key=spaces[best],
                                title=f"{best}: class centroid similarity")
""")

md(r"""
## Unsupervised structure

Clustering asks the same question without the labels: left to itself, does the
embedding rediscover the functional classes? `cluster_annotation_enrichment`
then scores each cluster against the known labels.
""")

code(r"""
protein_space = tl.cluster_embeddings(protein_space, obsm_key=spaces[best],
                                      method="leiden", resolution=1.0,
                                      n_neighbors=8, key_added="cluster")
print(protein_space.obs["cluster"].value_counts().sort_index().to_string())

enrichment = tl.cluster_annotation_enrichment(protein_space, cluster_key="cluster",
                                              annotation_key="family", top_k=3)
display(enrichment)
""")

code(r"""
pl.dendrogram(protein_space, obsm_key=spaces[best], metric="cosine",
              title=f"{best}: hierarchical structure of the panel")
""")

md(r"""
## Annotate the proteins

`tl.annotate_proteins` fetches UniProt and InterPro records: subcellular
location, domains, PTMs, disease links, GO terms, interaction counts, isoforms.
Compact summaries land in `.obs` as `prot_*`, full records in
`.uns["protein_annotations"]`.

This is real measured metadata, which makes it usable both for colouring plots
and — in the next section — as a *prediction target*.
""")

code(r"""
protein_space = tl.annotate_proteins(
    protein_space, column="symbol", id_type="symbol",
    sources=["metadata", "function", "location", "domains", "ptms",
             "diseases", "go", "interactions", "isoforms"],
    copy=True,
)

prot_cols = [c for c in protein_space.obs.columns if c.startswith("prot_")]
display(protein_space.obs[["family", *prot_cols]].head(10))
print("annotation records in .uns:", len(protein_space.uns["protein_annotations"]))
""")

code(r"""
numeric_annotations = [c for c in prot_cols
                       if pd.api.types.is_numeric_dtype(protein_space.obs[c])]
display(protein_space.obs.groupby("family", observed=True)[numeric_annotations].mean().round(1))
""")

code(r"""
# The same embedding, coloured by several annotation fields at once.
panel_keys = [c for c in ["family", "prot_location", "prot_n_domains", "prot_n_interactions"]
              if c in protein_space.obs]
pl.embedding_color_panel(protein_space, color_keys=panel_keys,
                         obsm_key=spaces[best], method="pca", ncols=2,
                         annotate=False, title=f"{best} coloured by annotation")
""")

md(r"""
## Which layer should you take?

Conventional wisdom says the last layer is specialised toward the pretraining
objective and so is often *not* the best for transfer. That is a claim to test per
task rather than assume — and on this panel it turns out **not** to hold, which is
exactly why the sweep is worth running. `rank_layers` scores every layer against
the final one and against its predecessor, so you can see where the representation
stabilises.

This sweep needs one full forward pass per layer, so it uses **`esm2_8M`** — the
smallest ESM-2. The question is where information sits *with depth*, and the
depth profile is what generalises, not the checkpoint size.
""")

code(r"""
SWEEP_MODEL, SWEEP_BLOCKS = "esm2_8M", 6   # embedding layer + 6 transformer blocks

layer_spaces = {}
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    for layer in range(SWEEP_BLOCKS + 1):
        out = embedder.embed(
            protein_space.copy(), entity_type="protein", id_type="symbol",
            obs_column="symbol", model=SWEEP_MODEL, layer=layer,
            output="anndata", attach_to="obs", key="X_layer",
        )
        layer_spaces[layer] = np.asarray(out.obsm["X_layer"])
embedder.clear_model_cache()

display(tl.rank_layers(layer_spaces).round(3))
""")

code(r"""
# Does the class signal itself peak before the last layer? Score every layer the
# same way the models were scored above.
layer_scores = []
for layer, matrix in layer_spaces.items():
    probe_space = protein_space.copy()
    probe_space.obsm["X_probe"] = matrix
    wb = pl.within_vs_between_similarity(probe_space, label_key="family", obsm_key="X_probe")
    layer_scores.append({"layer": layer,
                         "separation": wb["mean_within"] - wb["mean_between"],
                         "p_value": wb["p_value"]})

layer_table = pd.DataFrame(layer_scores).set_index("layer")
display(layer_table.round(4))
best_layer = layer_table["separation"].idxmax()
print(f"\nbest-separating layer: {best_layer} of {SWEEP_BLOCKS} (final = {SWEEP_BLOCKS})")
if best_layer == SWEEP_BLOCKS:
    print("The final layer wins here, so the usual 'prefer an intermediate layer' "
          "advice does not apply to this panel -- which is why it is worth checking.")
else:
    gain = layer_table.loc[best_layer, "separation"] - layer_table.loc[SWEEP_BLOCKS, "separation"]
    print(f"An intermediate layer beats the final one by {gain:.4f} separation.")
""")

Path(sys.argv[1]).write_text(json.dumps(CELLS))
print(f"part 3: {len(CELLS)} cells")
