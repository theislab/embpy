---
name: embpy-layer-selection
description: Choose which layer of a transformer model (ESM-2, ProtT5, Nucleotide Transformer, HyenaDNA, text models) gives the most useful embedding for a task with embpy - embed every layer, score each with a cross-validated probe against the user's labels and compare layers structurally, then embed with the chosen layer. Use when the user asks which layer to use, says final-layer embeddings perform poorly, or wants intermediate-layer embeddings.
---

# Choosing the embedding layer

The last layer of a foundation model is specialised for its pretraining task; intermediate
layers often transfer better. Measure it on the user's data instead of assuming.

Needs labels (or a numeric value) for at least a few dozen entities. Without labels,
`rank_layers` still reports how similar each layer is to its neighbours and to the last layer,
but cannot say which is best.

## 1. Embed every layer

```python
import embpy as ep

embedder = ep.BioEmbedder(device="cpu")
n_layers = embedder.get_model("esm2_8M").get_num_layers()
per_layer = {}
for layer in range(1, n_layers + 1):
    adata = embedder.embed_fasta("data/proteins.fasta", model="esm2_8M", seq_type="protein", target_layer=layer, obsm_key="X")
    per_layer[layer] = adata.obsm["X"]
labels = adata.obs["description"].str.extract(r"class=(\S+)")[0].to_numpy()
print(n_layers, {layer: m.shape for layer, m in per_layer.items()})
```

- Layers are numbered `1` to `n_layers` (transformer blocks); negative numbers count from the
  end, so `-1` is the default last layer.
- For gene, protein-ID, molecule or text inputs use the same loop with
  `embedder.embed(..., model=..., layer=layer, output="anndata")`.
- Fingerprints, static lookup tables and some wrappers have no layers; they raise a
  `ValueError` saying so.

## 2. Score the layers

```python
scores = ep.tl.rank_layers(per_layer, target=labels, cv=3)
print(scores)
best = int(scores["probe_score"].idxmax())
if scores.loc[best, "probe_score"] - scores.loc[n_layers, "probe_score"] < 0.02:
    best = n_layers  # no clear gain over the default last layer
print("chosen layer:", best)
```

The default last layer is kept unless another layer is clearly better (here by at least 0.02).
Ties are common when the task is easy; `idxmax` alone would then pick the first layer for no
reason. Adjust the margin to the fold-to-fold spread on the user's data.

Columns, one row per layer:
- `probe_score`: cross-validated accuracy (categorical labels) or R^2 (numeric target) of a
  ridge / logistic probe. Higher is better.
- `to_final`: alignment with the last layer. Low values mean the layer keeps information the
  last layer dropped.
- `to_target`: how well the layer's geometry matches the labels.
- `to_previous`: alignment with the layer before. Near 1 means little changes between them.

Use `cv` folds that fit the smallest class (at least 2 examples per class per fold).

## 3. Embed with the chosen layer

```python
final = embedder.embed_fasta("data/proteins.fasta", model="esm2_8M", seq_type="protein", target_layer=best)
final.write_h5ad("data/proteins_best_layer.h5ad")
```

## Reporting

Show the `probe_score` per layer with the number of entities and folds, and the chosen layer.
If the scores differ by less than their fold-to-fold noise, say the layers are equivalent for
this task and keep the default.
