"""Part 4 of docs/notebooks/cells.ipynb -- cell-eval on a perturbation dataset."""
from __future__ import annotations
import json, sys
from pathlib import Path

CELLS: list[tuple[str, str]] = []
def md(s: str) -> None: CELLS.append(("markdown", s.strip("\n")))
def code(s: str) -> None: CELLS.append(("code", s.strip("\n")))

md(r"""
## 2. cell-eval: is the perturbation preserved?

This section runs on a **different dataset**, and the reason is the mirror image
of the rule that governs [section 1](#1-scib-does-integration-preserve-the-biology).

scIB needs a technical covariate to remove and biological labels to keep, so it
needs an atlas. cell-eval scores agreement *per perturbation*, so it needs an
experiment with many perturbations -- and an atlas has none. Running cell-eval on
the atlas would produce numbers, and they would mean nothing.

[04_benchmark_models](04_benchmark_models.ipynb) shows the degenerate case
concretely. It used seven cell types as stand-in perturbations, and every
discrimination score came back exactly `1.000` for all five models:

| model | discrimination_score_l2 | discrimination_score_cosine |
| --- | --- | --- |
| pca | 1.000 | 1.0 |
| scvi | 1.000 | 1.0 |
| scgpt | 1.000 | 1.0 |
| geneformer_v2_12L | 1.000 | 1.0 |
| state | 1.000 | 1.0 |

A metric that returns the same perfect score for every model has ranked
nothing. Telling a monocyte from a T cell is trivial; the discrimination family
only becomes informative when the perturbations are numerous and some of them
are subtle.
""")

md(r"""
### The dataset

**Norman et al. 2019** -- CRISPRa in K562, single and combinatorial. 111,255
cells over 19,018 genes, with 237 perturbation levels of which 236 are real
perturbations and one is the control.

Three properties earn it this section:

* **Combinatorial perturbations.** Levels are single (`KLF1`) and pairwise
  (`CEBPE+RUNX1T1`). A pair whose two members overlap in effect is genuinely
  hard to distinguish from either alone, which is what gives the discrimination
  metrics something to fail at.
* **Every level has at least 50 cells.** So nothing is dropped for size, and a
  `DROPPED` list that comes back empty below is a real result rather than a
  filter that silently did nothing.
* **A named control.** The column carries a literal `"control"` level with
  11,835 cells, which cell-eval needs as its baseline.

One trap, and it is the same one the atlas loader guards against: **`.X` here is
log-normalised, not counts** (`X.max()` is 8.5, non-integral). The raw counts are
in `layers["counts"]`. Two of the four models in this section declare
`input_layer="counts"`, so taking `.X` at face value would fit a
negative-binomial likelihood to non-integers and never complain.
""")

code(r"""
PERT_KEY = "perturbation_name"     # 237 levels; see the note on the alternatives
CONTROL_VALUE = "control"          # 11,835 cells
N_PERT_CELLS = 4000                # subsample; the full set is 111,255 cells
MIN_CELLS_PER_PERT = 4             # cell-eval needs each level on both sides
PERT_PATH = DATA_DIR / "norman2019.h5ad"


def load_norman():
    # Two other columns look like candidates and both are unusable:
    # `guide_ids` has the same 237 levels but labels the control as the empty
    # string, and `guide_identity` has 290 levels of raw guide pairs
    # (NegCtrl10_NegCtrl0__NegCtrl10_NegCtrl0) with no clean control level.
    if PERT_PATH.exists():
        return sc.read_h5ad(PERT_PATH)
    import pertpy as pt  # noqa: PLC0415 - fallback only

    fetched = pt.data.norman_2019()
    fetched.write_h5ad(PERT_PATH)
    return fetched


pert_adata = load_norman()
print(f"loaded {pert_adata.n_obs} cells x {pert_adata.n_vars} genes")

# .X is log-normalised; the counts are in a layer. Take the counts, or the two
# scvi-tools models silently fit a count likelihood to non-integers.
assert COUNTS_LAYER in pert_adata.layers, f"expected raw counts in .layers[{COUNTS_LAYER!r}]"
pert_adata.X = pert_adata.layers[COUNTS_LAYER].copy()

# Drop the paper's own embeddings. Carrying them into a notebook that scores
# embeddings invites mistaking one for ours.
for key in list(pert_adata.obsm):
    del pert_adata.obsm[key]

X = pert_adata.X
print(f"counts integral after switching layers: "
      f"{float(np.abs(X.data - np.round(X.data)).max()) == 0.0}")
print(f"{PERT_KEY}: {pert_adata.obs[PERT_KEY].nunique()} levels")
print(f"control cells: {int((pert_adata.obs[PERT_KEY] == CONTROL_VALUE).sum())}")
""")

md(r"""
Subsampling has to happen **within** perturbation, not across the whole object.
A flat random subsample of 4,000 from 111,255 cells would drop most of the 237
levels entirely and leave the rest with a handful of cells each -- which would
turn the level count, the thing that makes this dataset worth using, into the
thing the subsample destroyed.
""")

code(r"""
rng = np.random.default_rng(SEED)

# Take an equal quota per level rather than a proportional sample, so the rare
# perturbations survive and the abundant ones stop dominating the averages.
levels = pert_adata.obs[PERT_KEY].astype(str)
per_level = max(MIN_CELLS_PER_PERT, N_PERT_CELLS // levels.nunique())

keep = []
for level in levels.unique():
    idx = np.flatnonzero((levels == level).to_numpy())
    if len(idx) > per_level:
        idx = rng.choice(idx, per_level, replace=False)
    keep.append(idx)
keep = np.sort(np.concatenate(keep))

pert_adata = pert_adata[keep].copy()
pert_adata.obs[PERT_KEY] = pert_adata.obs[PERT_KEY].astype(str).astype("category")
pert_adata.layers[COUNTS_LAYER] = pert_adata.X.copy()

counts_per_level = pert_adata.obs[PERT_KEY].value_counts()
print(f"{pert_adata.n_obs} cells, {counts_per_level.size} levels, "
      f"{per_level} cells per level target")
print(f"cells per level: min {counts_per_level.min()}, "
      f"median {int(counts_per_level.median())}, max {counts_per_level.max()}")
display(counts_per_level.head(8).to_frame("n_cells"))
""")

md(r"""
### A reduced roster, and why

Section 1 swept eight models. This section runs **four** -- `pca`, `scvi`,
`geneformer_v2_12L` and `state` -- and the reason is worth stating rather than
leaving a reader to wonder where the other four went.

The question here is what the *metric family* measures, not which of eight
models wins. Four models are enough to show a spread, and these four are the
ones already proven end to end, spanning a classical baseline, a VAE and two
foundation models. Embedding 237 perturbation levels eight times to make the
same point would cost hours and add nothing.
""")

code(r"""
PERT_ROSTER = ["pca", "scvi", "geneformer_v2_12L", "state"]

PERT_SWEEP: list[str] = []
PERT_FAILURES: dict[str, str] = {}

prepared_pert = pp.preprocess_counts(
    pert_adata, pipeline="standard", min_genes=0, min_cells=0,
    target_sum=1e4, log_transform=True, n_top_genes=2000, select_hvg=True,
    scale=False, copy=True,
)

for model_key in PERT_ROSTER:
    t0 = time.perf_counter()
    try:
        embedder.embed_cells(
            prepared_pert, models=[model_key], preprocessing="none", batch_size=8
        )
    except Exception as exc:  # noqa: BLE001 - a failed model is data, not a stop
        PERT_FAILURES[f"X_{model_key}"] = f"{type(exc).__name__}: {exc}"
        print(f"  {model_key:20} FAILED  {type(exc).__name__}")
    else:
        PERT_SWEEP.append(f"X_{model_key}")
        print(f"  {model_key:20} ok      {time.perf_counter() - t0:6.1f}s")
    embedder.clear_model_cache()

PERT_EMBEDDINGS = [k for k in PERT_SWEEP if k in prepared_pert.obsm]
print(f"\n{len(PERT_EMBEDDINGS)} of {len(PERT_ROSTER)} models embedded")
if PERT_FAILURES:
    display(pd.Series(PERT_FAILURES, name="error").to_frame())
""")

# ---------------------------------------------------------------- the pair
md(r"""
### Building a pair cell-eval will accept

cell-eval compares a **(predicted, real)** pair of AnnData objects. This notebook
has no prediction model, so the pair is two halves of the same dataset -- which
measures an *agreement ceiling*, not prediction accuracy. Say that plainly,
because the numbers look like accuracy and are not.

The construction has one hard requirement that is easy to violate: both objects
must carry the **same set of perturbation levels**, or cell-eval raises
`Perturbation mismatch`. A plain random half-split fails that as soon as any
level is small enough to land entirely on one side. So the split runs *within*
each level, and levels too small to appear on both sides are dropped with the
count shown.
""")

code(r"""
levels = prepared_pert.obs[PERT_KEY].astype(str)
level_counts = levels.value_counts()
USABLE = level_counts[level_counts >= MIN_CELLS_PER_PERT].index.tolist()
DROPPED = sorted(set(level_counts.index) - set(USABLE))

paired = prepared_pert[levels.isin(USABLE).to_numpy()].copy()
is_pred = np.zeros(paired.n_obs, dtype=bool)
for level in USABLE:
    idx = np.flatnonzero((paired.obs[PERT_KEY].astype(str) == level).to_numpy())
    rng.shuffle(idx)
    is_pred[idx[: len(idx) // 2]] = True

real = paired[~is_pred].copy()
pred = paired[is_pred].copy()

print(f"real {real.n_obs} cells | pred {pred.n_obs} cells")
print(f"levels on both sides: {len(USABLE)}")
print(f"dropped (fewer than {MIN_CELLS_PER_PERT} cells): {DROPPED or 'none'}")
assert set(real.obs[PERT_KEY].astype(str)) == set(pred.obs[PERT_KEY].astype(str)), \
    "cell-eval requires identical level sets on both sides"
assert CONTROL_VALUE in set(real.obs[PERT_KEY].astype(str)), "control level missing"
print("level sets match")
""")

md(r"""
### Pointing every metric at the embedding

This is the part that silently goes wrong, and it went wrong in
[04_benchmark_models](04_benchmark_models.ipynb) before it was caught.

cell-eval's metrics fall into families. The `MetricType.ANNDATA_PAIR` ones can
score *either* the expression matrix or an `.obsm` embedding, and which one they
use is decided per metric by an `embed_key` entry in `metric_configs`. A metric
you forget to configure does not error -- it scores `.X` instead, and its number
lands in the same table beside the ones that scored your embedding.

nb04 configured 4 of 10 pair metrics. Once the remaining six were pointed at the
embedding, `pearson_delta` moved from 0.777 to 0.854 -- so the original table was
a mixture of two different measurements presented as one.

The fix is to stop hand-listing metric names and **enumerate the registry**, then
print the count so a reader can see none were missed.

One metric resists this. `discrimination_score_l1` hardcodes `embed_key = None`
upstream, so it cannot be made to read `.obsm` no matter what you pass. It is
skipped explicitly rather than left in: a single expression-space column sitting
in an embedding-space table is precisely the mixed result this whole subsection
is about avoiding.
""")

code(r"""
from cell_eval import MetricType, metrics_registry

# Enumerate rather than hand-list: a metric added by a future cell-eval release
# is picked up automatically instead of silently scoring .X.
ALL_PAIR = set(metrics_registry.list_metrics(MetricType.ANNDATA_PAIR))

# discrimination_score_l1 hardcodes embed_key = None upstream, so it always
# scores .X. Skip it rather than let one expression-space column sit in an
# embedding-space table.
UNCONFIGURABLE = {"discrimination_score_l1"}
PAIR_METRICS = sorted(ALL_PAIR - UNCONFIGURABLE)

print(f"ANNDATA_PAIR metrics: {len(ALL_PAIR)}")
print(f"configurable on an embedding: {len(PAIR_METRICS)} -> {PAIR_METRICS}")
print(f"skipped (ignores embed_key): {sorted(UNCONFIGURABLE)}")
""")

code(r"""
frames = {}
for key in PERT_EMBEDDINGS:
    # Every pair metric gets the same embed_key. Configuring some and not
    # others is what produced nb04's mixed table.
    metric_configs = {name: {"embed_key": key} for name in PAIR_METRICS}
    real.obsm[key] = prepared_pert.obsm[key][~is_pred]
    pred.obsm[key] = prepared_pert.obsm[key][is_pred]

    t0 = time.perf_counter()
    try:
        per_pert, agg = tl.cell_eval(
            adata_pred=pred,
            adata_real=real,
            control_pert=CONTROL_VALUE,
            pert_col=PERT_KEY,
            profile="anndata",       # runs every ANNDATA_PAIR metric, which is
                                     # exactly the family that can see an .obsm
            skip_de=True,            # DE metrics score .X by construction; this
                                     # goes to the evaluator *constructor*
            metric_configs=metric_configs,
            skip_metrics=sorted(UNCONFIGURABLE),
        )
    except Exception as exc:  # noqa: BLE001
        print(f"  {key:22} FAILED  {type(exc).__name__}: {exc}")
        continue
    frames[key] = per_pert
    print(f"  {key:22} ok  {len(per_pert)} rows  {time.perf_counter() - t0:5.1f}s")

print(f"\nconfigured {len(PAIR_METRICS)} pair metrics per embedding, "
      f"{len(frames)} embeddings scored")
""")

md(r"""
`tl.cell_eval` returns a **tuple** of two frames -- per-perturbation and
aggregate -- not one. That matters because the pipeline-friendly variant behaves
differently in a way its own docstring gets wrong, which is worth one short
subsection.
""")

md(r"""
### What `run_cell_eval` does differently

`tl.run_cell_eval` is the pipeline form of the same call. Its summary line says
it returns *"a single DataFrame combining per-perturbation and aggregate
results"*. It does not. The body returns only the per-perturbation frame and
writes the aggregate into `adata_real.uns["cell_eval_agg"]` -- **mutating the
object you passed in as `real`**.

Neither half of that is a problem once you know it. Both halves are a problem if
you trust the summary line: you lose the aggregate, and an object you thought was
read-only has grown a key.
""")

code(r"""
if PERT_EMBEDDINGS:
    probe_key = PERT_EMBEDDINGS[0]
    real_probe = real.copy()
    before = set(real_probe.uns)
    tidy = tl.run_cell_eval(
        adata_pred=pred, adata_real=real_probe,
        control_pert=CONTROL_VALUE, pert_col=PERT_KEY, profile="anndata",
        skip_de=True,
        metric_configs={n: {"embed_key": probe_key} for n in PAIR_METRICS},
        skip_metrics=sorted(UNCONFIGURABLE),
    )
    print(f"run_cell_eval returned: {type(tidy).__name__} with {len(tidy)} rows")
    print(f"keys added to the object passed as `real`: "
          f"{sorted(set(real_probe.uns) - before)}")
    del real_probe
""")

# ---------------------------------------------------------------- results
md(r"""
### The result

One row per embedding, averaged over perturbations. The columns present depend
on the installed cell-eval version, so this takes whatever is numeric rather
than assuming a fixed set -- a hand-written column list is how a notebook breaks
on a dependency bump.
""")

code(r"""
if frames:
    summary = {}
    for key, frame in frames.items():
        numeric = frame.select_dtypes(include=[np.number])
        summary[key] = numeric.mean()
    CELL_EVAL = pd.DataFrame(summary).T
    CELL_EVAL.index.name = "embedding"
    display(CELL_EVAL.round(3))
else:
    CELL_EVAL = pd.DataFrame()
    print("no cell-eval results to show")
""")

md(r"""
Compare the discrimination columns against nb04's, reprinted at the top of this
section. There, with seven well-separated stand-in perturbations, every model
scored exactly 1.000. Here, with 236 real perturbations including combinatorial
ones, the same metric has room to separate them -- and if it still does not, that
is now a finding about the models rather than an artefact of the task being
trivial.
""")

code(r"""
if not CELL_EVAL.empty:
    # Discrimination scores and error metrics point in opposite directions, so
    # they get separate panels rather than a shared axis that would flatter one.
    higher_better = [c for c in CELL_EVAL.columns if "discrimination" in c
                     or c.startswith("pearson")]
    lower_better = [c for c in CELL_EVAL.columns
                    if c.startswith(("mse", "mae")) or "edistance" in c]

    panels = [(higher_better, "higher is better"), (lower_better, "lower is better")]
    panels = [(cols, title) for cols, title in panels if cols]
    fig, axes = plt.subplots(1, len(panels), figsize=(7 * len(panels), 3.6))
    axes = np.atleast_1d(axes)
    for ax, (cols, title) in zip(axes, panels):
        CELL_EVAL[cols].plot.bar(ax=ax, width=0.8)
        ax.set_title(title)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=20)
        ax.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.show()
""")

md(r"""
### Does it find the strong perturbations?

An aggregate over 236 perturbations can hide the thing that matters most: does
the metric respond to how large the perturbation actually is? A score that rates
a near-null perturbation as highly as a strong one is not measuring the
perturbation -- it is measuring something else and averaging it.

So rank the perturbations by a measured effect size, computed independently of
cell-eval, and check whether the per-perturbation scores track it.
""")

code(r"""
if frames:
    # Effect size computed independently of cell-eval: the L2 distance between
    # each perturbation's mean expression profile and the control's.
    ctrl_mask = (prepared_pert.obs[PERT_KEY].astype(str) == CONTROL_VALUE).to_numpy()
    log_layer = prepared_pert.layers.get("log_normalized", prepared_pert.X)
    log_layer = np.asarray(
        log_layer.todense() if hasattr(log_layer, "todense") else log_layer
    )
    ctrl_mean = log_layer[ctrl_mask].mean(axis=0)

    effect = {}
    for level in USABLE:
        if level == CONTROL_VALUE:
            continue
        m = (prepared_pert.obs[PERT_KEY].astype(str) == level).to_numpy()
        effect[level] = float(np.linalg.norm(log_layer[m].mean(axis=0) - ctrl_mean))
    EFFECT = pd.Series(effect, name="effect_l2").sort_values(ascending=False)

    print(f"effect size across {len(EFFECT)} perturbations: "
          f"{EFFECT.min():.2f} to {EFFECT.max():.2f}")
    display(pd.concat([EFFECT.head(5), EFFECT.tail(5)]).to_frame().round(3))
""")

code(r"""
if frames:
    # Correlate each metric against the independent effect size, per embedding.
    rows = []
    for key, frame in frames.items():
        idx_col = frame.columns[0] if frame.index.name is None else None
        f = frame.copy()
        if idx_col is not None and f[idx_col].dtype == object:
            f = f.set_index(idx_col)
        common = [i for i in f.index.astype(str) if i in EFFECT.index]
        if len(common) < 5:
            continue
        aligned = EFFECT.loc[common]
        for col in f.select_dtypes(include=[np.number]).columns:
            rows.append({
                "embedding": key,
                "metric": col,
                "spearman_vs_effect": float(
                    pd.Series(f.loc[common, col].values, index=common)
                    .corr(aligned, method="spearman")
                ),
            })

    if rows:
        TRACKS_EFFECT = (
            pd.DataFrame(rows)
            .pivot(index="metric", columns="embedding", values="spearman_vs_effect")
        )
        display(TRACKS_EFFECT.round(3))
        print("A metric near zero here does not respond to perturbation strength.")
    else:
        print("not enough shared perturbation levels to correlate")
""")

md(r"""
### Regression is a third question, and a third function

Two entry points have now been used, and there is a third that answers something
different again. `tl.benchmark_embeddings` trains ordinary regressors --
linear, ridge, k-NN, random forest -- on the embedding and reports `mse`, `r2`,
`pearson`, `spearman`.

It contains **no scIB path** and no cell-eval path: `grep -in scib
src/embpy/tl/benchmark.py` returns nothing. So the package offers three separate
front doors and no single "benchmark this embedding" call, which is worth
knowing before you go looking for one.

| Function | Question | Needs |
| --- | --- | --- |
| `tl.compute_scib_metrics` | does integration keep the biology? | labels + a batch covariate |
| `tl.cell_eval` | do two objects agree per perturbation? | a (pred, real) pair |
| `tl.benchmark_embeddings` | how much signal about a target does this carry? | a prediction target |
""")

Path(sys.argv[1] if len(sys.argv) > 1 else "cells_part4.json").write_text(json.dumps(CELLS))
print(f"part 4: {len(CELLS)} cells")
