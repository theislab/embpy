# World Model

The world-model training path is intentionally small now:

1. Start from one perturbation AnnData file.
2. Make sure it already has state embeddings in `.obsm`.
3. Attach one action embedding matrix to `.obsm`.
4. Train/evaluate within that same AnnData.

There is no cross-dataset transfer, zero-shot, fine-tune, LOEO, or ablation
entry point in this tree. Multiple embeddings are handled by launching the
same within-dataset run once per embedding.

## AnnData Contract

Training consumes only pre-attached AnnData fields:

```text
adata.obs[data.perturbation_key]       # per-cell action / perturbation label
data.control_label                     # exact obs label used for controls
adata.obsm[data.state_obsm_key]        # per-cell state embeddings, default X_state
adata.obsm[action_embedding.obsm_key]  # per-cell action embeddings, e.g. X_pert_esm2_650M
```

The dataloader does not derive model inputs from `adata.X`, external NPZ/CSV
tables, or online BioEmbedder calls. CSV/NPZ tables and BioEmbedder models are
offline sources used by `world_model.scripts.embed_perturbations` to write an
AnnData copy with the requested action `.obsm` key.

In embpy itself, perturbation/action embeddings can live canonically in
`adata.uns["embpy"]["perturbations"]` as one row per unique perturbation. Use
`embpy.io.materialize_perturbation_obsm(...)` to expand that table into
`.obsm` before training; the world model keeps `.obsm` as the strict tensor
input so batch construction stays simple and row-aligned.

`data.dataset` is only a human-readable label for run names and split-cache
files. It is not a Python loader switch; Nadig, Replogle, or any compatible
perturbation AnnData use the same generic loader.

## Configs

Canonical YAML files:

```text
src/world_model/world_model/configs/bases/single.yaml
src/world_model/world_model/configs/datasets/nadig.yaml
src/world_model/world_model/configs/datasets/replogle.yaml
src/world_model/world_model/configs/experiments/smoke.yaml
```

Dataset configs are convenience examples that inherit from `bases/single.yaml`.
Per-embedding runs should use CLI overrides rather than separate YAML files:

```bash
pixi run -e gpu python -m world_model.scripts.train \
  --config src/world_model/world_model/configs/datasets/replogle.yaml \
  data.h5ad_path=runs/_cache/action_h5ad/replogle_esm2_650M.h5ad \
  data.state_obsm_key=X_state \
  action_embedding.source=anndata_obsm \
  action_embedding.obsm_key=X_pert_esm2_650M \
  action_embedding.model_name=esm2_650M
```

## Attach Embeddings

Attach action embeddings from a BioEmbedder model:

```bash
pixi run -e gpu python -m world_model.scripts.embed_perturbations \
  --dataset replogle \
  --h5ad data/crispr_datasets/replogle/replogle_2022_k562_essential.h5ad \
  --model esm2_650M \
  --output runs/_cache/action_embeddings/replogle_esm2_650M.npz \
  --output-h5ad runs/_cache/action_h5ad/replogle_esm2_650M.h5ad \
  --obsm-key X_pert_esm2_650M
```

Attach action embeddings from a symbol-indexed CSV/NPZ table:

```bash
pixi run -e gpu python -m world_model.scripts.embed_perturbations \
  --dataset replogle \
  --h5ad data/crispr_datasets/replogle/replogle_2022_k562_essential.h5ad \
  --table data/embeddings/gene_embeddings/genept/embeddings_3072.csv \
  --output runs/_cache/action_embeddings/replogle_genept.npz \
  --output-h5ad runs/_cache/action_h5ad/replogle_genept.h5ad \
  --obsm-key X_pert_genept
```

Attach state embeddings with `encode_cells` or any external workflow that writes
an `(n_obs, d)` matrix to `adata.obsm["X_state"]`.

## Cluster Commands

Single dataset, one embedding:

```bash
DATASET=replogle EMB=esm2_650M STATE_OBSM_KEY=X_state \
  bash src/world_model/world_model/scripts/submit/submit_gene_embeddings.sh
```

All usable embeddings for one dataset:

```bash
DATASET=replogle STATE_OBSM_KEY=X_state \
  bash src/world_model/world_model/scripts/submit/submit_embedding_sweep.sh
```

Both datasets independently:

```bash
DATASET=both STATE_OBSM_KEY=X_state \
  bash src/world_model/world_model/scripts/submit/submit_embedding_sweep.sh
```

The submitter creates action-attached AnnData copies under
`runs/_cache/action_h5ad/`, trains with `train_embedding.sbatch`, runs baselines,
then writes comparison/report artifacts. The sweep submitter only loops the
single-embedding submitter and aggregates finished `comparison.csv` files.

## Local MPS Sweep

```bash
DATASETS="nadig replogle" EMBEDDINGS="esm2_650M minilm_l6_v2" \
  pixi run -e mps wm-sweep-mps
```

This is the same within-dataset attach-then-train flow, run sequentially on
Apple Silicon.

## Useful CLIs

```bash
python -m world_model.configs.run_matrix catalog --target slurm
python -m world_model.configs.run_matrix default-embeddings --target slurm
python -m world_model.scripts.sweeps.aggregate_sweep --datasets replogle nadig
python -m world_model.scripts.smoke_test
```
