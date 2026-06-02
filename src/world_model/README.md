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

In in-context mode only, a second action table can be configured for the
held-out query triplet:

```text
adata.obsm[query_action_embedding.obsm_key]  # query action embeddings
```

If `query_action_embedding.obsm_key` is empty, support and query actions use
the same table exactly as before.

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

## Cell-Eval Metrics

The world-model evaluation uses ArcInstitute `cell-eval` when
`eval.use_cell_eval=true`. The `gpu` and `dev` Pixi environments install it
through the `cell_eval` feature:

```bash
pixi install -e gpu
pixi run -e gpu python -c "import cell_eval; print(cell_eval.__file__)"
```

For cluster reports, prefer hard-failing if the package is missing:

```bash
pixi run -e gpu python -m world_model.scripts.eval \
  --config runs/path/to/config.yaml \
  --checkpoint runs/path/to/model_final.pt \
  eval.require_cell_eval=true \
  eval.cell_eval_profile=full \
  eval.cell_eval_num_threads=8
```

When `eval.require_cell_eval=false`, embpy keeps a small internal fallback so
smoke tests and laptops without `cell-eval` can still run. Fallback results are
not the full benchmark suite and should not be used for final comparisons.

The SLURM stability submitter runs full `cell-eval` on CPU by default. The GPU
training job sets `train.run_eval_after_fit=false`, releases the GPU after
training, and submits a dependent CPU job through
`scripts/slurm/eval_only_cpu.sbatch`. Override with `RUN_EVAL_ON_CPU=0` if you
want the historical inline post-training evaluation.

## Attach Embeddings

Attach action embeddings from a BioEmbedder model:

```bash
pixi run -e gpu python -m world_model.scripts.embed_perturbations \
  --dataset replogle \
  --h5ad data/crispr_datasets/replogle/replogle_2022_k562_essential.h5ad \
  --model esm2_650M \
  --output-h5ad runs/_cache/action_h5ad/replogle_esm2_650M.h5ad \
  --obsm-key X_pert_esm2_650M
```

Attach SubCell perturbation morphology embeddings through the public
`BioEmbedder.embed(..., entity_type="perturbation")` path:

```bash
pixi run -e gpu python -m world_model.scripts.embed_perturbations \
  --dataset replogle \
  --h5ad runs/_cache/action_h5ad/replogle_esm2_650M.h5ad \
  --model subcell_mae_rybg \
  --entity-type perturbation \
  --output-h5ad runs/_cache/action_h5ad/replogle_esm2_650M_subcell.h5ad \
  --obsm-key X_pert_subcell_mae_rybg \
  --morphology-dataset hpa \
  --max-images 5
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

Cross-modality in-context run, dry-run first:

```bash
DRYRUN=1 DATASET=both \
  bash src/world_model/world_model/scripts/submit/submit_cross_modality_incontext.sh
```

Submit the full Replogle + Nadig workflow:

```bash
DATASET=both \
  STACK_CHECKPOINT=/lustre/groups/ml01/workspace/goncalo.pinto/embpy/data/checkpoints/stack/bc_large.ckpt \
  STACK_GENELIST=/lustre/groups/ml01/workspace/goncalo.pinto/embpy/data/checkpoints/stack/basecount_1000per_15000max.pkl \
  bash src/world_model/world_model/scripts/submit/submit_cross_modality_incontext.sh
```

Single dataset:

```bash
DATASET=replogle bash src/world_model/world_model/scripts/submit/submit_cross_modality_incontext.sh
DATASET=nadig bash src/world_model/world_model/scripts/submit/submit_cross_modality_incontext.sh
```

Final AnnData inputs are written to:

```text
runs/_cache/cross_modality_incontext_h5ad/nadig_stack_esm2_650M_subcell_mae_rybg.h5ad
runs/_cache/cross_modality_incontext_h5ad/replogle_stack_esm2_650M_subcell_mae_rybg.h5ad
```

Training outputs and aggregate plots live under:

```text
runs/world_model/cross_modality_incontext/
```

The training overrides are:

```text
data.context_mode=incontext_set
dynamics.kind=incontext_tokens
action_embedding.obsm_key=X_pert_esm2_650M
query_action_embedding.obsm_key=X_pert_subcell_mae_rybg
```

In-context latent-stability experiment matrix:

```bash
DRYRUN=1 DATASET=nadig EMB=borzoi_v0 STATE_OBSM_KEY=X_stack \
  bash src/world_model/world_model/scripts/submit/submit_incontext_stability_experiments.sh
```

Submit after inspecting the dry run:

```bash
DATASET=nadig EMB=borzoi_v0 STATE_OBSM_KEY=X_stack \
  bash src/world_model/world_model/scripts/submit/submit_incontext_stability_experiments.sh
```

By default this submits a single standard setup named `standard`:
`dynamics.kind=incontext_tokens`, `dynamics.latent_normalization=layer_norm`,
`dynamics.prediction_mode=absolute`, and random same-bucket support sampling.
If the ready AnnData is not at
`runs/_cache/action_h5ad/<dataset>_<EMB>.h5ad`, pass `H5AD_PATH=/path/file.h5ad`
for one dataset or `H5AD_TEMPLATE='/path/{dataset}_file.h5ad'` for both.
Core switches are:

```text
dynamics.latent_normalization=none|layer_norm|l2
dynamics.prediction_mode=absolute|residual_delta
data.incontext_support_strategy=random|action_similarity
```

Generate STACK state embeddings and action perturbation embeddings first, then
write final world-model-ready AnnData files:

```bash
pixi install -e arc-gpu
pixi install -e gpu
pixi install -e caduceus
pixi install -e evo2

DATASET=both EMBEDDINGS=all \
  STACK_CHECKPOINT=/path/to/bc_large.ckpt \
  STACK_GENELIST=/path/to/basecount_1000per_15000max.pkl \
  bash src/world_model/world_model/scripts/submit/submit_world_model_ready_adatas.sh
```

This creates:

```text
runs/_cache/world_model_ready_h5ad/nadig_stack_all_gene_embeddings.h5ad
runs/_cache/world_model_ready_h5ad/replogle_stack_all_gene_embeddings.h5ad
```

Each output has cell/state embeddings in `.obsm["X_stack"]`, one action
perturbation matrix per selected embedding in `.obsm["X_pert_<embedding>"]`,
and the canonical unique-perturbation embpy payloads in
`.uns["embpy"]["perturbations"]`.
`EMBEDDINGS=all` expands to every usable catalog entry from
`world_model.configs.run_matrix default-embeddings --target slurm`; entries
marked `convert` are excluded. Use `DRYRUN=1` to print the `sbatch` commands
without submitting them. Set `RUN_TRAIN=1` if you also want the script to chain
`train_embedding.sbatch` after each ready AnnData is created.

To run a smaller subset, pass a space-separated list:

```bash
DATASET=both EMBEDDINGS="genept esm2_650M minilm_l6_v2" \
  STACK_CHECKPOINT=/path/to/bc_large.ckpt \
  STACK_GENELIST=/path/to/basecount_1000per_15000max.pkl \
  bash src/world_model/world_model/scripts/submit/submit_world_model_ready_adatas.sh
```

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
