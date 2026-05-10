# Perturbation World Model for Transcriptomics

A modular implementation of an autoregressive (state, action) world
model for single-cell transcriptomics. The model takes a sequence of
gene-expression observations and a sequence of genetic perturbations
and predicts the next-state observation autoregressively.

It adapts the world-model design proposed in
[Learning World Models for Unconstrained Goal Navigation](https://arxiv.org/pdf/2405.18193)
to perturb-seq data, and follows the modular implementation patterns
of [In-Context-Symmetries](https://github.com/Sharut/In-Context-Symmetries).

## Architecture

![Perturbation world model architecture](assets/architecture.png)

Pipeline overview:

1. **State stack encoder.** Each step `t` stacks `K` past
   gene-expression vectors `x in R^G`. A linear projection lifts each
   to `R^d`, then a small transformer with a CLS token aggregates the
   stack into a single state token `s_t in R^d`.
2. **Gene-embedding action encoder.** The perturbed gene's symbol is
   looked up in a pretrained gene embedding table (e.g. GenePT,
   gene2vec). Multi-gene perturbations are mean-pooled. A linear
   projection brings the embedding to `R^d`, producing `a_t in R^d`.
3. **GPT autoregressive dynamics.** State and action tokens are
   interleaved (`s_0, a_0, s_1, a_1, ...`) and fed through a causal
   transformer. The model reads next-state predictions
   `s_hat_{t+1}` off the action-token output positions.
4. **Decoder + loss.** An MLP decoder projects state tokens back to
   gene space (`R^d -> R^G`). The training objective combines
   next-state latent MSE, optional decoder MSE, and an optional
   InfoNCE term.

## Why this design?

### Mapping the reference paper to transcriptomics

| Paper component                | Transcriptomics counterpart                 |
| ------------------------------ | ------------------------------------------- |
| Stack of K image frames        | Stack of K gene-expression vectors          |
| ResNet trunk + flat token      | Per-frame Linear + Stack Transformer + CLS  |
| Discrete agent action          | Gene-embedding of the perturbed gene(s)     |
| GRU / world-model dynamics     | Causal GPT over `(s, a)` tokens             |
| Pixel reconstruction loss      | Gene-expression reconstruction (MSE)        |
| Latent consistency loss        | Latent next-state MSE (`s_hat` vs `s_next`) |
| Goal-conditioning              | Future work: condition on target cell-state |

### State-stack encoder

scRNA-seq observations are extremely noisy and zero-inflated. Stacking
`K` cells from the same condition averages out per-cell sampling noise
the same way frame stacking averages out per-frame motion blur in
Atari. Two encoder variants are provided:

* `TransformerStateStackEncoder` (default): per-frame linear projection
  followed by a small transformer encoder + CLS pooling. Closest
  analogue of the paper's ResNet trunk.
* `MLPStateStackEncoder`: cheap baseline that flattens `(K * G)` and
  runs an MLP. Useful as an ablation.

### Action encoding

The action representation is the embedding of the perturbed gene(s),
sharing the same vector space as any pretrained gene embedding the
user wants to drop in (GenePT, gene2vec, scGPT gene-token weights,
etc.). Index 0 is reserved for the control / non-targeting / padding
token, which lets the model learn an "identity action" as well.
Multi-gene perturbations are mean-pooled by default.

### Training objectives

The total loss is a weighted sum of:

* `latent_mse` -- MSE between predicted next-state token `s_hat` and
  the encoder's output on the true next stack `s_next` (with stop-grad
  on the target, BYOL-style). This is the primary signal.
* `decoder_mse` -- MSE between the decoded gene-expression
  reconstruction and the true mean expression vector. Anchors the
  state space to gene space.
* `info_nce` (optional) -- contrastive alignment between predicted and
  true tokens within a batch. Helpful when paired data is sparse.

### Key shapes (defaults)

```
B = batch size                      = 64
T = sequence length                 = 8
K = stack size                      = 4
G = number of HVGs                  = 5000
d = model token width               = 256
E_g = pretrained gene embed dim     = 3072 (GenePT)

obs_stack         : (B, T, K, G)
action_indices    : (B, T, n_pert)        (long, in [0, n_genes_pert])
state tokens      : (B, T, d)
action tokens     : (B, T, d)
interleaved seq   : (B, 2T, d)
s_hat             : (B, T, d)
x_hat             : (B, T, G)
```

## File tree

```
src/embpy/world_model/
  __init__.py                              top-level package
  README.md                                this file
  assets/
    architecture.png                       the diagram embedded above
  configs/
    __init__.py
    base.py                                dataclass schema + YAML loader + CLI overrides
    nadig.yaml                             legacy single-config (kept for back-compat)
    replogle.yaml                          legacy single-config
    experiments/
      single_nadig.yaml                    Setup 1: 80/20 on Nadig
      single_replogle.yaml                 Setup 2: 80/20 on Replogle
      transfer.yaml                        Setup 3: pretrain Nadig -> fine-tune p% Replogle
      smoke.yaml                           Tiny CPU smoke-test config
  data/
    __init__.py
    dataloader.py                          build_dataloaders(...) -> DataArtifacts
    preprocessing.py                       HVG, log1p, collate
    splits.py                              perturbation- and cell-aware splits (NPZ cached)
    datasets/
      __init__.py
      base.py                              PerturbationSequenceDataset (now with .subset())
      nadig.py                             Nadig adapter
      replogle.py                          Replogle adapter
  models/
    __init__.py
    blocks.py                              MLP, attention, transformer block
    world_model.py                         composed WorldModel + factory
    encoders/state_stack_encoder.py        transformer + MLP variants
    action/gene_embedding_action.py        pretrained-embedding action encoder
    dynamics/gpt_autoregressive.py         Decision-Transformer-style dynamics
    decoders/expression_decoder.py         d -> G MLP decoder
  training/
    __init__.py
    losses.py                              latent / delta / Gaussian / InfoNCE
    schedulers.py                          cosine + linear warmup
    hooks.py                               Hook ABC + Console / CSV / TB / Plot / Checkpoint
    trainer.py                             WorldModelTrainer (hook-based)
  evaluation/
    __init__.py
    metrics.py                             L2, cosine, R^2, delta-Pearson (latent-space)
    rollouts.py                            imagined_rollout(...)
    perturbation_eval.py                   end-to-end harness (model + every baseline)
    cell_eval_runner.py                    cell_eval wrapper + internal fallback metrics
    prep.py                                AnnData prep for cell_eval
    plots.py                               PNG + SVG plotters used by train.py
    report.py                              report.md generator
    baselines/
      __init__.py                          ALL_BASELINES registry
      base.py                              Baseline ABC + BaselineTrainData
      identity.py                          IdentityBaseline (sanity floor)
      control_mean.py                      ControlMeanBaseline
      mean.py                              MeanBaseline (global perturbed mean)
      additive.py                          AdditiveBaseline (control + delta_p)
      linear.py                            LinearRegressionBaseline (Ridge)
  utils/
    checkpoint.py / logging.py / seeding.py
  scripts/
    __init__.py
    train.py                               single + transfer; full eval at the end
    eval.py                                eval_only entry point
    run_baselines.py                       fit + evaluate all baselines on the same split
    smoke_test.py                          end-to-end CPU smoke test
    submit_all.sh                          submit every SLURM setup in one command
    slurm/
      train_single_nadig.sbatch
      train_single_replogle.sbatch
      train_transfer.sbatch                parameterised over FRACTION
      run_baselines.sbatch
      eval_only.sbatch
  tests/
    __init__.py
    test_state_stack_encoder.py
    test_gene_embedding_action.py
    test_gpt_autoregressive.py
    test_world_model.py
    test_losses.py
    test_dataset.py
    test_baselines.py                      every baseline on synthetic data
    test_splits.py                         determinism + roundtrip
    test_hooks.py                          hook lifecycle + CSV writer
```

## Module-by-module

### `configs`

Plain dataclasses (`WorldModelConfig`, `DataConfig`, `EncoderConfig`,
`DynamicsConfig`, `LossConfig`, `OptimConfig`, `TrainConfig`) loaded
from YAML via `load_yaml_config(path)`.

**Why dataclasses, not Hydra?** Zero new dependencies, IDE-friendly
type checking, and trivial YAML interop. Hydra is overkill for a
single training script and complicates config-driven imports inside
notebooks.

### `data`

* `data.preprocessing` -- log-normalisation, dependency-free HVG
  selection, and a custom collate fn that stacks dataset samples into
  the world-model batch dict.
* `data.datasets.base` -- `GeneIndexer` (gene symbol <-> int row),
  `load_gene_embedding_table` (CSV / NPZ), and the core
  `PerturbationSequenceDataset` that emits `(T, K, G)` sequences.
* `data.datasets.nadig` and `data.datasets.replogle` -- thin
  `from_h5ad(...)` adapters that load the AnnData, run preprocessing,
  and return `(dataset, gene_table, indexer, gene_symbols)`.
* `data.dataloader.build_dataloaders(cfg)` -- one-call setup that
  takes a `DataConfig` and returns `train_loader`, `val_loader`, and
  the artefacts needed to instantiate the model.

### `models`

* `blocks` -- shared building blocks (`MLP`, `CausalSelfAttention`,
  `TransformerBlock`).
* `encoders.state_stack_encoder` -- `TransformerStateStackEncoder`
  (default) and `MLPStateStackEncoder`. Both subclass
  `StateStackEncoder` and accept `(B, T, K, G)`, return `(B, T, d)`.
* `action.gene_embedding_action` -- `GeneEmbeddingAction` looks up
  perturbed-gene rows in a pretrained embedding table, mean-pools
  multi-gene perturbations and projects to `d_model`.
* `dynamics.gpt_autoregressive` -- `GPTAutoregressiveDynamics`,
  Decision-Transformer-style causal transformer over interleaved
  `(s, a)` tokens. The next-state prediction is read off the
  action-token output positions.
* `decoders.expression_decoder` -- MLP `d_model -> n_genes`. Optional
  log-variance head for Gaussian NLL training.
* `world_model.WorldModel` -- composes the four pieces and exposes
  `encode`, `encode_action`, `predict_next`, `decode`, `forward`,
  `loss` and `rollout`. A `build_world_model(...)` factory wires the
  defaults.

### `training`

* `losses` -- `latent_mse`, `delta_mse`, `gaussian_nll`, `info_nce`.
* `schedulers.build_scheduler` -- cosine + warmup, linear warmup,
  constant.
* `trainer.WorldModelTrainer` -- minimal training loop with AdamW,
  AMP on CUDA, gradient clipping and per-epoch checkpoints.

### `evaluation`

* `metrics` -- `latent_l2_error`, `cosine_similarity`,
  `expression_r2`, `delta_pearson` (the standard headline metric in
  the perturbation-prediction literature).
* `rollouts.imagined_rollout` -- runs autoregressive rollouts on a
  loader and aggregates metrics.

### `utils`

* `seeding.seed_everything(seed)` -- python / numpy / torch seeds.
* `logging.setup_logging` + `get_logger`.
* `checkpoint.save_checkpoint` / `load_checkpoint`.

### `scripts`

* `train.py` and `eval.py` -- thin `python -m` entry-points wrapping
  the pieces above.

## Install

```bash
# inside the embpy repo
pip install -e ".[torch]"

# optional: anndata / scanpy for real data loading
pip install anndata pyyaml
```

## Prepare data

This project expects the two datasets to live under `data/` at the
repo root:

```
data/
  datasets/
    nadig/NadigOConner2024_jurkat.h5ad
    nadig/NadigOConner2024_hepg2.h5ad
    replogle/replogle_2022_k562_essential.h5ad
    replogle/replogle_2022_k562_gwps.h5ad
    replogle/replogle_2022_rpe1.h5ad
  embeddings/
    gene_embeddings/
      genept/embeddings_3072.csv      first column = gene symbol
```

The defaults in `configs/replogle.yaml` and `configs/nadig.yaml`
already point at these locations. Edit them to retarget.

## Train

The package now ships **three training setups**, all driven by the same
`scripts/train.py` entry point. The YAML config alone selects which one
runs (no special branches in the code):

| Setup                       | Config                                                | Mode flag      |
| --------------------------- | ----------------------------------------------------- | -------------- |
| `single_nadig`              | `configs/experiments/single_nadig.yaml`               | `mode: single` |
| `single_replogle`           | `configs/experiments/single_replogle.yaml`            | `mode: single` |
| `transfer_nadig_to_replogle`| `configs/experiments/transfer.yaml`                   | `mode: transfer` |

### Locally

```bash
# single-dataset (Nadig)
pixi run -e gpu python -m embpy.world_model.scripts.train \
    --config src/embpy/world_model/configs/experiments/single_nadig.yaml

# single-dataset (Replogle)
pixi run -e gpu python -m embpy.world_model.scripts.train \
    --config src/embpy/world_model/configs/experiments/single_replogle.yaml

# transfer: pretrain on Nadig, fine-tune on 10% of Replogle
pixi run -e gpu python -m embpy.world_model.scripts.train \
    --config src/embpy/world_model/configs/experiments/transfer.yaml \
    transfer.finetune_fraction=0.10
```

Dotted CLI overrides (`encoder.d_model=512`, `train.n_epochs=20`,
`split.split_by=cell`, etc.) are accepted as positional arguments after
`--config`; unknown keys raise immediately so typos surface.

### On SLURM (pixi gpu env)

The launchers under `world_model/scripts/slurm/` activate the existing
pixi `gpu` env exactly like `submission_scripts/jupyter_pixi.sbatch`:

```bash
# single setups
sbatch src/embpy/world_model/scripts/slurm/train_single_nadig.sbatch
sbatch src/embpy/world_model/scripts/slurm/train_single_replogle.sbatch

# transfer at p in {1, 5, 10, 25, 50}%
for p in 0.01 0.05 0.10 0.25 0.50; do
    FRACTION=$p sbatch src/embpy/world_model/scripts/slurm/train_transfer.sbatch
done

# everything in one shot: train (3 setups) -> baselines -> compare/report
# chained via `sbatch --dependency=afterok:...`.
bash src/embpy/world_model/scripts/submit_all.sh
```

`submit_all.sh` submits, for each of the three setups:

```
train_<setup>  -->  run_baselines  -->  compare + make_report
```

so when the chain finishes, every `outputs/<run_id>/` ends up with a
`comparison.csv`, `comparison.png`, and `report.md` automatically.
Skip a setup with `SKIP_NADIG=1`, `SKIP_REPLOGLE=1`, `SKIP_TRANSFER=1`.

Defaults: `--gres=gpu:1`, `--cpus-per-task=8`, `--mem=64G`,
`--time=24:00:00`. The partition / qos lines are commented; uncomment
and / or set via env (`PARTITION=gpu_p QOS=gpu_normal sbatch ...`)
to match your cluster.

Per-script summary:

| Script                          | Purpose                                                              |
| ------------------------------- | -------------------------------------------------------------------- |
| `train_single_nadig.sbatch`     | Setup 1: train on Nadig only.                                        |
| `train_single_replogle.sbatch`  | Setup 2: train on Replogle only.                                     |
| `train_transfer.sbatch`         | Setup 3: pretrain on Nadig, fine-tune on `FRACTION` of Replogle.     |
| `run_baselines.sbatch`          | Fit + evaluate every baseline against the saved split.               |
| `eval_only.sbatch`              | Re-evaluate a finished checkpoint without retraining.                |
| `compare.sbatch`                | Build `comparison.csv` + `comparison.png` + `report.md`.             |

### Train/test split policy

The default and recommended split is **by perturbation identity**
(`split.split_by: perturbation`): test perturbations are *unseen* by
the model, which is the scientifically meaningful generalisation
setting. Set `split.split_by: cell` for a per-cell sanity check (any
sufficiently expressive model trivially memorises this -- never
report headline numbers from cell-level splits).

Splits are computed once and persisted to
`outputs/<run_id>/splits/<dataset>.npz`. Every subsequent run
(world model, baselines, eval-only) reuses the same file, so model and
baselines are compared on byte-identical train/test indices.

## Evaluate

```bash
# evaluate a checkpoint (model + baselines + plots + report)
pixi run -e gpu python -m embpy.world_model.scripts.eval \
    --config src/embpy/world_model/configs/experiments/single_replogle.yaml \
    --checkpoint outputs/world_model/single_replogle/single_replogle_final.pt

# or via SLURM
CONFIG=src/embpy/world_model/configs/experiments/single_replogle.yaml \
CKPT=outputs/world_model/single_replogle/single_replogle_final.pt \
    sbatch src/embpy/world_model/scripts/slurm/eval_only.sbatch
```

The full pipeline runs:

1. cell-eval-style metrics (MSE, MAE, R^2, Pearson, Spearman, DEG
   overlap@K) per perturbation and aggregated.
2. Plots: loss curves, predicted-vs-real scatter, per-perturbation R^2
   violin, DEG overlap bar, baseline-vs-model comparison. Saved as
   PNG + SVG under `outputs/<run_id>/plots/`.
3. `outputs/<run_id>/report.md` -- self-contained markdown summary
   with config dump, metric tables, and embedded plots.

## Baselines

Five small baselines under `evaluation/baselines/` share a
`Baseline` interface (`fit`, `predict`, `name`):

| Baseline           | Predicts                                          | Beats identity when                                              |
| ------------------ | ------------------------------------------------- | ---------------------------------------------------------------- |
| `IdentityBaseline` | `control_template`                                | never (sanity floor)                                             |
| `ControlMeanBaseline` | train-control mean                              | never (no perturbation signal)                                   |
| `MeanBaseline`     | mean of all train-perturbed cells                  | global response direction is informative                          |
| `AdditiveBaseline` | `control + delta(p)` if `p` seen in train          | only for cell-level splits or when p was observed                |
| `LinearRegressionBaseline` | Ridge: action embedding -> per-gene delta  | gene embedding carries the perturbation signal                   |

Run them against the same split as a world-model run:

```bash
pixi run -e gpu python -m embpy.world_model.scripts.run_baselines \
    --config src/embpy/world_model/configs/experiments/single_replogle.yaml \
    --checkpoint outputs/world_model/single_replogle/single_replogle_final.pt

# or via SLURM
CONFIG=src/embpy/world_model/configs/experiments/single_replogle.yaml \
CKPT=outputs/world_model/single_replogle/single_replogle_final.pt \
    sbatch src/embpy/world_model/scripts/slurm/run_baselines.sbatch
```

Outputs:

* `outputs/<run_id>/baselines.csv`  -- one row per (baseline, metric).
* `outputs/<run_id>/comparison.csv` -- wide format, world model + every baseline.
* `outputs/<run_id>/plots/comparison.png` -- bar chart per metric.

## Comparison and final report (standalone)

After `train.py` and `run_baselines.py` have produced their CSVs the
final comparison + report can be regenerated independently:

```bash
# build comparison.csv + comparison.png
pixi run -e gpu python -m embpy.world_model.scripts.compare \
    --run-dir outputs/world_model/single_replogle

# render report.md from whatever already exists in run-dir
pixi run -e gpu python -m embpy.world_model.scripts.make_report \
    --run-dir outputs/world_model/single_replogle
```

These two scripts read only files on disk (`config.yaml`,
`baselines.csv`, `world_model_metrics.csv`, `comparison.csv`,
`plots/*.png`, `eval/per_pert_*.csv`); they require no model state, so
they are safely re-runnable.

### Adding a new baseline

1. Drop a file `evaluation/baselines/my_baseline.py` defining a
   subclass of `Baseline`. Implement `fit(data)` and
   `predict(perturbations, control_template)`.
2. Register it in `evaluation/baselines/__init__.py` by adding it to
   the `ALL_BASELINES` registry. That's it -- it now participates in
   `run_baselines.py` and the plotting / report pipeline.

### Adding a new metric

Two layers exist:

1. Reusable numpy helpers in `evaluation/metrics.py`
   (`mse`, `mae`, `r2_score`, `pearson_corr`, `spearman_corr`,
   `deg_overlap_top_k`). Any new helper added there should also be
   re-exported from `evaluation/__init__.py`.
2. The metric *set* used by the comparison pipeline lives in
   `evaluation/cell_eval_runner.py::_internal_metrics`. Add a column
   there to make the new metric flow through `comparison.csv`,
   `comparison.png`, and `report.md` automatically. When `cell_eval`
   is installed and `eval.use_cell_eval=true`, metrics come from
   cell-eval directly -- add it upstream there for a real STATE
   evaluation.

Minimal template:

```python
# evaluation/metrics.py
def my_metric(real: np.ndarray, pred: np.ndarray) -> float:
    """One-line description."""
    return float(...)

# evaluation/cell_eval_runner.py::_internal_metrics
rows.append({
    ...,
    "my_metric": my_metric(real_mean, pred_mean),
    ...,
})
```

## How do I test that it runs?

A complete end-to-end smoke test that finishes in well under five
minutes on a laptop without GPU:

```bash
pixi run -e gpu python -m embpy.world_model.scripts.smoke_test
```

This runs the same code path as a real run on a tiny Nadig subset
(256 HVGs, 2 epochs, 64 sequences/epoch, batch size 8, all baselines,
the full eval pipeline) and asserts that all expected output files
(`train_log.csv`, `report.md`, `comparison.csv`, `plots/loss_curves.png`,
`plots/comparison.png`) exist before exiting.

## Output layout

Every run / baseline pass / comparison job writes into the same
`outputs/<run_id>/` directory so chaining and re-runs stay coherent:

```
outputs/<run_id>/
  config.yaml                        resolved config (used by make_report.py)
  train.log / baselines.log / ...    one log file per script
  splits/<dataset>.npz               deterministic train/test indices
  ckpt_epoch<NNN>.pt                 periodic checkpoints
  <run_name>_final.pt                end-of-training checkpoint
  train_log.csv                      per-epoch metrics (CSV; mirror of TB)
  tb/                                TensorBoard event files
  eval/
    real.h5ad                        test cells (true expression)
    pred_<name>.h5ad                 model / baseline predictions
    per_pert_<name>.csv              per-perturbation metric tables
  baselines.csv                      long format: (baseline, metric, value)
  world_model_metrics.csv            single-row world-model summary
  comparison.csv                     wide format: world model + every baseline
  plots/
    loss_curves.png/.svg
    scatter_world_model.png/.svg
    perpert_r2.png/.svg
    deg_overlap.png/.svg
    comparison.png/.svg
  report.md                          single self-contained markdown summary
```

## How to read the plots and report

* `plots/loss_curves.png` -- per-epoch train/val loss. Both should
  decrease together; a growing gap indicates overfitting (drop
  `train.n_epochs` or raise `optim.weight_decay` / `dynamics.dropout`).
* `plots/scatter_world_model.png` -- predicted vs real *mean*
  expression per perturbation. Points should cluster around the
  identity line; systematic bias appears as a slope.
* `plots/perpert_r2.png` -- distribution of R^2 across test
  perturbations. A wide tail of negative values means the model is
  worse than control on those perturbations.
* `plots/deg_overlap.png` -- top-K differentially expressed gene
  overlap per perturbation. A robust target is ~0.4-0.6 for K=50;
  random baselines hover around K/G.
* `plots/comparison.png` -- world model vs every baseline on each
  aggregated metric. The world model should beat every baseline on
  at least the headline ones (R^2, delta-cosine, DEG overlap).
* `report.md` -- one self-contained markdown file embedding the
  config, the metric tables, and links to all plots.

## Training diagnostics

The trainer routes everything through pluggable hooks
(`training/hooks.py`):

* per-step train loss, learning rate, gradient norm, every loss
  component -> stdout (every `train.log_every_n_steps`),
* per-epoch train/val loss + components -> CSV at
  `outputs/<run_id>/train_log.csv` and TensorBoard at
  `outputs/<run_id>/tb/`,
* loss curves + report rendered automatically at end of training.

Drop hooks by passing `hooks=[]` to `WorldModelTrainer.__init__`; add
custom ones by subclassing `Hook`.

## Tests

```bash
pytest src/embpy/world_model/tests
```

Each test file targets a single component on tiny synthetic inputs, so
the suite finishes in seconds on CPU.

## References

* Hu et al. *Learning World Models for Unconstrained Goal Navigation.*
  arXiv 2024. <https://arxiv.org/pdf/2405.18193>
* Garg et al. *In-Context-Symmetries.* GitHub.
  <https://github.com/Sharut/In-Context-Symmetries>
* Replogle et al. *Mapping information-rich genotype-phenotype
  landscapes with genome-scale Perturb-seq.* Cell 2022.
* Nadig & O'Connor et al. *Transcriptome-wide characterization of
  genetic perturbations.* 2024.
* Cui et al. *GenePT: A simple but hard-to-beat foundation model for
  genes and cells built from ChatGPT.* 2023.

## Debug checklist (component-by-component)

Every component can be exercised in isolation. Commands assume the
repo root and `pixi run -e gpu` for the GPU env.

### Data

```bash
pixi run -e gpu python -m pytest -x src/embpy/world_model/tests/test_dataset.py
pixi run -e gpu python -m pytest -x src/embpy/world_model/tests/test_splits.py
```

Expected: every test passes. Sanity-check a real dataset interactively:

```python
from embpy.world_model.configs import DataConfig, SplitConfig
from embpy.world_model.data import build_dataloaders
art = build_dataloaders(DataConfig(...), split_cfg=SplitConfig())
print(art.split.summary())
print(art.train_dataset[0]["obs_stack"].shape)  # (T, K, G)
```

### Encoder

```bash
pixi run -e gpu python -m pytest -x src/embpy/world_model/tests/test_state_stack_encoder.py
```

Expected output shape: `(B, T, d_model)`. Encoder collapse check:
`encoder(batch["obs_stack"]).std() ~ 1.0` after a short training run.

### Action encoder

```bash
pixi run -e gpu python -m pytest -x src/embpy/world_model/tests/test_gene_embedding_action.py
```

Confirm two different perturbations produce *distinct* tokens (cosine
similarity below 1).

### Dynamics

```bash
pixi run -e gpu python -m pytest -x src/embpy/world_model/tests/test_gpt_autoregressive.py
```

`test_dynamics_is_causal` enforces the causal mask: the dynamics
output at position `t` must not depend on tokens at `t' > t`.

### Loss

```bash
pixi run -e gpu python -m pytest -x src/embpy/world_model/tests/test_losses.py
pixi run -e gpu python -m pytest -x src/embpy/world_model/tests/test_world_model.py
```

`test_world_model.py` exercises the composite loss: `loss.backward()`
must succeed end-to-end.

### Trainer hooks

```bash
pixi run -e gpu python -m pytest -x src/embpy/world_model/tests/test_hooks.py
```

Checks: lifecycle order (start -> epoch loop -> end), CSV header +
rows, console logger does not crash without TensorBoard.

### Baselines

```bash
pixi run -e gpu python -m pytest -x src/embpy/world_model/tests/test_baselines.py
```

Each baseline is run on a tiny synthetic dataset where the perturbation
effect is a deterministic shift; the linear baseline must beat the
identity baseline given the true action embeddings.

### Baselines

```bash
pixi run -e gpu python -m pytest -x src/embpy/world_model/tests/test_baselines.py
```

Each baseline is exercised on a tiny synthetic dataset where the
perturbation effect is a deterministic shift; the linear baseline
must beat the identity baseline given the true action embeddings.

### Numpy metric helpers

```bash
pixi run -e gpu python -m pytest -x src/embpy/world_model/tests/test_metrics.py
```

Closed-form expected values for `mse`, `mae`, `r2_score`,
`pearson_corr`, `spearman_corr`, `deg_overlap_top_k` are checked.

### AnnData prep + plotting

```bash
pixi run -e gpu python -m pytest -x src/embpy/world_model/tests/test_prep.py
pixi run -e gpu python -m pytest -x src/embpy/world_model/tests/test_plots.py
```

Verifies that `prep.build_real_anndata` / `prep.build_pred_anndata`
produce aligned AnnData objects with the expected obs / var layout, and
that every plotting helper writes both PNG and SVG.

### Eval pipeline end-to-end (smoke test)

```bash
pixi run -e gpu python -m embpy.world_model.scripts.smoke_test
```

Must finish under five minutes on CPU and leave behind:

```
outputs/world_model/smoke/
  config.yaml
  train_log.csv
  report.md
  comparison.csv
  baselines.csv
  world_model_metrics.csv
  plots/loss_curves.png
  plots/comparison.png
```

### Compare / report (post-hoc)

```bash
# regenerate comparison.csv + comparison.png from a finished run
pixi run -e gpu python -m embpy.world_model.scripts.compare \
    --run-dir outputs/world_model/smoke

# regenerate report.md from whatever exists in run-dir
pixi run -e gpu python -m embpy.world_model.scripts.make_report \
    --run-dir outputs/world_model/smoke
```

### SLURM submission

Syntax-check every launcher without submitting (no actual job is queued):

```bash
for f in src/embpy/world_model/scripts/slurm/*.sbatch; do
    bash -n "$f" && echo OK "$f"
done
bash -n src/embpy/world_model/scripts/submit_all.sh && echo OK submit_all.sh
```

Expected: every line prints `OK <path>`. Submit smallest first
(`train_single_nadig.sbatch`); `logs/wm-nadig_<jobid>.out` should print
"Run single_nadig -- output_dir=outputs/world_model/single_nadig"
within seconds of the job starting.
