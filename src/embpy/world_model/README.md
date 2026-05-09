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
    base.py                                dataclass schema + YAML loader
    nadig.yaml                             Nadig 2024 Jurkat config
    replogle.yaml                          Replogle 2022 K562 essential config
  data/
    __init__.py
    dataloader.py                          build_dataloaders(...)
    preprocessing.py                       HVG, log1p, collate
    datasets/
      __init__.py
      base.py                              PerturbationSequenceDataset, GeneIndexer
      nadig.py                             Nadig adapter
      replogle.py                          Replogle adapter
  models/
    __init__.py
    blocks.py                              MLP, attention, transformer block
    world_model.py                         composed WorldModel + factory
    encoders/
      __init__.py
      state_stack_encoder.py               transformer + MLP variants
    action/
      __init__.py
      gene_embedding_action.py             pretrained-embedding action encoder
    dynamics/
      __init__.py
      gpt_autoregressive.py                Decision-Transformer-style dynamics
    decoders/
      __init__.py
      expression_decoder.py                d -> G MLP decoder
  training/
    __init__.py
    losses.py                              latent / delta / Gaussian / InfoNCE
    schedulers.py                          cosine + linear warmup
    trainer.py                             WorldModelTrainer
  evaluation/
    __init__.py
    metrics.py                             L2, cosine, R^2, delta-Pearson
    rollouts.py                            imagined_rollout(...)
  utils/
    __init__.py
    checkpoint.py                          save / load model bundles
    logging.py                             setup_logging / get_logger
    seeding.py                             seed_everything
  scripts/
    __init__.py
    train.py                               python -m ... .scripts.train
    eval.py                                python -m ... .scripts.eval
  tests/
    __init__.py
    test_state_stack_encoder.py
    test_gene_embedding_action.py
    test_gpt_autoregressive.py
    test_world_model.py
    test_losses.py
    test_dataset.py
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

```bash
python -m embpy.world_model.scripts.train \
    --config src/embpy/world_model/configs/replogle.yaml
```

This will:

1. Load the AnnData, run HVG selection + `log1p`,
2. Build the gene embedding table aligned to the perturbed genes,
3. Construct `(state, action, next_state)` sequence batches,
4. Train the world model with the multi-term objective,
5. Save checkpoints under `outputs/world_model/<run_name>/`.

## Evaluate

```bash
python -m embpy.world_model.scripts.eval \
    --config src/embpy/world_model/configs/replogle.yaml \
    --checkpoint outputs/world_model/replogle_k562_essential/replogle_k562_essential_final.pt
```

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

## Next steps / how to debug each component in isolation

The package is designed so every component is independently testable.
A typical debugging flow:

1. **Encoder.** Run `tests/test_state_stack_encoder.py`; on a real
   batch verify `encoder(batch["obs_stack"]).std()` is in a sane range
   (target ~ 1.0 once trained -- if it collapses to ~0, lower the
   encoder dropout or increase the encoder LR).
2. **Action encoder.** Run `tests/test_gene_embedding_action.py`;
   inspect `action_encoder(batch["action_indices"])` and confirm two
   different perturbations produce *different* tokens (cosine
   similarity well below 1.0).
3. **Dynamics.** Run `tests/test_gpt_autoregressive.py`; the
   `test_dynamics_is_causal` test exercises the causal mask. Once
   training, monitor `latent_mse` -- it should drop below the
   variance of the encoder output (`s_target.var()`).
4. **Decoder.** Replace the trained encoder/dynamics by their
   identity and confirm `decoder(s_target)` reconstructs `x_target`
   well; if not, the decoder is the bottleneck.
5. **Dataset.** Use `tests/test_dataset.py` as a template, then call
   `len(dataset)` and `dataset[0]` in a notebook to verify shapes and
   labels make sense for your data layout.
6. **Trainer.** Set `n_epochs=1`, `n_top_genes=512`,
   `batch_size=8` and `n_sequences_per_epoch=64` for a smoke run.
   Check that `loss.backward()` does not fail (already exercised in
   `tests/test_world_model.py`).
7. **Rollouts.** Once `latent_mse` is meaningful, run
   `imagined_rollout(model, val_loader)` and watch
   `delta_pearson` -- this is the standard headline metric.
