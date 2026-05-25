# Prompt: Migrate the world model onto modern embpy + boxplot reporting

You are working in the `embpy` monorepo at
`/lustre/groups/ml01/workspace/goncalo.pinto/embpy` (branch `vibe_embpy`).
The repo contains two packages: the `embpy` embedding library
(`src/embpy/`) and the `world_model` perturbation world model
(`src/world_model/world_model/`). The world model is **out of sync** with
recent embpy additions and must be brought up to date, re-run end to end,
and produce richer plots. Do this to the highest software-engineering
standard: typed, tested, debuggable, interpretable, fail-loud-not-silent.

Work in small, reviewable commits. Do **not** push or open a PR unless I
ask. Use `pixi run -e gpu -- ...` for anything that imports torch/embedder.

---

## Background you must respect (already true in the codebase)

- **New embpy embedding infrastructure** (the world model does *not* use it yet):
  - `embpy.io` — canonical in-memory result layer: `EmbeddingResult`,
    `EmbeddingProvenance`, plus `normalize_embedding_input`, `harmonize`,
    and exporters (`to_anndata`, `to_table`, …). One `EmbeddingResult` is
    the single source of truth; exporters turn it into AnnData/table at the
    edges. See `src/embpy/io/__init__.py` and `src/embpy/io/result.py`.
  - `embpy.store` — reusable embedding universe: `EmbeddingStore`,
    `EmbeddingBlock`, `RelationTable`, and the `adata.embpy` accessor
    (`EmbpyAccessor`, registered on import). On-disk format is `.emstore`
    (manifest + entities + per-embedding matrix/index + relations), with
    memory-mapped reads via `EmbeddingStore.read(..., backed=True)`. See
    `src/embpy/store/core.py`, `src/embpy/store/accessor.py`, and the
    tutorial `docs/notebooks/14_embedding_store_and_accessor_tutorial.ipynb`.
  - Key accessor methods to build on: `register_store`, `register_embedding`,
    `setup_conditions`, **`compile_actions`** (compiles obs-level
    perturbation/action embeddings into `.obsm`, handles combos like
    `"g1+g2"` and mean/sum aggregation), `make_splits`, `make_torch_dataset`.
  - `.X` is never touched; generated embeddings live in `.obsm`/`.varm`/store.

- **What the world model uses today (legacy, must be retired):**
  - Its own provider stack in `src/world_model/world_model/data/embeddings/`:
    `ActionEmbeddingProvider` ABC (`provider.py`) with two backends —
    `PrecomputedProvider` (CSV/NPZ, `precomputed.py`) and
    `BioEmbedderProvider` (`bio_embedder.py`, bridges
    `embpy.embedder.BioEmbedder.embed_genes_batch`). Selected via
    `registry.build_provider`.
  - Configs point at `data.gene_embedding_path` CSVs, e.g.
    `data/embeddings/gene_embeddings/genept/embeddings_3072.csv`
    (`configs/experiments/single_nadig.yaml`). This legacy CSV route is the
    thing to remove.
  - Useful invariants already in place that you MUST preserve:
    status-aware embedding (`EmbeddingStatus` RESOLVED/CONTROL/UNRESOLVED,
    deterministic CONTROL sentinel vectors, loud WARNINGs on UNRESOLVED),
    typed errors with stable `.exit_code`s (see `embpy/errors.py` and the
    exit-code table in `scripts/embed_perturbations.py`), lazy `import embpy`
    (PEP-562 in `src/embpy/__init__.py` — keep `import embpy` cheap), and
    run provenance written to `run_info.json` (`scripts/train.py`).

- **Job identifiers already half-exist:** `scripts/train.py::_apply_auto_suffix`
  appends `__job{SLURM_JOB_ID}__{timestamp}` to `output_dir`. The existing
  `runs/world_model/` is inconsistent (some dirs have the suffix, some
  don't). `aggregate_sweep.py::_emb_name` already strips the suffix to group
  by logical config — keep that grouping contract.

- **The three setups to KEEP exactly as setups** (names/semantics unchanged):
  `single_nadig`, `single_replogle`, and `transfer` (Nadig → p% Replogle).
  Orchestrated by `scripts/submit_all.sh` (train → baselines → compare DAG)
  and `aggregate_sweep.py` (cross-embedding view).

- **Other models are running concurrently** — your changes must not break
  them: (a) other world-model variants/configs (`single_*_incontext_tokens`,
  `single_replogle_stack_borzoi`, `single_replogle_state`, the ablation/LOEO
  branches in `submit_all.sh`); and (b) embpy's other-modality embedding
  pipelines (`submission_scripts/protein|dna|molecule/...`). Treat embpy as a
  shared library: additive, backward-compatible changes only; no signature
  breaks to public embpy APIs used elsewhere.

---

## Deliverables

### 1. Full migration of the world model onto modern embpy embeddings
- Make `adata.embpy` + `EmbeddingResult` + `EmbeddingStore`/`.emstore` the
  **single source of truth** for action (perturbation) embeddings. The world
  model should: build/load an `EmbeddingStore` for the gene embeddings,
  link it to the dataset AnnData, and use `adata.embpy.compile_actions(...)`
  (or an equivalent headless store API) to produce the action table the
  dynamics model consumes.
- **Retire the precomputed CSV path** (`PrecomputedProvider`, the
  `data.gene_embedding_path` config field, and the fallback logic in
  `registry.build_provider`). Provide a one-shot migration helper that
  converts any existing CSV/NPZ gene-embedding table into a `.emstore` so we
  don't lose access to gene2vec/genept/etc., and update the configs
  accordingly.
- Update the `BioEmbedder` integration to current embpy semantics: the
  embedder's canonical output is now `EmbeddingResult` — route through it
  rather than raw ndarrays. Verify `BioEmbedder.embed`/`embed_genes_batch`
  signatures and fix any drift in `bio_embedder.py` / `embed_perturbations.py`.
- Keep the status-aware contract end to end (CONTROL sentinel, UNRESOLVED =
  loud, never silent zeros). Persist the status/provenance sidecars
  (`action_embedding_meta.json`, `*.status.json`) and the embedding
  provenance (model, region, pooling, organism, ckpt/dataset hashes) into
  each run dir so a stale embedding is one `cat` away from being noticed.
- Back the action embeddings with a shared, content-addressed `.emstore`
  cache keyed by `(model, region, pooling, organism)` (mirror the existing
  `EmbeddingCacheKey`) so re-runs and the multi-seed matrix reuse vectors and
  only invoke the foundation model on genuinely new symbols.
- Make any **embpy-side changes needed to make training smooth**, but keep
  them additive and covered by tests: e.g. a clean headless
  store→action-table API usable without round-tripping through AnnData, robust
  combo/control handling in `compile_actions`, and clear errors when a
  perturbation has no resolvable target.

### 2. Boxplots instead of bar charts (per-perturbation, pooled across seeds)
- Replace the bar charts with **boxplots** where each box is the distribution
  of a metric **over perturbations, pooled across seed replicates**, one box
  per model/embedding and per baseline, faceted by metric and by dataset.
  Order boxes by median; overlay per-seed points/medians; annotate n.
  Concretely convert:
  - `evaluation/plots.py`: `plot_deg_overlap_bar` (`ax.bar`) and
    `plot_baseline_comparison` (`ax.barh`) → boxplot equivalents. Keep the
    existing `_save` (PNG + SVG) and the matplotlib-missing graceful skip.
    (`plot_per_perturbation_metric` already uses violin — align its style.)
  - `evaluation/ablation/aggregate.py` (`ax.bar` at the per-metric plots).
  - `scripts/aggregate_sweep.py` ("grouped bar per embedding" →
    grouped/faceted boxplots across embeddings and baselines).
- Emit the **underlying tidy/long-form CSV** next to every boxplot (one row
  per `dataset × model × baseline × seed × perturbation × metric`) so the
  figure is fully reproducible and inspectable. This long table is the
  "multitude of results / fine-grained data" the plots summarize.
- This requires per-perturbation metrics to be available per run (they are —
  `plot_per_perturbation_metric` consumes a per-pert frame). Make sure
  compare/aggregate persist and stitch per-perturbation tables across seeds.

### 3. Per-job identifiers, enforced consistently
- Every submitted job (train, baselines, compare, transfer pretrain/finetune,
  sweep, ablation) must write into a directory carrying a unique identifier
  **`__job{SLURM_JOB_ID}__{YYYYmmdd_HHMMSS}`** (fall back to
  `__local__{ts}` off-SLURM). Make `_apply_auto_suffix` the single chokepoint
  and ensure it's applied on *all* entry points, not just `train`. Record the
  job id + timestamp inside `run_info.json` too (it already snapshots SLURM
  env via `_slurm_provenance`).
- Aggregation (`aggregate_sweep`, the new boxplots) must group by **logical
  config** (suffix stripped) and pool seeds/jobs — never collide or
  double-count.

### 4. Seeds are user-definable
- Add a `--seeds`/`SEEDS=` parameter to `submit_all.sh` (and the seed knob to
  configs) that launches one job per `(setup, embedding, seed)`. **Default to
  a single seed for now** (I just want results); `SEEDS="0 1 2"` or
  `"0 1 2 3 4"` should later fan out replicates that the boxplots pool. The
  seed must flow into `seed_everything`, the split seed, and the run-name
  suffix so seed runs are distinguishable yet group together.

### 5. Wipe stale runs
- Delete everything under `/lustre/groups/ml01/workspace/goncalo.pinto/embpy/runs/world_model`
  (the ~117 stale run dirs). Do it as an explicit, logged step. Nothing there
  needs preserving. Recreate the empty dir so submission scripts have a target.

---

## Engineering standards (non-negotiable)

- **Types & lint:** full type hints on new/changed code; pass the repo's
  basedpyright/ruff config. No `Any` leakage across module boundaries.
- **Tests:** add/extend pytest coverage for: the store→action-table path,
  `compile_actions` with combos + controls + unresolved, the CSV→`.emstore`
  migration, the run-id suffix on every entry point, seed fan-out, and the
  boxplot builders (assert the long-form CSV schema + that a figure file is
  produced). Put world-model tests under the existing test tree; embpy tests
  under `tests/embpy/`. Keep the `smoke.yaml` config working as a fast
  end-to-end check.
- **Debuggable & interpretable:** structured logging at decision points
  (cache hit/miss counts, n_resolved/n_control/n_unresolved, chosen
  device/dtype, seed, run dir). Every failure path raises a *typed* embpy
  error with the right `.exit_code`; never swallow exceptions into silent
  zeros or `pass`. Keep the "loud on UNRESOLVED" behavior.
- **Provenance everywhere:** each run dir self-describes (config snapshot,
  git sha + dirty flag, SLURM ids, embedding model/provenance, seed, package
  versions). A reviewer should reconstruct any figure from the run dir alone.
- **Backward-compatibility for shared code:** `import embpy` stays cheap
  (don't break the lazy `__getattr__`); don't change public embpy signatures
  other modalities depend on; gate any behavior change behind additive params
  with sane defaults.
- **Docs:** update `src/world_model/README.md` (the action-encoding section
  describes the old gene-table lookup — bring it in line with the
  store/`compile_actions` flow), add a short `CHANGELOG.md` entry, and a
  migration note for the retired CSV path.

---

## Suggested execution order

1. Read `embpy.io.result`, `embpy.store.core`, `embpy.store.accessor`
   (esp. `compile_actions`/`make_torch_dataset`) and notebook 14 in full;
   confirm current `BioEmbedder` output semantics.
2. Land the embpy-side helpers (headless store→action API, hardened
   `compile_actions`) + tests, additively. Verify other modalities unaffected.
3. Migrate the world model data path to the store/`compile_actions` route;
   add the CSV→`.emstore` migrator; retire `PrecomputedProvider` + the
   `gene_embedding_path` config; update `single_nadig`/`single_replogle`/
   `transfer` configs. Get `smoke.yaml` green.
4. Enforce the `__job…__ts` suffix on all entry points; add `--seeds`
   fan-out (default 1); make aggregation group-by-logical-config + pool seeds.
5. Convert bar→boxplot in `plots.py`, `ablation/aggregate.py`,
   `aggregate_sweep.py`; emit the long-form CSVs.
6. Delete `runs/world_model/*`; recreate empty.
7. Dry-run `submit_all.sh` (print the DAG without submitting), then a single
   real smoke job per setup; confirm run dirs carry ids, boxplots + CSVs land,
   provenance is complete.

## Acceptance criteria
- `single_nadig`, `single_replogle`, `transfer` train, evaluate, and compare
  end to end using **only** the new embpy embedding path (no CSV route).
- Every run dir is uniquely identified by job id + timestamp; aggregation
  pools seeds per logical config without collisions.
- Reports show **boxplots** (per-perturbation, seed-pooled) with a matching
  long-form CSV; no bar charts remain in the world-model reporting path.
- `--seeds` defaults to 1 and cleanly fans out to N.
- `runs/world_model` is empty except for new runs.
- Other world-model variants and embpy's protein/dna/molecule pipelines still
  import and run; `import embpy` stays lazy/cheap; test suite passes.
