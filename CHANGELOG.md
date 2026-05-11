# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog][],
and this project adheres to [Semantic Versioning][].

[keep a changelog]: https://keepachangelog.com/en/1.0.0/
[semantic versioning]: https://semver.org/spec/v2.0.0.html

## [Unreleased]

### Added

- Two top-level Python packages under `src/`: **`embpy`** (infrastructure -- embeddings, resolvers, annotations, plotting, analysis, foundation-model wrappers) and **`world_model`** (perturbation modelling, built on top of `embpy`). The boundary is one-way: `world_model` depends on `embpy`, never the reverse. Enforced by `tests/embpy/test_boundary.py`. See `docs/audit/package_split.md` for the migration record and the fallback build-system trade-off.
- `world_model[state]`, `world_model[stack]`, `world_model[seqmodels]` optional dependency extras (modality-specific extras stay on `embpy`).
- Pixi env `gpu+state+stack` wiring the previously-unused `[feature.state]` + `[feature.stack]`; `embpy-shell` and `wm-shell` pixi tasks.
- `tests/embpy/test_boundary.py` -- guards `import embpy` against accidentally pulling in `world_model`, and exercises the lazy `embpy.world_model` shim path.
- `tests/world_model/test_post_split_smoke.py` (gated by `--run-smoke-regression`) -- byte-equivalent CSV / JSON comparison plus pixel-equal PNG comparison against `tests/_snapshots/pre_split/` from commit `f1b51cb`.
- `embpy.resources.gene.control.ControlPolicy`, `world_model.data.embeddings.sentinel.EmbeddingStatus`, and the `embed_with_status` API on `ActionEmbeddingProvider` (Part A). See `docs/audit/embpy_audit.md`.

### Changed

- `embpy/__init__.py` is now a PEP-562 lazy facade (audit step 1). `import embpy` no longer eagerly walks `dt`, `models`, `pl`, `pp`, `resources`, `tl`, or pulls `embpy.embedder` (3,600 lines, transformers + torch + numpy). Cold `import embpy` drops from minutes (lustre, observed) to milliseconds. The public API is unchanged: `embpy.BioEmbedder`, `embpy.GeneResolver`, `from embpy import tl`, etc. still resolve -- they load on first access via a module-level `__getattr__` (PEP 562). Static analysis is preserved through a `TYPE_CHECKING` import block. Guarded by `tests/embpy/test_boundary.py::{test_import_embpy_does_not_eagerly_load_subpackages, test_import_embpy_does_not_eagerly_load_embedder, test_lazy_bioembedder_attr_access_still_works}`.

### Deprecated

- `embpy.world_model` -- moved to the top-level package `world_model`. The old import path keeps working through `src/embpy/world_model/__init__.py`, which re-routes via `sys.modules[__name__] = world_model` and emits a `DeprecationWarning` on first access. **Removal target: 0.1.0.** Update `from embpy.world_model.X import Y` to `from world_model.X import Y`.
- `ActionEmbeddingProvider.embed(symbols)` -- replaced by `embed_with_status(symbols)`. Will be removed alongside the `embpy.world_model` shim in 0.1.0.
