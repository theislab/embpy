# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog][],
and this project adheres to [Semantic Versioning][].

[keep a changelog]: https://keepachangelog.com/en/1.0.0/
[semantic versioning]: https://semver.org/spec/v2.0.0.html

## [Unreleased]

### Added

- `embpy.store.actions.compile_action_table` is now the shared headless action compiler used by the AnnData accessor and world-model store provider, with explicit RESOLVED / CONTROL / UNRESOLVED status accounting and JSON sidecars.
- World-model boxplot reporting now writes tidy per-perturbation CSV sidecars next to comparison and DEG-overlap plots.
- World-model run identity helpers add consistent `__job{id}__YYYYmmdd_HHMMSS` / `__local__YYYYmmdd_HHMMSS` suffix handling plus logical-run grouping.
- Two top-level Python packages under `src/`: **`embpy`** (infrastructure -- embeddings, resolvers, annotations, plotting, analysis, foundation-model wrappers) and **`world_model`** (perturbation modelling, built on top of `embpy`). The boundary is one-way: `world_model` depends on `embpy`, never the reverse. Enforced by `tests/embpy/test_boundary.py`. See `docs/audit/package_split.md` for the migration record and the fallback build-system trade-off.
- `world_model[state]`, `world_model[stack]`, `world_model[seqmodels]` optional dependency extras (modality-specific extras stay on `embpy`).
- Pixi env `gpu-state-stack` wiring the previously-unused `[feature.state]` + `[feature.stack]`; `embpy-shell` and `wm-shell` pixi tasks.
  (Renamed from the original `gpu+state+stack` because pixi rejects `+` in environment names.)
- `tests/embpy/test_boundary.py` -- guards `import embpy` against accidentally pulling in `world_model`, and (after the shim was removed) asserts that `embpy.world_model` no longer exists as an attribute or as an importable module.
- `tests/world_model/test_post_split_smoke.py` (gated by `--run-smoke-regression`) -- byte-equivalent CSV / JSON comparison plus pixel-equal PNG comparison against `tests/_snapshots/pre_split/` from commit `f1b51cb`.
- `embpy.resources.gene.control.ControlPolicy`, `world_model.data.embeddings.sentinel.EmbeddingStatus`, and the `embed_with_status` API on `ActionEmbeddingProvider` (Part A). See `docs/audit/embpy_audit.md`.

### Changed

- World-model action embeddings now default to store-backed `.emstore` inputs. Legacy CSV/NPZ tables should be migrated once with `python -m embpy.store.migrate` and then referenced with `action_embedding.source: store`.
- `StoreProvider` delegates combo/control/unresolved pooling to the shared embpy action compiler instead of carrying its own copy of that logic.
- World-model Apple MPS execution keeps integer action indices on CPU and performs frozen action-table lookup before projecting float action tokens on-device, avoiding MPS long-index corruption in the action encoder.
- `embpy/__init__.py` is now a PEP-562 lazy facade (audit step 1). `import embpy` no longer eagerly walks `dt`, `models`, `pl`, `pp`, `resources`, `tl`, or pulls `embpy.embedder` (3,600 lines, transformers + torch + numpy). Cold `import embpy` drops from minutes (lustre, observed) to milliseconds. The public API is unchanged: `embpy.BioEmbedder`, `embpy.GeneResolver`, `from embpy import tl`, etc. still resolve -- they load on first access via a module-level `__getattr__` (PEP 562). Static analysis is preserved through a `TYPE_CHECKING` import block. Guarded by `tests/embpy/test_boundary.py::{test_import_embpy_does_not_eagerly_load_subpackages, test_import_embpy_does_not_eagerly_load_embedder, test_lazy_bioembedder_attr_access_still_works}`.
- `MODEL_REGISTRY` and the three DNA species sets (`HUMAN_ONLY_MODELS`, `MOUSE_ONLY_MODELS`, `MULTI_SPECIES_DNA`) moved from `src/embpy/embedder.py` to the new `src/embpy/embedder_registry/flat.py` (audit step 2). The optional-dep wrapper imports (Evo, Evo2, Boltz2) and the `_HAVE_*` flags moved alongside them. `from embpy.embedder import MODEL_REGISTRY` continues to return the exact same `dict` object (re-export via `from .embedder_registry.flat import ...`). Net delta in `embedder.py`: -216 lines.
- Registry split into per-modality submodules under `src/embpy/embedder_registry/` (audit step 3). `flat.py` is now a 25-line merge of `dna.py` (46 entries + the three DNA species sets), `protein.py` (23 entries, including Boltz-2 structural), `molecule.py` (14 entries), `text.py` (5 entries), `morphology.py` (11 entries), `singlecell.py` (0 entries today; reserved for the upcoming SingleCellEmbedder facade), and `api.py` (7 entries). Total: 106 entries; the public `MODEL_REGISTRY` dict has identical keys and identical `(Wrapper, path)` tuples vs. the pre-split snapshot (set + value byte-equivalence; iteration order changes only because GENA-LM / NT / HyenaDNA / Caduceus -- previously appended after morphology -- now group with the other DNA entries). Optional-dep gating (`EvoWrapper if _HAVE_EVO else None`) stays inside the per-modality submodule that owns the entry, so importing one modality's submodule never pays the cost of loading another's optional dependency. Guarded by `tests/embpy/test_registry_split.py` (snapshot of 106 entries + disjoint-modality + merge-equals-flat + species-sets-in-DNA + same-object-re-export tests).

### Removed

- `src/embpy/world_model/` -- the deprecation shim that re-routed `embpy.world_model.X` to the top-level `world_model.X` (introduced in commit `882be61` with a removal target of 0.1.0) was removed earlier than planned. The codemod in commit `18896cf` rewrote every internal caller, and embpy is not externally distributed, so the shim had zero remaining consumers; keeping it contradicted the very separation the Part C split established. `import embpy.world_model` now raises `ModuleNotFoundError`; `embpy.world_model` as an attribute raises `AttributeError`. Update any straggling `from embpy.world_model.X import Y` to `from world_model.X import Y`. Guarded by `tests/embpy/test_boundary.py::{test_embpy_world_model_attribute_no_longer_exists, test_dotted_embpy_world_model_import_fails}`.

### Deprecated

- `ActionEmbeddingProvider.embed(symbols)` -- replaced by `embed_with_status(symbols)`. Scheduled for removal in 0.1.0.
