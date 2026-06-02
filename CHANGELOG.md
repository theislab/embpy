# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog][],
and this project adheres to [Semantic Versioning][].

[keep a changelog]: https://keepachangelog.com/en/1.0.0/
[semantic versioning]: https://semver.org/spec/v2.0.0.html

## [Unreleased]

### Added

- `embpy.store.actions.compile_action_table` is now the shared headless action compiler, with explicit RESOLVED / CONTROL / UNRESOLVED status accounting and JSON sidecars.
- `embpy.resources.gene.control.ControlPolicy` and status-aware embedding APIs add clearer handling for control and unresolved identifiers.

### Changed

- `embpy/__init__.py` is now a PEP-562 lazy facade (audit step 1). `import embpy` no longer eagerly walks `dt`, `models`, `pl`, `pp`, `resources`, `tl`, or pulls `embpy.embedder` (3,600 lines, transformers + torch + numpy). Cold `import embpy` drops from minutes (lustre, observed) to milliseconds. The public API is unchanged: `embpy.BioEmbedder`, `embpy.GeneResolver`, `from embpy import tl`, etc. still resolve -- they load on first access via a module-level `__getattr__` (PEP 562). Static analysis is preserved through a `TYPE_CHECKING` import block. Guarded by `tests/embpy/test_boundary.py::{test_import_embpy_does_not_eagerly_load_subpackages, test_import_embpy_does_not_eagerly_load_embedder, test_lazy_bioembedder_attr_access_still_works}`.
- `MODEL_REGISTRY` and the three DNA species sets (`HUMAN_ONLY_MODELS`, `MOUSE_ONLY_MODELS`, `MULTI_SPECIES_DNA`) moved from `src/embpy/embedder.py` to the new `src/embpy/embedder_registry/flat.py` (audit step 2). The optional-dep wrapper imports (Evo, Evo2, Boltz2) and the `_HAVE_*` flags moved alongside them. `from embpy.embedder import MODEL_REGISTRY` continues to return the exact same `dict` object (re-export via `from .embedder_registry.flat import ...`). Net delta in `embedder.py`: -216 lines.
- Registry split into per-modality submodules under `src/embpy/embedder_registry/` (audit step 3). `flat.py` is now a 25-line merge of `dna.py` (46 entries + the three DNA species sets), `protein.py` (23 entries, including Boltz-2 structural), `molecule.py` (14 entries), `text.py` (5 entries), `morphology.py` (11 entries), `singlecell.py` (0 entries today; reserved for the upcoming SingleCellEmbedder facade), and `api.py` (7 entries). Total: 106 entries; the public `MODEL_REGISTRY` dict has identical keys and identical `(Wrapper, path)` tuples vs. the pre-split snapshot (set + value byte-equivalence; iteration order changes only because GENA-LM / NT / HyenaDNA / Caduceus -- previously appended after morphology -- now group with the other DNA entries). Optional-dep gating (`EvoWrapper if _HAVE_EVO else None`) stays inside the per-modality submodule that owns the entry, so importing one modality's submodule never pays the cost of loading another's optional dependency. Guarded by `tests/embpy/test_registry_split.py` (snapshot of 106 entries + disjoint-modality + merge-equals-flat + species-sets-in-DNA + same-object-re-export tests).

### Deprecated

- `ActionEmbeddingProvider.embed(symbols)` -- replaced by `embed_with_status(symbols)`. Scheduled for removal in 0.1.0.
