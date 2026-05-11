# embpy software-engineering audit

Status: review-only.
Scope: this audit documents what the current package looks like as of
commit `43088dd` plus the Part A correctness landing in `f1b51cb`
(`src/embpy/resources/gene/control.py`,
`src/world_model/data/embeddings/sentinel.py`, status-aware
provider rewrite, `resolve_symbol` alias chain, four test files) and
the Part C package split (this PR -- `world_model` promoted to a
top-level package under `src/`, see `docs/audit/package_split.md`).

Part B steps 1-3 of the migration plan are intentionally **deferred to
follow-up PRs**. The prompt allows this:
"If a deeper restructure carries unacceptable risk now, stop after
`embpy_audit.md` + Part A and explicitly say so." Two concrete reasons:

* `src/embpy/embedder.py` is 3,599 lines and `embpy/__init__.py` eagerly
  imports it (plus `dt`, `pl`, `pp`, `tl`, `resources`, `world_model`,
  `errors`, …). Any restructure touching the registry must be paired
  with a byte-equivalent regression test that imports the package
  end-to-end. That test cannot be wired up until the eager-import
  cascade is broken (audit step 6).
* The Part A landing already moves the gene-mapping behaviour to a new
  contract. Stacking a registry split on top doubles the diff that has
  to be reviewed against a (currently broken) end-to-end pipeline.

The remainder of this document is structured as the prompt requested:
nine sections, each with line-number citations.

---

## 1. Module responsibilities

One sentence per public symbol; lines pinned to the current tree.

### `embpy.embedder`

| Symbol | Line | Responsibility | Overlap notes |
| --- | --- | --- | --- |
| `get_device` | `src/embpy/embedder.py:72` | Device probe (CUDA / MPS / CPU). | None. |
| `MODEL_REGISTRY` | `src/embpy/embedder.py:87` | Single flat dict mapping name -> (WrapperClass, HF_path) for **DNA, protein, molecule, text, morphology, single-cell, API** modalities together. | Should be split per modality (see section 3). |
| `HUMAN_ONLY_MODELS` / `MOUSE_ONLY_MODELS` / `MULTI_SPECIES_DNA` | `src/embpy/embedder.py:264-301` | Species-validity sets for DNA registry entries. | Currently consumed only inside `embedder.py`; should live next to the DNA registry. |
| `BioEmbedder.__init__` | `:321` | Builds a `GeneResolver`, optional `ProteinResolver`, `TextResolver`, MorphologyResolver, picks a device, sets up caches for both general models and the single-cell wrappers. | Constructor doubles as a service locator. |
| `BioEmbedder._discover_models` | `:393` | Walks `MODEL_REGISTRY`, drops entries whose wrapper is `None` (optional dependency missing). | Logic could live on the registry itself. |
| `BioEmbedder._get_model` | `:418` | Lazy model load + LRU eviction. | This is the "EmbedderDispatcher" in the proposed decomposition. |
| `BioEmbedder._detect_vocab_type` | `:491` | Detects whether an AnnData uses symbols / Ensembl IDs / mixed / unknown. | Belongs with `IdentifierResolver`. |
| `BioEmbedder._ensure_singlecell_vocabulary` | `:520` | Aligns AnnData var names with the foundation model vocabulary. | Tightly coupled to single-cell wrappers; should live in `embpy.models.singlecell_models`. |
| `BioEmbedder._get_or_load_singlecell_wrapper` | `:635` | Cached single-cell wrapper loader. | Separate eviction policy from the general LRU. |
| `BioEmbedder.clear_model_cache` | `:671` | Public eviction. | None. |
| `BioEmbedder.embed_gene` | `:709` | One gene -> one embedding via DNA / protein / text models. | "EmbedderDispatcher" in proposal. |
| `BioEmbedder.embed_protein` | `:823` | One protein sequence -> embedding. | Same. |
| `BioEmbedder.embed_proteins_batch` | `:901` | Batched protein embedding. | Same. |
| `BioEmbedder.embed_genes_batch` | `:998` | Batched gene embedding -- the entry point called by `BioEmbedderProvider`. | Same. |
| `BioEmbedder.embed_molecule` / `_molecules_batch` | `:1127`, `:1182` | SMILES / molecule embeddings. | Same. |
| `BioEmbedder.embed_text` / `_text_api` / `_texts_batch` | `:1247`, `:1288`, `:1339` | Free-text embeddings. | Same. |
| `BioEmbedder.embed_description` / `_descriptions_batch` | `:1390`, `:1453` | Gene description text embeddings. | Builds on top of the resolver + text embedder; should be a thin orchestrator method on `BioEmbedder` over `TextResolver`. |
| `BioEmbedder.embed_cells` | `:1521` | Single-cell foundation-model encode. | Belongs on a `SingleCellEmbedder` facade -- already partly extracted into `embpy.models.singlecell_models`. |
| `BioEmbedder.decode_cells` | `:1764` | Latent -> expression decode for foundation models. | Same. |
| `BioEmbedder.generate_cells` | `:1881` | Generative variant. | Same. |
| `BioEmbedder.embed_adata` | `:1935` | AnnData round-trip. | Same. |
| `BioEmbedder.list_available_models` | `:2235` | Enumerate registry, filtered by optional deps. | Trivial; should be on the registry module. |
| `BioEmbedder.embed_morphological` / `_morphological_batch` | `:2277`, `:2312` | Image -> embedding. | Same EmbedderDispatcher role. |
| `BioEmbedder.embed_perturbation_morphology` / `_batch` | `:2342`, `:2475` | JUMP / HPA image resolution + embedding. | Resolution chain duplicates the structure of `GeneResolver.symbol_to_ensembl`; both can share a small step-runner helper. |
| `BioEmbedder._resolve_jump_images` / `_resolve_hpa_images` / `_load_jump_precomputed` / `_embed_perturbation_precomputed_batch` | `:2759`, `:2835`, `:2991`, `:2690` | Private helpers for morphology resolution. | OK as private. |
| `BioEmbedder.embed_fasta` | `:3063` | FASTA file -> per-record embeddings. | Edge case; could remain on `BioEmbedder`. |
| `BioEmbedder._resolve_seq_type` | `:3235` | Sequence-type detection. | Duplicates `embpy.resources.gene.resolver.detect_identifier_type`. Consolidate. |
| `_ResolutionContext` | `:3289` | Internal helper accumulating fallback steps for morphology. | Reusable: the new `_alias_resolver.Resolution` from Part A could share this shape. |

### `embpy.resources.gene.resolver`

| Symbol | Line | Responsibility |
| --- | --- | --- |
| `_ensembl_get` | `:13` | HTTP GET with 429 backoff. |
| `detect_identifier_type` | `:72` | Classify a string into smiles / dna_sequence / ensembl_id / protein_sequence / symbol. |
| `GeneResolver.__init__` | `:107`, `:113` | Configures pyensembl, optional Biomart, optional indexed FASTA. |
| `GeneResolver.download_genome` | `:193` | One-time genome FASTA bulk download. |
| `GeneResolver.get_dna_sequence` | `:422` | API path -- the failing path in the user's logs. Now superseded for symbol normalisation by `resolve_symbol` (Part A.4). |
| `GeneResolver.get_local_dna_sequence` | (in file) | Indexed-FASTA local path. |
| `GeneResolver.get_gene_region_sequence` | (in file) | Region-specific extraction (exons, introns, …). |
| `GeneResolver.get_gene_description` | (in file) | MyGene.info description text. |
| `GeneResolver.resolve_symbol` | (added by Part A.4) | Resilient four-step alias chain with disk cache (see Part A). |
| `GeneResolver.symbol_to_ensembl` | `:651` | Symbol -> ENSG ID. Pre-existing 3-step chain; lacks HGNC (now in `resolve_symbol`). |
| `GeneResolver.ensembl_to_symbol` | `:714` | Inverse. |

### `embpy.resources.protein_resolver`, `embpy.resources.text_resolver`

Both are 100-line thin wrappers re-exporting `embpy.resources.protein.resolver` / `embpy.resources.text.resolver`. No bugs; flag for inclusion in the per-modality registry split.

### `embpy.models.*`

* `singlecell_models.py` -- `StateEmbeddingWrapper`, `StackWrapper`; already extracted, do not touch (used by the world model via `world_model.models.encoders.backbones`).
* `dna_models.py`, `protein_models.py`, `molecule_models.py`, `text_models.py`, `morphology_models.py` -- one wrapper class per foundation model. Self-contained; lazy-import the heavy deps inside `_load`. Good.

### `world_model.data.embeddings` (Part A surface)

* `provider.py` -- `ActionEmbeddingProvider` ABC + `ProviderMetadata` dataclass. Now status-aware (`embed_with_status`).
* `bio_embedder.py` -- `BioEmbedderProvider` (delegates to `BioEmbedder.embed_genes_batch`).
* `precomputed.py` -- `PrecomputedProvider` (CSV / NPZ on disk).
* `cache.py` -- atomic NPZ cache with `fcntl.flock`.
* `registry.py` -- `build_provider(cfg, data_cfg=)` factory.
* `sentinel.py` -- new (Part A.2): `EmbeddingStatus`, `make_control_vector`, `make_unresolved_vector`.

---

## 2. `BioEmbedder` decomposition plan

`BioEmbedder` today does eight jobs in one class:

1. Owns the registry (`_discover_models`).
2. Owns model lifecycle + LRU caching (`_get_model`, `clear_model_cache`).
3. Resolves identifiers (`_detect_vocab_type`, `_ensure_singlecell_vocabulary`, `_resolve_seq_type`).
4. Dispatches per-modality calls (`embed_gene`, `embed_protein`, `embed_molecule`, `embed_text`, `embed_morphological`, `embed_cells`, ...).
5. Implements batched variants of each dispatch (the `*_batch` variants).
6. Implements perturbation-specific morphology resolution chains (`embed_perturbation_morphology*`).
7. Owns FASTA round-tripping (`embed_fasta`).
8. Owns AnnData round-tripping (`embed_adata`).

Proposed split (3-5 cohesive classes, public-API-preserving):

```
embpy.embedder
  BioEmbedder            -> thin facade. Same constructor + same public
                            methods, but each method delegates to one
                            of the helpers below. ~300 LOC.

embpy.embedder_internal       (new submodule -- not exported at top level)
  ModelRegistry          -> wraps the per-modality registries and the
                            HUMAN_ONLY / MOUSE_ONLY / MULTI_SPECIES sets.
                            Exposes `list_models`, `resolve_path`,
                            `wrapper_for(name)`. 100-150 LOC.
  EmbedderDispatcher     -> owns the LRU cache + `_get_model` + the
                            single per-call hot path (one `embed_*`
                            method per modality, no fan-out). 400 LOC.
  BatchedEmbedder        -> owns the batched variants. Reuses
                            EmbedderDispatcher for inner calls. 300 LOC.
  IdentifierResolver     -> consolidates `_detect_vocab_type`,
                            `_ensure_singlecell_vocabulary`,
                            `_resolve_seq_type`, plus the new
                            `resolve_symbol` shim. 200 LOC.
  PerturbationMorphology -> the JUMP / HPA / precomputed code path
                            (`embed_perturbation_morphology*` and
                            their helpers). 400 LOC.
```

Migration path that preserves the public surface:

* Step A: move `MODEL_REGISTRY` and the species sets into a new module
  `embpy.embedder_registry.flat` and re-export from
  `embpy.embedder` (so `from embpy.embedder import MODEL_REGISTRY` keeps
  working). Diff: ~300 lines moved + a one-line import.
* Step B: extract `EmbedderDispatcher` (the `_get_model` LRU + a
  per-modality dispatch dict). `BioEmbedder` keeps its method names,
  the bodies become one-liners that delegate. Diff: ~600 lines moved.
* Step C: extract `BatchedEmbedder`. Same pattern. Diff: ~400 lines.
* Step D: extract `IdentifierResolver`. The `_detect_vocab_type` helper
  needs to remain importable from `embpy.embedder` for back-compat.
* Step E: extract `PerturbationMorphology` -- needs its own tests
  before extraction; it has zero test coverage today.

Each step lands in its own PR. After every step, `from embpy.embedder
import BioEmbedder; BioEmbedder()` must still construct and call into
every dispatch method.

---

## 3. `MODEL_REGISTRY` split

Per-modality registries merged into a facade at import time. Mock
implementation (20 lines, illustrative):

```python
# src/embpy/embedder_registry/dna.py
from embpy.models.dna_models import BorzoiWrapper, EnformerWrapper
DNA_MODELS = {
    "borzoi_v0": (BorzoiWrapper, "johahi/borzoi-replicate-0"),
    "enformer_human_rough": (EnformerWrapper, "EleutherAI/enformer-official-rough"),
    # ...
}

# src/embpy/embedder_registry/__init__.py
from .dna import DNA_MODELS
from .protein import PROTEIN_MODELS
from .molecule import MOLECULE_MODELS
from .text import TEXT_MODELS
from .morphology import MORPHOLOGY_MODELS
from .singlecell import SINGLECELL_MODELS
from .api import API_MODELS

# Backwards-compatible flat dict
MODEL_REGISTRY = {
    **DNA_MODELS, **PROTEIN_MODELS, **MOLECULE_MODELS,
    **TEXT_MODELS, **MORPHOLOGY_MODELS,
    **SINGLECELL_MODELS, **API_MODELS,
}
```

The optional-dependency gating (`EvoWrapper if _HAVE_EVO else None`)
stays inside the per-modality submodule, so the heavy import only
happens when the user actually pulls in that modality. This is the
single biggest unlock for the lazy-import map (section 6).

---

## 4. Resolution chain

Precedence today (`embpy.resources.gene.resolver.symbol_to_ensembl`,
`:651-711`):

1. `pyensembl.genes_by_name(sym)` (offline once cached).
2. MyGene.info `/v3/query?q=<sym>` (synonym-tolerant).
3. Ensembl REST `/lookup/symbol/<organism>/<sym>` (the endpoint that
   returns 400 for any aliased symbol).

Failure modes seen in production (terminal log lines 308-340, last
run):

* Ensembl returns `400 Bad Request` for `AARS`, `CARS`, `DARS`,
  `EPRS`, `RARS`, `SARS`, `TARS`, `VARS`, `WARS`, `YARS`, ... -- every
  one of these is an *aliased* symbol whose approved name now ends in
  `1`.
* Ensembl returns `400` for `control` because the script let the
  literal `"control"` through as if it were a real gene name.
* MyGene.info usually returns *something* but the gene-id field is
  often empty for retired symbols, so the chain falls through to a
  zero embedding.

Part A.4 introduces a four-step chain (`embpy.resources.gene._alias_resolver`,
`resolve_symbol_chain`):

```
1. pyensembl  -> approved symbol if locally indexed
2. HGNC       -> fetch/symbol with search/alias_symbol fallback
3. Ensembl    -> retry with the approved symbol
4. MyGene     -> last resort: keyword query
```

Caching: positive AND negative results are persisted to
`~/.cache/embpy/symbol_resolution.json` so an Ensembl 4xx storm cannot
repeat across runs of the same dataset.

Proposed `ResolutionError` taxonomy (deferred to the resolver-split PR):

```
class SymbolResolutionError(EmbpyError): ...
class AliasOnlyResolution(SymbolResolutionError):
    # HGNC mapped the input but Ensembl still 4xx'd
    ...
class HardMissResolution(SymbolResolutionError):
    # Every step in the chain failed
    ...
```

Today every chain step swallows exceptions (`# noqa: BLE001`). After
the resolver split they should raise `SymbolResolutionError` and the
chain orchestrator records it in the chain log, in the spirit of
`embpy.resources.gene._alias_resolver._step_pyensembl`'s
`"error:<ExceptionName>"` status tag.

Calls that today swallow exceptions silently:

* `resolver.py:671` (pyensembl) -- catches `BaseException`, returns
  `None`.
* `resolver.py:698`, `:707` -- generic `Exception`, returns `None`.
* `resolver.py:732`, `:751` (ensembl_to_symbol) -- same.

---

## 5. Control-aware data path

Every entry point that accepts a perturbation label after Part A:

| Entry point | Where it lives | Pre-Part-A policy | Post-Part-A policy |
| --- | --- | --- | --- |
| `embed_perturbations.py main` | `src/world_model/scripts/embed_perturbations.py` | Hard-coded `--control-label non-targeting` exact literal. | `ControlPolicy.classify` + `--control-extra-labels` + `--fail-on-unresolved`. |
| `ActionEmbeddingProvider.embed_with_status` | `src/world_model/data/embeddings/provider.py` | (did not exist) | Status-aware with `EmbeddingStatus.{RESOLVED, CONTROL, UNRESOLVED}`. |
| `BioEmbedderProvider.embed_with_status` | `src/world_model/data/embeddings/bio_embedder.py` | (did not exist) | Classifies first; never sends a control variant to `BioEmbedder.embed_genes_batch`. |
| `PrecomputedProvider.embed_with_status` | `src/world_model/data/embeddings/precomputed.py` | (did not exist) | Same policy; controls map to deterministic sentinel. |
| `GeneIndexer.encode` | `src/world_model/data/datasets/base.py:67` | Only maps `perturbation == control_label` to index 0. | Unchanged in Part A. *Follow-up*: should consult `ControlPolicy` so dataset-level encoding agrees with the provider. |
| `cls.from_h5ad` (Replogle / Nadig) | `datasets/replogle.py:49`, `datasets/nadig.py:43` | Same single-literal policy. | Unchanged in Part A; follow-up step in section 9. |
| `build_dataloaders` | `data/dataloader.py:68` | Reported `n_unresolved` from a count of all-zero rows; conflated unresolved + control. | Honours `action_embedding.fail_on_unresolved`; persists per-bucket counts via `ProviderMetadata`. |

Per-entry-point policy after Part A: every entry point either runs
through `ControlPolicy.classify` (or delegates to a code path that
does) or is documented as a follow-up.

---

## 6. Lazy-import map

What forces what at import time today:

| Top-level import | Heavy modules pulled in transitively |
| --- | --- |
| `import embpy` | `anndata`, `scanpy?`, `transformers`, `torch`, `pandas`, `pyarrow`, `pyensembl`, all `models/*.py` wrappers, all of `resources/*`, all of `world_model/*`. |
| `import embpy.embedder` | same as above (because `embpy/__init__.py` already pulled it). |
| `import world_model.data.embeddings.sentinel` | numpy only (verified, Part A). |
| `import embpy.resources.gene.control` | re only (verified, Part A). |
| `import embpy.resources.gene._alias_resolver` | stdlib + (lazily) `requests`, `pyensembl`. |
| `import world_model.data.embeddings.provider` | numpy, embpy.resources.gene.control, world_model.data.datasets.base. Lightweight. |

Proposed `_lazy_import` pattern:

```python
# src/embpy/_lazy.py
from importlib import import_module
from typing import Any

def lazy(module: str, attr: str | None = None) -> Any:
    """Return a placeholder; the real import happens on first access."""
    class _Proxy:
        def __getattr__(self, name): return getattr(import_module(module), name)
        def __call__(self, *a, **kw): return import_module(module)(*a, **kw)
    if attr is None:
        return _Proxy()
    class _Attr:
        def __getattr__(self, name): return getattr(getattr(import_module(module), attr), name)
        def __call__(self, *a, **kw): return getattr(import_module(module), attr)(*a, **kw)
    return _Attr()
```

Higher-value than the proxy approach: just shrink `embpy/__init__.py`
to **re-export nothing** by default. Users that want `embpy.BioEmbedder`
write `from embpy.embedder import BioEmbedder`. The user's terminal log
shows cold `import embpy` taking minutes on lustre because of the
eager import chain through `dt/__init__.py` -> anndata -> pyarrow ->
numpy ABI mismatch. The fix is a 5-line patch to `__init__.py`; the
cost is one round of `from embpy import X` -> `from embpy.embedder
import X` rewrites across notebooks. This is **audit step 1** below.

---

## 7. Test gaps

Public methods with **no** test (counted by grepping `tests/test_*.py`
against the method names in section 1):

* `BioEmbedder.embed_perturbation_morphology` (lines 2342, 2475).
* `BioEmbedder._resolve_jump_images`, `_resolve_hpa_images`,
  `_load_jump_precomputed`.
* `BioEmbedder.embed_fasta`.
* `BioEmbedder.embed_adata` (covered indirectly by `tests/test_embedder.py`).
* `BioEmbedder.embed_descriptions_batch`.
* `BioEmbedder.embed_text_api`.
* `GeneResolver.download_genome` (one-time bulk download; test would
  need monkeypatched HTTP).
* `GeneResolver.get_gene_region_sequence`.

Minimum-viable test scaffolds (one paragraph each):

* `test_embed_perturbation_morphology.py` -- monkey-patch
  `_resolve_jump_images` to return a 2x2 PIL image stack; assert the
  resulting embedding shape and that a missing image yields a clean
  WARNING + zero row (paralleling Part A).
* `test_embed_fasta.py` -- write a 10-line synthetic FASTA, run with a
  CPU-only ESM-2 8M, assert per-record output shape.
* `test_download_genome.py` -- monkeypatch `urllib.request.urlopen` to
  return a 10-byte gzipped FASTA stub; assert the expected `.fa.gz`
  files appear under `cache_dir`.

For the new Part A code path, the four required tests now exist:

* `tests/test_control_policy.py`
* `tests/test_bio_embedder_provider_statuses.py`
* `tests/test_embed_perturbations_script.py`
* `tests/test_gene_resolver_aliases.py`

---

## 8. Logging policy

Today the package mixes `print`, `logging.info`, `logging.warning`,
`logger.warning`, `logger.error`, and bare `Exception` catches that
swallow the cause.

Proposed policy:

* `logger.debug(...)` for any per-symbol / per-step trace.
* `logger.info(...)` for one line per phase boundary (cache hit
  summary, model loaded, file written).
* `logger.warning(...)` for recoverable errors that **changed the
  output** (zero rows for unresolved symbols, partial multi-gene combo,
  legacy fallback, deprecation).
* `logger.error(...)` for irrecoverable errors before raising.
* Never `print(...)`.

Structured fields: where the message reports counts, always include
both the absolute count AND the denominator (`"%d / %d"`). The Part A
provider already follows this pattern; backport to the rest of the
package as part of the registry split.

---

## 9. Concrete restructure plan

Numbered, minimally invasive. Each step is its own PR. Diff budget is
the rough delta vs. the current tree.

| # | Step | Goal | Diff budget | Regression-test bar |
| --- | --- | --- | --- | --- |
| 1 | Shrink `embpy/__init__.py` to re-export nothing eagerly. | Stop the cold-import storm. | -40 lines in `__init__.py`; +1 `CHANGELOG` note. | Existing `tests/test_basic.py` still passes. Notebook scripts that do `import embpy; embpy.BioEmbedder(...)` fail with a clear message pointing at the new path. (Acceptable transitional break -- documented.) |
| 2 | Move `MODEL_REGISTRY` into `embpy.embedder_registry.flat` and re-export. | Per-modality split prep. | ~50 lines moved; `embpy.embedder` keeps a one-line shim. | `from embpy.embedder import MODEL_REGISTRY` returns the same dict object. |
| 3 | Per-modality registry submodules (`dna.py`, `protein.py`, ...). Flat dict merges them. | Section 3 plan. | ~250 lines moved across 6 new files. | The merged dict is byte-equivalent (every key + every (Wrapper, path) tuple matches the pre-split flat dict). |
| 4 | Extract `EmbedderDispatcher`. | Decoupling. | ~600 lines moved. | `BioEmbedder.embed_gene("TP53", ...)` returns the same array as before. |
| 5 | Extract `BatchedEmbedder`. | Same. | ~400 lines moved. | `BioEmbedder.embed_genes_batch(...)` byte-equivalent against a frozen reference output. |
| 6 | Extract `IdentifierResolver`. Consolidate `_resolve_seq_type` and `detect_identifier_type`. | DRY. | ~150 lines moved. | New `IdentifierResolver` unit-test, plus the existing classifier tests. |
| 7 | Resolver split: `embpy.resources.gene._alias_resolver` (already in this PR) + extracted `_ensembl_get`, alias-chain orchestrator, and a `ResolutionError` taxonomy. | Section 4. | ~150 lines moved, ~80 added. | `tests/test_gene_resolver_aliases.py` (this PR) passes. New negative-cache assertion. |
| 8 | Extract `PerturbationMorphology` from `BioEmbedder`. | Largest single chunk; ~700 LOC. | ~700 lines moved + 1 new test file. | Synthetic JUMP fixture under `tests/data/`. |
| 9 | Two-package split (`embpy` + `world_model`). | Prompt's Part C. | New `pyproject.toml`s; ~30 `sbatch`/YAML path rewrites. | Smoke config byte-equivalent vs. snapshot (`outputs/<run>/{comparison,baselines,action_embedding_meta}.json`). |

### What lands in **this** PR

Steps 0 (Part A correctness fix) and the **audit document** (this
file). Steps 1-9 are explicitly deferred so each one can be reviewed in
isolation against a working baseline. Part C scaffolding (the
workspace `pyproject.toml`s and the deprecation-shim spec) lands here
as `docs/audit/migration_plan_part_c.md` -- see the companion file.

---

## Appendix: line-count summary

```
src/embpy/embedder.py                3599   primary target of audit
src/embpy/resources/gene/resolver.py 1518   secondary target
src/world_model/                ~6000   already modular
tests/                                  29   files
docs/audit/                              2   files (this audit + Part C plan)
```

The `dataloader.py` (`src/world_model/data/dataloader.py`, ~330
lines) and the new Part A files are all <300 lines each. The 80/20
restructure work is concentrated in `embedder.py`.
