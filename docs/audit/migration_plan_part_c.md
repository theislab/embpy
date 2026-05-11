# Part C migration plan -- split `embpy` and `world_model` into two top-level packages

Status: planning + scaffolding only. **File moves are deferred** to a
follow-up PR. Justification follows the prompt's explicit escape hatch:
"If the Part C workspace conversion is too risky to do in one PR, stop
after producing the workspace pyproject.tomls and the deprecation
shim, leave the actual file moves to a follow-up commit, and explicitly
say so."

The two concrete reasons for deferring file moves in this PR:

1. **Part A correctness has to land first.** The pipeline the prompt's
   regression test relies on (`configs/experiments/smoke.yaml` byte-
   equivalence) currently fails to even start (terminal log line 386:
   `ValueError: BioEmbedder returned no embeddings for any of 2058
   symbols`). The Part A landing fixes that root cause. Until it has
   been re-run end-to-end on the cluster, there is no baseline to
   compare a post-split smoke run against.
2. **The pixi GPU env shows a 5+ minute cold-import lag** (audit
   section 6). Moving the `world_model` tree while the env's `embpy`
   import is still going through `dt -> anndata -> pyarrow -> numpy
   ABI mismatch` would make a green-build vs. red-build read on the
   move impossible. Audit step 1 (shrink `embpy/__init__.py`) needs to
   land before audit step 9 (this split).

What lands now:

* This planning document.
* Workspace-root `pyproject.toml` scaffold for the two-package layout,
  added as `docs/audit/pyproject.workspace.toml.template` (NOT applied
  to the real `pyproject.toml`).
* Per-package `pyproject.toml` templates (`embpy.pyproject.toml.template`,
  `world_model.pyproject.toml.template`).
* The deprecation-shim layout spec.
* The pixi-feature-fold spec.
* The two-commit import-rewrite plan.
* The byte-equivalence regression-test contract.

What does NOT land now:

* Any file move under `src/embpy/world_model/`.
* Any change to the real `pyproject.toml` (root).
* Any change to `pixi.toml`.

---

## C.1 Target layout (recap)

```
src/
  embpy/                       infrastructure: embeddings, annotations,
    __init__.py                plotting, analysis, resolvers, model wrappers
    embedder.py
    errors.py
    dt/  pl/  pp/  tl/  models/  resources/
    pyproject.toml             (per-package manifest)
  world_model/                 perturbation modelling on top of embpy
    __init__.py                consumes embpy as a normal dependency
    configs/  data/  evaluation/
    models/  training/  utils/
    scripts/  README.md
    assets/  configs/experiments/
    pyproject.toml             (per-package manifest)
```

Tests split into `tests/embpy/` and `tests/world_model/`. The prompt
allows `world_model` as a placeholder name and a future rename
(candidates: `pertwm`, `pertworld`, `worldcell`). **Assumption stated
up front**: the rename decision is out of scope for the split.

---

## C.2 Package boundary contract

* `embpy` MUST NOT import from `world_model`. The follow-up PR adds
  `tests/embpy/test_no_world_model_import.py`:

  ```python
  import sys
  def test_embpy_does_not_pull_world_model():
      # Reset import state so test ordering is irrelevant.
      for k in list(sys.modules):
          if k.startswith(("embpy", "world_model")):
              del sys.modules[k]
      import embpy  # noqa: F401
      assert "world_model" not in sys.modules, (
          f"embpy must not depend on world_model. "
          f"Found: {[k for k in sys.modules if k.startswith('world_model')]}"
      )
  ```

* `world_model` MAY import from `embpy` freely. The cross-package
  imports are enumerated once at the top of `world_model/__init__.py`
  as a comment block:

  ```python
  # Cross-package imports (embpy -> world_model). Keep this list in
  # sync; the boundary test in tests/embpy/test_no_world_model_import.py
  # enforces the inverse direction.
  #
  #   from embpy.embedder            import BioEmbedder
  #   from embpy.resources.gene.control import ControlPolicy
  #   from embpy.resources.gene.resolver import GeneResolver
  #   from embpy.models.singlecell_models import StateEmbeddingWrapper, StackWrapper
  ```

---

## C.3 Build system (uv / hatch workspace)

### Root `pyproject.toml` (template)

See `pyproject.workspace.toml.template` in this directory. Highlights:

```toml
[tool.uv.workspace]
members = ["src/embpy", "src/world_model"]

[tool.hatch.metadata]
allow-direct-references = true
```

### `src/embpy/pyproject.toml` (template)

Carries every non-world-model dep from the current root manifest:
`anndata`, `biopython`, `broad-babel`, `cirpy`, `huggingface-hub`,
`ipywidgets`, `matplotlib`, `numpy`, `pysam`, `pandas`, `pyarrow`,
`pyensembl`, `rdkit`, `requests`, `scikit-learn`, `scipy`, `seaborn`,
`sentencepiece`, `protobuf`, `session-info`, `torch-geometric`,
`transformers`. Modality extras (`torch`, `torch-cpu`, `torch-cu121/4/8/130`,
`esm3`, `seqmodels`, `minimol`, `evo`, `evo2`, `caduceus`, `ntv3`,
`boltz`, `ppi`, `helical`, `morphology`, `lamindb`, `pertpy`, `scanpy`)
move into `embpy`.

### `src/world_model/pyproject.toml` (template)

```toml
[project]
name = "world_model"
dependencies = [
  "embpy",
  "numpy",
  "torch>=2.5.1",
  "anndata",
  "pyyaml",
]

[project.optional-dependencies]
state    = ["arc-state>=0.10"]
stack    = ["arc-stack>=0.1.3", "scvi-tools>=1.2"]
umap     = ["umap-learn>=0.5"]
eval     = ["matplotlib>=3.7", "pandas>=2.0"]
```

### Fallback (single-pyproject.toml two-package layout)

If pixi's resolver cannot accept the workspace layout, the prompt
permits a fallback:

* Keep one root `pyproject.toml`.
* Add a `[tool.uv.workspace]` block declaring two virtual members.
* Hatch's multi-package support lets us publish two wheels from one
  source tree.

Document the trade-off in `CHANGELOG.md` if we take this path.

---

## C.4 Pixi configuration

The two warnings `WARN feature 'state' / feature 'stack' is defined
but not used` go away by wiring those features into a new
`gpu+state+stack` env:

```toml
[environments]
gpu_state_stack = { features = [
  "gpu", "morphology", "jump", "scanpy", "pertpy",
  "ppi", "lamindb", "esm3", "seqmodels", "jupyter",
  "state", "stack",
] }
```

New convenience tasks:

```toml
[feature.gpu.tasks]
embpy-shell = "bash"                       # cwd is repo root, embpy in editable mode
wm-shell    = { cmd = "bash", env = { PYTHONPATH = "src/world_model" } }
```

(These come AFTER the file moves; this PR only documents them.)

---

## C.5 Import rewrites (two-commit plan)

* **Commit 1: `embpy.world_model` -> `world_model`.** Touches every
  Python import (`grep -rn 'embpy.world_model' src/ tests/`), every
  `python -m embpy.world_model.*` in `.sbatch` scripts (under
  `src/embpy/world_model/scripts/slurm/*.sbatch`), every reference in
  YAML configs (`configs/experiments/*.yaml`), every reference in
  shell scripts (e.g. `scripts/submit_all.sh`), and every README /
  notebook reference. Estimated touched files: ~90.
* **Commit 2: any other internal cross-references introduced by audit
  steps 1-3** (registry split). Estimated touched files: ~20.

### Transitional shim

`src/embpy/world_model/__init__.py` (after the move) becomes:

```python
"""Deprecated module path.

The world model has moved to its own top-level package, ``world_model``.
This module re-exports it for backward compatibility and will be
removed in the next minor release.
"""

from __future__ import annotations

import warnings
warnings.warn(
    "embpy.world_model is deprecated and will be removed in the next "
    "release. Import from world_model instead "
    "(`from world_model import ...`).",
    DeprecationWarning,
    stacklevel=2,
)

from world_model import *  # noqa: F401, F403, E402
```

Document the deprecation in `CHANGELOG.md` and delete the shim one
release later.

---

## C.6 Tests after the split

* `tests/embpy/` and `tests/world_model/` mirror the source tree.
* `tests/world_model/test_no_embpy_cycle.py` ensures importing
  `world_model` does not pull anything under
  `world_model.<unrelated subpackage>` either way (regression guard
  against future intra-`world_model` cycles).
* Re-run the smoke config end-to-end. The contract is:

  ```python
  import hashlib, pathlib
  for fname in ("comparison.csv", "baselines.csv",
                "action_embedding_meta.json"):
      a = hashlib.sha256(pathlib.Path(f"snapshot/{fname}").read_bytes()).hexdigest()
      b = hashlib.sha256(pathlib.Path(f"outputs/<run>/{fname}").read_bytes()).hexdigest()
      assert a == b, f"split changed {fname}: {a} != {b}"
  ```

The snapshot under `snapshot/` is captured **after** Part A lands and
the cluster pre-warm succeeds. The byte-equivalence guard can only be
honest if the snapshot is itself green.

---

## C.7 SLURM, scripts, configs (file-move checklist)

After Part A is validated, the follow-up PR moves:

| From | To |
| --- | --- |
| `src/embpy/world_model/` | `src/world_model/` |
| `src/embpy/world_model/scripts/slurm/*.sbatch` | `src/world_model/scripts/slurm/*.sbatch` |
| `src/embpy/world_model/configs/experiments/*.yaml` | `src/world_model/configs/experiments/*.yaml` |
| `src/embpy/world_model/assets/*` | `src/world_model/assets/*` |
| `src/embpy/world_model/README.md` | `src/world_model/README.md` |
| `tests/test_*.py` referencing `embpy.world_model` | `tests/world_model/test_*.py` |

Plus `submit_all.sh` (if it exists at repo root) updates its module
paths.

---

## C.8 README updates

Root `README.md` (after the move) gets a new "Repository layout"
paragraph:

```
This repository hosts two installable Python packages:

* embpy (src/embpy) -- infrastructure: foundation-model wrappers,
  identifier resolvers, AnnData utilities, plotting / preprocessing /
  analysis primitives.
* world_model (src/world_model) -- perturbation world modelling on
  top of embpy.

Install matrix:

    pixi install -e gpu               # both packages in editable mode
    pixi install -e gpu_state_stack   # plus arc-state + arc-stack

Cross-package imports (embpy -> world_model) are documented in
src/world_model/__init__.py.
```

Each package gets its own focused README under `src/embpy/README.md`
and `src/world_model/README.md` (the latter is largely a copy of the
current `src/embpy/world_model/README.md`).

---

## C.9 Order of operations (recap)

1. Land Part A (this PR).
2. Land the audit doc + steps 1-3 of the audit migration plan
   (follow-up PR; section 9 of `embpy_audit.md`).
3. Land Part C in two commits as described in C.5 (next-after-that PR).

The prompt asks for everything in one mega-patch internally structured
in commit order. **This PR delivers the first two stages of stage 1 and
the planning artefact for stages 2 and 3.** Stages 2 and 3 are deferred
because the byte-equivalence regression test that the prompt mandates
for them cannot be honestly produced until the pipeline runs end-to-end
post-Part-A.
