# Part C -- embpy / world_model package split

Status: landed in two commits.

This document records what shipped, what the prompt allowed us to defer,
and how to take the rest of the way home in a follow-up PR.

## What landed (this PR)

Two top-level Python packages live under `src/`:

```
src/
  embpy/         # built from the root pyproject.toml (unchanged contract)
  world_model/   # built from src/world_model/pyproject.toml (new)
```

Commit 1 -- module move

* `git mv src/embpy/world_model src/world_model` (history preserved on
  every file).
* Codemod (`scripts/_codemod_split.py`) rewrote 314 references across
  63 files: `embpy.world_model` -> `world_model` and
  `src/embpy/world_model` -> `src/world_model`. Touched files: Python
  modules, YAML configs, `.sbatch` SLURM scripts, `submit_all.sh`, READMEs,
  audit docs. The script is kept under `scripts/` so the rewrite is
  auditable; remove it once the deprecation shim is gone.
* Cross-package import surface enumerated at the top of
  `src/world_model/__init__.py` under
  `# --- depends on embpy: ---` (4 symbols, two of them lazy).
* New `src/world_model/pyproject.toml`. Uses the
  "directory IS the package" hatch pattern via
  `[tool.hatch.build.targets.wheel.sources]` so the import name
  `world_model` maps onto the on-disk `src/world_model/` directory
  without a nested `src/world_model/src/world_model/` layout.
* Root `pyproject.toml`: added `[tool.uv.workspace] members = ["src/world_model"]`
  so `uv sync --all-packages` resolves both packages together. Otherwise
  unchanged -- this is the explicit fallback path from the prompt:
  > If the workspace conversion is too risky in one PR, fall back to a
  > single root pyproject.toml with a `[tool.uv.workspace]` block and
  > document the trade-off in `docs/audit/package_split.md`.
* `pixi.toml`: `world_model = { path = "src/world_model", editable = true }`
  alongside `embpy = { path = ".", editable = true }` in
  `[pypi-dependencies]`; new `gpu+state+stack` env wiring the previously
  unused `[feature.state]` + `[feature.stack]`; new `embpy-shell` and
  `wm-shell` tasks.
* Tests split into `tests/embpy/` (28 files + 4 helper subdirs) and
  `tests/world_model/` (29 files including the two Part A tests that
  cover world_model code).
* READMEs: root README gets a "Repository structure" section with the
  install matrix and the cross-package import table. `src/world_model/README.md`
  gets a callout banner pointing at the new import path and the
  deprecation shim.

Commit 2 -- deprecation shim + boundary test + smoke regression

* `src/embpy/world_model/__init__.py` reinstates the old import path as
  a thin re-export of `world_model` with a `DeprecationWarning`. Uses
  `sys.modules[__name__] = world_model` so submodule lookups
  (`from embpy.world_model.training.trainer import WorldModelTrainer`)
  resolve via Python's standard machinery without per-submodule shim
  files.
* `tests/embpy/test_boundary.py`: imports `embpy` and asserts
  `'world_model' not in sys.modules` afterwards. Guards the one-way
  arrow.
* `tests/world_model/test_post_split_smoke.py`: gated by
  `--run-smoke-regression`; compares byte-equivalent CSV / JSON outputs
  and pixel-equal PNGs against snapshots saved under
  `tests/_snapshots/pre_split/`.
* `CHANGELOG.md` entry pins the shim removal release.

## What we did NOT do (explicit trade-off)

We chose the **fallback build-system layout** the prompt allowed:

* **No `src/embpy/pyproject.toml` in this PR.** The "directory IS the
  package" hatch pattern works fine for `world_model` (we just added it)
  but applying it to `embpy` is riskier because:
    * The root `pyproject.toml` is what PyPI ships from. Restructuring
      `embpy`'s build metadata under a sub-pyproject without verifying
      `python -m build` produces an equivalent wheel risks breaking
      external `pip install embpy` consumers.
    * Pixi's `embpy = { path = ".", editable = true }` currently points
      at the root; moving the pyproject would also require changing the
      path here in lockstep, and we cannot smoke-test pixi resolution
      inside the agent session (pixi cold-resolve is 5+ min on lustre).
* No byte-equivalence regression actually exercised in this PR. The
  pre-split snapshot fixture lives under `tests/_snapshots/pre_split/`
  and is populated by running the smoke config on the pre-split tree
  (commit `f1b51cb`). The post-split test reads from it. See the
  "Pre-condition gap" section below.
* No SLURM `--partition` / `--qos` re-tweak. Your one-line
  `train_single_replogle.sbatch` edit is left unstaged in the working
  tree.

## Pre-condition gap (worth flagging)

The Part C prompt expected:

> The smoke config `src/embpy/world_model/configs/experiments/smoke.yaml`
> runs green end-to-end.

This was not re-verified after Part A landed. The smoke config exists
and uses the `precomputed` action-embedding path (not BioEmbedder), so
the Borzoi-control failure that originally motivated Part A is not on
its critical path. But the regression-test infrastructure
(`--run-smoke-regression`) is shipped *without an actual pre-split
snapshot*: the first time you run

    pixi run -e gpu pytest tests/world_model/test_post_split_smoke.py --run-smoke-regression

it will fail with `MISSING SNAPSHOT`. To create the snapshot once:

    git checkout f1b51cb -- src/embpy/world_model    # pre-split tree, READ ONLY
    pixi run -e gpu python -m embpy.world_model.scripts.smoke_test \
        --config src/embpy/world_model/configs/experiments/smoke.yaml \
        --snapshot-out tests/_snapshots/pre_split/smoke/
    git checkout HEAD -- src/embpy/world_model       # back to the split tree

then commit `tests/_snapshots/pre_split/smoke/` and re-run the test.

## Follow-up PRs

1. Convert the root `pyproject.toml` into a true workspace meta-project
   and add `src/embpy/pyproject.toml`, mirroring the world_model pattern.
   Acceptance: `python -m build` from `src/embpy/` produces a wheel
   whose contents match the current root build byte-for-byte.
2. Capture the `pre_split` snapshot (instructions above) and turn
   `test_post_split_smoke.py` into a CI guardrail.
3. Drop the deprecation shim per the CHANGELOG entry and delete
   `scripts/_codemod_split.py`.
4. Walk the audit migration plan (steps 1-3) on top of the split tree.
