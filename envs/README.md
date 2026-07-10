# Per-model-family uv environments

embpy's models split into **dependency islands** that cannot share one resolution
(`uv lock` on the whole project fails: e.g. `ntv3` needs `transformers>=5` but the base pins
`transformers<5`; `boltz` pins `numpy<2`; `esm3` pins `transformers<4.48.2`; `seqmodels` `<4.51`;
`helical` `>=4.53`). So instead of one impossible lockfile, **each family is its own uv project with
its own `uv.lock`** — fully isolated, pip/pixi-free.

```
envs/
  <family>/
    pyproject.toml   # depends on embpy[<extra>]; embpy pulled as an editable path dep
    uv.lock          # this family's pinned, reproducible env (commit it)
```

## Run a family (this is the whole contract)

```bash
uv run --project envs/<family> python -m embpy.scripts.embed ...      # or the family's entrypoint
uv lock   --project envs/<family>                                     # (re)generate its lock
uv sync   --project envs/<family>                                     # materialize its venv
```

`uv run --project envs/<family>` resolves+syncs only that family's lock, so its `transformers` /
`numpy` / `torch` pins never touch another family. This is the same one-lockfile-per-unit rule the
ML handbook argues for, applied per model family. A Snakemake pipeline calls `uv run --project` per
node — Snakemake owns the DAG, uv owns the env, files own the data.

## Families

| Family    | embpy extra | Why isolated                              | Locks on macOS? |
|-----------|-------------|-------------------------------------------|-----------------|
| core      | torch,ppi,morphology,scanpy | base transformers 4.45–<5, numpy 2 (ESM2, ProtT5, HF-DNA, molecule LMs, text, morphology) | yes |
| esm3      | esm3        | `esm>=3.2` pins transformers<4.48.2       | yes |
| seqmodels | seqmodels   | borzoi pins transformers<4.51             | yes |
| ntv3      | ntv3        | NucleotideTransformer v3 needs transformers>=5 | yes |
| boltz     | boltz       | `boltz>=2` pins numpy<2                    | yes |
| helical   | helical     | needs transformers>=4.53 (+ igraph C lib; may need system libs) | maybe |
| state     | state       | arc-state (torch>=2.7)                     | yes |
| stack     | stack       | arc-stack + scvi-tools                     | yes |
| caduceus  | caduceus    | mamba-ssm CUDA extension (linux+CUDA)      | **no — cluster only** |
| minimol   | minimol     | torch-sparse/scatter build chain (linux+CUDA) | **no — cluster only** |
| evo2      | evo2        | transformer_engine → **needs a container**, not uv-only (see lab handbook Ch. 9) | **no — container** |

CUDA-build families restrict their lock to linux (`[tool.uv].environments`) and inject torch into the
build isolation of their C-extensions (`[tool.uv.extra-build-dependencies]`), matching embpy's root
`[tool.uv]`. `evo2` is the documented hard holdout: it needs a CUDA image, so its "env" is a
container, not a uv.lock.
