# Notebook generators

`docs/notebooks/proteins.ipynb` is generated, not hand-edited. To change it, edit
the part files and rebuild:

```bash
cd /Users/grpinto/Documents/embpy
SP=scripts/notebook_generators
for n in 1 2 3 4 5; do python3 $SP/proteins_part$n.py $SP/proteins_part$n.json; done
python3 $SP/assemble.py $SP/proteins_part{1,2,3,4,5}.json docs/notebooks/proteins.ipynb
```

Then execute it end to end before committing (cluster, ~25 min):

```bash
ssh hpc 'cd /lustre/groups/ml01/workspace/goncalo.pinto/embpy && \
  srun --partition=interactive_cpu_p --qos=interactive_cpu --cpus-per-task=8 \
       --mem=16G --time=04:00:00 bash -c "source .venv-test/bin/activate && \
  cd docs/notebooks && jupyter nbconvert --to notebook --execute --inplace \
       --ExecutePreprocessor.timeout=2400 proteins.ipynb"'
```

`genes.ipynb`, `small_molecules.ipynb` and `cells.ipynb` have build wrappers that
use a `mktemp` dir, so no intermediate JSON lands in the tree:

```bash
scripts/notebook_generators/build_genes.sh
scripts/notebook_generators/build_small_molecules.sh
scripts/notebook_generators/build_cells.sh
```

All three honour `PY=` to override the interpreter. `build_genes.sh` and
`build_small_molecules.sh` still default to `.pixi/envs/default/bin/python`;
`build_cells.sh` defaults to `uv run --no-project python`, because the project
environment cannot be resolved at all -- `embpy[esm3]` and `embpy[helical]` carry
mutually unsatisfiable pins -- and the generators import nothing from embpy.

The small-molecules notebook runs locally in ~15 min -- no GPU, no cluster -- but
PubChem and ChEMBL are load-bearing:

```bash
cd docs/notebooks && jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=2400 small_molecules.ipynb
```

`cd docs/notebooks` first: the notebooks write to `Path("outputs")`, a relative
path, and `cells.ipynb` reads `Path("data")` the same way.

## Two guards the build runs

`assemble.py` parses every code cell with `ast.parse` before writing the
notebook, and refuses to assemble if any fails. A generator that emits
unparseable code otherwise produces a notebook that looks fine and dies on
execution minutes later, pointing at a cell rather than at the part file. Nested
triple quotes inside the `r"""` builders are the recurring cause.

`check_contract.py` catches the other recurring class: a name bound inside a
function body is not available to a later cell. It walks the parts in order,
tracking module-level bindings only, and reports any name a cell reads that no
earlier cell bound. `build_cells.sh` runs it before assembling; the other
wrappers do not yet.

```bash
uv run --no-project python scripts/notebook_generators/check_contract.py part1.json part2.json ...
```

## Building and executing `cells.ipynb`

Five parts, 95 cells. It needs the single-cell stack, so it runs on the cluster
in `.venv-sc` behind the `embpy-sc` kernel -- not in `.venv` or `.venv-test`,
neither of which has scib, cell-eval, helical or arc-state.

Two datasets must be staged into `docs/notebooks/data/` first, because `pertpy`
is deliberately absent from `.venv-sc` (adding it risks moving `numpy`/`scanpy`
in an environment that took real work to resolve):

```bash
cd docs/notebooks/data
curl -sSL -o scIBPancreas.h5ad https://ndownloader.figshare.com/files/46763269
curl -sSL -o norman2019.h5ad \
  https://scverse-exampledata.s3.eu-west-1.amazonaws.com/pertpy/norman_2019.h5ad
```

Then execute on an H100. **Not** `interactive_gpu_p` -- those nodes are V100s
(sm_70) and `.venv-sc`'s torch is a CUDA 13 build with no kernels for them, so
`torch.cuda.is_available()` returns True and kernel launches then fail:

```bash
sbatch .job_tmp/exec_cells.sbatch     # gpu_p, gpu:h100:1, 64G, 12h
```

Runtime is dominated by scIB, whose silhouette and LISI metrics are quadratic in
cell count, across a ten-embedding roster. `N_CELLS` in part 1 is the knob; 1,400
keeps it near an hour.

This directory is tracked. Intermediate JSON is not -- if you see
`cells_part*.json` in the tree, something wrote them outside a `mktemp` dir.
