# Notebook generators

`docs/notebooks/proteins.ipynb` is generated, not hand-edited. To change it, edit
the part files and rebuild:

```bash
cd /Users/grpinto/Documents/embpy
SP=scripts/notebook_generators
for n in 1 2 3 4 5; do python3 $SP/proteins_part$n.py $SP/cells_part$n.json; done
python3 $SP/assemble.py $SP/cells_part{1,2,3,4,5}.json docs/notebooks/proteins.ipynb
```

Then execute it end to end before committing (cluster, ~25 min):

```bash
ssh hpc 'cd /lustre/groups/ml01/workspace/goncalo.pinto/embpy && \
  srun --partition=interactive_cpu_p --qos=interactive_cpu --cpus-per-task=8 \
       --mem=16G --time=04:00:00 bash -c "source .venv-test/bin/activate && \
  cd docs/notebooks && jupyter nbconvert --to notebook --execute --inplace \
       --ExecutePreprocessor.timeout=2400 proteins.ipynb"'
```

This directory is untracked; `git add` it if the generators should be versioned.
