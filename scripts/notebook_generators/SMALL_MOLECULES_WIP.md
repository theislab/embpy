# Small-molecules deep dive -- WIP handoff

Nothing is committed. All work is in the working tree on `vibe_embpy`.

## Done and verified

**Library (complete, lint-clean, tested).**

- `src/embpy/resources/molecule/chembl.py` -- new `ChEMBLAnnotator`. Covers the
  ChEMBL_37 drug record `MoleculeAnnotator` never reached: `drug`,
  `drug_indication`, `drug_warning`, `atc_class`, `molecule_synonyms`,
  `metabolism`, `molecule_form`, `target`/`target_component`/
  `protein_classification`, `similarity`, and the full `activity` field set.
  Plus derived `get_target_profile` / `build_target_profile` and
  `summarize_selectivity`.
- `MoleculeAnnotator` gains source groups `drug`, `target_profile`,
  `metabolism`, `analogs`; `sources="all"` includes the first three.
  Result lands under `result["drug"]`. `annotate_adata` writes 38 `drug_*`
  columns plus `drug_n_indications_at_limit`.
- `tl.annotate_drug_perturbations(adata, column=...)` -- new entry point.
- Exports wired: `resources/molecule/__init__.py`, `resources/__init__.py`,
  `tl/__init__.py`, `docs/api.md`. CHANGELOG updated.
- `tests/embpy/test_chembl_annotator.py` -- 117 tests. Suite: 178 pass across
  the three molecule test files. Full suite: 1924 pass; the 3 failures
  (`test_plotting.py` leiden) and 5 errors (`test_local_genome.py`) are
  pre-existing env gaps (missing igraph/leidenalg, pysam fixture), unrelated.

**Four bugs found and fixed, each verified against the live API.**

1. `get_mechanism_of_action` returned `[]` for any drug curated against a salt.
   It filtered `mechanism` on `molecule_chembl_id`; drug-level rows hang off the
   **parent**. Imatinib: 0 mechanisms before, 4 after. Indications 52 -> 134.
2. `get_mechanism_of_action` always returned `target_name == ""` -- the
   `mechanism` payload has no such field. Now resolved via `target`.
3. `_resolve_to_smiles` resolved every *name* to `None`, so all `mol_*` columns
   were empty for name-based input. PubChem renamed its SMILES properties:
   a `CanonicalSMILES` request answers under `ConnectivitySMILES`, an
   `IsomericSMILES` request under `SMILES`.
4. `DrugResolver._extract_smiles` did not know the `SMILES` key, so the direct
   name lookup always fell through to the slower CID path
   (`source="pubchem_cid"` where it should be `pubchem_name`).

### Fifth bug, found later and fixed

`build_aliases` in `src/embpy/io/_canon.py` had no `molecule` branch, so a
non-canonical SMILES could not be attached back to an AnnData indexed by it.
Fixed; `01_embed_any_model.ipynb` no longer imports RDKit to pre-canonicalise.

**`small_molecules.ipynb` cell 3 still defines a hand-rolled
`canonicalize_smiles` using RDKit.** Not patched, because that notebook is being
regenerated -- the spec routes resolution through `DrugResolver` instead. Drop
the helper when part 2 is written.

## Not done -- the notebook

Spec: `/private/tmp/claude-501/-Users-grpinto-Documents-embpy/904438ec-5ff2-4c74-9e19-5eb6fef97fc0/scratchpad/notebook_spec.md`
**Copy that file somewhere durable before the scratchpad is cleaned.**

Generator parts present: `1, 3, 4, 6` plus `6_attn` (hand-written, verified to
emit 22 cells). **Parts 2 and 5 were never written** -- the generation workflow
was stopped early.

- part 2 = compound resolution + tokenizer audit + the embedding sweep
- part 5 = annotation, the ChEMBL clinical depth (the part the notebook exists
  for)

### Renumbering still owed

`6_attn` is meant to be part 6 (attention), and the existing part 6
(downstream + molecule-only + save + "What we found") becomes part 7:

```bash
cd scripts/notebook_generators
mv small_molecules_part6.py small_molecules_part7.py
mv small_molecules_part6_attn.py small_molecules_part6.py
```

Then extend the `for n in 1 2 3 4 5 6` loop in `build_small_molecules.sh` to 7.

### Environment facts to keep

- Build/execute with `.pixi/envs/default/bin/python` (3.13.13, torch 2.10.0,
  transformers 4.48.1). Tests need `.pixi/envs/dev/bin/python` (has pytest).
- `uv sync` on this project **fails** -- `embpy[esm3]` and `embpy[helical]`
  pin incompatible transformers. Use pixi, or a hand-built venv with
  `--no-deps -e .`.
- `BioEmbedder()` needs torch+transformers even for the RDKit fingerprints:
  `embedder_registry/flat.py` imports every modality's registry.
- `molformer_base` **cannot load** here: its `trust_remote_code` module imports
  `transformers.masking_utils`, absent in 4.48.1. Root cause is two re-wraps
  deep (`ModuleNotFoundError` -> `RuntimeError` at
  `molecule_models.py:287` -> `ModelLoadError`). Part 6 turns this into an
  honest-failure section rather than pretending it works.
- `chemberta2MLM` is **768**-wide with 12 blocks, not 384 -- the two ChemBERTa
  checkpoints differ in width and tokenisation (BPE vs character-level), not
  just objective.
- `extract_attention` takes tokenised `input_ids`, not a SMILES string.

### Build and execute

```bash
scripts/notebook_generators/build_small_molecules.sh
cd docs/notebooks && jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=2400 small_molecules.ipynb
```

`cd docs/notebooks` first -- the notebook writes to `Path("outputs")`.

## Open question left on the table

`summarize_selectivity` defaults (`rank_by="best_pchembl"`, `min_measurements=1`)
report imatinib's primary target as **ERBB2** off a single 10.22 assay, against
43 ABL1 measurements with a ~7.9 median. `min_measurements=5` gives DDR1;
neither default gives ABL1. The parameters and a `Warning` docstring section
are in place, but the right *default* is a judgement call that was not made.
