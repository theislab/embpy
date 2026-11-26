# Agent Skills

The MCP server gives an assistant the *tools*. The Agent Skills in
[`skills/`](https://github.com/theislab/embpy/tree/main/skills) tell it *how to use them*:
which tool to call first, which arguments matter, and how to read each result. Claude
loads a skill when a request matches its description.

| Skill | Use it for |
|---|---|
| `embpy-single-cell-workflow` | inspect -> preprocess -> embed cells -> UMAP -> scIB, and perturbation benchmarks |
| `embpy-entity-embeddings` | gene, perturbation, DNA, protein, molecule, text and morphology embeddings from a chosen column |
| `embpy-identifiers-and-annotation` | gene and compound identifier mapping (species rules, ambiguous aliases) and annotation |
| `embpy-similarity-search` | nearest neighbours and similarity rankings of genes, perturbations or compounds |
| `embpy-embedding-clustering` | Leiden / k-means clustering, agreement with known labels, cluster enrichment |
| `embpy-compare-embedding-models` | CKA, TSI/QSI, mutual-kNN and kNN-overlap comparisons between models |
| `embpy-embedding-visualization` | UMAP / t-SNE / PCA plots, embedding grids, label-separation diagnostics |
| `embpy-perturbation-screen-analysis` | CRISPR / Perturb-seq screens: phenotypic activity (mAP) and phenocopies |
| `embpy-compound-screen-analysis` | compound screens: names -> SMILES, fingerprints, RDKit and ChEMBL annotation |
| `embpy-perturbation-prediction-metrics` | scoring predicted vs observed perturbation responses against a baseline |
| `embpy-embedding-task-benchmark` | which embedding best predicts a readout (IC50, expression) |
| `embpy-cross-species-proteins` | ortholog embeddings across species vs sequence identity |
| `embpy-variant-effects` | SNP effect signals from DNA language models, single variants and VCFs |
| `embpy-mcp-setup` | installing, connecting and troubleshooting the MCP server |
| `embpy-fasta-embeddings` | embedding every sequence in a FASTA/FASTQ file, keeping header annotations |
| `embpy-export-and-reuse-embeddings` | NPZ/CSV/Zarr export with provenance, reading exports back, common dimensions |
| `embpy-layer-selection` | scoring every model layer against the user's labels and picking one |
| `embpy-differential-expression` | DE genes per perturbation / cluster vs control or rest, clean tables |
| `embpy-embedding-quality-checks` | NaN / constant / duplicate checks, collapse detection, cross-model comparison |

The first three skills drive the MCP tools. The other sixteen combine the tools with embpy's
Python API (`embpy.tl`, `embpy.pl`), which the assistant runs through its shell.

## Installing

The repository is a Claude Code plugin: `.claude-plugin/plugin.json` registers the MCP
server, and `skills/` holds the skills. Claude Code loads it from a folder on disk, a git clone
of the repository:

```bash
git clone https://github.com/theislab/embpy.git
cd embpy
# Until the MCP server is released, it is on the branch of pull request #52:
git fetch origin pull/52/head:mcp
git checkout mcp
# Later, to update that branch: git pull origin pull/52/head
```

The plugin starts the server through `scripts/embpy-mcp-launcher` (`embpy-mcp-launcher.cmd`
on Windows). It uses the first of these Pythons that has embpy's MCP server and the `mcp`
package:

1. the Python named by the environment variable `EMBPY_PYTHON`;
2. the environment that `uv tool install` creates (below);
3. `python3`, then `python`, on PATH, for example from an environment activated before
   starting Claude.

Install embpy from the clone, so that the server matches the skills, in one of two ways:

```bash
# a) In your own environment (venv, conda, ...). models = PyTorch, needed by every embedding tool.
pip install -e ".[mcp,models]"
python -c "import sys; print(sys.executable)"   # the value for EMBPY_PYTHON

# b) With uv, in an environment of its own (takes a few minutes the first time)
uv tool install --python 3.12 --editable ".[mcp,models]"
```

With a), set `EMBPY_PYTHON` to that Python unless you start Claude from a shell where the
environment is active: `export EMBPY_PYTHON=/path/to/env/bin/python` (macOS, Linux),
`setx EMBPY_PYTHON C:\path\to\env\Scripts\python.exe` (Windows venv, for new terminals), or for every
session in `~/.claude/settings.json`: `{"env": {"EMBPY_PYTHON": "/path/to/env/bin/python"}}`.
Then load the plugin and check that the server connects:

```bash
claude --plugin-dir /path/to/embpy             # load for one session
claude --plugin-dir /path/to/embpy mcp list    # plugin:embpy:embpy should be connected
```

If no Python can run the server, `claude mcp list` shows it as failed with the reason and these
steps. A server also registered with `claude mcp add` gives every tool twice; keep one of them.
Other clients (Claude Desktop, Cursor) use only the MCP server; see
[Client Setup](client-setup.md).

## How they are checked

- `tests/test_skills.py` parses every `SKILL.md` and checks the frontmatter; that every example
  tool call uses a registered tool with existing parameters and allowed values (models are
  checked against the modality, annotation types and sources against the entity type); that
  every embpy Python call (`ep.tl.*`, `ep.pl.*`, `BioEmbedder` methods) exists and binds to its
  signature; and that every name quoted in backticks exists in the embpy source or model
  registry. The annotation and modality tables are compared with the tool's own tables.
- `tests/test_skill_mcp_examples.py` runs every tool call of the three MCP skills
  (single-cell workflow, entity embeddings, identifiers and annotation) as written in their
  `SKILL.md`, in order, on small generated files, with the web services replaced by recorded
  answers; any other network request fails the test. It is required in CI (job `test-mcp`).
  The two entity-embedding calls that download a model (`gene2vec`, `esm2_8M`) run only with
  `EMBPY_RUN_SKILL_EXAMPLES=1`.
- `tests/test_skill_examples.py` runs the examples of the sixteen Python-API skills, block by
  block, on small generated datasets (MCP calls go to the real tool functions). It downloads
  small models and calls web services, so it runs only with `EMBPY_RUN_SKILL_EXAMPLES=1`
  (CI job `skill-examples`, non-blocking). Blocks marked `# Requires:` (a local genome FASTA)
  are checked statically only.
- `claude plugin validate .` validates the manifest.
- The skills were run in headless Claude Code sessions (`claude -p ... --plugin-dir .`) to
  check that the right skill is chosen for a request and its tool calls are correct.

## Not covered yet

- Workflows needing a GPU or large downloads (scGPT, Geneformer, STATE, ESM-2 650M and larger,
  Enformer) were not run; the examples use small CPU models.
- The MCP skills' offline run uses generated data and recorded service answers; scVI runs
  only where scvi-tools is installed (otherwise the call is checked to report it as failed).
- There is no automated eval suite measuring how reliably each skill triggers across many
  prompts; the `evals/` folder holds two starter prompts only.
