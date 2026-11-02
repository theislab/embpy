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
server, and `skills/` holds the skills.

```bash
pip install "embpy[mcp]"
claude --plugin-dir /path/to/embpy      # load for one session
```

The plugin starts the server with `python -m embpy.mcp.server`, so run Claude from an
environment where that `python` has embpy installed. Other clients (Claude Desktop, Cursor)
use only the MCP server; see [Client Setup](client-setup.md).

## How they are checked

- `tests/test_skills.py` parses every `SKILL.md` and checks the frontmatter; that every example
  tool call uses a registered tool with existing parameters and allowed values (models are
  checked against the modality, annotation types and sources against the entity type); that
  every embpy Python call (`ep.tl.*`, `ep.pl.*`, `BioEmbedder` methods) exists and binds to its
  signature; and that every name quoted in backticks exists in the embpy source or model
  registry. The annotation and modality tables are compared with the tool's own tables.
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
- The examples of the first three skills reference user datasets and optional models (scVI)
  and are checked statically, not executed.
- There is no automated eval suite measuring how reliably each skill triggers across many
  prompts; the `evals/` folder holds two starter prompts only.
