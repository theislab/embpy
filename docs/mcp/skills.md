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

- `tests/test_skills.py` parses every `SKILL.md` and checks the frontmatter, that every example
  tool call uses a registered tool with existing parameters and allowed values (models are
  checked against the modality, annotation types and sources against the entity type), and that
  every name quoted in backticks exists in the tool code or model registry. The annotation and
  modality tables are compared with the tool's own tables.
- `claude plugin validate .` validates the manifest.
- The skills were run in headless Claude Code sessions (`claude -p ... --plugin-dir .`) on
  molecule embedding, gene identifier mapping and single-cell embedding + scIB requests.

## Not covered yet

- Workflows needing a GPU or large downloads (scGPT, Geneformer, STATE, ESM-2 650M and larger)
  were not run through the skills.
- There is no automated eval suite measuring how reliably each skill triggers across many
  prompts; the `evals/` folder holds two starter prompts only.
