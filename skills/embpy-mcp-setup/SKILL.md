---
name: embpy-mcp-setup
description: Install, connect, check and troubleshoot the embpy MCP server and its optional extras - Claude Code, Claude Desktop and Cursor configuration, missing tools, server hangs, file-not-found errors from relative paths, network warnings from Ensembl or MyGene, and ImportError for scIB, cell-eval, pyensembl or sequence models. Use when the embpy tools are missing or failing, the user asks how to set up embpy with an assistant, or a tool returns an install hint or connection warning.
---

# Setting up and troubleshooting the embpy MCP server

## Install

```bash
pip install "embpy[mcp]"
```

Optional extras, installed only when a task needs them:

| Extra | Enables |
|---|---|
| `embpy[scib]` | `benchmark_scib` |
| `embpy[eval]` | `benchmark_perturbations` (cell-eval) |
| `embpy[genome]` | offline gene lookups with pyensembl (then `pyensembl install --release 109 --species homo_sapiens`) |
| `embpy[seqmodels]` | Enformer and Borzoi |
| `embpy[pertpy]` | cell-line annotation |

## Connect

- Claude Code: `claude mcp add embpy -- python -m embpy.mcp.server`, or load this repository as a
  plugin with `claude --plugin-dir <path-to-embpy>`.
- Claude Desktop / Cursor: add a server whose `command` is the **full path** of the Python that
  has embpy installed, with `args` `["-m", "embpy.mcp.server"]` (see `docs/mcp/client-setup.md`).

The `python` that starts the server must be the environment where embpy is installed. A plain
`python` on PATH is often a different one.

## Check that it works

```python
list_model_registry(entity_type="molecule")
```

A `status: "success"` reply with `compute_device` and `optional_extras_status` means the server
runs; `optional_extras_status` shows which extras are installed.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| No embpy tools available | The server did not start: wrong Python in the config, or embpy/mcp not installed there. Run `python -m embpy.mcp.server` in a terminal with that Python and read the error. |
| Server listed but "connecting" or failed, client log says the connection timed out | The first start of a large environment (cold disk cache) can exceed the client's 30 s limit. Start the session again, or raise the limit for Claude Code with the environment variable `MCP_TIMEOUT=60000`. |
| `FileNotFoundError` for a file that exists | Relative paths resolve against the **server's** working directory, not yours. Pass absolute paths. |
| First call that loads scipy never returns (Windows, MCP SDK 1.x) | Fixed in this version: the server preloads scipy at startup. Upgrade embpy if it still happens. |
| `warning` / `action_required` mentioning Ensembl or MyGene | The lookup service was slow or unreachable; unmapped genes are not proven absent. Retry, or install `embpy[genome]` for offline lookups. |
| `ImportError` / `suggestion` with a pip command | An optional extra is missing; give the user that exact command. |
| `ModelOOMError` | Smaller model, smaller `batch_size`, or `device="cpu"`. |
| Very slow embeddings on `compute_device: "cpu"` | Large foundation models need a GPU; suggest a small model (`pca`, `morgan_fp`, `esm2_8M`) for a first pass. |

Report the exact `error_type` and `error` from the tool result to the user rather than
paraphrasing, and never retry a failing call unchanged more than once.
