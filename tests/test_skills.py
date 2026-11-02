"""Check the Agent Skills in ``skills/`` against the code they describe.

A skill is prose an assistant follows literally, so a wrong parameter name, a
model the registry does not have, or a result field the tool never returns
sends it down a dead end. These tests parse every SKILL.md and verify:

* the frontmatter is valid for Claude Code / the Agent Skills format;
* every example tool call names a registered MCP tool, uses only parameters
  that tool has, passes allowed values, and (for compute_embeddings and
  annotate_metadata) a model / annotation type valid for the stated modality;
* every identifier quoted in backticks exists: a tool, parameter, value or
  result field in the MCP tool source, a model key, or a reviewed extra.
"""

from __future__ import annotations

import ast
import inspect
import re
import typing
from pathlib import Path

import pytest

from embpy.mcp import tools as mcp_tools

SKILLS_DIR = Path(__file__).resolve().parents[1] / "skills"
SKILL_FILES = sorted(SKILLS_DIR.glob("*/SKILL.md"))

# The tools create_server registers (see embpy/mcp/server.py).
TOOLS = {
    name: getattr(mcp_tools, name)
    for name in (
        "inspect_data",
        "resolve_identifiers",
        "preprocess_counts",
        "list_model_registry",
        "compute_embeddings",
        "annotate_metadata",
        "compute_umap",
        "benchmark_scib",
        "benchmark_perturbations",
    )
}

# Backticked words that are neither in the tool source nor model keys, each checked by hand:
# skill names, AnnData slots, dataset columns used in examples, external package names,
# and values that come from embpy internals the tools pass through.
EXTRA_KNOWN = {
    "embpy",
    "embpy-mcp",
    "embpy-single-cell-workflow",
    "obs_names",
    "var_names",
    ".X",
    ".obs",
    ".obsm",
    ".varm",
    ".uns",
    ".h5ad",
    "X_pca",
    "X_scvi",
    "X_umap_pca",
    "cell_type",
    "batch",
    "target_gene",
    "gene_symbol",
    "smiles",
    "ENSMUSG...",
    "TP53",
    "Trp53",
    "Egfr",
    "P53",
    "AIM1",
    "ModelOOMError",
    "ImportError",
    "ValueError",
    "partial_success",
    "scvi-tools",
    "total",
}


def _frontmatter(text: str) -> dict[str, str]:
    m = re.match(r"^---\n(.*?)\n---\n", text, re.S)
    assert m, "SKILL.md must start with a --- frontmatter block"
    fields = {}
    for line in m.group(1).splitlines():
        key, _, value = line.partition(":")
        fields[key.strip()] = value.strip()
    return fields


def _tool_calls(text: str):
    for block in re.findall(r"```python\n(.*?)```", text, re.S):
        for node in ast.walk(ast.parse(block)):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                yield node


def _allowed_values(annotation) -> set | None:
    """Literal values allowed by an annotation, or None if it is not Literal-typed."""
    values: set = set()
    stack = [annotation]
    found = False
    while stack:
        ann = stack.pop()
        origin = typing.get_origin(ann)
        if origin is typing.Literal:
            values.update(typing.get_args(ann))
            found = True
        elif origin is not None:
            stack.extend(typing.get_args(ann))
    return values if found else None


def _known_names() -> set[str]:
    from embpy.embedder_registry.flat import MODEL_REGISTRY
    from embpy.models.singlecell_models import list_singlecell_models
    from embpy.pp.static_embeddings import static_embedding_keys

    names: set[str] = set(TOOLS) | set(MODEL_REGISTRY) | set(list_singlecell_models()) | set(static_embedding_keys("gene"))
    names |= set(mcp_tools.MODEL_ALIASES)
    for fn in TOOLS.values():
        names |= set(inspect.signature(fn).parameters)
    source = Path(mcp_tools.__file__).read_text(encoding="utf-8")
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            names.add(node.value)
    return names | EXTRA_KNOWN


def test_skills_exist():
    assert {p.parent.name for p in SKILL_FILES} >= {
        "embpy-single-cell-workflow",
        "embpy-entity-embeddings",
        "embpy-identifiers-and-annotation",
    }


@pytest.mark.parametrize("path", SKILL_FILES, ids=lambda p: p.parent.name)
def test_frontmatter(path):
    meta = _frontmatter(path.read_text(encoding="utf-8"))
    assert meta.get("name") == path.parent.name
    assert re.fullmatch(r"[a-z0-9]+(-[a-z0-9]+)*", meta["name"]) and len(meta["name"]) <= 64
    description = meta.get("description", "")
    assert 50 <= len(description) <= 1024
    assert "<" not in description and ">" not in description
    assert "Use when" in description, "the description should say when the skill applies"


@pytest.mark.parametrize("path", SKILL_FILES, ids=lambda p: p.parent.name)
def test_example_tool_calls_match_the_tools(path):
    from embpy.mcp.tools import _ANNOTATION_SOURCES, _validate_embedding_model

    calls = list(_tool_calls(path.read_text(encoding="utf-8")))
    assert calls, "each skill should show at least one tool call"
    for call in calls:
        name = call.func.id
        assert name in TOOLS, f"{name} is not an embpy MCP tool"
        params = inspect.signature(TOOLS[name]).parameters
        hints = typing.get_type_hints(TOOLS[name])
        kwargs = {}
        for kw in call.keywords:
            assert kw.arg in params, f"{name}() has no parameter {kw.arg!r}"
            value = ast.literal_eval(kw.value)
            kwargs[kw.arg] = value
            allowed = _allowed_values(hints.get(kw.arg))
            if allowed is not None:
                for v in value if isinstance(value, list) else [value]:
                    assert v in allowed, f"{name}({kw.arg}={v!r}) is not one of {sorted(map(str, allowed))}"
        required = [p for p, spec in params.items() if spec.default is inspect.Parameter.empty]
        assert set(required) <= set(kwargs), f"{name}() call is missing {set(required) - set(kwargs)}"

        if name == "compute_embeddings":
            modality = kwargs.get("entity_type", "single_cell")
            models = kwargs.get("models", "pca")
            for model in [models] if isinstance(models, str) else models:
                err = _validate_embedding_model(modality, mcp_tools._canonical_model_name(model))
                assert err is None, err
        if name == "annotate_metadata":
            supported = _ANNOTATION_SOURCES[kwargs["entity_type"]]
            types = kwargs.get("annotation_types")
            for t in [types] if isinstance(types, str) else types:
                assert t in supported, (kwargs["entity_type"], t)
            for s in kwargs.get("sources", []):
                assert any(s in supported[t] for t in ([types] if isinstance(types, str) else types)), s


@pytest.mark.parametrize("path", SKILL_FILES, ids=lambda p: p.parent.name)
def test_quoted_identifiers_exist(path):
    known = _known_names()
    text = re.sub(r"```.*?```", "", path.read_text(encoding="utf-8"), flags=re.S)
    unknown = []
    for token in re.findall(r"`([^`\n]+)`", text):
        # Code spans that are expressions, paths or commands are checked by reading, not here.
        if not re.fullmatch(r"[A-Za-z_.][A-Za-z0-9_.\-]*", token):
            continue
        # A nested result field (expression_nature.detected_nature) is checked part by part.
        parts = [token] if token.startswith(".") else token.split(".")
        if not all(part in known for part in parts):
            unknown.append(token)
    assert not unknown, f"backticked names not found in the tools, registry or reviewed extras: {unknown}"


def test_annotation_table_matches_the_tool():
    """The annotation table in the identifiers skill lists exactly what annotate_metadata supports."""
    from embpy.mcp.tools import _ANNOTATION_SOURCES

    text = (SKILLS_DIR / "embpy-identifiers-and-annotation" / "SKILL.md").read_text(encoding="utf-8")
    for entity, types in _ANNOTATION_SOURCES.items():
        row = next(line for line in text.splitlines() if line.startswith(f"| `{entity}` |"))
        listed = set(re.findall(r"`([a-z_]+)`", row.split("|")[2]))
        assert listed == set(types), (entity, listed, set(types))
        for sources in types.values():
            for source in sources:
                assert source in row, (entity, source)


def test_modality_table_matches_the_tool():
    from embpy.mcp.tools import _EMBED_ROUTES

    text = (SKILLS_DIR / "embpy-entity-embeddings" / "SKILL.md").read_text(encoding="utf-8")
    rows = {m.group(1) for m in re.finditer(r"^\| `([a-z_]+)` \|", text, re.M)} - {"entity_type", "error_type"}
    assert rows == set(_EMBED_ROUTES) - {"single_cell"}
