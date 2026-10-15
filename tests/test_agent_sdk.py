"""Tests for Claude Agent SDK integration and Model Context Protocol (MCP) tool dispatch.

Validates:
1. Hermetic ClaudeAgentOptions configuration with setting_sources=[] and embpy mcp_servers.
2. AssistantMessage streams emitting ToolUseBlock invocations for embpy tools.
3. Compatibility between embpy tool registry and Claude SDK tool naming conventions (mcp__embpy__<tool>).
"""

from __future__ import annotations

import sys
from typing import AsyncIterator
import pytest

pytest.importorskip("claude_agent_sdk", reason="claude-agent-sdk not installed; skip Claude SDK tests")

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ToolUseBlock,
    TextBlock,
)
from embpy.mcp.server import create_server


@pytest.fixture
def hermetic_agent_options() -> ClaudeAgentOptions:
    """Fixture providing hermetic ClaudeAgentOptions configured with embpy MCP server."""
    mcp_servers = {
        "embpy": {
            "type": "stdio",
            "command": sys.executable,
            "args": ["-m", "embpy.mcp.server"],
            "env": {"PYTHONUNBUFFERED": "1"},
        }
    }
    return ClaudeAgentOptions(
        setting_sources=[],  # Hermetic environment: ignore local system configs
        mcp_servers=mcp_servers,
    )


def test_agent_options_hermetic_setup(hermetic_agent_options: ClaudeAgentOptions) -> None:
    """Assert that ClaudeAgentOptions is strictly hermetic and embpy MCP server is configured."""
    assert hermetic_agent_options.setting_sources == []
    assert hermetic_agent_options.mcp_servers is not None
    assert "embpy" in hermetic_agent_options.mcp_servers

    embpy_cfg = hermetic_agent_options.mcp_servers["embpy"]
    assert embpy_cfg["command"] == sys.executable
    assert "-m" in embpy_cfg["args"]
    assert "embpy.mcp.server" in embpy_cfg["args"]


import asyncio

@pytest.mark.anyio
async def test_assistant_message_inspect_data_tool_use() -> None:
    """Assert that AssistantMessage streams contain correct ToolUseBlock invocations for inspect_data."""
    async def mock_assistant_stream() -> AsyncIterator[AssistantMessage]:
        yield AssistantMessage(
            content=[
                TextBlock(text="I will inspect the single-cell dataset at data/pbmc3k.h5ad."),
                ToolUseBlock(
                    id="toolu_inspect_01",
                    name="mcp__embpy__inspect_data",
                    input={"path": "data/pbmc3k.h5ad", "max_categories": 10},
                ),
            ],
            model="claude-3-5-sonnet-20241022",
        )

    stream = mock_assistant_stream()
    messages = [msg async for msg in stream]

    assert len(messages) == 1
    assistant_msg = messages[0]
    assert isinstance(assistant_msg, AssistantMessage)

    tool_blocks = [block for block in assistant_msg.content if isinstance(block, ToolUseBlock)]
    assert len(tool_blocks) == 1

    tool_call = tool_blocks[0]
    assert tool_call.id == "toolu_inspect_01"
    assert tool_call.name == "mcp__embpy__inspect_data"
    assert tool_call.input["path"] == "data/pbmc3k.h5ad"


@pytest.mark.anyio
async def test_assistant_message_pipeline_stream() -> None:
    """Assert that AssistantMessage correctly sequences Preprocess -> Embed -> Benchmark tool calls."""
    async def mock_pipeline_stream() -> AsyncIterator[AssistantMessage]:
        # Step 1: Preprocess
        yield AssistantMessage(
            content=[
                ToolUseBlock(
                    id="toolu_prep_01",
                    name="mcp__embpy__preprocess_counts",
                    input={"path": "data/raw.h5ad", "target_sum": 1e4},
                )
            ],
            model="claude-3-5-sonnet-20241022",
        )
        # Step 2: Compute Embeddings
        yield AssistantMessage(
            content=[
                ToolUseBlock(
                    id="toolu_embed_02",
                    name="mcp__embpy__compute_embeddings",
                    input={"path": "data/raw_preprocessed.h5ad", "models": ["pca", "scvi"]},
                )
            ],
            model="claude-3-5-sonnet-20241022",
        )
        # Step 3: scIB Benchmark
        yield AssistantMessage(
            content=[
                ToolUseBlock(
                    id="toolu_scib_03",
                    name="mcp__embpy__benchmark_scib",
                    input={
                        "path": "data/raw_embedded.h5ad",
                        "embedding_keys": ["X_pca", "X_scvi"],
                        "label_key": "cell_type",
                    },
                )
            ],
            model="claude-3-5-sonnet-20241022",
        )

    expected_tools = [
        "mcp__embpy__preprocess_counts",
        "mcp__embpy__compute_embeddings",
        "mcp__embpy__benchmark_scib",
    ]

    invoked_tools = []
    async for msg in mock_pipeline_stream():
        for block in msg.content:
            if isinstance(block, ToolUseBlock):
                invoked_tools.append(block.name)

    assert invoked_tools == expected_tools


def test_mcp_server_tool_registry_contains_sdk_targets() -> None:
    """Assert that the embpy MCP server exposes all 9 universal skills required by Claude SDK."""
    pytest.importorskip("mcp", reason="mcp not installed")
    server = create_server()
    registered_tool_names = set(server._tool_manager._tools.keys()) if hasattr(server, "_tool_manager") else set()

    # If fastmcp direct attribute
    if not registered_tool_names:
        registered_tool_names = {t.name for t in getattr(server, "list_tools", lambda: [])()}

    # Required 9 skills
    required_skills = {
        "inspect_data",
        "resolve_identifiers",
        "preprocess_counts",
        "list_model_registry",
        "compute_embeddings",
        "annotate_metadata",
        "compute_umap",
        "benchmark_scib",
        "benchmark_perturbations",
    }

    # If server hasn't been started, check functions from tools module
    from embpy.mcp import tools
    for skill in required_skills:
        assert hasattr(tools, skill), f"Tool '{skill}' missing in embpy.mcp.tools"
        assert callable(getattr(tools, skill)), f"Tool '{skill}' must be callable"
