# Client Setup & Configuration Guide

The `embpy` MCP server communicates over standard input/output (`stdio`) using JSON-RPC 2.0. This guide provides ready-to-copy configuration for **Claude Code**, **Claude Desktop**, **Cursor IDE**, **Windsurf**, and custom Python LLM agent frameworks.

---

## 1. Prerequisites

Install `embpy` with the `mcp` and `models` extras in a Python 3.11-3.13 virtual environment:

```bash
pip install "embpy[mcp,models]"
```

- `mcp` brings the MCP SDK, scanpy and scikit-misc; `models` brings PyTorch, which every
  embedding model needs (PCA included). On a Linux machine without a GPU, install the CPU
  build of PyTorch first: `pip install torch --index-url https://download.pytorch.org/whl/cpu`.
- `benchmark_scib` also needs `embpy[scib]`, and `benchmark_perturbations` needs `embpy[eval]`.

Check the command, and note the full path of this environment's Python for the client configuration:

```bash
embpy-mcp --help
python -c "import sys; print(sys.executable)"
```

Started by hand, `embpy-mcp` prints nothing and waits for JSON-RPC messages on stdin; that is
normal. MCP clients start it themselves.

---

## 2. Claude Code Setup

Register the server once, with the full path of the Python that has embpy installed:

```bash
# macOS / Linux
claude mcp add embpy --scope user -- /path/to/env/bin/python -m embpy.mcp.server
# Windows
claude mcp add embpy --scope user -e PYTHONUTF8=1 -- C:\path\to\env\Scripts\python.exe -m embpy.mcp.server
```

`claude mcp list` then shows `embpy` as connected, and `/mcp` inside a session lists its tools.
`--scope user` makes the server available in every project; without it, it is added to the
current project only. To load the [Agent Skills](skills.md) as well, start Claude with
`claude --plugin-dir /path/to/embpy` from an environment whose `python` has embpy installed.

---

## 3. Claude Desktop Setup

Claude Desktop looks for server definitions in its configuration file:

- **macOS:** `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Linux:** `~/.config/Claude/claude_desktop_config.json`
- **Windows:** `%APPDATA%\Claude\claude_desktop_config.json` (typically `C:\Users\<Username>\AppData\Roaming\Claude\claude_desktop_config.json`)

### Ready-to-Copy Configuration

=== "Windows (`%APPDATA%\Claude\claude_desktop_config.json`)"

    ```json
    {
      "mcpServers": {
        "embpy": {
          "command": "C:\\Users\\<Username>\\anaconda3\\envs\\myenv\\python.exe",
          "args": [
            "-m",
            "embpy.mcp.server"
          ],
          "env": {
            "PYTHONUNBUFFERED": "1",
            "PYTHONIOENCODING": "utf-8"
          }
        }
      }
    }
    ```

    > **Windows Note:** Ensure all backslashes in paths are escaped (`\\`) or forward slashes (`/`) are used.

=== "macOS / Linux (`~/.config/Claude/claude_desktop_config.json`)"

    ```json
    {
      "mcpServers": {
        "embpy": {
          "command": "/Users/<Username>/miniconda3/envs/myenv/bin/python",
          "args": [
            "-m",
            "embpy.mcp.server"
          ],
          "env": {
            "PYTHONUNBUFFERED": "1"
          }
        }
      }
    }
    ```

=== "Using Entrypoint Binary Directly"

    If your active environment is in your system `PATH`:

    ```json
    {
      "mcpServers": {
        "embpy": {
          "command": "embpy-mcp"
        }
      }
    }
    ```

After updating `claude_desktop_config.json`, fully quit and restart Claude Desktop. The tools menu will list the 9 `embpy` tools (plus 6 backward-compatible aliases).

---

## 4. Cursor IDE Setup

In Cursor, MCP servers can be configured globally in UI settings or committed directly to a workspace.

### Option A: UI Settings
1. Navigate to **Settings** &rarr; **Features** &rarr; **MCP**.
2. Click **Add New MCP Server**.
3. Configure the fields:
   - **Name:** `embpy`
   - **Type:** `command`
   - **Command:** `python -m embpy.mcp.server` (or point directly to your virtual environment Python binary).

### Option B: Workspace Configuration (`.cursor/mcp.json`)
Create or edit `.cursor/mcp.json` in your project root:

```json
{
  "mcpServers": {
    "embpy": {
      "command": "python",
      "args": [
        "-m",
        "embpy.mcp.server"
      ],
      "env": {
        "PYTHONUNBUFFERED": "1"
      }
    }
  }
}
```

If your project uses a virtual environment located at `./.venv`:

```json
{
  "mcpServers": {
    "embpy": {
      "command": "./.venv/bin/python",
      "args": [
        "-m",
        "embpy.mcp.server"
      ]
    }
  }
}
```

On Windows with a local `.venv`:

```json
{
  "mcpServers": {
    "embpy": {
      "command": ".\\.venv\\Scripts\\python.exe",
      "args": [
        "-m",
        "embpy.mcp.server"
      ]
    }
  }
}
```

---

## 5. Windsurf IDE Setup

Windsurf uses `mcp_config.json` to manage tool providers.

- **Global Path:** `~/.codeium/windsurf/mcp_config.json` (or `%USERPROFILE%\.codeium\windsurf\mcp_config.json` on Windows).
- **Workspace Path:** `.windsurf/mcp_config.json` in your repository root.

### Ready-to-Copy Configuration (`mcp_config.json`)

```json
{
  "mcpServers": {
    "embpy": {
      "command": "embpy-mcp",
      "env": {
        "PYTHONUNBUFFERED": "1"
      }
    }
  }
}
```

Alternatively, targeting an explicit Python environment:

```json
{
  "mcpServers": {
    "embpy": {
      "command": "python",
      "args": [
        "-m",
        "embpy.mcp.server"
      ],
      "env": {
        "PYTHONUNBUFFERED": "1"
      }
    }
  }
}
```

---

## 6. Custom Python / LLM Agent Frameworks

If you are developing custom agent pipelines with Python, LangChain, LlamaIndex, or the official `mcp` client SDK, you can spawn `embpy-mcp` via `stdio`:

```python
import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def run_agent():
    server_params = StdioServerParameters(
        command="python",
        args=["-m", "embpy.mcp.server"],
        env={"PYTHONUNBUFFERED": "1"},
    )
    
    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            
            # List all tools
            tools_result = await session.list_tools()
            print([t.name for t in tools_result.tools])
            
            # Call inspect_anndata
            result = await session.call_tool(
                "inspect_anndata",
                arguments={"path": "data/pbmc3k.h5ad"},
            )
            print(result.content[0].text)

asyncio.run(run_agent())
```

---

## 7. Verifying Connectivity & Diagnostics

If your client reports that tools are missing or disconnected:

1. **Verify Python Path:** Test running the command in terminal:
   ```bash
   python -m embpy.mcp.server
   ```
   The process should wait silently for JSON-RPC input without printing errors or crashing.
2. **Check Stderr Output:** The server prints diagnostic initialization messages to `stderr`. Check your client's developer console or logs to see confirmation of the 9 registered tools (plus 6 backward-compatible aliases).
3. **Encoding Issues on Windows:** If you encounter Unicode issues, ensure `PYTHONIOENCODING=utf-8` is set in the `env` block of your client config.
