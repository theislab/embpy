# Client Setup & Configuration Guide

The `embpy` MCP server communicates over standard input/output (`stdio`) using JSON-RPC 2.0. This guide provides ready-to-copy configuration files to integrate `embpy` with **Claude Desktop**, **Cursor IDE**, **Windsurf**, and custom Python LLM agent frameworks.

---

## 1. Prerequisites

First, ensure that `embpy` is installed with the `[mcp]` extra inside your Python virtual environment:

```bash
pip install "embpy[mcp]"
```

Verify that the CLI command is accessible:

```bash
embpy-mcp --help
```

---

## 2. Claude Desktop Setup

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

After updating `claude_desktop_config.json`, fully quit and restart Claude Desktop. The hammer icon in the prompt box will show the 7 `embpy` bioinformatics tools.

---

## 3. Cursor IDE Setup

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

## 4. Windsurf IDE Setup

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

## 5. Custom Python / LLM Agent Frameworks

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

## 6. Verifying Connectivity & Diagnostics

If your client reports that tools are missing or disconnected:

1. **Verify Python Path:** Test running the command in terminal:
   ```bash
   python -m embpy.mcp.server
   ```
   The process should wait silently for JSON-RPC input without printing errors or crashing.
2. **Check Stderr Output:** The server prints diagnostic initialization messages to `stderr`. Check your client's developer console or logs to see confirmation of the 7 registered tools.
3. **Encoding Issues on Windows:** If you encounter Unicode issues, ensure `PYTHONIOENCODING=utf-8` is set in the `env` block of your client config.
