"""Tests for the Claude Code plugin and for what the MCP server exposes.

* ``.claude-plugin/plugin.json`` starts the server through ``scripts/embpy-mcp-launcher``
  (``embpy-mcp-launcher.cmd`` on Windows). The launcher picks the Python that has embpy
  and, when there is none, tells the MCP client what to do.

The launcher tests run the real scripts against fake interpreters: the POSIX script
wherever ``sh`` exists (Linux, macOS, Git for Windows), the ``.cmd`` script on Windows.
"""

from __future__ import annotations

import asyncio
import json
import os
import queue
import shutil
import subprocess
import sys
import threading
import venv
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / ".claude-plugin" / "plugin.json"
LAUNCHER = ROOT / "scripts" / "embpy-mcp-launcher"
LAUNCHER_CMD = ROOT / "scripts" / "embpy-mcp-launcher.cmd"
WINDOWS = sys.platform == "win32"
SH = shutil.which("sh")


def _request(request_id) -> str:
    """The initialize request as Claude Code writes it: one line, the id last."""
    params = {"protocolVersion": "2025-11-25", "capabilities": {}, "clientInfo": {"name": "test", "version": "0"}}
    return json.dumps({"method": "initialize", "params": params, "jsonrpc": "2.0", "id": request_id}) + "\n"


# Claude Code 2.1.322 may first probe a server with server/discover (the id second) and,
# after an error, send initialize on the same connection.
_DISCOVER = (
    json.dumps(
        {
            "jsonrpc": "2.0",
            "id": "server-discover-probe-1",
            "method": "server/discover",
            "params": {"_meta": {"io.modelcontextprotocol/clientInfo": {"name": "claude-code", "version": "2"}}},
        },
        separators=(",", ":"),
    )
    + "\n"
)
_NOTIFICATION = '{"jsonrpc":"2.0","method":"notifications/cancelled","params":{"requestId":1}}\n'


def _converse(argv: list[str], env: dict) -> tuple[list[dict], bool, int]:
    """Send a discover probe, a notification and initialize, each after the previous reply.

    Returns the parsed replies, whether the launcher was still waiting for the client
    afterwards, and its exit code once stdin is closed.
    """
    process = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
    lines: queue.Queue = queue.Queue()
    threading.Thread(
        target=lambda: [lines.put(line) for line in iter(process.stdout.readline, b"")], daemon=True
    ).start()
    threading.Thread(target=process.stderr.read, daemon=True).start()
    replies = []
    try:
        for message in (_DISCOVER, _NOTIFICATION, _request(0)):
            process.stdin.write(message.encode())
            process.stdin.flush()
            if '"id"' in message:
                replies.append(json.loads(lines.get(timeout=60)))
        waiting = process.poll() is None
        process.stdin.close()
        return replies, waiting, process.wait(timeout=60)
    finally:
        if process.poll() is None:
            process.kill()


def _require_mcp():
    """Skip without the mcp package, unless EMBPY_REQUIRE_MCP=1 (set by the MCP CI job)."""
    if os.environ.get("EMBPY_REQUIRE_MCP") == "1":
        import mcp  # noqa: F401
    else:
        pytest.importorskip("mcp", reason="mcp not installed")


# ---------------------------------------------------------------------------
# Plugin manifest and launcher
# ---------------------------------------------------------------------------


def test_manifest_starts_the_server_through_the_launcher():
    server = json.loads(MANIFEST.read_text(encoding="utf-8"))["mcpServers"]["embpy"]
    # No extension: Claude Code runs the shell script on macOS and Linux, and on Windows
    # it finds embpy-mcp-launcher.cmd through PATHEXT.
    assert server["command"] == "${CLAUDE_PLUGIN_ROOT}/scripts/embpy-mcp-launcher"
    assert not server.get("args")
    script = LAUNCHER.read_bytes()
    assert script.startswith(b"#!/bin/sh\n") and b"\r" not in script
    # cmd.exe can miss labels in a batch file with LF line endings.
    assert b"\r\n" in LAUNCHER_CMD.read_bytes() and b"\n" not in LAUNCHER_CMD.read_bytes().replace(b"\r\n", b"")
    if not WINDOWS:
        assert os.access(LAUNCHER, os.X_OK)


@pytest.mark.parametrize(
    "path", ["plugin.json", "graders", "scratch/build_tutorial_nb.py", "tutorials/mcp_agent_workflows.ipynb"]
)
def test_no_stray_plugin_files(path):
    # A second manifest at the repository root, a copy of evals/graders, a duplicate
    # tutorial notebook and the script that generated it.
    assert not (ROOT / path).exists()


FAKE_PYTHON_SH = r"""#!/bin/sh
# Fake interpreter: logs how it was called; passes the launcher's check if "$0.ok" exists.
printf '%s\n' "$0 $*" >> "$FAKE_LOG"
case $1 in
  *embpy_mcp_check.py)
    [ -f "$0.ok" ] && exit 0
    echo "embpy is not installed"
    exit 1 ;;
esac
exit 0
"""

FAKE_UV_SH = r"""#!/bin/sh
printf '%s\n' "uv $*" >> "$FAKE_LOG"
if [ "$1" = tool ] && [ "$2" = dir ]; then printf '%s\n' "$FAKE_UV_TOOL_DIR"; fi
exit 0
"""


class _ShWorld:
    """Fake interpreters and a fake uv for the POSIX launcher, with nothing else on PATH."""

    def __init__(self, tmp_path: Path):
        self.tmp = tmp_path
        self.bin = tmp_path / "bin"
        self.bin.mkdir()
        self.log = tmp_path / "calls.log"
        self.env = {"PATH": str(self.bin), "FAKE_LOG": self.log.as_posix(), "HOME": str(tmp_path)}
        for key in ("SYSTEMROOT", "WINDIR", "TEMP", "TMP"):  # Git for Windows' sh needs them
            if key in os.environ:
                self.env[key] = os.environ[key]

    def python(self, path: Path, *, has_embpy: bool) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(FAKE_PYTHON_SH.encode())
        path.chmod(0o755)
        if has_embpy:
            Path(f"{path}.ok").touch()
        return path

    def uv(self, tool_dir: Path) -> None:
        path = self.bin / "uv"
        path.write_bytes(FAKE_UV_SH.encode())
        path.chmod(0o755)
        self.env["FAKE_UV_TOOL_DIR"] = tool_dir.as_posix()

    def run(self, *args: str, request: str = _request(7)) -> subprocess.CompletedProcess:
        return subprocess.run(
            [SH, LAUNCHER.as_posix(), *args], input=request, capture_output=True, text=True, env=self.env, timeout=60
        )

    def server_starts(self) -> list[str]:
        """Log lines of the interpreters that were asked to start the server."""
        lines = self.log.read_text().splitlines() if self.log.exists() else []
        return [line for line in lines if " -m embpy.mcp.server" in line]


needs_sh = pytest.mark.skipif(SH is None, reason="needs a POSIX sh")


@needs_sh
def test_launcher_uses_embpy_python_first(tmp_path):
    world = _ShWorld(tmp_path)
    explicit = world.python(tmp_path / "explicit" / "bin" / "python", has_embpy=True)
    world.python(world.bin / "python3", has_embpy=True)
    world.uv(tmp_path / "tools")
    world.python(tmp_path / "tools" / "embpy" / "bin" / "python", has_embpy=True)
    world.env["EMBPY_PYTHON"] = explicit.as_posix()

    result = world.run("--version")

    assert result.returncode == 0, result.stderr
    (start,) = world.server_starts()
    assert start.endswith("explicit/bin/python -m embpy.mcp.server --version"), start


@needs_sh
def test_launcher_stops_at_an_unusable_embpy_python(tmp_path):
    world = _ShWorld(tmp_path)
    explicit = world.python(tmp_path / "explicit" / "bin" / "python", has_embpy=False)
    world.python(world.bin / "python3", has_embpy=True)
    world.env["EMBPY_PYTHON"] = explicit.as_posix()

    result = world.run()

    assert result.returncode == 1
    assert world.server_starts() == []  # no silent fallback to python3
    reply = json.loads(result.stdout)
    assert reply["id"] == 7
    assert "EMBPY_PYTHON" in reply["error"]["message"]
    assert "embpy is not installed" in reply["error"]["message"]
    assert "EMBPY_PYTHON" in result.stderr


@needs_sh
def test_launcher_uses_the_uv_tool_environment(tmp_path):
    world = _ShWorld(tmp_path)
    world.uv(tmp_path / "tools")
    world.python(tmp_path / "tools" / "embpy" / "bin" / "python", has_embpy=True)
    world.python(world.bin / "python3", has_embpy=True)

    result = world.run()

    assert result.returncode == 0, result.stderr
    (start,) = world.server_starts()
    assert start.endswith("tools/embpy/bin/python -m embpy.mcp.server"), start


@needs_sh
def test_launcher_falls_back_to_a_python_on_path_that_has_embpy(tmp_path):
    world = _ShWorld(tmp_path)
    world.uv(tmp_path / "no-tools")  # uv installed, but no embpy tool environment
    world.python(world.bin / "python3", has_embpy=False)
    world.python(world.bin / "python", has_embpy=True)

    result = world.run()

    assert result.returncode == 0, result.stderr
    (start,) = world.server_starts()
    assert start.endswith("bin/python -m embpy.mcp.server"), start


@needs_sh
@pytest.mark.parametrize("request_id", [7, "req-1"])
def test_launcher_tells_the_client_what_to_do_without_a_python(tmp_path, request_id):
    world = _ShWorld(tmp_path)
    world.uv(tmp_path / "no-tools")
    world.python(world.bin / "python3", has_embpy=False)

    result = world.run(request=_request(request_id))

    assert result.returncode == 1
    assert world.server_starts() == []
    reply = json.loads(result.stdout)  # the answer to the client's initialize request
    assert reply["jsonrpc"] == "2.0" and reply["id"] == request_id
    message = reply["error"]["message"]
    for hint in ("EMBPY_PYTHON", "uv tool install", 'embpy[mcp,models]"', "embpy is not installed"):
        assert hint in message, (hint, message)
    assert "EMBPY_PYTHON" in result.stderr


@needs_sh
def test_launcher_answers_a_discover_probe_and_then_initialize(tmp_path):
    world = _ShWorld(tmp_path)

    replies, waiting, code = _converse([SH, LAUNCHER.as_posix()], world.env)

    assert [reply["id"] for reply in replies] == ["server-discover-probe-1", 0]
    assert all("EMBPY_PYTHON" in reply["error"]["message"] for reply in replies)
    assert waiting  # exits when the client closes stdin, so that no reply is lost
    assert code == 1


# The Windows launcher looks interpreters up as .exe files, so its fakes are real Python
# environments whose interpreter only logs how it was called (a .pth hook runs first).
FAKE_HOOK = r"""
import os, sys
_argv = getattr(sys, "orig_argv", sys.argv)
with open(os.environ["FAKE_LOG"], "a", encoding="utf-8") as _log:
    _log.write(" ".join([sys.prefix, *_argv[1:]]) + "\n")
if any(a.endswith("embpy_mcp_check.py") for a in _argv):
    _ok = os.path.exists(os.path.join(sys.prefix, "has_embpy"))
    if not _ok:
        print("embpy is not installed", flush=True)
    os._exit(0 if _ok else 1)
os._exit(0)
"""

FAKE_UV_CMD = "\r\n".join(
    [
        "@echo off",
        '>>"%FAKE_LOG%" echo uv %*',
        'if /i "%~1"=="tool" if /i "%~2"=="dir" echo %FAKE_UV_TOOL_DIR%',
        "exit /b 0",
        "",
    ]
)


@pytest.fixture(scope="module")
def fake_env_template(tmp_path_factory):
    path = tmp_path_factory.mktemp("fake") / "env"
    venv.EnvBuilder(with_pip=False).create(path)
    site_packages = path / "Lib" / "site-packages"
    (site_packages / "_fake_python.py").write_text(FAKE_HOOK, encoding="utf-8")
    (site_packages / "_fake_python.pth").write_text("import _fake_python\n", encoding="utf-8")
    return path


class _CmdWorld:
    """Fake environments and a fake uv for the Windows launcher, with nothing else on PATH."""

    def __init__(self, tmp_path: Path, template: Path):
        self.tmp = tmp_path
        self.template = template
        self.bin = tmp_path / "bin"
        self.bin.mkdir()
        self.log = tmp_path / "calls.log"
        self.path = [self.bin]
        self.env = {
            "SystemRoot": os.environ["SystemRoot"],
            "COMSPEC": os.environ["COMSPEC"],
            "PATHEXT": ".COM;.EXE;.BAT;.CMD",
            "TEMP": str(tmp_path),
            "TMP": str(tmp_path),
            "FAKE_LOG": str(self.log),
        }

    def environment(self, path: Path, *, has_embpy: bool, on_path: bool = False) -> Path:
        shutil.copytree(self.template, path)
        if has_embpy:
            (path / "has_embpy").touch()
        if on_path:
            self.path.append(path / "Scripts")
        return path

    def uv(self, tool_dir: Path) -> None:
        (self.bin / "uv.cmd").write_bytes(FAKE_UV_CMD.encode())
        self.env["FAKE_UV_TOOL_DIR"] = str(tool_dir)

    def run(self, *args: str, request: str = _request(7)) -> subprocess.CompletedProcess:
        env = {**self.env, "PATH": os.pathsep.join(map(str, self.path))}
        return subprocess.run(
            [str(LAUNCHER_CMD), *args], input=request, capture_output=True, text=True, env=env, timeout=120
        )

    def server_starts(self) -> list[str]:
        lines = self.log.read_text(encoding="utf-8").splitlines() if self.log.exists() else []
        return [line for line in lines if " -m embpy.mcp.server" in line]


needs_windows = pytest.mark.skipif(not WINDOWS, reason="the .cmd launcher runs on Windows")


@needs_windows
def test_windows_launcher_uses_embpy_python_first(tmp_path, fake_env_template):
    world = _CmdWorld(tmp_path, fake_env_template)
    explicit = world.environment(tmp_path / "explicit", has_embpy=True)
    world.environment(tmp_path / "onpath", has_embpy=True, on_path=True)
    world.uv(tmp_path / "tools")
    world.environment(tmp_path / "tools" / "embpy", has_embpy=True)
    world.env["EMBPY_PYTHON"] = str(explicit / "Scripts" / "python.exe")

    result = world.run("--version")

    assert result.returncode == 0, result.stderr
    (start,) = world.server_starts()
    assert start == f"{explicit} -m embpy.mcp.server --version"


@needs_windows
def test_windows_launcher_stops_at_an_unusable_embpy_python(tmp_path, fake_env_template):
    world = _CmdWorld(tmp_path, fake_env_template)
    explicit = world.environment(tmp_path / "explicit", has_embpy=False)
    world.environment(tmp_path / "onpath", has_embpy=True, on_path=True)
    world.env["EMBPY_PYTHON"] = str(explicit / "Scripts" / "python.exe")

    result = world.run()

    assert result.returncode == 1
    assert world.server_starts() == []
    reply = json.loads(result.stdout)
    assert reply["id"] == 7
    assert "EMBPY_PYTHON" in reply["error"]["message"]
    assert "embpy is not installed" in reply["error"]["message"]
    assert "EMBPY_PYTHON" in result.stderr


@needs_windows
def test_windows_launcher_uses_the_uv_tool_environment(tmp_path, fake_env_template):
    world = _CmdWorld(tmp_path, fake_env_template)
    world.uv(tmp_path / "tools")
    tool_env = world.environment(tmp_path / "tools" / "embpy", has_embpy=True)
    world.environment(tmp_path / "onpath", has_embpy=True, on_path=True)

    result = world.run()

    assert result.returncode == 0, result.stderr
    (start,) = world.server_starts()
    assert start == f"{tool_env} -m embpy.mcp.server"


@needs_windows
def test_windows_launcher_falls_back_to_a_python_on_path_that_has_embpy(tmp_path, fake_env_template):
    world = _CmdWorld(tmp_path, fake_env_template)
    world.uv(tmp_path / "no-tools")
    on_path = world.environment(tmp_path / "onpath", has_embpy=True, on_path=True)

    result = world.run()

    assert result.returncode == 0, result.stderr
    (start,) = world.server_starts()
    assert start == f"{on_path} -m embpy.mcp.server"


@needs_windows
@pytest.mark.parametrize("request_id", [7, "req-1"])
def test_windows_launcher_tells_the_client_what_to_do_without_a_python(tmp_path, fake_env_template, request_id):
    world = _CmdWorld(tmp_path, fake_env_template)
    world.uv(tmp_path / "no-tools")
    world.environment(tmp_path / "onpath", has_embpy=False, on_path=True)

    result = world.run(request=_request(request_id))

    assert result.returncode == 1
    assert world.server_starts() == []
    reply = json.loads(result.stdout)
    assert reply["jsonrpc"] == "2.0" and reply["id"] == request_id
    message = reply["error"]["message"]
    for hint in ("EMBPY_PYTHON", "uv tool install", 'embpy[mcp,models]"', "embpy is not installed"):
        assert hint in message, (hint, message)
    assert "EMBPY_PYTHON" in result.stderr


@needs_windows
def test_windows_launcher_answers_a_discover_probe_and_then_initialize(tmp_path, fake_env_template):
    world = _CmdWorld(tmp_path, fake_env_template)
    env = {**world.env, "PATH": os.pathsep.join(map(str, world.path))}

    # The message holds Windows paths, so this also checks that backslashes are escaped.
    replies, waiting, code = _converse([str(LAUNCHER_CMD)], env)

    assert [reply["id"] for reply in replies] == ["server-discover-probe-1", 0]
    assert all(str(ROOT) in reply["error"]["message"] for reply in replies)
    assert waiting
    assert code == 1


def _platform_launcher() -> str:
    if WINDOWS:
        return str(LAUNCHER_CMD)
    return str(LAUNCHER)


def _client_run(env: dict, scenario):
    from mcp import StdioServerParameters
    from mcp.client.session import ClientSession
    from mcp.client.stdio import stdio_client

    async def main():
        params = StdioServerParameters(command=_platform_launcher(), args=[], env=env)
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                return await scenario(session)

    return asyncio.run(main())


def test_launcher_starts_a_working_server():
    _require_mcp()
    env = {**os.environ, "EMBPY_PYTHON": sys.executable, "PYTHONUNBUFFERED": "1"}
    if not WINDOWS:
        assert os.access(LAUNCHER, os.X_OK)

    async def scenario(session):
        init = await session.initialize()
        tools = await session.list_tools()
        return init, {t.name for t in tools.tools}

    init, names = _client_run(env, scenario)
    info = getattr(init, "server_info", None) or init.serverInfo
    assert info.name == "embpy-mcp"
    assert "inspect_data" in names


def test_launcher_error_reaches_the_client(tmp_path):
    _require_mcp()
    empty = tmp_path / "empty"
    empty.mkdir()
    env = {"PATH": str(empty), "HOME": str(tmp_path), "TEMP": str(tmp_path), "TMP": str(tmp_path)}
    for key in ("SystemRoot", "SYSTEMROOT", "COMSPEC", "WINDIR"):
        if key in os.environ:
            env[key] = os.environ[key]

    async def scenario(session):
        await session.initialize()

    with pytest.raises(BaseException) as excinfo:  # the SDK's McpError, maybe inside an ExceptionGroup
        _client_run(env, scenario)
    assert any("EMBPY_PYTHON" in str(exc) for exc in _flatten(excinfo.value)), excinfo.value


def _flatten(exc: BaseException) -> list[BaseException]:
    found = [exc]
    for inner in getattr(exc, "exceptions", ()):
        found += _flatten(inner)
    return found
