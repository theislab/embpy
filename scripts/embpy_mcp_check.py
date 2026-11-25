"""Tell embpy-mcp-launcher whether this Python can run the embpy MCP server.

The launcher runs ``<python> embpy_mcp_check.py`` for each candidate interpreter. Exit
code 0 means embpy (with its ``embpy.mcp`` server) and the ``mcp`` SDK are installed;
otherwise the reason is printed on one line. It only looks the packages up, without
importing them, so it takes a fraction of a second. The syntax stays valid for old
Pythons, so that they report their version instead of a SyntaxError.
"""

import os
import sys


def problem():
    """Return why this Python cannot run the server, or None if it can."""
    if sys.version_info < (3, 11):  # noqa: UP036 -- runs under any Python the launcher finds
        return "Python " + sys.version.split()[0] + " is too old (embpy needs 3.11 or newer)"
    import importlib.util

    spec = importlib.util.find_spec("embpy")
    if spec is None or not spec.submodule_search_locations:
        return "embpy is not installed"
    if not any(os.path.isfile(os.path.join(p, "mcp", "server.py")) for p in spec.submodule_search_locations):
        return "its embpy has no MCP server (embpy.mcp); install a newer embpy"
    if importlib.util.find_spec("mcp") is None:
        return "the mcp package is missing (it comes with the extra: embpy[mcp,models])"
    return None


if __name__ == "__main__":
    reason = problem()
    if reason:
        print(reason)
    sys.exit(1 if reason else 0)
