"""One-shot codemod for the embpy / world_model split (Part C).

Rewrites two unambiguous tokens in tracked files:

  embpy.world_model  ->  world_model
  src/embpy/world_model -> src/world_model

Run from the repo root:

    python scripts/_codemod_split.py            # dry run, prints summary
    python scripts/_codemod_split.py --apply    # write changes in place

The script is intended to run once and then be deleted. It only touches
files that git already tracks, never produces partial edits, and prints
a per-file diff count so the change set is auditable.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

MODULE_RE = re.compile(r"\bembpy\.world_model\b")
PATH_RE = re.compile(r"\bsrc/embpy/world_model\b")
EXTS = {".py", ".yaml", ".yml", ".sbatch", ".sh", ".md", ".toml", ".cfg", ".ini", ".txt", ".rst", ".json"}
SKIP_DIRS = {"node_modules", "__pycache__", ".git", ".pixi", ".venv"}
SELF = Path(__file__).resolve()


def list_tracked_files(root: Path) -> list[Path]:
    out = subprocess.check_output(["git", "ls-files"], cwd=root, text=True)
    files = []
    for line in out.splitlines():
        p = Path(line)
        if any(part in SKIP_DIRS for part in p.parts):
            continue
        if p.suffix not in EXTS:
            continue
        files.append(p)
    return files


def rewrite_text(text: str) -> tuple[str, int]:
    new, n1 = MODULE_RE.subn("world_model", text)
    new, n2 = PATH_RE.subn("src/world_model", new)
    return new, n1 + n2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args(argv)

    root = Path(subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip())
    files = list_tracked_files(root)
    totals = 0
    touched = 0
    for rel in files:
        fp = root / rel
        if fp.resolve() == SELF:
            continue
        try:
            text = fp.read_text()
        except (UnicodeDecodeError, OSError):
            continue
        if "embpy.world_model" not in text and "src/embpy/world_model" not in text:
            continue
        new, n = rewrite_text(text)
        if n == 0:
            continue
        totals += n
        touched += 1
        if args.apply:
            fp.write_text(new)
            print(f"[apply] {rel}: {n} replacements")
        else:
            print(f"[dry]   {rel}: {n} replacements")
    mode = "applied" if args.apply else "would apply"
    print(f"-- {mode} {totals} replacements across {touched} files --")
    return 0


if __name__ == "__main__":
    sys.exit(main())
