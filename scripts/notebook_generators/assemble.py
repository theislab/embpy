"""Assemble the part files into docs/notebooks/proteins.ipynb."""
import ast
import json, sys
from pathlib import Path

parts = [json.loads(Path(p).read_text()) for p in sys.argv[1:-1]]
out = Path(sys.argv[-1])

# A generator that emits unparseable code is worse than one that crashes: the
# notebook assembles fine and fails on execution, minutes later. Parse every
# code cell here, where the error still points at a part file. Nested triple
# quotes inside the r""" builders are the recurring cause.
_bad = []
for pi, part in enumerate(parts, start=1):
    for ci, (kind, source) in enumerate(part):
        if kind != "code":
            continue
        try:
            ast.parse(source)
        except SyntaxError as e:
            _bad.append(f"  part {pi}, code cell {ci}, line {e.lineno}: {e.msg}\n"
                        f"    {(source.splitlines() or [''])[max(0, (e.lineno or 1) - 1)]}")
if _bad:
    raise SystemExit("refusing to assemble -- unparseable code cells:\n" + "\n".join(_bad))

cells = []
for part in parts:
    for kind, source in part:
        lines = source.splitlines(keepends=True)
        cell = {"id": f"cell-{len(cells):03d}", "cell_type": kind,
                "metadata": {}, "source": lines}
        if kind == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
        cells.append(cell)

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3 (ipykernel)",
                       "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12.0",
                          "file_extension": ".py", "mimetype": "text/x-python",
                          "nbconvert_exporter": "python",
                          "pygments_lexer": "ipython3"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
out.write_text(json.dumps(nb, indent=1) + "\n")
n_code = sum(1 for c in cells if c["cell_type"] == "code")
print(f"wrote {out}: {len(cells)} cells ({n_code} code, {len(cells)-n_code} markdown)")
