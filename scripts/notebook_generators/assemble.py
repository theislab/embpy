"""Assemble the part files into docs/notebooks/proteins.ipynb."""
import json, sys
from pathlib import Path

parts = [json.loads(Path(p).read_text()) for p in sys.argv[1:-1]]
out = Path(sys.argv[-1])

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
