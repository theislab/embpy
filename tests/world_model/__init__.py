"""Unit tests for the world_model package.

Each test module exercises a single component (encoder, action encoder,
dynamics, decoder, dataset, loss). They run on tiny synthetic inputs so
the suite finishes quickly on CPU.

Run with::

    pytest tests/world_model -q
"""

from __future__ import annotations

from pathlib import Path

# Pytest imports this directory as a package named ``world_model`` during
# collection, which can shadow the real editable package. Extend the package
# search path so ``import world_model.configs`` still resolves to
# ``src/world_model/world_model`` when the local editable install is absent.
__path__.append(str(Path(__file__).resolve().parents[2] / "src" / "world_model" / "world_model"))  # type: ignore[name-defined]
