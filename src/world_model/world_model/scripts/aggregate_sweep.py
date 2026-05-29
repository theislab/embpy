"""Compatibility wrapper for :mod:`world_model.scripts.sweeps.aggregate_sweep`."""

from __future__ import annotations

import sys
import warnings

from world_model.scripts.sweeps import aggregate_sweep as _impl

warnings.warn(
    "world_model.scripts.aggregate_sweep is deprecated; use "
    "world_model.scripts.sweeps.aggregate_sweep.",
    DeprecationWarning,
    stacklevel=2,
)

if __name__ == "__main__":  # pragma: no cover
    _impl.main()

sys.modules[__name__] = _impl

