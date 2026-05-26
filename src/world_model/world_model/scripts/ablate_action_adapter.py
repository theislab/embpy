"""Compatibility wrapper for :mod:`world_model.scripts.sweeps.ablate_action_adapter`."""

from __future__ import annotations

import sys
import warnings

from world_model.scripts.sweeps import ablate_action_adapter as _impl

warnings.warn(
    "world_model.scripts.ablate_action_adapter is deprecated; use "
    "world_model.scripts.sweeps.ablate_action_adapter.",
    DeprecationWarning,
    stacklevel=2,
)

if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_impl.main())

sys.modules[__name__] = _impl

