"""Compatibility wrapper for :mod:`world_model.scripts.sweeps.leave_one_encoder_out`."""

from __future__ import annotations

import sys
import warnings

from world_model.scripts.sweeps import leave_one_encoder_out as _impl

warnings.warn(
    "world_model.scripts.leave_one_encoder_out is deprecated; use "
    "world_model.scripts.sweeps.leave_one_encoder_out.",
    DeprecationWarning,
    stacklevel=2,
)

if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_impl.main())

sys.modules[__name__] = _impl

