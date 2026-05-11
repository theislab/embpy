"""Deprecation shim -- ``embpy.world_model`` has moved to ``world_model``.

The Part C package split (see ``docs/audit/package_split.md``) promoted
the world model to a top-level Python package under ``src/world_model/``.
This module is a transitional re-export so existing code paths like ::

    from embpy.world_model.training import WorldModelTrainer
    from embpy.world_model.data import build_dataloaders

keep working for one more release. Each import of ``embpy.world_model``
emits a ``DeprecationWarning``; please update to::

    from world_model.training import WorldModelTrainer
    from world_model.data import build_dataloaders

The removal target is recorded in ``CHANGELOG.md``.

Implementation note
-------------------

We deliberately use the ``sys.modules[__name__] = world_model`` swap
pattern instead of pre-creating per-submodule shim files
(``configs.py``, ``data/__init__.py``, ``training/trainer.py``, ...).
Two reasons:

1. **Lazy by design.** The prompt's "Lazy-import stays the rule"
   constraint rules out a wall of ``from world_model.<sub> import *``
   files; those would eagerly import every submodule on first access.
   The swap touches only the parent package -- every sub-module under
   ``embpy.world_model.X.Y[...]`` is resolved through Python's normal
   import machinery against ``world_model.X.Y[...]`` at the moment the
   user imports it, not before.

2. **Single source of truth.** With one file we cannot drift out of
   sync with ``world_model``'s public surface. New submodules added to
   ``world_model`` automatically become importable through the shim.
"""

from __future__ import annotations

import sys as _sys
import warnings as _warnings

_warnings.warn(
    "`embpy.world_model` has moved to the top-level `world_model` package. "
    "Update your imports (e.g. `from world_model.training import WorldModelTrainer`). "
    "This shim will be removed in the next release; see CHANGELOG.md.",
    DeprecationWarning,
    stacklevel=2,
)

import world_model as _wm  # noqa: E402

_sys.modules[__name__] = _wm
