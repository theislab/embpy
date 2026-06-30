# Backward compatibility -- code lives in tl/genomics/snp_utils.py
from .genomics.snp_utils import *  # noqa: F401,F403

# `import *` skips underscore-prefixed names; re-export the internal helpers
# explicitly so existing call sites (and unit tests) that imported them from
# this shim keep working.
from .genomics.snp_utils import _apply_snp, _extract_context  # noqa: F401
