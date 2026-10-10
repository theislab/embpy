# Backward compatibility -- code lives in resources/gene/resolver.py
from .gene.resolver import *  # noqa: F403

# `import *` skips underscore-prefixed names; re-export the internal helpers
# explicitly so existing call sites (and unit tests) keep working.
from .gene.resolver import (  # noqa: F401
    _ensembl_get,
    _is_ensembl_id,
    _looks_like_smiles,
)
