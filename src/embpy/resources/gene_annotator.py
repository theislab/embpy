# Backward compatibility -- code lives in resources/gene/annotator.py
from .gene.annotator import *  # noqa: F401,F403

# `import *` skips underscore-prefixed names; re-export the internal helper
# explicitly so existing call sites (and unit tests) keep working.
from .gene.annotator import _is_ensembl_gene_id  # noqa: F401
