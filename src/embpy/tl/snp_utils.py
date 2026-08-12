# Backward compatibility -- code lives in tl/genomics/snp_utils.py
# (re-exports private helpers too, since existing tests import e.g. `_apply_snp`
# directly from this module).
from .genomics import snp_utils as _snp_utils

globals().update({k: v for k, v in vars(_snp_utils).items() if not k.startswith("__")})
del _snp_utils
