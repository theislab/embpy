"""decode_cells returns different scales per wrapper; that must be discoverable.

STATE returns per-gene log-probabilities, the scVI family returns NB/ZINB rates, and
PCA returns a linear inverse transform. A caller averaging decoded matrices across
wrappers would silently mix log-probabilities with rates, so every decoding wrapper
must declare its scale.
"""

from __future__ import annotations

import pytest

from embpy.models.singlecell_models import (
    Cell2SentenceWrapper,
    GeneformerWrapper,
    PCAEmbedding,
    ScGPTWrapper,
    ScVIToolsWrapper,
    SingleCellWrapper,
    StateEmbeddingWrapper,
    TranscriptFormerWrapper,
    UCEWrapper,
)

VALID_SCALES = {"log_prob", "rate", "linear"}

DECODERS = [
    (StateEmbeddingWrapper, "log_prob"),
    (ScVIToolsWrapper, "rate"),
    (PCAEmbedding, "linear"),
]

NON_DECODERS = [ScGPTWrapper, GeneformerWrapper, UCEWrapper, TranscriptFormerWrapper, Cell2SentenceWrapper]


class TestDecodeScaleDeclared:
    def test_base_default_is_none(self):
        assert SingleCellWrapper.decode_scale is None

    @pytest.mark.parametrize(("cls", "expected"), DECODERS)
    def test_decoding_wrappers_declare_their_scale(self, cls, expected):
        assert cls.decode_scale == expected, (
            f"{cls.__name__} must declare decode_scale={expected!r} so callers can tell log-probabilities from rates"
        )

    @pytest.mark.parametrize(("cls", "expected"), DECODERS)
    def test_the_three_scales_are_distinct(self, cls, expected):
        """The whole hazard is that one method name hides three scales."""
        others = {e for c, e in DECODERS if c is not cls}
        assert expected not in others

    @pytest.mark.parametrize("cls", NON_DECODERS)
    def test_non_decoders_declare_no_scale(self, cls):
        assert cls.decode_scale is None


class TestConsistencyWithSupportsDecode:
    """decode_scale and supports_decode must not drift apart."""

    @pytest.mark.parametrize(("cls", "_expected"), DECODERS)
    def test_declaring_a_scale_implies_supports_decode(self, cls, _expected):
        assert cls.supports_decode is True

    @pytest.mark.parametrize("cls", NON_DECODERS)
    def test_no_scale_implies_no_decode_support(self, cls):
        assert cls.supports_decode is False

    def test_every_decoding_wrapper_in_the_module_declares_a_scale(self):
        """Catches a future wrapper that implements decode_cells but forgets the scale."""
        import inspect

        from embpy.models import singlecell_models as mod

        offenders = []
        for _name, obj in inspect.getmembers(mod, inspect.isclass):
            if not issubclass(obj, SingleCellWrapper) or obj is SingleCellWrapper:
                continue
            if getattr(obj, "supports_decode", False) and obj.decode_scale not in VALID_SCALES:
                offenders.append(obj.__name__)
        assert not offenders, f"decoding wrappers missing decode_scale: {offenders}"
