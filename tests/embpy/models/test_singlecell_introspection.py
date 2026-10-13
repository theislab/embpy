"""SingleCellWrapper can now reach layer/attention introspection.

SingleCellWrapper does not inherit BaseModelWrapper, so these models previously had
no route to extract_attention at all. The capability is delegated to a thin adapter
so the extraction logic stays in one tested place.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from embpy.models.singlecell_models import (
    GeneformerWrapper,
    ScGPTWrapper,
    SingleCellWrapper,
    StateEmbeddingWrapper,
    TranscriptFormerWrapper,
    UCEWrapper,
)


class _Bare(SingleCellWrapper):
    """Minimal concrete subclass; tests drive _model directly."""

    def load(self, device: str = "cpu") -> None:
        self.device = device

    def embed_cells(self, adata):
        return np.zeros((1, 4), dtype=np.float32)


class _AttnBlock(torch.nn.Module):
    def forward(self, x):
        seq = x.shape[1]
        return x, torch.softmax(torch.rand(1, 2, seq, seq), dim=-1)


class _Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([_AttnBlock() for _ in range(3)])
        self.proj = torch.nn.Linear(4, 4)  # gives the module parameters (=> a device)

    def forward(self, x):
        for layer in self.layers:
            out = layer(x)
            x = out[0] if isinstance(out, tuple) else out
        return x


class TestApiSurface:
    def test_methods_exist_on_the_base_class(self):
        for name in ("extract_attention", "extract_hidden_states", "torch_module"):
            assert hasattr(SingleCellWrapper, name), f"{name} missing"

    def test_default_has_attention_is_true(self):
        assert SingleCellWrapper.has_attention is True


class TestTorchModuleResolution:
    def test_returns_none_before_load(self):
        assert _Bare().torch_module() is None

    def test_finds_a_direct_module(self):
        w = _Bare()
        w._model = _Net()
        assert isinstance(w.torch_module(), torch.nn.Module)

    def test_finds_a_module_nested_one_level(self):
        """helical wrappers keep the network at ._model.model."""

        class _Helical:
            def __init__(self):
                self.model = _Net()

        w = _Bare()
        w._model = _Helical()
        assert isinstance(w.torch_module(), torch.nn.Module)

    def test_unresolvable_module_raises_clearly(self):
        w = _Bare()
        w._model = object()  # nothing module-like
        with pytest.raises(NotImplementedError, match="does not expose a torch module"):
            w.extract_attention(torch.zeros(1, 5, 4))


class TestDelegation:
    def test_extracts_attention_through_the_adapter(self):
        w = _Bare()
        w._model = _Net()
        w.device = "cpu"
        out = w.extract_attention(torch.rand(1, 5, 4))
        assert set(out) == {0, 1, 2}
        for tensor in out.values():
            assert tensor.shape[-1] == tensor.shape[-2] == 5

    def test_attention_free_declaration_is_respected(self):
        class _Free(_Bare):
            has_attention = False

        w = _Free()
        w._model = _Net()
        w.device = "cpu"
        with pytest.raises(NotImplementedError, match="attention-free"):
            w.extract_attention(torch.rand(1, 5, 4))

    def test_layer_subset_is_forwarded(self):
        w = _Bare()
        w._model = _Net()
        w.device = "cpu"
        assert set(w.extract_attention(torch.rand(1, 5, 4), layers=[0, 2])) == {0, 2}


class TestFeasibilityFlagsMatchTheStudy:
    """docs/attention_extraction.md records why each of these is set."""

    @pytest.mark.parametrize("cls", [ScGPTWrapper, StateEmbeddingWrapper])
    def test_fused_kernel_models_declare_no_attention(self, cls):
        assert cls.has_attention is False

    @pytest.mark.parametrize("cls", [GeneformerWrapper, UCEWrapper, TranscriptFormerWrapper])
    def test_materialising_models_keep_attention(self, cls):
        assert cls.has_attention is True
