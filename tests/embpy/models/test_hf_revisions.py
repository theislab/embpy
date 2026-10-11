"""Tests for the pinned revisions of ``trust_remote_code`` model loads (no network)."""

from __future__ import annotations

import ast
import importlib.machinery
import re
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
import transformers

import embpy.models as models_pkg
from embpy.embedder_registry.dna import DNA_MODELS
from embpy.embedder_registry.molecule import MOLECULE_MODELS
from embpy.models.dna_models import (
    CaduceusWrapper,
    GENALMWrapper,
    HyenaDNAWrapper,
    NucleotideTransformerV3Wrapper,
    NucleotideTransformerWrapper,
)
from embpy.models.hf_revisions import REMOTE_CODE_REVISIONS, resolve_revision
from embpy.models.molecule_models import MolformerWrapper

MOLFORMER = "ibm/MoLFormer-XL-both-10pct"
# Last MoLFormer commit that loads on transformers 4.x (later ones import create_bidirectional_mask).
MOLFORMER_SHA = "7b12d946c181a37f6012b9dc3b002275de070314"


@pytest.fixture
def hf_mocks(monkeypatch):
    """Replace ``from_pretrained`` on every transformers Auto class the wrappers call."""
    mocks = {}
    for name in ("AutoTokenizer", "AutoConfig", "AutoModel", "AutoModelForMaskedLM", "AutoModelForCausalLM"):
        mock = MagicMock(name=f"{name}.from_pretrained")
        monkeypatch.setattr(getattr(transformers, name), "from_pretrained", mock)
        mocks[name] = mock
    return mocks


def _set_transformers_version(monkeypatch, version):
    """Pretend ``version`` of transformers is installed, so tests do not depend on the version in the test env.

    Patches ``sys.modules["transformers"]`` (what ``import transformers`` in ``resolve_revision`` returns)
    rather than this module's ``transformers`` reference: transformers' own ``direct_transformers_import``
    re-executes the package, so after importing ``embpy.models`` the two can be different module objects.
    """
    monkeypatch.setattr(sys.modules["transformers"], "__version__", version)


@pytest.fixture
def transformers_4(monkeypatch):
    _set_transformers_version(monkeypatch, "4.57.6")


@pytest.fixture
def transformers_5(monkeypatch):
    _set_transformers_version(monkeypatch, "5.0.0")


def _assert_calls_use_revision(hf_mocks, repo, revision):
    """Every from_pretrained call made during load() must target ``repo`` at ``revision``."""
    called = [m for m in hf_mocks.values() if m.called]
    assert len(called) >= 2, "expected at least a tokenizer and a model load"
    for mock in called:
        for call in mock.call_args_list:
            assert call.args[0] == repo
            assert call.kwargs["trust_remote_code"] is True
            assert call.kwargs["revision"] == revision


class TestResolveRevision:
    def test_molformer_is_pinned(self, transformers_4):
        assert REMOTE_CODE_REVISIONS[MOLFORMER] == MOLFORMER_SHA
        assert resolve_revision(MOLFORMER) == MOLFORMER_SHA

    def test_molformer_pin_skipped_on_transformers_5(self, transformers_5):
        # The pinned (March 2024) files import transformers.onnx, which transformers 5 removed; main works there.
        assert resolve_revision(MOLFORMER) is None

    @pytest.mark.parametrize("version", ["4.57.6", "5.0.0"])
    def test_explicit_revision_wins(self, monkeypatch, version):
        _set_transformers_version(monkeypatch, version)
        assert resolve_revision(MOLFORMER, "main") == "main"

    @pytest.mark.parametrize("version", ["4.57.6", "5.0.0"])
    def test_other_pins_apply_on_any_transformers_version(self, monkeypatch, version):
        _set_transformers_version(monkeypatch, version)
        repo = "LongSafari/hyenadna-tiny-1k-seqlen-hf"
        assert resolve_revision(repo) == REMOTE_CODE_REVISIONS[repo]

    def test_repo_id_match_is_case_insensitive(self, transformers_4):
        assert resolve_revision(MOLFORMER.lower()) == MOLFORMER_SHA
        repo = "LongSafari/hyenadna-tiny-1k-seqlen-hf"
        assert resolve_revision(repo.swapcase()) == REMOTE_CODE_REVISIONS[repo]

    @pytest.mark.parametrize("name", ["some-org/unpinned-model", "/local/path/to/model", "", None])
    def test_unpinned_is_none(self, name):
        assert resolve_revision(name) is None

    def test_pins_are_commit_shas(self):
        for repo, sha in REMOTE_CODE_REVISIONS.items():
            assert re.fullmatch(r"[0-9a-f]{40}", sha), f"{repo} is not pinned to a 40-char commit sha"


class TestMolformerRevision:
    def test_load_passes_pinned_revision(self, hf_mocks, transformers_4):
        MolformerWrapper().load(torch.device("cpu"))

        tok = hf_mocks["AutoTokenizer"]
        tok.assert_called_once()
        assert tok.call_args.args[0] == MOLFORMER
        assert tok.call_args.kwargs["trust_remote_code"] is True
        assert tok.call_args.kwargs["revision"] == MOLFORMER_SHA

        model = hf_mocks["AutoModel"]
        model.assert_called_once()
        assert model.call_args.args[0] == MOLFORMER
        assert model.call_args.kwargs["trust_remote_code"] is True
        assert model.call_args.kwargs["revision"] == MOLFORMER_SHA

    def test_load_uses_main_on_transformers_5(self, hf_mocks, transformers_5):
        MolformerWrapper().load(torch.device("cpu"))

        assert hf_mocks["AutoTokenizer"].call_args.kwargs["revision"] is None
        assert hf_mocks["AutoModel"].call_args.kwargs["revision"] is None

    def test_revision_kwarg_overrides_pin(self, hf_mocks, transformers_4):
        MolformerWrapper(revision="abc123").load(torch.device("cpu"))

        assert hf_mocks["AutoTokenizer"].call_args.kwargs["revision"] == "abc123"
        assert hf_mocks["AutoModel"].call_args.kwargs["revision"] == "abc123"


class TestDnaWrappersRevision:
    @pytest.mark.parametrize(
        ("wrapper_cls", "registry_key"),
        [
            (GENALMWrapper, "gena_lm_bert_base"),
            (NucleotideTransformerWrapper, "nt_v2_100m"),
            (HyenaDNAWrapper, "hyenadna_tiny_1k"),
            (CaduceusWrapper, "caduceus_ps_131k"),
        ],
    )
    def test_load_passes_pinned_revision(self, hf_mocks, monkeypatch, wrapper_cls, registry_key):
        if wrapper_cls is CaduceusWrapper:
            # CaduceusWrapper.load checks for the Mamba backend before loading anything.
            fake_mamba = types.ModuleType("mamba_ssm")
            fake_mamba.__spec__ = importlib.machinery.ModuleSpec("mamba_ssm", None)
            monkeypatch.setitem(sys.modules, "mamba_ssm", fake_mamba)
        repo = DNA_MODELS[registry_key][1]
        wrapper_cls(model_path_or_name=repo).load(torch.device("cpu"))

        _assert_calls_use_revision(hf_mocks, repo, REMOTE_CODE_REVISIONS[repo])

    def test_revision_kwarg_overrides_pin(self, hf_mocks):
        repo = DNA_MODELS["nt_v2_100m"][1]
        NucleotideTransformerWrapper(model_path_or_name=repo, revision="abc123").load(torch.device("cpu"))

        _assert_calls_use_revision(hf_mocks, repo, "abc123")

    def test_gated_ntv3_is_unpinned_but_accepts_revision(self, hf_mocks):
        repo = DNA_MODELS["ntv3_100m_pre"][1]
        NucleotideTransformerV3Wrapper(model_path_or_name=repo).load(torch.device("cpu"))
        _assert_calls_use_revision(hf_mocks, repo, None)

        for mock in hf_mocks.values():
            mock.reset_mock()
        NucleotideTransformerV3Wrapper(model_path_or_name=repo, revision="abc123").load(torch.device("cpu"))
        _assert_calls_use_revision(hf_mocks, repo, "abc123")


class TestPinCoverage:
    def test_every_remote_code_call_passes_revision(self):
        """Static guard: a new ``trust_remote_code=True`` load must not forget ``revision=``."""
        found = 0
        for path in sorted(Path(models_pkg.__file__).parent.glob("*.py")):
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                kwargs = {kw.arg for kw in node.keywords}
                if "trust_remote_code" in kwargs:
                    found += 1
                    assert "revision" in kwargs, f"{path.name}:{node.lineno} loads remote code unpinned"
        assert found > 0

    def test_registered_remote_code_repos_are_pinned(self):
        """Every registry repo loaded with trust_remote_code is pinned (NTv3 is gated, so it is exempt)."""
        remote_code_wrappers = (GENALMWrapper, NucleotideTransformerWrapper, HyenaDNAWrapper, CaduceusWrapper)
        entries = [*DNA_MODELS.items(), *MOLECULE_MODELS.items()]
        checked = 0
        for key, (wrapper_cls, repo) in entries:
            if wrapper_cls in (*remote_code_wrappers, MolformerWrapper):
                checked += 1
                assert repo in REMOTE_CODE_REVISIONS, f"registry model '{key}' ({repo}) has no pinned revision"
        assert checked > 0
