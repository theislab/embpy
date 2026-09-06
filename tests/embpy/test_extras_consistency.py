"""No optional extra may contradict the base requirements.

``[ntv3]`` pinned ``transformers>=5.0.0`` while the base requirements pin
``transformers>=4.45.0,<5.0.0``. The intersection is empty, so
``uv pip install "embpy[ntv3]"`` failed outright -- with or without ``[cpu]`` or
``[gpu]`` -- and none of the five ``ntv3_*`` model keys could be installed at all.
Nothing caught it, because every test that touched extras checked that the *names*
existed rather than that the *versions* could co-exist.

These tests resolve the specifier intersection offline, so they fail on the
declaration rather than waiting for a user to hit it.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.version import Version

PYPROJECT = Path(__file__).resolve().parents[2] / "pyproject.toml"

# A grid dense enough to find a satisfying release if one could exist. Real
# packages do not use version 100, so an empty result here means "no version of
# this distribution can satisfy both constraints".
_GRID = [
    Version(f"{major}.{minor}.{patch}")
    for major in range(0, 12)
    for minor in (0, 1, 2, 4, 5, 8, 10, 20, 45, 48, 50, 57, 99)
    for patch in (0, 1, 6)
]


def _project() -> dict:
    return tomllib.loads(PYPROJECT.read_text())["project"]


def _requirements(entries: list[str]) -> dict[str, SpecifierSet]:
    """Map distribution name -> combined specifier, ignoring self-referential extras."""
    out: dict[str, SpecifierSet] = {}
    for raw in entries:
        if raw.startswith("embpy["):
            continue  # a self-reference; its own contents are checked separately
        try:
            req = Requirement(raw)
        except Exception:
            continue
        if req.url:
            continue  # git/URL requirement carries no comparable specifier
        out[req.name.lower()] = out.get(req.name.lower(), SpecifierSet()) & req.specifier
    return out


def _satisfiable(spec: SpecifierSet) -> bool:
    return any(v in spec for v in _GRID)


def _model_extras() -> list[str]:
    """Extras that exist to enable a model backend."""
    opt = _project()["optional-dependencies"]
    skip = {"dev", "doc", "test", "all", "all-cpu", "all-cu121", "all-cu124", "all-cu128"}
    return [name for name in opt if name not in skip]


class TestExtrasDoNotContradictBase:
    @pytest.mark.parametrize("extra", _model_extras())
    @pytest.mark.parametrize("against", ["dependencies", "cpu", "gpu"])
    def test_extra_can_coexist_with_the_supported_installs(
        self, extra: str, against: str
    ) -> None:
        """Compare against the base list *and* against [cpu]/[gpu].

        Comparing only against ``project.dependencies`` would have missed the ntv3
        regression entirely: ``transformers`` is not a base requirement, it is
        declared in ``[cpu]`` and ``[gpu]``. Since those two are the supported
        installs, an extra that cannot co-exist with them is broken in practice
        even if the base list is silent about the distribution.
        """
        project = _project()
        if extra == against:
            pytest.skip("comparing an extra with itself")
        if against == "dependencies":
            reference = _requirements(project["dependencies"])
        else:
            reference = _requirements(project["optional-dependencies"][against])
        extra_reqs = _requirements(project["optional-dependencies"][extra])

        for dist, extra_spec in extra_reqs.items():
            if dist not in reference:
                continue
            combined = reference[dist] & extra_spec
            assert _satisfiable(combined), (
                f"extra [{extra}] requires {dist}{extra_spec} but {against} pins "
                f"{dist}{reference[dist]}; the intersection is empty, so "
                f'`uv pip install "embpy[{against if against != "dependencies" else "cpu"},{extra}]"` '
                "cannot resolve."
            )

    def test_ntv3_specifically_resolves_against_base(self) -> None:
        """The regression that motivated this file."""
        project = _project()
        base = _requirements(project["dependencies"])
        ntv3 = _requirements(project["optional-dependencies"]["ntv3"])
        # transformers lives in [cpu]/[gpu] rather than the base list, so compare
        # against that too -- it is the pin ntv3 actually collided with.
        cpu = _requirements(project["optional-dependencies"]["cpu"])
        for source in (base, cpu):
            if "transformers" in ntv3 and "transformers" in source:
                combined = source["transformers"] & ntv3["transformers"]
                assert _satisfiable(combined), (
                    f"[ntv3] transformers{ntv3['transformers']} cannot co-exist with "
                    f"transformers{source['transformers']}"
                )


class TestCpuGpuParity:
    def test_cpu_and_gpu_differ_only_in_torch(self) -> None:
        """They are documented as identical bar the torch flavour; keep them so."""
        opt = _project()["optional-dependencies"]
        strip = lambda items: {  # noqa: E731
            i for i in items if "torch" not in i.lower()
        }
        assert strip(opt["cpu"]) == strip(opt["gpu"])

    @pytest.mark.parametrize("extra", ["cpu", "gpu"])
    def test_extra_is_internally_satisfiable(self, extra: str) -> None:
        reqs = _requirements(_project()["optional-dependencies"][extra])
        for dist, spec in reqs.items():
            assert _satisfiable(spec), f"[{extra}] pins {dist}{spec}, which no version satisfies"


class TestEveryModelExtraIsDeclared:
    def test_wrapper_extras_all_exist(self) -> None:
        from embpy.embedder_registry.extras import WRAPPER_EXTRAS

        opt = _project()["optional-dependencies"]
        unknown = {e for e in WRAPPER_EXTRAS.values() if e not in opt}
        assert not unknown, f"WRAPPER_EXTRAS names extras that pyproject does not declare: {unknown}"


class TestIsolatedBackendsStayOut:
    """[cpu] and [gpu] must never pull a backend that cannot install everywhere.

    Each of these has a hard reason it cannot live in the standard install:
    flash-attn is NVIDIA-only and wheel-less, evo2 caps Python below 3.13, boltz
    pins numpy<2, and the esm SDK caps transformers. Adding any of them to [cpu] or
    [gpu] would turn a working install into a broken one for most users.
    """

    FORBIDDEN = ["flash-attn", "flash_attn", "evo2", "evo-model", "boltz", "esm", "mamba-ssm"]

    @pytest.mark.parametrize("extra", ["cpu", "gpu"])
    def test_supported_installs_pull_no_isolated_backend(self, extra: str) -> None:
        reqs = _requirements(_project()["optional-dependencies"][extra])
        leaked = sorted(set(reqs) & {f.lower() for f in self.FORBIDDEN})
        assert not leaked, (
            f"[{extra}] pulls {leaked}, which cannot be installed on every supported "
            "platform/interpreter. Keep these behind their own extra."
        )

    def test_flashzoi_extra_exists_and_requests_the_flash_variant(self) -> None:
        """The 4 flashzoi_* keys need borzoi-pytorch's own `flash` extra."""
        opt = _project()["optional-dependencies"]
        assert "flashzoi" in opt, "the flashzoi_* keys need a named install path"
        assert any("borzoi-pytorch[flash]" in item for item in opt["flashzoi"]), (
            "flashzoi must request borzoi-pytorch[flash]; flash-attn is declared by "
            "borzoi-pytorch under that extra"
        )

    def test_seqmodels_does_not_request_flash(self) -> None:
        """seqmodels covers the 8 CPU-capable borzoi keys; it must stay installable."""
        opt = _project()["optional-dependencies"]
        assert not any("[flash]" in item for item in opt["seqmodels"])


class TestFlashzoiIsGatedAtRuntime:
    def test_flashzoi_keys_are_registered(self) -> None:
        pytest.importorskip("torch")
        from embpy.embedder_registry.flat import MODEL_REGISTRY

        keys = [k for k in MODEL_REGISTRY if k.startswith("flashzoi")]
        assert len(keys) == 4, f"expected 4 flashzoi keys, found {keys}"

    def test_missing_flash_attn_names_the_package_and_the_fix(self) -> None:
        """Without the gate this raised a generic 'Could not load Borzoi'."""
        pytest.importorskip("torch")
        # The flash_attn gate lives *after* the borzoi_pytorch check, so without
        # borzoi installed load() fails earlier with "borzoi_pytorch not installed"
        # and this gate is never reached (the `full` CI env has neither package).
        pytest.importorskip("borzoi_pytorch")
        import importlib.util

        if importlib.util.find_spec("flash_attn") is not None:
            pytest.skip("flash_attn is installed; the gate is not exercised here")

        import torch

        from embpy.models.dna_models import BorzoiWrapper

        wrapper = BorzoiWrapper("johahi/flashzoi-replicate-0")
        with pytest.raises(ImportError) as excinfo:
            wrapper.load(torch.device("cpu"))
        message = str(excinfo.value)
        assert "flash_attn" in message
        assert "embpy[flashzoi]" in message
        assert "--no-build-isolation" in message
        # and it must not have tried to download the checkpoint first
        assert wrapper.model is None
