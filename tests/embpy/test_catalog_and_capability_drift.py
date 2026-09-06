"""Three lists that used to disagree, and one capability flag that overpromised.

Each test here pins a bug that static review missed and only surfaced when the
gene deep-dive notebook was executed end to end:

* ``model_catalog("static")`` advertised ``ccle`` / ``ccle_ensembl``, which have
  no download specification at all, so ``embed`` raised ``FileNotFoundError``.
* the same table also holds two STRING tables keyed by protein ids. They must
  stay *out* of the gene roster: routing gene symbols to them makes the loader
  resolve ~19,000 ``9606.ENSP...`` identifiers one request at a time.
* ``has_attention`` defaults to ``True``, and four non-HuggingFace DNA wrappers
  inherited it while ``extract_attention`` could not possibly succeed.
"""

from __future__ import annotations

import pytest

from embpy.pp.static_embeddings import _DEFAULT_SOURCE_SPECS, static_embedding_keys


class TestStaticCatalogueMatchesSpecs:
    def test_advertised_set_is_derived_from_the_specs(self) -> None:
        """The roster and the download table must be the same list, not two lists."""
        pytest.importorskip("torch")
        from embpy.embedder import DEFAULT_STATIC_EMBEDDING_MODELS

        assert DEFAULT_STATIC_EMBEDDING_MODELS == static_embedding_keys("gene")

    def test_no_advertised_key_lacks_a_specification(self) -> None:
        pytest.importorskip("torch")
        from embpy.embedder import DEFAULT_STATIC_EMBEDDING_MODELS

        spec_keys = {spec["key"] for spec in _DEFAULT_SOURCE_SPECS.values()}
        phantom = DEFAULT_STATIC_EMBEDDING_MODELS - spec_keys
        assert not phantom, (
            f"advertised with no way to fetch them: {sorted(phantom)}. "
            "These raise FileNotFoundError from embed()."
        )

    def test_every_gene_spec_is_advertised(self) -> None:
        """No gene table may be downloadable-but-unreachable."""
        pytest.importorskip("torch")
        from embpy.embedder import DEFAULT_STATIC_EMBEDDING_MODELS

        gene_specs = {
            spec["key"]
            for spec in _DEFAULT_SOURCE_SPECS.values()
            if spec.get("entity_type", "gene") == "gene"
        }
        stranded = gene_specs - DEFAULT_STATIC_EMBEDDING_MODELS
        assert not stranded, (
            f"downloadable but not routed to the static path: {sorted(stranded)}. "
            "These fall through to Hugging Face and raise ModelNotFoundError."
        )

    @pytest.mark.parametrize("key", ["ccle", "ccle_ensembl"])
    def test_phantom_keys_are_gone(self, key: str) -> None:
        pytest.importorskip("torch")
        from embpy.embedder import DEFAULT_STATIC_EMBEDDING_MODELS

        assert key not in DEFAULT_STATIC_EMBEDDING_MODELS

    @pytest.mark.parametrize("key", ["string_functional_9606", "string_node2vec_9606"])
    def test_protein_tables_stay_out_of_the_gene_roster(self, key: str) -> None:
        """They are keyed by STRING protein ids, not by gene."""
        pytest.importorskip("torch")
        from embpy.embedder import DEFAULT_STATIC_EMBEDDING_MODELS

        assert key not in DEFAULT_STATIC_EMBEDDING_MODELS
        assert key in static_embedding_keys("protein")


class TestAttentionCapabilityIsHonest:
    """``has_attention`` must not promise what ``extract_attention`` cannot do."""

    @pytest.mark.parametrize(
        "wrapper_name",
        ["EnformerWrapper", "BorzoiWrapper", "EvoWrapper", "Evo2Wrapper"],
    )
    def test_non_hf_dna_wrappers_declare_no_attention(self, wrapper_name: str) -> None:
        pytest.importorskip("torch")
        from embpy.models import dna_models

        wrapper = getattr(dna_models, wrapper_name)
        assert wrapper.has_attention is False, (
            f"{wrapper_name} inherits has_attention=True from BaseModelWrapper but is "
            "not a HuggingFace model and does not override _get_layer_modules(), so "
            "extract_attention() cannot succeed."
        )

    @pytest.mark.parametrize(
        "wrapper_name", ["NucleotideTransformerWrapper", "GENALMWrapper"]
    )
    def test_hf_dna_wrappers_still_declare_attention(self, wrapper_name: str) -> None:
        """The fix must not over-correct: these two genuinely work."""
        pytest.importorskip("torch")
        from embpy.models import dna_models

        assert getattr(dna_models, wrapper_name).has_attention is True

    def test_every_dna_wrapper_declaring_attention_can_reach_it(self) -> None:
        """A wrapper may claim attention only if HF-backed or it finds its own layers.

        This is the general form of the bug: the flag defaults to True, so any new
        non-HuggingFace wrapper inherits a promise unless it opts out.
        """
        pytest.importorskip("torch")
        import inspect

        from embpy.models import dna_models
        from embpy.models.base import BaseModelWrapper

        offenders = []
        for name, obj in vars(dna_models).items():
            if not inspect.isclass(obj) or not issubclass(obj, BaseModelWrapper):
                continue
            if obj is BaseModelWrapper or not obj.has_attention:
                continue
            # Declares attention: it must at least say where its layers are, unless
            # it is one of the HuggingFace-backed wrappers whose eager path works.
            overrides_layers = "_get_layer_modules" in vars(obj)
            hf_backed = name in {
                "NucleotideTransformerWrapper",
                "NucleotideTransformerV3Wrapper",
                "GENALMWrapper",
            }
            if not (overrides_layers or hf_backed):
                offenders.append(name)
        assert not offenders, (
            f"declare has_attention=True with no way to extract it: {offenders}"
        )
