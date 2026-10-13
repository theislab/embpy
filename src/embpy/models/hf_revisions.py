"""Pinned Hugging Face revisions for models loaded with ``trust_remote_code=True``.

Those repos ship Python files (modelling / tokenizer code) that ``transformers``
downloads from the Hub and executes, so an upstream commit can change or break
embpy's behaviour without any embpy release.  For example, ``ibm/MoLFormer-XL-both-10pct``
was updated to import ``create_bidirectional_mask``, which only exists in
transformers 5, and stopped loading on transformers 4.x.

``REMOTE_CODE_REVISIONS`` maps ``{hf_repo_id: commit_sha}``.  The wrappers look their
repo id up here (case-insensitively, like the Hub) and pass the result as ``revision=`` to
every ``from_pretrained`` call (tokenizer, config and model alike); see :func:`resolve_revision`.
Repos that are not listed load from ``main`` as before.  A wrapper's ``revision`` kwarg
overrides the pin, but ``BioEmbedder`` does not forward kwargs, so that override only works
for directly constructed wrappers.

The MolFormer pin only applies on transformers 4.x (see ``_TRANSFORMERS_4_ONLY``): on
transformers >= 5 that old commit fails (it imports ``transformers.onnx``, removed in 5) while
``main`` works, so there ``molformer_base`` loads from ``main``.

Unless noted otherwise, each sha is the head of ``main`` on 2026-10-11 as returned by
``huggingface_hub.HfApi().model_info(repo_id).sha``, i.e. it freezes the behaviour at
that date.  To bump a pin, look up the new sha, check the model still loads and embeds,
and update the entry.
"""

from __future__ import annotations

import logging

REMOTE_CODE_REVISIONS: dict[str, str] = {
    # MolFormer: *not* the head of main (361063d0).  The 2026 upstream commits (a14249e5, 361063d0) import
    # create_bidirectional_mask, which only exists in transformers 5; this is the last commit before them,
    # verified to load and embed on transformers 4.57.6.  Applied on transformers 4.x only (_TRANSFORMERS_4_ONLY).
    "ibm/MoLFormer-XL-both-10pct": "7b12d946c181a37f6012b9dc3b002275de070314",
    # GENA-LM (AIRI-Institute)
    "AIRI-Institute/gena-lm-bert-base-t2t": "4f1352bd4e820f1dba341047f54ad7795e083bf2",
    "AIRI-Institute/gena-lm-bert-large-t2t": "f997a8a7c1ba5feb6d11de46354a41c51ffe9660",
    "AIRI-Institute/gena-lm-bert-base-lastln-t2t": "3177322cb31140be33a692bfb968263cbd3ef89a",
    "AIRI-Institute/gena-lm-bert-base-t2t-multi": "4633e5a1ada905bb7afee6877d71cc12578a95a5",
    "AIRI-Institute/gena-lm-bigbird-base-t2t": "f155a24ef5cefd7218a1e4c6888aedb574509cc3",
    "AIRI-Institute/gena-lm-bigbird-base-sparse-t2t": "79b367060cd3aa22308865b78c5644d79004e998",
    # Nucleotide Transformer v1 / v2 (InstaDeepAI)
    "InstaDeepAI/nucleotide-transformer-500m-human-ref": "f87b5d7233295242e79c951873d290f4cf992045",
    "InstaDeepAI/nucleotide-transformer-500m-1000g": "a1fcafab929cf0f835adeb6fb7dd305607dc6349",
    "InstaDeepAI/nucleotide-transformer-2.5b-1000g": "46f2210107b149b19cc8fe5605533f89eadcda61",
    "InstaDeepAI/nucleotide-transformer-2.5b-multi-species": "b746b125aacd1b0970c05a32fd71ba726754542e",
    "InstaDeepAI/nucleotide-transformer-v2-50m-multi-species": "81b29e5786726d891dbf929404ef20adca5b36f1",
    "InstaDeepAI/nucleotide-transformer-v2-100m-multi-species": "f34324c6fde36a4f635f0f1f06cac5d25acd6798",
    "InstaDeepAI/nucleotide-transformer-v2-250m-multi-species": "c0f0359229f36ff6bc3a021247eefe0a9c344bd1",
    "InstaDeepAI/nucleotide-transformer-v2-500m-multi-species": "06615c1660c892fc199840c18123f8385b3542a8",
    # Nucleotide Transformer v3 (InstaDeepAI/NTv3_*) is intentionally not pinned: those repos are gated
    # and their files cannot be fetched without an accepted licence and an authorised HF token.
    # HyenaDNA (LongSafari)
    "LongSafari/hyenadna-tiny-1k-seqlen-hf": "e8c1effa8673814e257e627d2e1eda9ea5a373f6",
    "LongSafari/hyenadna-small-32k-seqlen-hf": "8fe770c78eb13fe33bf81501612faeddf4d6f331",
    "LongSafari/hyenadna-medium-160k-seqlen-hf": "7ebf71773d22c0ede2cc55cb2be15ee8c289e1ce",
    "LongSafari/hyenadna-medium-450k-seqlen-hf": "42dedd4d374eac0fb8168549e546a3472fbd27ae",
    "LongSafari/hyenadna-large-1m-seqlen-hf": "0a629abf9c7f85b4ec9aa6a1aefa3adcf1907446",
    # Caduceus (kuleshov-group)
    "kuleshov-group/caduceus-ph_seqlen-131k_d_model-256_n_layer-16": "b0477522ac5d044ad03578aa724ec8e4bdbd405b",
    "kuleshov-group/caduceus-ps_seqlen-131k_d_model-256_n_layer-16": "d89eeb853136ea64da7feb3d0c8e909771b17ae6",
}

# Pins that only fit transformers 4.x and are skipped (``main`` is used) on transformers >= 5.
_TRANSFORMERS_4_ONLY: frozenset[str] = frozenset({"ibm/MoLFormer-XL-both-10pct"})


def resolve_revision(model_name: str | None, revision: str | None = None) -> str | None:
    """Return the revision to pass to ``from_pretrained`` for ``model_name``.

    Parameters
    ----------
    model_name
        Hugging Face repo id (or local path) the wrapper was created with.  Repo ids are matched
        case-insensitively.
    revision
        Explicit revision (branch, tag or commit sha), e.g. the ``revision`` kwarg given to a
        wrapper.  Takes precedence over the pinned one.

    Returns
    -------
    str or None
        ``revision`` if given, else the commit pinned in :data:`REMOTE_CODE_REVISIONS`, else
        ``None`` (``from_pretrained`` then uses ``main``, which is also what it does for local paths).
        Repos in ``_TRANSFORMERS_4_ONLY`` also get ``None`` when transformers >= 5 is installed.
    """
    if revision:
        return revision
    name = (model_name or "").lower()
    pins = {repo.lower(): sha for repo, sha in REMOTE_CODE_REVISIONS.items()}
    if name not in pins:
        return None
    if name in {repo.lower() for repo in _TRANSFORMERS_4_ONLY}:
        import transformers

        if int(transformers.__version__.split(".")[0]) >= 5:
            logging.debug(f"transformers {transformers.__version__} >= 5: not pinning '{model_name}', using main.")
            return None
    return pins[name]
