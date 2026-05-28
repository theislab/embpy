from __future__ import annotations

import logging
import os
import pathlib
import re
import traceback
from collections.abc import Mapping, Sequence
from hashlib import sha1
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import torch
from rdkit import Chem

from .errors import (
    ConfigError,
    ContextOverflowError,
    DependencyError,
    EmbpyError,
    IdentifierError,
    ModelLoadError,
    ModelNotFoundError,
    ModelOOMError,
)
from .models.base import BaseModelWrapper
from .observability import log_event, time_block
from .reporting import ResolutionReport
from .retry import embed_batch_with_oom_recovery

if TYPE_CHECKING:
    pass
# MODEL_REGISTRY + the three DNA species sets live in
# `embpy.embedder_registry` (audit steps 2 + 3). The DNA / Protein /
# Molecule / Text / Morphology / Single-cell / API entries are split
# across per-modality submodules; `embedder_registry.flat` merges them
# into a single dict. Re-exported here so `from embpy.embedder import
# MODEL_REGISTRY` (and every BioEmbedder method body that references
# HUMAN_ONLY_MODELS etc.) keeps working byte-equivalently against the
# pre-split snapshot. The Evo / Evo2 / Boltz-2 wrapper aliases and
# their `_HAVE_*` gating flags stayed inside the per-modality files
# (dna.py / protein.py) because they are an implementation detail of
# the registry; no consumer outside embpy.embedder_registry references
# them.
from .embedder_registry.flat import (
    HUMAN_ONLY_MODELS,
    MODEL_REGISTRY,
    MOUSE_ONLY_MODELS,
)
from .models.api_models import APIEmbeddingWrapper
from .models.morphology_models import SubCellWrapper
from .models.text_models import TextLLMWrapper
from .resources.gene_resolver import GeneResolver
from .resources.protein_resolver import ProteinResolver
from .resources.text_resolver import TextResolver


# Helper function (can be moved to utils later)
def get_device() -> torch.device:  # type: ignore[name-defined]
    """Selects the best available device: CUDA, MPS (for Apple Silicon), or CPU."""
    if torch.cuda.is_available():  # type: ignore[attr-defined]
        logging.info("CUDA device found, using GPU.")
        return torch.device("cuda")  # type: ignore[attr-defined]
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():  # type: ignore[attr-defined]
        logging.info("MPS device found, using Apple Silicon GPU.")
        return torch.device("mps")  # type: ignore[attr-defined]
    else:
        logging.info("No GPU found, using CPU.")
        return torch.device("cpu")  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Exception classification helpers
# ---------------------------------------------------------------------------
#
# These bridge between "whatever the underlying library raised" and our
# typed embpy.errors taxonomy. They are deliberately defensive: when a
# raw exception's category cannot be determined we fall back to the
# generic EmbpyError so the upstream entry point still gets a typed
# exception (and structured exit code) instead of a bare RuntimeError.

_OOM_PATTERNS = (
    "out of memory",
    "outofmemoryerror",
    "cuda error: out of memory",
    "cudnn_status_alloc_failed",
)

_CONTEXT_OVERFLOW_PATTERNS = (
    # GENA-LM / BERT positional buffer mismatch
    "the expanded size of the tensor",
    "must match the existing size",
    # NT / some HF position checks
    "sequence length is longer than",
    # Generic "input too long" guards
    "exceeds maximum length",
)


def _guess_missing_package(msg: str) -> str | None:
    """Best-effort extraction of a missing module name from an ImportError.

    Handles the two common spellings:
        "No module named 'mamba_ssm'"
        "cannot import name 'X' from 'Y' (path)"
    Returns ``None`` if no module name can be confidently extracted; the
    caller falls back to a generic "unknown" placeholder.
    """
    m = re.search(r"No module named ['\"]([^'\"]+)['\"]", msg)
    if m:
        return m.group(1).split(".")[0]
    m = re.search(r"from ['\"]([^'\"]+)['\"]", msg)
    if m:
        return m.group(1).split(".")[0]
    return None


def _parse_oom_attempted_bytes(msg: str) -> int | None:
    """Extract ``B`` from CUDA OOM messages like "Tried to allocate 33.87 GiB".

    Returns the number in bytes (int) or None if unparseable. Used as a
    diagnostic field on ``ModelOOMError`` -- not for control flow.
    """
    m = re.search(r"Tried to allocate ([\d.]+)\s*(GiB|MiB|KiB|GB|MB|KB|B)", msg)
    if not m:
        return None
    val, unit = float(m.group(1)), m.group(2).lower()
    scale = {
        "gib": 1024**3,
        "mib": 1024**2,
        "kib": 1024,
        "gb": 1000**3,
        "mb": 1000**2,
        "kb": 1000,
        "b": 1,
    }.get(unit, 1)
    return int(val * scale)


def _classify_embedder_exception(
    exc: BaseException,
    *,
    model_name: str,
    n_inputs: int,
    device: str | None = None,
) -> EmbpyError:
    """Map an arbitrary exception from ``inst.embed_batch`` to a typed embpy error.

    Detection order matters: OOM is the most specific (it inherits from
    RuntimeError in PyTorch); context-overflow is the next most specific
    (recognised by its tensor-expansion message); missing-deps are
    ImportErrors; everything else collapses to a generic EmbpyError
    that still carries the underlying class name in its message.
    """
    msg = str(exc)
    msg_lower = msg.lower()

    # 1) CUDA OOM. PyTorch exposes torch.OutOfMemoryError on >= 2.4; for
    #    older versions the same allocation failure surfaces as a plain
    #    RuntimeError whose message contains "out of memory". We check
    #    both.
    is_oom = isinstance(exc, getattr(torch, "OutOfMemoryError", ()))  # type: ignore[arg-type]
    if not is_oom:
        is_oom = any(p in msg_lower for p in _OOM_PATTERNS)
    if is_oom:
        return ModelOOMError(
            model_name=model_name,
            device=device,
            attempted_bytes=_parse_oom_attempted_bytes(msg),
        )

    # 2) Context overflow: the model's positional buffer or attention
    #    mask got expanded to a sequence length it cannot represent.
    if isinstance(exc, RuntimeError) and any(p in msg_lower for p in _CONTEXT_OVERFLOW_PATTERNS):
        # Best-effort length parsing from messages like
        # "Target sizes: [16, 16596]. Tensor sizes: [1, 512]"
        m = re.search(r"\[\d+,\s*(\d+)\][\s\S]*\[\d+,\s*(\d+)\]", msg)
        input_len = int(m.group(1)) if m else -1
        ctx = int(m.group(2)) if m else -1
        return ContextOverflowError(
            model_name=model_name,
            input_length=input_len,
            context_window=ctx,
        )

    # 3) Missing python package surfaced through the embed call (e.g.
    #    deferred import inside the forward pass).
    if isinstance(exc, ImportError):
        return DependencyError(
            package=_guess_missing_package(msg) or "unknown",
            feature=f"model '{model_name}'",
        )

    # 4) Default: typed but generic. Preserves model_name + cause class.
    return ModelLoadError(
        model_name=model_name,
        cause=f"{type(exc).__name__}: {exc}",
        message=(
            f"embed_batch failed for model='{model_name}' on {n_inputs} inputs. Root cause: {type(exc).__name__}: {exc}"
        ),
    )


class BioEmbedder:
    """
    Central class for generating biological embeddings.

    Manages different embedding models and provides a unified interface
    for embedding genes (via DNA or protein sequences) and small molecules.

    Attributes
    ----------
    device : torch.device
        The computing device (CPU or GPU) used for model inference.
    model_cache : dict[str, BaseModelWrapper]
        A cache to store loaded model instances.
    gene_resolver : GeneResolver
        An instance to handle gene identifier resolution and sequence fetching.
    """

    def __init__(
        self,
        device: str | torch.device | None = "auto",  # type: ignore[name-defined]
        organism: str = "human",
        resolver_backend: Literal["api", "local"] = "api",
        mart_file: str | None = None,
        chromosome_folder: str | None = None,
    ):
        """
        Initializes the BioEmbedder.

        Args:
            device: 'auto', 'cuda', 'mps', or 'cpu', or torch.device.
            organism: Default organism for sequence resolution and annotation
                (e.g. 'human', 'mouse', 'zebrafish'). Any species supported
                by Ensembl can be used.
            resolver_backend: 'api' to use online APIs, 'local' to use local FASTAs.
            mart_file: path to Mart CSV (required if resolver_backend='local').
            chromosome_folder: path to folder with chr*.fa files (required if 'local').
        """
        self.organism = organism

        # Device setup
        if isinstance(device, str):
            if device == "auto":
                self.device = get_device()
            else:
                self.device = torch.device(device)  # type: ignore[attr-defined]
        elif isinstance(device, torch.device):  # type: ignore[attr-defined]
            self.device = device
        else:
            raise ConfigError("Invalid device; use 'auto','cpu','cuda','mps', or torch.device.")

        # GeneResolver setup
        self.resolver_backend = resolver_backend
        if resolver_backend == "local":
            if not mart_file or not chromosome_folder:
                raise ConfigError("mart_file and chromosome_folder must be provided for local resolver.")
            self.gene_resolver = GeneResolver(
                mart_file=mart_file,
                chromosome_folder=chromosome_folder,
            )
        else:
            self.gene_resolver = GeneResolver(species=organism)

        # Protein resolver
        self.protein_resolver = ProteinResolver(organism=organism)

        # Text resolver for description-based embeddings
        self.text_resolver = TextResolver(organism=organism)

        # Model cache and discovery.
        #
        # `model_cache` holds DNA/protein/molecule/text/morphology wrappers
        # that subclass `BaseModelWrapper`. Single-cell foundation-model
        # wrappers (scGPT, Geneformer, UCE, Tahoe, ...) subclass
        # `SingleCellWrapper` instead, so they live in a separate cache.
        # Both caches are keyed so that repeated calls to embed_cells /
        # embed_perturbation / etc. reuse an already-loaded torch model
        # instead of re-instantiating it on every call -- critical for
        # chunked inference over large datasets.
        self.model_cache: dict[str, BaseModelWrapper] = {}
        self._singlecell_cache: dict[tuple[str, str], Any] = {}
        self._available_models = self._discover_models()

        # Layer 2: per-call resolution report. Populated by
        # ``embed_genes_batch`` and exposed for callers (e.g.
        # ``BioEmbedderProvider`` and ``embed_perturbations.py``)
        # that want a per-identifier audit trail without parsing logs.
        # Each call to ``embed_genes_batch`` REPLACES this attribute,
        # so callers should grab it immediately after the call returns.
        self.last_report: ResolutionReport | None = None

        logging.info(
            "BioEmbedder initialized: device=%s, organism=%s, backend=%s",
            self.device,
            self.organism,
            self.resolver_backend,
        )

    def _discover_models(self) -> dict[str, tuple[type[BaseModelWrapper], str]]:
        """Filters the MODEL_REGISTRY based on available wrapper classes."""
        available = {}
        for name, (wrapper_class, model_path) in MODEL_REGISTRY.items():
            if wrapper_class is None:
                logging.debug(
                    f"Skipping model '{name}': Wrapper class not imported (optional dependency likely missing)."
                )
                continue
            if model_path is None:
                logging.warning(f"Skipping model '{name}': No model path/identifier defined in registry.")
                continue
            # Check if the wrapper class itself is valid (was imported successfully)
            if not issubclass(wrapper_class, BaseModelWrapper):
                logging.error(f"Internal Error: Registered item for '{name}' is not a valid BaseModelWrapper subclass.")
                continue

            available[name] = (wrapper_class, model_path)
            logging.debug(f"Registered model '{name}' using wrapper {wrapper_class.__name__} for path '{model_path}'")

        if not available:
            logging.warning("No models were successfully registered. Check imports and MODEL_REGISTRY definition.")

        return available

    def _get_model(self, model_name: str) -> BaseModelWrapper:
        """Loads a model or retrieves it from the cache using the registry or direct HF loading for text models."""
        if model_name in self.model_cache:
            return self.model_cache[model_name]

        is_human = self.organism.lower() in ("human", "homo_sapiens")
        is_mouse = self.organism.lower() in ("mouse", "mus_musculus")
        if not is_human and model_name in HUMAN_ONLY_MODELS:
            logging.warning(
                "Model '%s' was trained on human data only. Results for organism='%s' may not be meaningful.",
                model_name,
                self.organism,
            )
        if not is_mouse and model_name in MOUSE_ONLY_MODELS:
            logging.warning(
                "Model '%s' was trained on mouse data only. Results for organism='%s' may not be meaningful.",
                model_name,
                self.organism,
            )

        if model_name in self._available_models:
            logging.info(f"Loading registered model '{model_name}' onto device '{self.device}'...")
            WrapperClass, model_path_or_name = self._available_models[model_name]
            try:
                extra_kwargs: dict[str, Any] = {}
                if model_name == "boltz2_pairwise":
                    extra_kwargs["output_type"] = "pairwise"
                elif model_name == "boltz2_both":
                    extra_kwargs["output_type"] = "both"
                if WrapperClass is SubCellWrapper:
                    pass
                if WrapperClass is APIEmbeddingWrapper:
                    provider_map = {
                        "openai_small": "openai",
                        "openai_large": "openai",
                        "cohere_v3": "cohere",
                        "cohere_multilingual": "cohere",
                        "voyage_3": "voyage",
                        "voyage_3_lite": "voyage",
                        "google_embed": "google",
                    }
                    extra_kwargs["provider"] = provider_map.get(model_name, "openai")
                model_instance = WrapperClass(model_path_or_name=model_path_or_name, **extra_kwargs)
                model_instance.load(self.device)
                self.model_cache[model_name] = model_instance
                logging.info(f"Model '{model_name}' loaded successfully.")
                return model_instance
            except ImportError as e:
                # Missing optional dependency -- forward as typed DependencyError
                # so the operator (and SLURM exit code) can distinguish
                # "needs `pixi install` of a different env" from "the model
                # is genuinely broken on HF".
                pkg = _guess_missing_package(str(e)) or "unknown"
                logging.error(f"Failed to load model '{model_name}': missing dependency '{pkg}'.")
                raise DependencyError(package=pkg, feature=f"model '{model_name}'") from e
            except Exception as e:
                logging.error(
                    f"Failed to load model '{model_name}' using wrapper {WrapperClass.__name__} and path '{model_path_or_name}': {e}"
                )
                raise ModelLoadError(
                    model_name=model_name,
                    cause=f"{type(e).__name__}: {e}",
                ) from e

        # If not in registry, try to load as a text model from Hugging Face
        else:
            logging.info(f"Model '{model_name}' not in registry. Attempting to load as text model from Hugging Face...")
            try:
                model_instance = TextLLMWrapper(model_path_or_name=model_name)
                model_instance.load(self.device)
                self.model_cache[model_name] = model_instance
                logging.info(f"Successfully loaded text model '{model_name}' from Hugging Face.")
                return model_instance
            except Exception as e:
                available_model_names = self.list_available_models()
                message = (
                    f"Model '{model_name}' could not be loaded from Hugging Face and is not in the predefined registry."
                )
                if available_model_names:
                    message += f" Available predefined models: {available_model_names}"
                message += f" Original error: {str(e)}"
                raise ModelNotFoundError(message) from e

    @staticmethod
    def _detect_vocab_type(
        var_names: Sequence[str], sample: int = 200
    ) -> Literal["symbol", "ensembl_id", "mixed", "unknown"]:
        r"""Guess the gene-identifier convention of an AnnData's ``var_names``.

        Heuristic: sample up to ``sample`` names and check what fraction
        look like Ensembl gene IDs (``ENSG`` / ``ENSMUSG`` / ``ENSRNOG`` /
        ``ENSG0``... or any ``ENS[A-Z]+G\\d+`` prefix, optionally with a
        trailing version suffix).

        Returns
        -------
        str
            ``"ensembl_id"`` if >= 80% match the Ensembl pattern,
            ``"symbol"`` if <= 20% match (rest look like gene symbols),
            ``"mixed"`` if in between, ``"unknown"`` if empty input.
        """
        import re

        if len(var_names) == 0:
            return "unknown"
        head = list(var_names[: min(sample, len(var_names))])
        pattern = re.compile(r"^ENS[A-Z]*G\d+(\.\d+)?$")
        n_ens = sum(1 for v in head if pattern.match(str(v)))
        frac = n_ens / len(head)
        if frac >= 0.8:
            return "ensembl_id"
        if frac <= 0.2:
            return "symbol"
        return "mixed"

    def _ensure_singlecell_vocabulary(
        self,
        adata,
        model_key: str,
        organism: str = "human",
    ):
        """Auto-convert ``adata.var_names`` for a single-cell foundation model.

        Does nothing if the model's ``vocab_type`` is ``"any"`` or
        ``"either"`` (the wrapper handles it internally), or if the
        detected format already matches. Otherwise uses
        :class:`GeneResolver` to remap symbols <-> Ensembl IDs via
        pyensembl (fast, local) with MyGene.info and Ensembl REST
        fallbacks for misses.

        Parameters
        ----------
        adata
            AnnData to inspect / convert. If conversion is needed the
            returned AnnData is a sliced copy with rekeyed ``var_names``;
            the original is left untouched.
        model_key
            Key into the single-cell model registry.
        organism
            Species name passed to ``GeneResolver`` (default ``"human"``).

        Returns
        -------
        tuple[AnnData, dict]
            The (possibly rewritten) AnnData and a small report dict
            with the detected/target formats and number of mapped genes.
        """
        from .models.singlecell_models import _SC_MODEL_REGISTRY

        card = _SC_MODEL_REGISTRY.get(model_key)
        if card is None or card.vocab_type in ("any", "either"):
            return adata, {"action": "none", "reason": f"vocab_type={getattr(card, 'vocab_type', '?')}"}

        target = card.vocab_type  # "symbol" or "ensembl_id"
        current = self._detect_vocab_type(adata.var_names)
        if current == target or current == "unknown":
            return adata, {"action": "none", "reason": f"already {current}"}
        if current == "mixed":
            logging.warning(
                "adata.var_names for model '%s' look mixed (both symbols and "
                "Ensembl IDs). Leaving untouched; expect poor vocabulary match.",
                model_key,
            )
            return adata, {"action": "none", "reason": "mixed var_names"}

        # Need to convert from `current` to `target`.
        logging.info(
            "Auto-converting adata.var_names for model '%s': %s -> %s (%d genes)",
            model_key,
            current,
            target,
            adata.n_vars,
        )

        if current == "ensembl_id" and target == "symbol":
            mapping = self.gene_resolver.ensembl_to_symbols_batch(list(adata.var_names), organism=organism)
        elif current == "symbol" and target == "ensembl_id":
            mapping = self.gene_resolver.symbols_to_ensembl_batch(list(adata.var_names), organism=organism)
        else:
            return adata, {"action": "none", "reason": f"unhandled {current}->{target}"}

        new_names = [mapping.get(v) for v in adata.var_names]
        keep_mask = np.array([n is not None and n != "" for n in new_names], dtype=bool)
        n_mapped = int(keep_mask.sum())
        if n_mapped == 0:
            logging.error(
                "Vocabulary conversion %s -> %s produced 0 mapped genes for "
                "model '%s'. Returning original adata; embedding will likely fail.",
                current,
                target,
                model_key,
            )
            return adata, {"action": "failed", "detected": current, "target": target, "n_mapped": 0}

        out = adata[:, keep_mask].copy()
        new_names_arr = np.array([n for n, keep in zip(new_names, keep_mask, strict=False) if keep])
        # Drop duplicates (multiple Ensembl IDs can map to the same symbol
        # and vice versa). Keep first occurrence.
        seen: set[str] = set()
        unique_mask = np.zeros(len(new_names_arr), dtype=bool)
        for i, n in enumerate(new_names_arr):
            if n not in seen:
                seen.add(n)
                unique_mask[i] = True
        if (~unique_mask).any():
            logging.info(
                "Dropped %d duplicate names during %s -> %s conversion.",
                int((~unique_mask).sum()),
                current,
                target,
            )
            out = out[:, unique_mask].copy()
            new_names_arr = new_names_arr[unique_mask]

        # Preserve the original identifiers as a var column so they
        # remain recoverable (e.g. for round-trip mapping after embedding).
        orig_col = "original_ensembl_id" if current == "ensembl_id" else "original_gene_symbol"
        orig_names = np.asarray(adata.var_names)[keep_mask]
        out.var[orig_col] = orig_names[unique_mask] if (~unique_mask).any() else orig_names
        out.var_names = new_names_arr
        logging.info(
            "Vocabulary conversion OK: %d -> %d genes mapped (%s -> %s).",
            adata.n_vars,
            out.n_vars,
            current,
            target,
        )
        return out, {
            "action": "converted",
            "detected": current,
            "target": target,
            "n_mapped": n_mapped,
            "n_out": out.n_vars,
        }

    def _get_or_load_singlecell_wrapper(
        self,
        model_key: str,
        batch_size: int,
        device_str: str,
    ):
        """Return a cached single-cell foundation-model wrapper.

        The cache is keyed by ``(model_key, device_str)`` only;
        ``batch_size`` is applied on every call so it can change between
        chunks without forcing a reload.

        This is what makes chunked inference over large datasets cheap:
        a caller looping ``for chunk in chunks: embedder.embed_cells(chunk)``
        pays the model-instantiation cost once, not per chunk.
        """
        from .models.singlecell_models import get_singlecell_wrapper

        cache_key = (model_key, device_str)
        cached = self._singlecell_cache.get(cache_key)
        if cached is not None:
            # Cheap attribute update -- do not re-run `.load()`.
            try:
                cached.batch_size = batch_size
            except Exception:  # noqa: BLE001
                pass
            logging.debug("Reusing cached single-cell wrapper for '%s'", model_key)
            return cached

        logging.info("Loading single-cell wrapper for '%s' on %s ...", model_key, device_str)
        wrapper = get_singlecell_wrapper(model_key, batch_size=batch_size)
        wrapper.load(device_str)
        self._singlecell_cache[cache_key] = wrapper
        return wrapper

    def clear_model_cache(self, *, which: Literal["all", "singlecell", "other"] = "all") -> None:
        """Drop cached model wrappers and free associated GPU memory.

        Parameters
        ----------
        which
            Which cache(s) to clear. ``"all"`` drops both the single-cell
            foundation-model cache and the DNA/protein/molecule/text/
            morphology cache. ``"singlecell"`` clears only the single-cell
            cache (useful when switching from foundation-model inference
            to structure/morphology work). ``"other"`` clears only the
            non-single-cell cache.

        After dropping references, runs ``gc.collect()`` +
        ``torch.cuda.empty_cache()`` so the allocator actually releases
        the GPU memory back to the system.
        """
        import gc

        if which in ("all", "singlecell"):
            n = len(self._singlecell_cache)
            self._singlecell_cache.clear()
            logging.info("Cleared %d cached single-cell wrapper(s).", n)
        if which in ("all", "other"):
            n = len(self.model_cache)
            self.model_cache.clear()
            logging.info("Cleared %d cached model wrapper(s).", n)

        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception:  # noqa: BLE001
            pass

    # ------------------------------------------------------------------
    # Standardized output API (canonical EmbeddingResult + exporters)
    # ------------------------------------------------------------------

    def embed(
        self,
        identifiers: Any | None = None,
        *,
        entity_type: str | Sequence[str] | None = None,
        model: str | Sequence[str],
        id_type: str | Mapping[str, str] | None = None,
        organism: str | None = None,
        pooling_strategy: str = "mean",
        output: Literal["anndata", "table", "payload"] | None = None,
        target: Any = None,
        input_path: str | os.PathLike[str] | None = None,
        output_path: str | os.PathLike[str] | None = None,
        attach_to: Literal["auto", "obs", "var", "uns"] = "auto",
        harmonize_dim: int | None = None,
        path: str | os.PathLike[str] | None = None,
        fmt: Literal["csv", "npz", "zarr"] = "npz",
        missing: Literal["error", "nan"] = "error",
        id_column: str | None = None,
        identifier_column: str | None = None,
        anndata_axis: Literal["obs", "var"] | None = None,
        obs_column: str | None = None,
        var_column: str | None = None,
        whole_genome: bool = False,
        biotype: str = "protein_coding",
        show_progress: bool = False,
        key: str | None = None,
        metadata_mode: Literal["minimal", "full"] = "full",
        include_matrix: bool = True,
        random_state: int = 0,
        **embed_kwargs: Any,
    ):
        """Embed entities and return the standardized embpy output.

        This is the package-wide output path. It accepts direct in-memory
        inputs (sequences, NumPy arrays, pandas Series/DataFrames, AnnData)
        and CSV/TSV/Parquet paths, normalizes them before inference,
        canonicalizes identifiers by entity type, then exports either
        AnnData or a table. Generated embeddings are never stored in
        AnnData ``.X``; standalone AnnData uses sparse placeholder ``.X``
        and stores matrices in ``.obsm``, ``.varm`` or ``.uns``.
        ``output="payload"`` returns the entity-aligned embedding payload
        directly, including canonical ids, model/provenance metadata and
        the requested embedding matrix.

        Defaults are explicit: if ``output`` is omitted, an output path
        chooses compact NPZ/CSV/Zarr file output, otherwise AnnData is
        returned. Multiple models or multiple entity types return the
        same output family with deterministic keys that include entity
        type and model name.
        """
        from .io.exporters import route_output
        from .io.normalize import normalize_embedding_input

        org = organism or self.organism
        out_path = output_path if output_path is not None else path
        data = input_path if input_path is not None else identifiers

        if data is None and target is not None and not whole_genome:
            data = target
        if data is None and not whole_genome:
            raise ValueError(
                "input normalization: no identifiers or input_path were provided. "
                "Pass identifiers, an input file path, an AnnData object, or "
                "whole_genome=True."
            )

        if target is None and data is not None and self._is_anndata_like(data):
            target = data

        entity_types = self._resolve_entity_types(entity_type, data, whole_genome)
        models = self._as_string_list(model, arg_name="model")
        normalized_by_entity = {}
        for et in entity_types:
            if whole_genome:
                normalized_by_entity[et] = self._whole_genome_input(
                    entity_type=et,
                    organism=org,
                    biotype=biotype,
                )
                continue
            entity_data = data[et] if isinstance(data, Mapping) and et in data else data
            try:
                normalized_by_entity[et] = normalize_embedding_input(
                    entity_data,
                    entity_type=et,
                    id_column=id_column,
                    identifier_column=identifier_column,
                    anndata_axis=anndata_axis,
                    obs_column=obs_column,
                    var_column=var_column,
                )
            except Exception as exc:
                if str(exc).startswith(("input loading:", "input normalization:")):
                    raise
                raise ValueError(
                    f"input normalization: failed for entity_type={et!r}: {type(exc).__name__}: {exc}"
                ) from exc

        tasks = [(et, m) for et in entity_types for m in models]
        iterator = tasks
        if show_progress:
            from tqdm.auto import tqdm

            iterator = tqdm(tasks, desc="embpy embed", unit="result")

        results = []
        for et, model_name in iterator:
            result = self._embed_to_result(
                normalized_by_entity[et],
                entity_type=et,
                model=model_name,
                id_type=self._id_type_for_entity(id_type, et),
                organism=org,
                pooling_strategy=pooling_strategy,
                show_progress=show_progress,
                **embed_kwargs,
            )
            results.append(result)

        resolved_output = output or ("table" if out_path is not None else "anndata")
        return route_output(
            results,
            output=resolved_output,
            target=target,
            attach_to=attach_to,
            harmonize_dim=harmonize_dim,
            path=out_path,
            fmt=fmt,
            missing=missing,
            key=key,
            metadata_mode=metadata_mode,
            include_matrix=include_matrix,
            random_state=random_state,
        )

    def _embed_to_result(
        self,
        identifiers: Any,
        *,
        entity_type: str,
        model: str,
        id_type: str | None,
        organism: str,
        pooling_strategy: str,
        show_progress: bool = False,
        **embed_kwargs: Any,
    ):
        """Dispatch to a batch embedder + canonicalize -> EmbeddingResult.

        The numeric embedding comes from the existing ``embed_*_batch``
        methods; canonicalization reuses the existing resolvers (the sole
        owners of id-mapping logic). Rows whose embedding or id-resolution
        fails are dropped; duplicate canonical ids collapse to the first.
        """
        from .io._canon import SCHEME, build_aliases, canonicalize, drop_and_dedup
        from .io.normalize import NormalizedInput, normalize_embedding_input
        from .io.result import EmbeddingProvenance, EmbeddingResult

        norm = (
            identifiers
            if isinstance(identifiers, NormalizedInput)
            else normalize_embedding_input(identifiers, entity_type=entity_type)
        )
        ids = list(norm.identifiers)
        alias_cols = {k: list(v) for k, v in norm.alias_columns.items()}
        requested_n = len(ids)
        extra: dict[str, Any] = {
            "entity_type": entity_type,
            "organism": organism,
            "input_kind": norm.input_kind,
            "input_source": norm.source,
            "input_id_column": norm.id_column,
            "input_id_type": id_type,
            "n_requested_inputs": requested_n,
        }
        layer = (
            embed_kwargs.get("target_layer")
            or embed_kwargs.get("layer")
            or embed_kwargs.get("embedding_layer")
            or embed_kwargs.get("layer_name")
        )
        if "region" in embed_kwargs:
            extra["region"] = embed_kwargs["region"]
        if "isoform" in embed_kwargs:
            extra["isoform_mode"] = embed_kwargs["isoform"]
        whole_genome_sequences = norm.metadata.get("prefetched_sequences")
        output_entity_type = entity_type

        if entity_type == "gene":
            it = "ensembl_id" if norm.metadata.get("whole_genome") else (id_type or "symbol")
            vecs = self.embed_genes_batch(
                model=model,
                identifiers=ids,
                id_type=it,
                organism=organism,
                pooling_strategy=pooling_strategy,
                **embed_kwargs,
                fetch_all_dna=bool(norm.metadata.get("whole_genome")),
                biotype=str(norm.metadata.get("biotype", "protein_coding")),
                prefetched_sequences=whole_genome_sequences,
            )
            raw, matrix = self._aligned_matrix(ids, vecs)
            canon, keep = canonicalize(
                raw,
                "gene",
                organism,
                id_type=it,
                gene_resolver=self.gene_resolver,
            )
            scheme = SCHEME["gene"]
        elif entity_type == "molecule":
            it = id_type or "smiles"
            smiles, name_alias = self._molecule_inputs_to_smiles(ids, it, extra)
            vecs = self.embed_molecules_batch(
                smiles,
                model,
                pooling_strategy=pooling_strategy,
                **embed_kwargs,
            )
            raw, matrix = self._aligned_matrix(smiles, vecs)
            alias_cols = self._molecule_alias_columns(raw, name_alias)
            canon, keep = canonicalize(raw, "molecule", organism, id_type=it)
            scheme = SCHEME["molecule"]
        elif entity_type == "protein":
            it = id_type or "symbol"
            d = self.embed_proteins_batch(
                ids,
                model,
                id_type=it,
                organism=organism,
                pooling_strategy=pooling_strategy,
                **embed_kwargs,
            )
            raw, matrix, protein_aliases, output_entity_type, scheme = self._dict_matrix(d)
            if output_entity_type == "protein_isoform":
                extra.setdefault("isoform_mode", "all")
                canon = [str(x) for x in raw]
                keep = np.array([True] * len(canon), dtype=bool)
                aliases = protein_aliases
            else:
                canon, keep = canonicalize(
                    raw,
                    "protein",
                    organism,
                    id_type=it,
                    protein_resolver=self.protein_resolver,
                )
                aliases = protein_aliases
                scheme = SCHEME["protein"]
        elif entity_type == "perturbation":
            morph_kwargs = dict(embed_kwargs)
            morphology_dataset = morph_kwargs.pop(
                "morphology_dataset",
                morph_kwargs.pop("dataset", "hpa"),
            )
            morphology_source = morph_kwargs.pop(
                "morphology_source",
                morph_kwargs.pop("source", "subcell"),
            )
            morphology_local_dir = morph_kwargs.pop(
                "morphology_local_dir",
                morph_kwargs.pop("local_dir", None),
            )
            aggregate = morph_kwargs.pop(
                "aggregation",
                morph_kwargs.pop("aggregate", "mean"),
            )
            perturbation_type = morph_kwargs.pop("perturbation_type", "genetic")
            max_images = morph_kwargs.pop("max_images", 5)
            plate_type = morph_kwargs.pop("plate_type", None)
            jump_profiles_dir = morph_kwargs.pop("jump_profiles_dir", None)
            skip_failures = bool(morph_kwargs.pop("skip_failures", True))
            n_workers_value = morph_kwargs.pop("n_workers", None)
            if n_workers_value is None:
                n_workers_value = morph_kwargs.pop("morphology_workers", 8)
            n_workers = int(n_workers_value)
            verbose = bool(morph_kwargs.pop("verbose", True))
            matrix, successful_labels = self.embed_perturbation_morphology_batch(
                ids,
                perturbation_type=perturbation_type,
                dataset=morphology_dataset,
                source=morphology_source,
                model=model,
                pooling_strategy=pooling_strategy,
                local_dir=morphology_local_dir,
                aggregate=aggregate,
                max_images=max_images,
                plate_type=plate_type,
                jump_profiles_dir=jump_profiles_dir,
                verbose=verbose,
                skip_failures=skip_failures,
                n_workers=n_workers,
                **morph_kwargs,
            )
            raw = [str(x) for x in successful_labels]
            alias_cols = {}
            matrix = np.asarray(matrix, dtype=np.float32)
            if matrix.ndim != 2:
                raise ValueError(f"perturbation morphology embeddings must be 2D, got {matrix.shape!r}.")
            if matrix.shape[0] == 0:
                raise ValueError(
                    "embedding generation: no perturbation morphology embeddings were produced. "
                    "Check the morphology dataset/source, local cache, max_images, and perturbation labels."
                )
            canon = raw
            keep = np.ones((len(canon),), dtype=bool)
            scheme = "perturbation_label"
            aliases = {str(label): {"perturbation_label": str(label)} for label in successful_labels}
            extra.update(
                {
                    "morphology_dataset": morphology_dataset,
                    "morphology_source": morphology_source,
                    "morphology_local_dir": morphology_local_dir,
                    "aggregation": aggregate,
                    "perturbation_type": perturbation_type,
                    "max_images": max_images,
                    "plate_type": plate_type,
                    "jump_profiles_dir": jump_profiles_dir,
                    "skip_failures": skip_failures,
                    "n_workers": n_workers,
                }
            )
        elif entity_type in ("sequence", "text"):
            raw, matrix = self._embed_raw_inputs(
                ids,
                model=model,
                pooling_strategy=pooling_strategy,
                show_progress=show_progress,
                **embed_kwargs,
            )
            canon, keep = canonicalize(
                raw,
                entity_type,
                organism,
                id_type=entity_type,
            )
            scheme = SCHEME[entity_type]
            aliases = {}
        else:
            raise ValueError(
                "embedding generation: entity_type must be one of "
                "'gene', 'molecule', 'protein', 'sequence', 'text', or 'perturbation', "
                f"got {entity_type!r}."
            )

        n_embedding_failures = max(0, requested_n - len(raw))
        n_canonical_unresolved = int((~keep).sum())
        kept_canonical = [str(c) for c, ok in zip(canon, keep, strict=False) if ok and c]
        n_duplicate_canonical = len(kept_canonical) - len(set(kept_canonical))
        canon, matrix, alias_cols, kept_raw = drop_and_dedup(
            canon,
            matrix,
            keep,
            raw,
            alias_cols,
        )
        built_aliases = build_aliases(output_entity_type, canon, kept_raw, alias_cols)
        aliases = self._merge_aliases(built_aliases, aliases if "aliases" in locals() else {})
        extra.update(
            {
                "canonical_id_scheme": scheme,
                "n_successfully_embedded_entities": len(canon),
                "n_embedding_failures": n_embedding_failures,
                "n_unresolved_identifiers": n_canonical_unresolved,
                "duplicate_canonical_ids_dropped": n_duplicate_canonical,
                "n_dropped_or_unresolved_entities": (
                    n_embedding_failures + n_canonical_unresolved + n_duplicate_canonical
                ),
            }
        )
        prov = EmbeddingProvenance.create(
            model=model,
            pooling=pooling_strategy,
            layer=layer,
            extra=extra,
        )
        return EmbeddingResult(
            matrix=matrix,
            entity_ids=tuple(canon),
            entity_type=output_entity_type,
            id_scheme=scheme,
            provenance=prov,
            aliases=aliases or None,
        )

    @staticmethod
    def _as_string_list(value: str | Sequence[str], *, arg_name: str) -> list[str]:
        if isinstance(value, str):
            return [value]
        values = [str(x) for x in value]
        if not values:
            raise ValueError(f"input normalization: {arg_name} cannot be empty.")
        return values

    def _resolve_entity_types(
        self,
        entity_type: str | Sequence[str] | None,
        data: Any,
        whole_genome: bool,
    ) -> list[str]:
        if entity_type is None:
            if whole_genome:
                return ["gene"]
            if isinstance(data, Mapping):
                return [str(k) for k in data.keys()]
            raise ValueError(
                "input normalization: entity_type is required unless identifiers "
                "is a mapping keyed by entity type or whole_genome=True."
            )
        values = self._as_string_list(entity_type, arg_name="entity_type")
        allowed = {"gene", "molecule", "protein", "sequence", "text", "perturbation"}
        bad = [x for x in values if x not in allowed]
        if bad:
            raise ValueError(
                f"input normalization: unsupported entity_type(s) {bad}; expected one of {sorted(allowed)}."
            )
        if isinstance(data, Mapping):
            missing = [x for x in values if x not in data]
            if missing and not whole_genome:
                raise ValueError(
                    "input normalization: identifiers mapping is missing "
                    f"entity_type key(s) {missing}. Available keys: {list(data.keys())}."
                )
        return values

    @staticmethod
    def _id_type_for_entity(
        id_type: str | Mapping[str, str] | None,
        entity_type: str,
    ) -> str | None:
        if isinstance(id_type, Mapping):
            return id_type.get(entity_type)
        return id_type

    @staticmethod
    def _is_anndata_like(value: Any) -> bool:
        try:
            from anndata import AnnData
        except ImportError:  # pragma: no cover
            return False
        return isinstance(value, AnnData)

    def _whole_genome_input(
        self,
        *,
        entity_type: str,
        organism: str,
        biotype: str,
    ):
        from .io.normalize import NormalizedInput

        if entity_type != "gene":
            raise ValueError(
                "input loading: whole_genome=True is currently supported for "
                f"entity_type='gene' only, got {entity_type!r}."
            )
        try:
            sequences = self.gene_resolver.get_gene_sequences(biotype=biotype)
        except Exception as exc:
            raise RuntimeError(
                "input loading: whole-genome gene request failed while reading "
                f"the genome annotation resource for organism={organism!r}, "
                f"biotype={biotype!r}: {type(exc).__name__}: {exc}. "
                "Install/index the resolver backend (for example pyensembl or "
                "a local Mart/genome resource) or pass explicit identifiers."
            ) from exc
        if not sequences:
            raise RuntimeError(
                "input loading: whole-genome gene request could not load any "
                f"Ensembl gene IDs for organism={organism!r}, biotype={biotype!r}. "
                "The resolver backend or genome annotation resource is missing "
                "or empty; install/index the resource or pass explicit identifiers."
            )
        return NormalizedInput(
            identifiers=tuple(str(x) for x in sequences.keys()),
            input_kind="whole_genome",
            source=f"whole_genome:{organism}:{biotype}",
            id_column="ensembl_gene_id",
            metadata={
                "whole_genome": True,
                "organism": organism,
                "biotype": biotype,
                "prefetched_sequences": sequences,
            },
        )

    @staticmethod
    def _merge_aliases(
        left: dict[str, dict[str, str]],
        right: dict[str, dict[str, str]],
    ) -> dict[str, dict[str, str]]:
        out = {k: dict(v) for k, v in left.items()}
        for key, mapping in right.items():
            out.setdefault(key, {}).update(mapping)
        return out

    @staticmethod
    def _molecule_alias_columns(
        raw_smiles: list[str],
        name_alias: dict[str, str],
    ) -> dict[str, list[str]]:
        names = [name_alias.get(s) for s in raw_smiles]
        return {"name": list(names)} if any(names) else {}

    def _embed_raw_inputs(
        self,
        identifiers: list[str],
        *,
        model: str,
        pooling_strategy: str,
        show_progress: bool = False,
        **kwargs: Any,
    ) -> tuple[list[str], np.ndarray]:
        inst = self._get_model(model)
        raw_iter = identifiers
        rows: list[np.ndarray] = []
        kept: list[str] = []

        if hasattr(inst, "embed_batch"):
            try:
                batch = inst.embed_batch(
                    inputs=identifiers,
                    pooling_strategy=pooling_strategy,
                    **kwargs,
                )
            except TypeError:
                batch = inst.embed_batch(
                    input=identifiers,
                    pooling_strategy=pooling_strategy,
                    **kwargs,
                )
            return self._aligned_matrix(identifiers, list(batch))

        if show_progress:
            from tqdm.auto import tqdm

            raw_iter = tqdm(identifiers, desc=f"{model} inputs", unit="input")
        for item in raw_iter:
            try:
                rows.append(
                    np.asarray(
                        inst.embed(input=item, pooling_strategy=pooling_strategy, **kwargs),
                        dtype=np.float32,
                    ).ravel()
                )
                kept.append(item)
            except Exception as exc:  # noqa: BLE001
                logging.warning("Failed to embed input %r with %s: %s", item, model, exc)
        if not rows:
            raise ValueError("embedding generation: no embeddings were produced.")
        return kept, np.stack(rows, axis=0)

    @staticmethod
    def _aligned_matrix(
        ids: list[str],
        vecs: list[np.ndarray | None],
    ) -> tuple[list[str], np.ndarray]:
        """Drop None embeddings; return (kept_ids, stacked matrix)."""
        kept_ids = [i for i, v in zip(ids, vecs, strict=False) if v is not None]
        kept = [np.asarray(v, dtype=np.float32).ravel() for v in vecs if v is not None]
        if not kept:
            raise ValueError("embedding generation: no embeddings were produced (all inputs failed).")
        return kept_ids, np.stack(kept, axis=0)

    @staticmethod
    def _dict_matrix(
        d: dict,
    ) -> tuple[list[str], np.ndarray, dict[str, dict[str, str]], str, str]:
        """Flatten a {id: vector} batch-protein result into (ids, matrix)."""
        ids: list[str] = []
        rows: list[np.ndarray] = []
        aliases: dict[str, dict[str, str]] = {}
        has_isoforms = any(isinstance(v, dict) for v in d.values())

        for k, v in d.items():
            if isinstance(v, dict):
                for iso_id, iso_vec in v.items():
                    ids.append(str(iso_id))
                    rows.append(np.asarray(iso_vec, dtype=np.float32).ravel())
                    aliases.setdefault(str(iso_id), {})["parent_input_id"] = str(k)
                continue
            ids.append(str(k))
            rows.append(np.asarray(v, dtype=np.float32).ravel())
        if not rows:
            raise ValueError("embedding generation: no protein embeddings were produced.")
        if has_isoforms:
            return ids, np.stack(rows, axis=0), aliases, "protein_isoform", "uniprot_isoform"
        return ids, np.stack(rows, axis=0), aliases, "protein", "uniprot"

    def _molecule_inputs_to_smiles(
        self,
        ids: list[str],
        id_type: str,
        extra: dict[str, Any],
    ) -> tuple[list[str], dict[str, str]]:
        """Return SMILES to embed + a {smiles: name} alias map.

        For ``id_type="smiles"`` the inputs are passed through. For
        ``id_type="name"`` each name is resolved to SMILES via the drug
        resolver chain, recording which API answered (provenance).
        """
        if id_type in ("smiles", "canonical_smiles"):
            return ids, {}
        if id_type != "name":
            raise ValueError(
                f"canonicalization: molecule id_type must be 'smiles', 'canonical_smiles', or 'name', got {id_type!r}."
            )
        from .resources.molecule.resolver import DrugResolver

        resolver = DrugResolver()
        smiles: list[str] = []
        name_alias: dict[str, str] = {}
        sources: dict[str, int] = {}
        for name in ids:
            res = resolver.name_to_smiles_resolved(name)
            if res.smiles is None:
                logging.warning("Dropping drug name %r: could not resolve to SMILES.", name)
                continue
            smiles.append(res.smiles)
            name_alias[res.smiles] = name
            sources[res.source] = sources.get(res.source, 0) + 1
        if sources:
            extra["name_resolution_sources"] = ", ".join(f"{k}:{v}" for k, v in sorted(sources.items()))
        return smiles, name_alias

    def embed_gene(
        self,
        identifier: str,
        model: str,
        id_type: Literal["symbol", "ensembl_id", "uniprot_id", "sequence"] = "symbol",
        organism: str = "human",
        pooling_strategy: str = "mean",
        region: Literal["full", "exons", "introns"] = "full",
        protein_isoform: Literal["canonical", "all"] = "canonical",
        gene_description_format: str = "Gene: {identifier}. Type: {id_type}. Organism: {organism}.",
        **kwargs: Any,
    ) -> np.ndarray:
        """
        Generates an embedding for a single gene using the specified model.

        Determines whether to use sequence (DNA/protein) or text description
        based on the selected model's type.

        Args:
            identifier (str): The gene identifier (e.g., "TP53", "ENSG00000141510")
                or a raw sequence when ``id_type="sequence"``.
            model (str): The user-facing name of the model to use (e.g., "enformer_human_rough",
                        "esm2_650M", "minilm_l6_v2"). Must be a key in MODEL_REGISTRY.
            id_type (Literal): Type of the identifier provided. Use ``"sequence"`` to
                pass a raw DNA/protein/text string directly. Defaults to "symbol".
            organism (str): Organism name (e.g., "human", "mouse"). Used for sequence/description lookup.
                            Defaults to "human".
            pooling_strategy (str): Pooling strategy if the model outputs per-token/residue embeddings.
                                    Defaults to "mean". Check model's `available_pooling_strategies`.
            region (Literal): Gene region to embed. ``"full"`` uses the complete genomic
                sequence (default), ``"exons"`` concatenates only exonic regions, and
                ``"introns"`` concatenates only intronic regions. Only applies to DNA
                models when *id_type* is ``"symbol"`` or ``"ensembl_id"``.
            protein_isoform (Literal): For protein models: ``"canonical"`` (default)
                embeds only the canonical UniProt sequence; ``"all"`` is not valid
                here -- use :meth:`embed_protein_isoforms` for all isoforms.
            gene_description_format (str): Format string used to generate input for text models.
                                        Defaults to "Gene: {identifier}. Type: {id_type}. Organism: {organism}.".
            **kwargs: Additional arguments passed to the specific model's embed method
                    (e.g., `target_layer` for transformer models).

        Returns
        -------
            np.ndarray: The computed gene embedding.

        Raises
        ------
            ModelNotFoundError: If the requested model name is not registered or available.
            IdentifierError: If the gene identifier cannot be resolved or required sequence/description fetched.
            ValueError: If the model type is ambiguous or incompatible inputs are generated.
            RuntimeError: If model loading or inference fails.
        """
        inst = self._get_model(model)
        mtype = inst.model_type

        if id_type != "sequence" and mtype == "protein":
            from embpy.resources.gene_resolver import detect_identifier_type

            detected = detect_identifier_type(identifier)
            if detected == "protein_sequence":
                logging.info(
                    "Auto-detected raw protein sequence (id_type was %r); "
                    "embedding directly. Pass id_type='sequence' explicitly "
                    "to suppress this message.",
                    id_type,
                )
                id_type = "sequence"

        if id_type == "sequence":
            input_data = identifier
        elif mtype == "dna":
            if id_type not in ("symbol", "ensembl_id"):
                raise IdentifierError(
                    f"DNA models require id_type 'symbol', 'ensembl_id', or 'sequence', got '{id_type}'."
                )
            dna_id_type: Literal["symbol", "ensembl_id"] = id_type  # type: ignore[assignment]
            if region in ("exons", "introns"):
                seq = self.gene_resolver.get_gene_region_sequence(
                    identifier,
                    id_type=dna_id_type,
                    organism=organism,
                    region=region,
                )
            elif self.resolver_backend == "local":
                seq = self.gene_resolver.get_local_dna_sequence(identifier, dna_id_type)
            else:
                seq = self.gene_resolver.get_dna_sequence(identifier, dna_id_type, organism)
            if not seq:
                raise IdentifierError(f"DNA not found for {id_type}='{identifier}' (region={region})")
            input_data = seq
        elif mtype == "protein":
            prot = self.protein_resolver.get_canonical_sequence(identifier, id_type, organism)
            if not prot:
                raise IdentifierError(f"Protein not found for {id_type}='{identifier}'")
            input_data = prot
        elif mtype == "text":
            desc = self.gene_resolver.get_gene_description(
                identifier, id_type, organism, format_string=gene_description_format
            )
            if not desc:
                raise IdentifierError(f"Description not found for {id_type}='{identifier}'")
            input_data = desc
        elif mtype == "ppi":
            input_data = identifier
        else:
            raise ValueError(f"Unsupported model type '{mtype}' for embedding.")
        # Embed
        try:
            emb = inst.embed(input=input_data, pooling_strategy=pooling_strategy, **kwargs)
            return emb
        except Exception as e:
            logging.error(f"Embedding error for {identifier} with model {model}: {e}")
            raise RuntimeError(f"Embedding failed for {identifier}") from e

    def embed_protein(
        self,
        identifier: str,
        model: str,
        id_type: Literal["symbol", "ensembl_id", "uniprot_id", "sequence"] = "symbol",
        organism: str = "human",
        pooling_strategy: str = "mean",
        isoform: Literal["canonical", "all"] = "canonical",
        **kwargs: Any,
    ) -> np.ndarray | dict[str, np.ndarray]:
        """Embed a protein sequence using a protein language model.

        When ``isoform="canonical"``, returns a single embedding of the
        canonical UniProt sequence.  When ``isoform="all"``, returns a
        dict mapping isoform accession IDs to their embeddings.

        Parameters
        ----------
        identifier
            Gene symbol, Ensembl ID, UniProt accession, or raw amino
            acid sequence (when ``id_type="sequence"``).
        model
            Protein model name (e.g. ``"esm2_650M"``, ``"prot_t5_xl"``).
        id_type
            Type of identifier.
        organism
            Organism name (default ``"human"``).
        pooling_strategy
            Pooling strategy (``"mean"``, ``"max"``, ``"cls"``).
        isoform
            ``"canonical"`` for the canonical sequence only, ``"all"``
            to embed every UniProt isoform.
        **kwargs
            Forwarded to the model's ``embed`` method.

        Returns
        -------
        np.ndarray
            When ``isoform="canonical"``.
        dict[str, np.ndarray]
            When ``isoform="all"`` -- maps isoform accession to embedding.
        """
        inst = self._get_model(model)
        if inst.model_type != "protein":
            raise ValueError(f"Model '{model}' is not a protein model (type={inst.model_type}).")

        if id_type == "sequence":
            emb = inst.embed(input=identifier, pooling_strategy=pooling_strategy, **kwargs)
            return emb

        if isoform == "canonical":
            seq = self.protein_resolver.get_canonical_sequence(identifier, id_type, organism)
            if not seq:
                raise IdentifierError(f"Canonical protein not found for {id_type}='{identifier}'")
            return inst.embed(input=seq, pooling_strategy=pooling_strategy, **kwargs)

        # isoform == "all"
        isoforms = self.protein_resolver.get_isoforms(
            identifier,
            id_type,
            organism,
            include_canonical=True,
        )
        if not isoforms:
            raise IdentifierError(f"No isoforms found for {id_type}='{identifier}'")

        results: dict[str, np.ndarray] = {}
        for iso_id, seq in isoforms.items():
            try:
                results[iso_id] = inst.embed(
                    input=seq,
                    pooling_strategy=pooling_strategy,
                    **kwargs,
                )
            except Exception as e:  # noqa: BLE001
                logging.warning(f"Failed to embed isoform {iso_id}: {e}")
        return results

    def embed_proteins_batch(
        self,
        identifiers: list[str],
        model: str,
        id_type: Literal["symbol", "ensembl_id", "uniprot_id", "sequence"] = "symbol",
        organism: str = "human",
        pooling_strategy: str = "mean",
        isoform: Literal["canonical", "all"] = "canonical",
        **kwargs: Any,
    ) -> dict[str, np.ndarray] | dict[str, dict[str, np.ndarray]]:
        """Embed proteins for a batch of gene identifiers.

        Parameters
        ----------
        identifiers
            List of gene identifiers or raw amino-acid sequences
            (when ``id_type="sequence"``).
        model
            Protein model name.
        id_type
            Type of identifiers.  Use ``"sequence"`` to embed raw
            amino-acid strings directly without resolver lookup.
        organism
            Organism name.
        pooling_strategy
            Pooling strategy.
        isoform
            ``"canonical"`` or ``"all"``.  Ignored when
            ``id_type="sequence"``.
        **kwargs
            Forwarded to the model.

        Returns
        -------
        dict[str, np.ndarray]
            When ``isoform="canonical"`` or ``id_type="sequence"``
            -- maps identifier (or sequence) to embedding.
        dict[str, dict[str, np.ndarray]]
            When ``isoform="all"`` -- maps identifier to
            ``{isoform_accession: embedding}``.
        """
        inst = self._get_model(model)
        if inst.model_type != "protein":
            raise ValueError(f"Model '{model}' is not a protein model.")

        if id_type == "sequence":
            results: dict[str, np.ndarray] = {}
            for seq in identifiers:
                try:
                    results[seq] = inst.embed(
                        input=seq,
                        pooling_strategy=pooling_strategy,
                        **kwargs,
                    )
                except Exception as e:  # noqa: BLE001
                    logging.warning(f"Failed to embed protein sequence: {e}")
            return results

        if isoform == "canonical":
            seqs = self.protein_resolver.get_canonical_sequences_batch(
                identifiers,
                id_type,
                organism,
            )
            results = {}
            for ident, seq in seqs.items():
                try:
                    results[ident] = inst.embed(
                        input=seq,
                        pooling_strategy=pooling_strategy,
                        **kwargs,
                    )
                except Exception as e:  # noqa: BLE001
                    logging.warning(f"Failed to embed protein for {ident}: {e}")
            return results

        # isoform == "all"
        all_isoforms = self.protein_resolver.get_isoforms_batch(
            identifiers,
            id_type,
            organism,
            include_canonical=True,
        )
        results_iso: dict[str, dict[str, np.ndarray]] = {}
        for ident, isoforms_map in all_isoforms.items():
            results_iso[ident] = {}
            for iso_id, seq in isoforms_map.items():
                try:
                    results_iso[ident][iso_id] = inst.embed(
                        input=seq,
                        pooling_strategy=pooling_strategy,
                        **kwargs,
                    )
                except Exception as e:  # noqa: BLE001
                    logging.warning(f"Failed to embed isoform {iso_id} for {ident}: {e}")
        return results_iso

    def embed_genes_batch(
        self,
        model: str,
        identifiers: Sequence[str] | None = None,
        id_type: Literal["symbol", "ensembl_id", "uniprot_id", "sequence"] = "symbol",
        organism: str = "human",
        pooling_strategy: str = "mean",
        region: Literal["full", "exons", "introns"] = "full",
        gene_description_format: str | None = None,
        fetch_all_dna: bool = False,
        biotype: str = "protein_coding",
        prefetched_sequences: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> list[np.ndarray | None]:
        """
        Generates embeddings for a batch of genes.

        If `identifiers` is None and `fetch_all_dna` is True, it will automatically
        fetch ALL genes of the specified `biotype` and embed them.

        Set ``region`` to ``"exons"`` or ``"introns"`` to embed only those
        gene regions (DNA models only).
        """
        inst = self._get_model(model)
        mtype = inst.model_type

        # Dictionary to hold pre-fetched sequences (if any)
        prefetched_data: dict[str, str] = dict(prefetched_sequences or {})
        if prefetched_data and identifiers is None:
            identifiers = list(prefetched_data.keys())
            id_type = "ensembl_id"
            logging.info("Using %d prefetched whole-genome sequences.", len(prefetched_data))

        # --- LOGIC CHANGE: Discovery Mode ---
        # If identifiers is None OR we explicitly want to pre-fetch
        if (fetch_all_dna or identifiers is None) and not prefetched_data:
            if self.resolver_backend == "api":
                logging.info(f"Fetching list of all '{biotype}' genes via API...")
                # This returns {ensembl_id: dna_sequence} or None
                fetched = self.gene_resolver.get_gene_sequences(biotype=biotype)
                if fetched is not None:
                    prefetched_data = fetched

                if identifiers is None:
                    # USE DISCOVERED GENES
                    if prefetched_data:
                        identifiers = list(prefetched_data.keys())
                        logging.info(f"Auto-discovered {len(identifiers)} genes.")
                        # Force ID type to Ensembl ID as that's what get_gene_sequences returns
                        id_type = "ensembl_id"
                    else:
                        logging.error(f"No genes found for biotype '{biotype}'.")
                        return []
            else:
                if identifiers is None:
                    logging.error("Cannot auto-discover genes with 'local' backend. Provide identifiers.")
                    return []
                logging.warning("fetch_all_dna is ignored in local mode.")

        if not identifiers:
            self.last_report = ResolutionReport(model_name=model, organism=organism)
            return []

        # Layer 2: fresh ResolutionReport for this batch. Every input
        # gets exactly one ``ResolutionRecord`` regardless of whether
        # it resolves, fails the resolver, fails the embedder, or hits
        # the OOM bisector. Replaces ``self.last_report`` so callers
        # can grab it after the call returns.
        report = ResolutionReport(model_name=model, organism=organism)
        self.last_report = report

        input_data_list: list[str | None] = []
        # Track per-input resolver state so we can attach reason codes
        # AFTER the embed_batch result is known (a resolver hit that
        # later fails to embed should be reported separately from a
        # resolver miss).
        per_input_state: list[dict[str, Any]] = []
        logging.info(f"Batch: {len(identifiers)} items for '{model}' ({mtype})...")

        # Narrow id_type for DNA resolver methods (only accept symbol/ensembl_id)
        dna_id: Literal["symbol", "ensembl_id"] | None = (
            id_type if id_type in ("symbol", "ensembl_id") else None  # type: ignore[assignment]
        )

        for ident in identifiers:
            data = None
            # Per-input audit state for Layer 2 reporting.
            state: dict[str, Any] = {
                "source": None,
                "reason": None,
                "attempted": [],
                "latency_ms": 0.0,
            }
            try:
                with time_block(
                    "resolver_input",
                    model=model,
                    identifier=ident,
                    id_type=id_type,
                    mtype=mtype,
                ) as ev:
                    if id_type == "sequence":
                        data = ident
                        state["source"] = "passthrough"
                    elif mtype == "dna":
                        state["attempted"].append("gene_resolver_dna")
                        if region in ("exons", "introns") and dna_id is not None:
                            data = self.gene_resolver.get_gene_region_sequence(
                                ident,
                                id_type=dna_id,
                                organism=organism,
                                region=region,
                            )
                            state["source"] = f"gene_resolver_region:{region}"
                        else:
                            if prefetched_data:
                                if id_type == "ensembl_id":
                                    data = prefetched_data.get(ident)
                                    if data is not None:
                                        state["source"] = "prefetched"

                            if data is None:
                                if dna_id is None:
                                    logging.warning(
                                        "DNA models require id_type 'symbol', 'ensembl_id', or 'sequence', "
                                        "got '%s'; skipping %s.",
                                        id_type,
                                        ident,
                                    )
                                    state["reason"] = f"resolver:unsupported_id_type:{id_type}"
                                elif self.resolver_backend == "local":
                                    data = self.gene_resolver.get_local_dna_sequence(ident, dna_id)
                                    state["source"] = "gene_resolver_local"
                                else:
                                    data = self.gene_resolver.get_dna_sequence(ident, dna_id, organism)
                                    state["source"] = "gene_resolver_api"

                    elif mtype == "protein":
                        state["attempted"].append("protein_resolver")
                        data = self.protein_resolver.get_canonical_sequence(ident, id_type, organism)
                        state["source"] = "protein_resolver"

                    elif mtype == "text":
                        # Default template combines the input identifier
                        # with MyGene.info fields. ``_SafeFormatDict`` in
                        # the resolver tolerates missing fields, so genes
                        # with no ``summary`` on MyGene still produce a
                        # thin but non-None description (e.g. just "Gene
                        # TP53 (human). TP53: tumor protein p53.").
                        state["attempted"].append("mygene")
                        fmt = gene_description_format or ("Gene {identifier} ({organism}). {symbol}: {name}. {summary}")
                        data = self.gene_resolver.get_gene_description(
                            ident,
                            id_type,
                            organism,
                            format_string=fmt,
                        )
                        state["source"] = "mygene_description"

                    if data is None:
                        logging.warning(f"No data for {ident}; skipping.")
                        ev["status"] = "no_hit"
                        if state["reason"] is None:
                            state["reason"] = f"{mtype}_resolver:no_hit"
            except Exception as e:  # noqa: BLE001
                # We deliberately swallow resolver-side exceptions per
                # input (the resolver wraps its own errors in None
                # returns; anything still escaping here is unexpected
                # but should not nuke a 2000-gene batch). Layer 2
                # records the failure so callers can audit.
                logging.warning(f"Error fetching {ident}: {e}")
                state["reason"] = f"resolver_exception:{type(e).__name__}"
                log_event(
                    "resolver_input_exception",
                    level="warn",
                    model=model,
                    identifier=ident,
                    error=type(e).__name__,
                    error_msg=str(e)[:200],
                )
            input_data_list.append(data)
            per_input_state.append(state)

        valid_inputs = [d for d in input_data_list if d is not None]
        valid_indices = [i for i, d in enumerate(input_data_list) if d is not None]

        if not valid_inputs:
            # Every input failed the resolver -- record each one as
            # UNRESOLVED with the per-input reason captured above and
            # return early. This is the failure mode that used to be
            # silently swallowed under "BioEmbedder returned no
            # embeddings"; now the sidecar carries the actionable
            # detail.
            for ident, state in zip(identifiers, per_input_state, strict=False):
                report.record(
                    ident,
                    "unresolved",
                    source=state["source"],
                    reason=state["reason"] or "resolver:no_data",
                    latency_ms=float(state["latency_ms"]),
                    attempted_sources=list(state["attempted"]),
                )
            return [None] * len(identifiers)

        # Layer 3: wrap the model forward pass in an OOM-bisection
        # helper. If the wrapper already does its own batching (most
        # of them do, after the protein / DNA fixes), the outer call
        # succeeds and the wrapper is responsible for chunking. If
        # the wrapper raises an OOM-shaped exception, the bisector
        # halves the input list and retries, isolating the specific
        # input that genuinely cannot fit. The single-item case that
        # still OOMs propagates so it can be classified as
        # ContextOverflowError / ModelOOMError below.
        def _do_embed(batch: Sequence[str]) -> list[Any]:
            return list(
                inst.embed_batch(
                    inputs=list(batch),
                    pooling_strategy=pooling_strategy,
                    **kwargs,
                )
            )

        try:
            with time_block(
                "model_forward",
                model=model,
                mtype=mtype,
                n_inputs=len(valid_inputs),
                device=str(self.device),
            ):
                batch_results = embed_batch_with_oom_recovery(
                    _do_embed,
                    valid_inputs,
                    on_split=lambda left, right: log_event(
                        "oom_bisect",
                        level="warn",
                        model=model,
                        left=left,
                        right=right,
                    ),
                )
        except EmbpyError:
            # Already a typed embpy exception (e.g. ModelOOMError raised by
            # a wrapper that classifies OOMs itself). Let it propagate
            # unchanged -- wrapping would only obscure the original fields.
            traceback.print_exc()
            raise
        except Exception as e:
            # Classify the failure so the upstream entry point can give a
            # category-specific exit code without parsing stderr. The
            # historical behaviour ("swallow + return [None]*N") masked
            # CUDA OOMs, context overflows, missing CUDA kernels, and
            # missing pip packages behind a single misleading "no
            # embeddings" error several layers up. We refuse to do that.
            logging.error(
                "Batch embed failed for model='%s' (n_inputs=%d): %s",
                model,
                len(valid_inputs),
                e,
            )
            traceback.print_exc()
            typed = _classify_embedder_exception(
                e,
                model_name=model,
                n_inputs=len(valid_inputs),
                device=str(self.device),
            )
            raise typed from e

        results: list[np.ndarray | None] = [None] * len(identifiers)
        for idx, emb in zip(valid_indices, batch_results, strict=False):
            results[idx] = emb

        # Layer 2: now that we know which inputs ended up with an
        # embedding (and which were resolver-misses), record one
        # ResolutionRecord per input. Embedder-side failures (OOM,
        # context overflow) raise typed errors above and never reach
        # this point, so anything in ``results[i] is None`` here is a
        # resolver miss with state already captured.
        valid_index_set = set(valid_indices)
        for i, (ident, state) in enumerate(zip(identifiers, per_input_state, strict=False)):
            if i in valid_index_set and results[i] is not None:
                report.record(
                    ident,
                    "resolved",
                    source=state["source"],
                    latency_ms=float(state["latency_ms"]),
                    attempted_sources=list(state["attempted"]),
                )
            else:
                report.record(
                    ident,
                    "unresolved",
                    source=state["source"],
                    reason=state["reason"] or f"{mtype}_resolver:no_data",
                    latency_ms=float(state["latency_ms"]),
                    attempted_sources=list(state["attempted"]),
                )

        return results

    def embed_molecule(
        self,
        identifier: str,  # Expecting SMILES string
        model: str,  # e.g. "chemberta_zinc_v1"
        pooling_strategy: str = "mean",
        **kwargs: Any,
    ) -> np.ndarray:
        """
        Generate an embedding for a single small molecule.

        Validates that the input is a proper SMILES string before embedding.

        Args:
            identifier (str):
                A SMILES string representing the molecule (e.g., "CCO" for ethanol).
            model (str):
                Key of the molecule embedding model to use (e.g., "chemberta_zinc_v1").
            pooling_strategy (str, optional):
                Pooling strategy to aggregate token‐level embeddings.
                Defaults to "mean".
            **kwargs:
                Additional keyword arguments forwarded to the model’s `embed` method.

        Returns
        -------
            np.ndarray:
                A 1D NumPy array representing the molecule embedding.

        Raises
        ------
            ModelNotFoundError:
                If the specified model key is not registered.
            ValueError:
                If the chosen model is not of type "molecule", or if the SMILES string is invalid.
            RuntimeError:
                If the embedding process itself fails.
        """
        inst = self._get_model(model)
        if inst.model_type != "molecule":
            raise ValueError(f"Model '{model}' is not a molecule embedder.")

        smiles = identifier
        # Validate SMILES
        if Chem.MolFromSmiles(smiles) is None:
            raise ValueError(f"Invalid SMILES string: '{smiles}'")

        logging.debug(f"Embedding molecule SMILES: {smiles} with model '{model}'")
        try:
            emb = inst.embed(input=smiles, pooling_strategy=pooling_strategy, **kwargs)
            logging.debug(f"Molecule embedding shape: {emb.shape}")
            return emb
        except Exception as e:
            logging.error(f"Embedding failed for SMILES '{smiles}': {e}")
            raise RuntimeError(f"Embedding failed for SMILES '{smiles}'") from e

    def embed_molecules_batch(
        self,
        identifiers: Sequence[str],
        model: str,
        pooling_strategy: str = "mean",
        **kwargs: Any,
    ) -> list[np.ndarray | None]:
        """
        Generate embeddings for a batch of small molecules.

        Invalid SMILES strings are skipped and return `None` in their positions.

        Args:
            identifiers (Sequence[str]):
                A list of SMILES strings to embed.
            model (str):
                Key of the molecule embedding model to use.
            pooling_strategy (str, optional):
                Pooling strategy to apply. Defaults to "mean".
            **kwargs:
                Additional keyword arguments forwarded to `embed_batch`.

        Returns
        -------
            List[Optional[np.ndarray]]:
                A list of embeddings (NumPy arrays) for valid SMILES, with `None`
                for any invalid or failed entries. Order aligns with input list.

        Raises
        ------
            ModelNotFoundError:
                If the specified model key is not registered.
            ValueError:
                If the chosen model is not of type "molecule".
        """
        inst = self._get_model(model)
        if inst.model_type != "molecule":
            raise ValueError(f"Model '{model}' is not a molecule embedder.")

        valid_inputs: list[str] = []
        valid_indices: list[int] = []
        for idx, smi in enumerate(identifiers):
            if Chem.MolFromSmiles(smi) is not None:
                valid_inputs.append(smi)
                valid_indices.append(idx)
            else:
                logging.warning(f"Skipping invalid SMILES: '{smi}'")

        # Prepare output list
        results: list[np.ndarray | None] = [None] * len(identifiers)
        if not valid_inputs:
            logging.warning("No valid SMILES provided; returning all None.")
            return results

        logging.info(f"Embedding {len(valid_inputs)} valid SMILES with model '{model}'")
        try:
            batch_embs = inst.embed_batch(input=valid_inputs, pooling_strategy=pooling_strategy, **kwargs)
            for out_idx, emb in zip(valid_indices, batch_embs, strict=False):
                results[out_idx] = emb
        except Exception as e:  # noqa: BLE001
            logging.error(f"Batch embedding failed: {e}")
            # Leave all as None

        return results

    def embed_text(
        self,
        text: str,
        model: str,  # Can be registry name OR any HF model identifier
        pooling_strategy: str = "mean",
        **kwargs: Any,
    ) -> np.ndarray:
        """
        Generates an embedding for an arbitrary text string using a text model.

        Args:
            text (str): The input text string.
            model (str): The name of the text embedding model. Can be either:
                        - A predefined model name (e.g., "minilm_l6_v2", "bert_base_uncased")
                        - Any Hugging Face model identifier (e.g., "sentence-transformers/all-MiniLM-L6-v2")
            pooling_strategy (str): Pooling strategy. Defaults to "mean".
            **kwargs: Additional arguments for the text model's embed method.

        Returns
        -------
            np.ndarray: The computed text embedding.

        Raises
        ------
            ModelNotFoundError: If the model cannot be loaded from HF or found in registry.
            ValueError: If the loaded model is not a text model.
            RuntimeError: If model loading or inference fails.
        """
        model_instance = self._get_model(model)
        if model_instance.model_type != "text":
            raise ValueError(f"Model '{model}' is not a text embedder. Use embed_gene or embed_molecule.")

        logging.debug(f"Embedding text: '{text[:100]}...' using model '{model}'")
        try:
            embedding = model_instance.embed(input=text, pooling_strategy=pooling_strategy, **kwargs)
            logging.debug(f"Text embedding generated with shape: {embedding.shape}")
            return embedding
        except Exception as e:
            logging.error(f"Error during text embedding generation for model '{model}': {e}")
            raise RuntimeError(f"Text embedding failed for input '{text[:50]}...'.") from e

    def embed_text_api(
        self,
        text: str,
        model: str = "text-embedding-3-small",
        provider: str = "openai",
        api_key: str | None = None,
        base_url: str | None = None,
        **kwargs: Any,
    ) -> np.ndarray:
        """Embed text using an API-based model with explicit provider config.

        Convenience method for one-off API calls where you want to
        specify the provider, API key, and/or base URL directly
        without pre-registering the model.

        Parameters
        ----------
        text
            Text string to embed.
        model
            Model name as expected by the API.
        provider
            ``"openai"``, ``"cohere"``, ``"voyage"``, ``"google"``,
            or ``"generic"`` (OpenAI-compatible endpoint).
        api_key
            API key (or set the provider's env var).
        base_url
            Custom API endpoint URL.
        **kwargs
            Forwarded to the API wrapper.

        Returns
        -------
        np.ndarray
            Embedding vector.
        """
        from .models.api_models import APIEmbeddingWrapper

        cache_key = f"_api_{provider}_{model}"
        if cache_key not in self.model_cache:
            wrapper = APIEmbeddingWrapper(
                model_path_or_name=model,
                provider=provider,
                api_key=api_key,
                base_url=base_url,
            )
            wrapper.load(self.device)
            self.model_cache[cache_key] = wrapper

        return self.model_cache[cache_key].embed(input=text, **kwargs)

    def embed_texts_batch(
        self,
        texts: Sequence[str],
        model: str,
        pooling_strategy: str = "mean",
        batch_size: int | None = None,
        **kwargs: Any,
    ) -> list[np.ndarray | None]:  # Return None for errors
        """
        Generates embeddings for a batch of arbitrary text strings.

        Args:
            texts (Sequence[str]): A list or tuple of text strings.
            model (str): The name of the text embedding model.
            pooling_strategy (str): Pooling strategy. Defaults to "mean".
            batch_size (Optional[int]): Maximum batch size for processing. If None, processes all texts at once.
                                       Use smaller values to avoid OOM errors with large datasets.
            **kwargs: Additional arguments for the model's embed_batch method.

        Returns
        -------
            list[Optional[np.ndarray]]: List of embeddings, with None for texts that failed.
        """
        model_instance = self._get_model(model)
        if model_instance.model_type != "text":
            raise ValueError(f"Model '{model}' is not a text embedder.")

        valid_inputs = list(texts)
        logging.info(f"Embedding batch of {len(valid_inputs)} texts using model '{model}'...")

        try:
            batch_results = model_instance.embed_batch(
                inputs=valid_inputs,
                pooling_strategy=pooling_strategy,
                batch_size=batch_size,
                **kwargs,
            )
            if len(batch_results) != len(valid_inputs):
                logging.error(
                    f"Batch embedding returned {len(batch_results)} results for {len(valid_inputs)} inputs. Mismatch!"
                )
                return [None] * len(texts)
            logging.info("Batch text embedding successful.")
            results: list[np.ndarray | None] = list(batch_results)

        except (ValueError, KeyError) as e:
            logging.error(f"Error during batch text embedding generation for model '{model}': {e}")
            results = [None] * len(texts)

        return results

    def embed_description(
        self,
        identifier: str,
        model: str = "minilm_l6_v2",
        entity_type: Literal["gene", "protein", "molecule", "auto"] = "auto",
        sources: list[str] | str = "all",
        pooling_strategy: str = "mean",
        **kwargs: Any,
    ) -> np.ndarray:
        """Fetch text description(s) for a biological entity and embed them.

        Resolves the identifier to rich text descriptions from public
        knowledge sources (MyGene, NCBI, Ensembl, UniProt, Wikipedia,
        PubChem), combines them, and embeds the result with a text model.

        Parameters
        ----------
        identifier
            Gene symbol, Ensembl ID, UniProt accession, drug name,
            or SMILES string.
        model
            Text embedding model (default ``"minilm_l6_v2"``).
        entity_type
            ``"gene"``, ``"protein"``, ``"molecule"``, or ``"auto"``
            to auto-detect.
        sources
            Knowledge sources to query. ``"all"`` uses the defaults
            for the detected entity type.
        pooling_strategy
            Pooling strategy for the text model.
        **kwargs
            Forwarded to the text model's ``embed`` method.

        Returns
        -------
        np.ndarray
            The text embedding vector.

        Examples
        --------
        >>> emb = embedder.embed_description("TP53", model="minilm_l6_v2")
        >>> emb.shape
        (384,)
        >>> emb = embedder.embed_description("aspirin", entity_type="molecule")
        """
        description = self.text_resolver.get_combined_description(
            identifier,
            entity_type=entity_type,
            sources=sources,
        )
        logging.info(
            "Description for '%s' (%d chars): %s...",
            identifier,
            len(description),
            description[:100],
        )
        return self.embed_text(
            text=description,
            model=model,
            pooling_strategy=pooling_strategy,
            **kwargs,
        )

    def embed_descriptions_batch(
        self,
        identifiers: Sequence[str],
        model: str = "minilm_l6_v2",
        entity_type: Literal["gene", "protein", "molecule", "auto"] = "auto",
        sources: list[str] | str = "all",
        pooling_strategy: str = "mean",
        batch_size: int | None = None,
        **kwargs: Any,
    ) -> list[np.ndarray | None]:
        """Fetch and embed text descriptions for a batch of entities.

        Parameters
        ----------
        identifiers
            List of biological entity identifiers.
        model
            Text embedding model.
        entity_type
            Entity type (applied to all identifiers).
        sources
            Knowledge sources to query.
        pooling_strategy
            Pooling strategy for the text model.
        batch_size
            Maximum batch size for embedding inference.
        **kwargs
            Forwarded to the text model.

        Returns
        -------
        List of embedding arrays, with ``None`` for failed lookups.
        """
        texts: list[str] = []
        valid_indices: list[int] = []

        for i, ident in enumerate(identifiers):
            try:
                desc = self.text_resolver.get_combined_description(
                    ident,
                    entity_type=entity_type,
                    sources=sources,
                )
                if desc:
                    texts.append(desc)
                    valid_indices.append(i)
                else:
                    logging.warning("No description found for '%s'", ident)
            except Exception as e:  # noqa: BLE001
                logging.warning("Failed to fetch description for '%s': %s", ident, e)

        if not texts:
            return [None] * len(identifiers)

        batch_embs = self.embed_texts_batch(
            texts=texts,
            model=model,
            pooling_strategy=pooling_strategy,
            batch_size=batch_size,
            **kwargs,
        )

        results: list[np.ndarray | None] = [None] * len(identifiers)
        for idx, emb in zip(valid_indices, batch_embs, strict=False):
            results[idx] = emb

        return results

    def embed_cells(
        self,
        adata,  # anndata.AnnData
        models: list[str] | str = "scgpt",
        preprocessing: Literal["raw", "standard", "none"] = "standard",
        *,
        # Preprocessing params (forwarded to preprocess_counts)
        target_sum: float | None = 1e4,
        n_top_genes: int = 2000,
        log_transform: bool = True,
        scale: bool = False,
        max_value: float | None = 10.0,
        min_genes: int = 200,
        min_cells: int = 3,
        max_pct_mito: float | None = None,
        # PCA-specific
        n_pca_components: int = 50,
        pca_use_hvg: bool = True,
        # scVI-specific
        n_latent: int = 30,
        n_layers_scvi: int = 2,
        n_hidden_scvi: int = 128,
        max_epochs: int = 200,
        early_stopping: bool = True,
        batch_key: str | None = None,
        labels_key: str | None = None,
        protein_expression_obsm_key: str | None = None,
        # General
        batch_size: int = 32,
        obsm_prefix: str = "X_",
        copy: bool = True,
        backend: Literal["cpu", "gpu"] = "cpu",
        vocab_conversion: Literal["auto", "off"] = "auto",
    ):
        """Embed single cells from an AnnData object.

        Orchestrates preprocessing and multi-model embedding in a single
        call.  Each requested model's embeddings are stored in
        ``adata.obsm`` under ``"{obsm_prefix}{model_name}"``.

        Parameters
        ----------
        adata : anndata.AnnData
            AnnData with raw counts in ``.X``.
        models : str or list[str]
            Model key(s) from the single-cell model registry.  Accepts
            any key from :func:`~embpy.models.singlecell_models.list_singlecell_models`,
            e.g. ``"scgpt"``, ``"geneformer_v2_12L"``, ``"pca"``,
            ``"scvi"``, ``"scanvi"``, ``"totalvi"``.
        preprocessing : {"raw", "standard", "none"}
            ``"standard"`` runs log-normalize + HVG.
            ``"raw"`` applies only QC filtering.
            ``"none"`` skips preprocessing entirely.
        target_sum
            Target total counts for normalization (standard pipeline).
        n_top_genes
            Number of highly variable genes (standard pipeline).
        log_transform
            Whether to log1p-transform (standard pipeline).
        scale
            Whether to scale to unit variance (standard pipeline).
        max_value
            Max value after scaling.
        min_genes
            Minimum genes per cell for QC.
        min_cells
            Minimum cells per gene for QC.
        max_pct_mito
            Maximum mitochondrial fraction (``None`` to skip).
        n_pca_components
            Number of PCA components (when ``"pca"`` is in models).
        pca_use_hvg
            Whether PCA restricts to HVGs.
        n_latent
            scvi-tools latent dimensionality.
        n_layers_scvi
            Number of hidden layers in scvi-tools encoder/decoder.
        n_hidden_scvi
            Number of nodes per hidden layer in scvi-tools.
        max_epochs
            Maximum training epochs for scvi-tools.
        early_stopping
            Whether to use early stopping for scvi-tools.
        batch_key
            Batch column in ``adata.obs`` for scvi-tools batch correction.
        labels_key
            Label column in ``adata.obs`` for scANVI.
        protein_expression_obsm_key
            Key in ``adata.obsm`` with protein counts for totalVI.
        batch_size
            Batch size for foundation model inference.
        obsm_prefix
            Prefix for ``.obsm`` keys (default ``"X_"``).
        copy
            If ``True``, operate on a copy of adata.

        Returns
        -------
        anndata.AnnData
            The AnnData with:

            - ``.X`` = original raw counts
            - ``.layers["counts"]`` = raw counts copy
            - ``.layers["log_normalized"]`` = processed (standard pipeline)
            - ``.obsm["{prefix}{model}"]`` = embeddings per model
            - ``.uns["embpy_cell_embeddings"]`` = metadata dict
        """
        from .models.singlecell_models import (
            PCAEmbedding,
            ScVIToolsWrapper,
            list_singlecell_models,
        )
        from .pp.sc_preprocessing import preprocess_counts

        if isinstance(models, str):
            models = [models]

        available = list_singlecell_models()
        for m in models:
            if m not in available:
                raise ValueError(f"Unknown single-cell model '{m}'. Available: {available}")

        if copy:
            adata = adata.copy()

        # ---- Preprocessing -----------------------------------------------
        if preprocessing != "none":
            adata = preprocess_counts(
                adata,
                pipeline=preprocessing,
                min_genes=min_genes,
                min_cells=min_cells,
                max_pct_mito=max_pct_mito,
                target_sum=target_sum,
                n_top_genes=n_top_genes,
                log_transform=log_transform,
                scale=scale,
                max_value=max_value,
                copy=False,
                backend=backend,
            )

        # ---- Embed with each model ---------------------------------------
        device_str = str(self.device)
        metadata: dict[str, dict[str, Any]] = {}

        for model_key in models:
            obsm_key = f"{obsm_prefix}{model_key}"
            logging.info("Embedding cells with '%s' ...", model_key)

            try:
                if model_key == "pca":
                    wrapper = PCAEmbedding(
                        n_components=n_pca_components,
                        use_hvg=pca_use_hvg,
                        layer="log_normalized",
                        scale=True,
                        backend=backend,
                    )
                    wrapper.load(device_str)
                    embs = wrapper.embed_cells(adata)
                    # Park the fitted wrapper so decode_cells() can reuse
                    # the sklearn/cuml PCA + scaler for inverse_transform.
                    self._singlecell_cache[(model_key, device_str)] = wrapper

                elif model_key in ("scvi", "scanvi", "totalvi"):
                    model_cls_name = model_key.upper()
                    if model_cls_name == "SCVI":
                        model_cls_name = "SCVI"
                    elif model_cls_name == "SCANVI":
                        model_cls_name = "SCANVI"
                    elif model_cls_name == "TOTALVI":
                        model_cls_name = "TOTALVI"

                    wrapper = ScVIToolsWrapper(
                        model_class=model_cls_name,
                        n_latent=n_latent,
                        n_layers=n_layers_scvi,
                        n_hidden=n_hidden_scvi,
                        max_epochs=max_epochs,
                        early_stopping=early_stopping,
                        batch_key=batch_key,
                        labels_key=labels_key,
                        protein_expression_obsm_key=protein_expression_obsm_key,
                        layer="counts",
                        batch_size=batch_size,
                    )
                    wrapper.load(device_str)
                    embs = wrapper.embed_cells(adata)
                    # Park the trained scvi-tools model so decode_cells()
                    # can route arbitrary latents through its generative
                    # module without retraining.
                    self._singlecell_cache[(model_key, device_str)] = wrapper

                else:
                    # Foundation models (scGPT, Geneformer, UCE, Tahoe, ...)
                    # are cached by (model_key, device) so repeated calls
                    # to embed_cells -- e.g. chunked inference over a large
                    # AnnData -- reuse the already-loaded torch model
                    # instead of re-instantiating it on every call.
                    wrapper = self._get_or_load_singlecell_wrapper(model_key, batch_size, device_str)
                    # Auto-adapt var_names to the model's vocabulary.
                    # This is the difference between a silent zero-match
                    # failure (e.g. passing Ensembl IDs to scGPT) and a
                    # working embedding. Set vocab_conversion="off" to
                    # opt out of the automatic GeneResolver roundtrip.
                    if vocab_conversion == "auto":
                        adata_for_model, report = self._ensure_singlecell_vocabulary(
                            adata, model_key, organism=self.organism
                        )
                        metadata[f"{model_key}__vocab"] = report
                    else:
                        adata_for_model = adata
                    embs = wrapper.embed_cells(adata_for_model)

                adata.obsm[obsm_key] = embs
                metadata[model_key] = {
                    "obsm_key": obsm_key,
                    "embedding_dim": embs.shape[1],
                    "n_cells": embs.shape[0],
                    "wrapper_class": type(wrapper).__name__,
                }
                logging.info(
                    "  -> stored in .obsm['%s'], shape %s",
                    obsm_key,
                    embs.shape,
                )

            except Exception as e:  # noqa: BLE001
                logging.error("Failed to embed with '%s': %s", model_key, e)
                metadata[model_key] = {"error": str(e)}

        adata.uns["embpy_cell_embeddings"] = metadata
        logging.info(
            "embed_cells complete: %d models, %d cells",
            len(models),
            adata.n_obs,
        )
        return adata

    def decode_cells(
        self,
        adata=None,
        *,
        latent=None,
        model: str = "scvi",
        obsm_key: str | None = None,
        gene_names: Sequence[str] | None = None,
        write_layer: str | None = None,
        **decode_kwargs: Any,
    ):
        """Decode cell embeddings back to gene-expression space.

        Companion to :meth:`embed_cells` that routes a latent matrix
        through the *same* fitted encoder-decoder wrapper that produced
        it. This is the decoder hook used by flow-matching / cellflow
        style perturbation pipelines: encode basal + perturbed cells to
        a shared latent, learn a transport map there, then decode the
        predicted latent back to counts with this method.

        Requires a prior :meth:`embed_cells` call with the same
        ``model`` key so the trained wrapper is cached for this device.
        For STATE / Stack, instantiate the wrapper manually (checkpoint +
        gene-list paths cannot be supplied through :meth:`embed_cells`).

        Parameters
        ----------
        adata : anndata.AnnData, optional
            AnnData that holds the latent to decode. Used both to pick
            the latent from ``obsm[obsm_key]`` and to recover
            ``var_names`` for gene-parametric decoders (STATE).
        latent : np.ndarray, optional
            Alternative to ``adata`` + ``obsm_key``: pass the latent
            matrix directly.
        model : str
            Encoder-decoder registry key: ``"pca"``, ``"scvi"``,
            ``"scanvi"``, ``"totalvi"``, or ``"state"``.
        obsm_key : str, optional
            Override key used to pull the latent from ``adata.obsm``.
            Defaults to ``f"X_{model}"``.
        gene_names : sequence of str, optional
            Target genes to decode to (STATE only). Defaults to
            ``adata.var_names`` when ``adata`` is supplied.
        write_layer : str, optional
            If set and ``adata`` is supplied, the decoded expression is
            written to ``adata.layers[write_layer]``.
        **decode_kwargs
            Forwarded to the wrapper's ``decode_cells`` (e.g.
            ``library_size``, ``batch_index`` for scVI, ``read_depth``
            for STATE).

        Returns
        -------
        np.ndarray of shape ``(n_cells, n_genes)``
            Decoded expression (NB mean / log-probs / linear recon,
            depending on the model; see the wrapper docstring).
        """
        from .models.singlecell_models import singlecell_info

        card = singlecell_info(model)
        if not card.supports_decode:
            raise ValueError(
                f"Model '{model}' does not expose a decoder "
                f"(wrapper_class={card.wrapper_class_name}). "
                "Supported: pca, scvi, scanvi, totalvi, state."
            )

        if latent is None:
            if adata is None:
                raise ValueError("decode_cells needs either `latent` or `adata` (+ optional `obsm_key`).")
            key = obsm_key or f"X_{model}"
            if key not in adata.obsm:
                raise KeyError(
                    f"{key!r} not present in adata.obsm. Run embed_cells "
                    f"with models=[{model!r}] first, or pass `obsm_key`."
                )
            latent_arr = np.asarray(adata.obsm[key])
        else:
            latent_arr = np.asarray(latent)

        if gene_names is None and adata is not None:
            gene_names = list(adata.var_names)

        device_str = str(self.device)
        cache_key = (model, device_str)
        wrapper = self._singlecell_cache.get(cache_key)
        if wrapper is None:
            raise RuntimeError(
                f"No fitted '{model}' wrapper cached for device "
                f"{device_str!r}. Call `embed_cells(adata, models=["
                f"'{model}'])` first (for STATE/Stack, instantiate the "
                "wrapper manually with the checkpoint path and pass it to "
                "`wrapper.embed_cells` / `wrapper.decode_cells` directly)."
            )

        decoded = wrapper.decode_cells(
            latent_arr,
            gene_names=gene_names,
            adata=adata,
            **decode_kwargs,
        )

        if write_layer is not None and adata is not None:
            if decoded.shape[1] != adata.n_vars:
                raise ValueError(
                    f"Decoded matrix has {decoded.shape[1]} genes but adata has "
                    f"{adata.n_vars} var_names; cannot write to adata.layers."
                )
            adata.layers[write_layer] = decoded

        return decoded

    def generate_cells(
        self,
        base_adata,
        test_adata,
        *,
        model: str = "stack",
        wrapper=None,
        **generate_kwargs: Any,
    ):
        """Run in-context generation to synthesise new cell profiles.

        Currently only supported by the Stack wrapper. Unlike
        :meth:`decode_cells`, this does NOT consume a latent matrix: it
        conditions on a donor-specific ``base_adata`` context and
        synthesises gene-expression for every cell in ``test_adata``.

        Parameters
        ----------
        base_adata : anndata.AnnData
            Donor-conditioned base context.
        test_adata : anndata.AnnData
            Cells to predict for.
        model : str
            Registry key (currently only ``"stack"``).
        wrapper : SingleCellWrapper, optional
            Already-constructed wrapper. Required for Stack because its
            checkpoint and gene-list paths cannot be expressed through
            ``embed_cells``.
        **generate_kwargs
            Forwarded to the wrapper's ``generate_cells`` (split_column,
            split_values, prompt_ratio, ...).
        """
        from .models.singlecell_models import singlecell_info

        card = singlecell_info(model)
        if not card.supports_generation:
            raise ValueError(f"Model '{model}' does not expose an in-context generation head. Supported: stack.")

        if wrapper is None:
            device_str = str(self.device)
            wrapper = self._singlecell_cache.get((model, device_str))
        if wrapper is None:
            raise RuntimeError(
                f"No '{model}' wrapper available. For Stack, instantiate "
                "it manually with `StackWrapper(checkpoint=..., "
                "genelist=...)`, call `.load(device)`, and pass it via "
                "the `wrapper=` kwarg."
            )

        return wrapper.generate_cells(base_adata, test_adata, **generate_kwargs)

    def embed_adata(
        self,
        adata,  # anndata.AnnData
        *,
        # --- Cell-level embeddings (from expression) ---
        cell_models: list[str] | str | None = None,
        preprocessing: Literal["raw", "standard", "none"] = "standard",
        target_sum: float | None = 1e4,
        n_top_genes: int = 2000,
        log_transform: bool = True,
        scale: bool = False,
        max_value: float | None = 10.0,
        min_genes: int = 200,
        min_cells: int = 3,
        max_pct_mito: float | None = None,
        n_pca_components: int = 50,
        pca_use_hvg: bool = True,
        n_latent: int = 30,
        n_layers_scvi: int = 2,
        n_hidden_scvi: int = 128,
        max_epochs: int = 200,
        early_stopping: bool = True,
        batch_key: str | None = None,
        labels_key: str | None = None,
        protein_expression_obsm_key: str | None = None,
        # --- Perturbation-level embeddings (gene / protein / molecule) ---
        perturbation_models: list[str] | str | None = None,
        perturbation_column: str | None = None,
        perturbation_type: Literal[
            "auto",
            "symbol",
            "ensembl_id",
            "uniprot_id",
            "smiles",
        ] = "auto",
        perturbation_organism: str = "human",
        pooling_strategy: str = "mean",
        # --- General ---
        batch_size: int = 32,
        obsm_prefix: str = "X_",
        copy: bool = True,
        backend: Literal["cpu", "gpu"] = "cpu",
    ):
        """Unified embedding of an AnnData -- cells and/or perturbations.

        Combines cell-level embeddings (from expression via single-cell
        foundation models, PCA, or scVI) with perturbation-level
        embeddings (from gene/protein/molecule identifiers in ``.obs``)
        in a single call.

        Parameters
        ----------
        adata : anndata.AnnData
            Input AnnData with raw counts in ``.X`` and (optionally)
            perturbation annotations in ``.obs``.

        cell_models : str or list[str], optional
            Single-cell model key(s) for expression-based cell
            embeddings (e.g. ``"scgpt"``, ``"pca"``, ``"scvi"``).
            ``None`` skips cell embedding.
        preprocessing
            Preprocessing pipeline for cell models.
        target_sum, n_top_genes, log_transform, scale, max_value,
        min_genes, min_cells, max_pct_mito
            Forwarded to :func:`~embpy.pp.preprocess_counts`.
        n_pca_components, pca_use_hvg
            PCA-specific parameters.
        n_latent, n_layers_scvi, n_hidden_scvi, max_epochs,
        early_stopping, batch_key, labels_key,
        protein_expression_obsm_key
            scvi-tools parameters.

        perturbation_models : str or list[str], optional
            Sequence/molecule model key(s) for perturbation embeddings
            (e.g. ``"esm2_650M"``, ``"enformer_human_rough"``,
            ``"chemberta2MTR"``).  ``None`` skips perturbation embedding.
        perturbation_column : str, optional
            Column in ``adata.obs`` containing perturbation identifiers.
            Required when *perturbation_models* is set.
        perturbation_type
            Type of identifiers in the perturbation column.
            ``"auto"`` auto-detects per identifier.
        perturbation_organism
            Organism for gene/protein resolution.
        pooling_strategy
            Pooling for sequence/molecule models.

        batch_size
            Batch size for model inference.
        obsm_prefix
            Prefix for ``.obsm`` keys (default ``"X_"``).
        copy
            If ``True``, operate on a copy of adata.

        Returns
        -------
        anndata.AnnData
            The enriched AnnData with:

            - ``.X`` = original raw counts
            - ``.layers["counts"]`` = raw counts copy
            - ``.layers["log_normalized"]`` = processed expression
              (standard pipeline)
            - ``.obsm["{prefix}{cell_model}"]`` = cell embeddings
            - ``.uns["perturbations"]["{prefix}{pert_model}"]``
              = perturbation embeddings, one row per unique perturbation
            - ``.uns["embpy_embeddings"]`` = metadata dict

        Examples
        --------
        >>> result = embedder.embed_adata(
        ...     adata,
        ...     cell_models=["pca", "scvi", "scgpt"],
        ...     perturbation_models=["esm2_650M", "chemberta2MTR"],
        ...     perturbation_column="perturbation",
        ...     perturbation_type="auto",
        ... )
        >>> result.obsm["X_scgpt"].shape  # cell embeddings
        (5000, 512)
        >>> result.uns["perturbations"]["X_esm2_650M"]["matrix"].shape
        (n_perturbations, 1280)
        """
        from .io.exporters import to_anndata
        from .io.result import EmbeddingProvenance, EmbeddingResult
        from .resources.gene_resolver import detect_identifier_type

        if copy:
            adata = adata.copy()

        metadata: dict[str, dict[str, Any]] = {}

        # ==================================================================
        # Cell-level embeddings (expression-based)
        # ==================================================================
        if cell_models is not None:
            adata = self.embed_cells(
                adata,
                models=cell_models,
                preprocessing=preprocessing,
                target_sum=target_sum,
                n_top_genes=n_top_genes,
                log_transform=log_transform,
                scale=scale,
                max_value=max_value,
                min_genes=min_genes,
                min_cells=min_cells,
                max_pct_mito=max_pct_mito,
                n_pca_components=n_pca_components,
                pca_use_hvg=pca_use_hvg,
                n_latent=n_latent,
                n_layers_scvi=n_layers_scvi,
                n_hidden_scvi=n_hidden_scvi,
                max_epochs=max_epochs,
                early_stopping=early_stopping,
                batch_key=batch_key,
                labels_key=labels_key,
                protein_expression_obsm_key=protein_expression_obsm_key,
                batch_size=batch_size,
                obsm_prefix=obsm_prefix,
                copy=False,
                backend=backend,
            )
            if "embpy_cell_embeddings" in adata.uns:
                metadata.update(adata.uns["embpy_cell_embeddings"])

        # ==================================================================
        # Perturbation-level embeddings (gene / protein / molecule)
        # ==================================================================
        if perturbation_models is not None:
            if perturbation_column is None:
                raise ValueError(
                    "perturbation_column is required when perturbation_models "
                    "is specified. Set it to the .obs column containing "
                    "perturbation identifiers (e.g. 'perturbation', 'gene', "
                    "'smiles')."
                )
            if perturbation_column not in adata.obs.columns:
                raise ValueError(
                    f"Column '{perturbation_column}' not found in adata.obs. Available: {list(adata.obs.columns)}"
                )

            if isinstance(perturbation_models, str):
                perturbation_models = [perturbation_models]

            pert_ids = adata.obs[perturbation_column].astype(str).values
            unique_perts = list(dict.fromkeys(pert_ids))
            logging.info(
                "Perturbation embedding: %d cells, %d unique perturbations from column '%s'",
                len(pert_ids),
                len(unique_perts),
                perturbation_column,
            )

            # Determine id_type per perturbation if auto
            if perturbation_type == "auto":
                id_types = {p: detect_identifier_type(p) for p in unique_perts}
            else:
                id_types = dict.fromkeys(unique_perts, perturbation_type)

            for model_key in perturbation_models:
                obsm_key = f"{obsm_prefix}{model_key}"
                logging.info(
                    "Embedding perturbations with '%s' ...",
                    model_key,
                )

                try:
                    # Embed each unique perturbation
                    pert_embs: dict[str, np.ndarray | None] = {}
                    for pert in unique_perts:
                        id_t = id_types[pert]
                        try:
                            if id_t == "smiles":
                                emb = self.embed_molecule(
                                    identifier=pert,
                                    model=model_key,
                                    pooling_strategy=pooling_strategy,
                                )
                            else:
                                emb = self.embed_gene(
                                    identifier=pert,
                                    model=model_key,
                                    id_type=id_t,
                                    organism=perturbation_organism,
                                    pooling_strategy=pooling_strategy,
                                )
                            pert_embs[pert] = np.asarray(
                                emb,
                                dtype=np.float32,
                            ).ravel()
                        except Exception as e:  # noqa: BLE001
                            logging.warning(
                                "Failed to embed perturbation '%s': %s",
                                pert,
                                e,
                            )
                            pert_embs[pert] = None

                    # Determine embedding dim
                    emb_dim = 0
                    for v in pert_embs.values():
                        if v is not None:
                            emb_dim = v.shape[0]
                            break

                    if emb_dim == 0:
                        logging.error(
                            "No perturbation embeddings succeeded for '%s'",
                            model_key,
                        )
                        metadata[model_key] = {
                            "error": "no embeddings succeeded",
                        }
                        continue

                    n_ok = sum(1 for v in pert_embs.values() if v is not None)
                    successful_perts = [p for p in unique_perts if pert_embs.get(p) is not None]
                    entity_matrix = np.stack(
                        [np.asarray(pert_embs[p], dtype=np.float32).ravel() for p in successful_perts],
                        axis=0,
                    )
                    aliases = {
                        str(p): {
                            "perturbation_label": str(p),
                            "input_id_type": str(id_types.get(p, perturbation_type)),
                        }
                        for p in successful_perts
                    }
                    result = EmbeddingResult(
                        matrix=entity_matrix,
                        entity_ids=tuple(str(p) for p in successful_perts),
                        entity_type="perturbation",
                        id_scheme="perturbation_label",
                        provenance=EmbeddingProvenance.create(
                            model=model_key,
                            pooling=pooling_strategy,
                            extra={
                                "entity_type": "perturbation",
                                "input_kind": "anndata_obs_column",
                                "input_source": "AnnData.obs",
                                "input_id_column": perturbation_column,
                                "perturbation_column": perturbation_column,
                                "perturbation_type": perturbation_type,
                                "organism": perturbation_organism,
                                "n_requested_inputs": len(unique_perts),
                                "n_successfully_embedded_entities": n_ok,
                                "n_embedding_failures": len(unique_perts) - n_ok,
                                "n_cells": int(adata.n_obs),
                            },
                        ),
                        aliases=aliases,
                    )
                    to_anndata(result, target=adata, attach_to="uns", key=obsm_key)

                    metadata[model_key] = {
                        "uns_key": obsm_key,
                        "storage": "uns",
                        "embedding_dim": emb_dim,
                        "n_cells": int(adata.n_obs),
                        "n_perturbations_total": len(unique_perts),
                        "n_perturbations_embedded": n_ok,
                        "type": "perturbation",
                        "perturbation_column": perturbation_column,
                    }
                    logging.info(
                        "  -> stored in .uns perturbations[%r], shape=(%d, %d) (%d/%d perturbations embedded)",
                        obsm_key,
                        entity_matrix.shape[0],
                        entity_matrix.shape[1],
                        n_ok,
                        len(unique_perts),
                    )

                except Exception as e:  # noqa: BLE001
                    logging.error(
                        "Failed perturbation embedding with '%s': %s",
                        model_key,
                        e,
                    )
                    metadata[model_key] = {"error": str(e)}

        adata.uns["embpy_embeddings"] = metadata
        logging.info("embed_adata complete.")
        return adata

    def list_available_models(
        self,
        category: Literal[
            "all",
            "dna",
            "protein",
            "molecule",
            "text",
            "morphology",
            "single_cell",
        ] = "all",
    ) -> list[str]:
        """Return available model names, optionally filtered by category.

        Parameters
        ----------
        category
            ``"all"`` returns every model (sequence + single-cell).
            ``"dna"``, ``"protein"``, ``"molecule"``, ``"text"``,
            ``"morphology"`` filter the sequence/structure model registry.
            ``"single_cell"`` returns single-cell foundation model keys.
        """
        from .models.singlecell_models import list_singlecell_models

        if category == "single_cell":
            return sorted(list_singlecell_models())

        if category == "all":
            seq_models = sorted(self._available_models.keys())
            sc_models = sorted(list_singlecell_models())
            return seq_models + sc_models

        result = []
        for name, (wrapper_cls, _) in self._available_models.items():
            if hasattr(wrapper_cls, "model_type") and wrapper_cls.model_type == category:
                result.append(name)
        return sorted(result)

    # ------------------------------------------------------------------
    # Morphological embedding API
    # ------------------------------------------------------------------

    def embed_morphological(
        self,
        image: str | np.ndarray | torch.Tensor,
        model: str = "subcell_mae_rybg",
        pooling_strategy: str = "attention_pool",
        **kwargs,
    ) -> np.ndarray:
        """Embed a single microscopy image using a morphology model.

        Parameters
        ----------
        image
            Path to a multi-channel image, or a ``(C, H, W)`` array/tensor.
        model
            Registered morphology model key (default ``"subcell_mae_rybg"``).
        pooling_strategy
            Pooling strategy for the model (``"cls"``, ``"mean"``,
            ``"attention_pool"``, ``"none"``).

        Returns
        -------
        np.ndarray
            1-D embedding vector.
        """
        wrapper = self._get_model(model)
        if getattr(wrapper, "model_type", None) != "morphology":
            raise ValueError(
                f"Model '{model}' is not a morphology embedder (type={getattr(wrapper, 'model_type', '?')})."
            )
        try:
            return wrapper.embed(image, pooling_strategy=pooling_strategy, **kwargs)
        except Exception as e:
            logging.error("Morphological embedding failed: %s", e)
            raise RuntimeError("Morphological embedding failed") from e

    def embed_morphological_batch(
        self,
        images: Sequence[str | np.ndarray | torch.Tensor],
        model: str = "subcell_mae_rybg",
        pooling_strategy: str = "attention_pool",
        **kwargs,
    ) -> list[np.ndarray]:
        """Embed a batch of microscopy images.

        Parameters
        ----------
        images
            Iterable of paths or ``(C, H, W)`` arrays/tensors.
        model
            Registered morphology model key.
        pooling_strategy
            Pooling strategy passed to the underlying model.

        Returns
        -------
        list[np.ndarray]
            One embedding per image.
        """
        wrapper = self._get_model(model)
        if getattr(wrapper, "model_type", None) != "morphology":
            raise ValueError(
                f"Model '{model}' is not a morphology embedder (type={getattr(wrapper, 'model_type', '?')})."
            )
        return wrapper.embed_batch(images, pooling_strategy=pooling_strategy, **kwargs)

    def embed_perturbation_morphology(
        self,
        perturbation: str,
        perturbation_type: str = "genetic",
        dataset: str = "jump",
        source: str = "subcell",
        model: str = "subcell_mae_rybg",
        pooling_strategy: str = "attention_pool",
        local_dir: str | None = None,
        aggregate: str = "mean",
        max_images: int | None = 5,
        plate_type: str | None = None,
        jump_profiles_dir: str | None = None,
        verbose: bool = True,
        **kwargs,
    ) -> np.ndarray:
        """Embed a perturbation from JUMP or HPA by resolving images automatically.

        The method handles identifier resolution (gene symbols via
        ``GeneResolver``, compound names via ``DrugResolver``), image
        fetching from CDN or local directories, preprocessing, and
        optional aggregation.

        Parameters
        ----------
        perturbation
            Gene symbol, compound name, JCP2022 ID, or Ensembl ID.
        perturbation_type
            ``"genetic"`` (CRISPR/ORF) or ``"compound"``.
        dataset
            ``"jump"`` or ``"hpa"``.
        source
            ``"subcell"`` -- run SubCell inference on images.
            ``"precomputed"`` -- load pre-computed JUMP CellProfiler profiles
            (only valid with ``dataset="jump"``).
        model
            Morphology model key (used when ``source="subcell"``).
        pooling_strategy
            Pooling strategy for the SubCell model.
        local_dir
            Path to a local image directory.  When images are already
            present they are loaded from disk; when they are fetched
            from the CDN they are cached here for future re-use.
            ``None`` = fetch from CDN without caching.
        aggregate
            ``"mean"`` averages embeddings across images; ``"none"``
            returns the first image embedding only.
        max_images
            Maximum number of images to download and embed.  Defaults
            to ``5`` to avoid long downloads.  Pass ``None`` to fetch
            all available images for the perturbation.
        plate_type
            JUMP plate type (``"crispr"``, ``"orf"``, ``"compound"``).
            Inferred from *perturbation_type* when ``None``.
        jump_profiles_dir
            Directory containing JUMP profile parquets (for
            ``source="precomputed"``).  When ``None``, defaults to
            ``data/embeddings/morphology_embeddings/JUMP/`` within the
            project.  If the parquet is not present it is automatically
            downloaded from the Cell Painting Gallery S3 bucket.
        verbose
            If ``True`` (default), prints a single summary message at the
            end indicating success/failure and the data source used.
            Set to ``False`` to suppress all output.

        Returns
        -------
        np.ndarray
            Embedding vector (shape depends on model and aggregation).
        """
        if source == "precomputed":
            if dataset != "jump":
                raise ValueError("source='precomputed' is only valid with dataset='jump'")
            result = self._load_jump_precomputed(
                perturbation,
                perturbation_type,
                plate_type,
                jump_profiles_dir,
                aggregate,
            )
            if verbose:
                print(
                    f"[embpy] {perturbation!r}: loaded precomputed JUMP profile (plate_type={plate_type or 'crispr'})"
                )
            return result

        # Use a resolution context to track steps without intermediate warnings
        resolution_ctx = _ResolutionContext()

        if dataset == "jump":
            images = self._resolve_jump_images(
                perturbation,
                perturbation_type,
                local_dir,
                plate_type,
                max_images,
                resolution_ctx,
            )
        elif dataset == "hpa":
            if perturbation_type == "compound":
                raise ValueError("perturbation_type='compound' is not supported for dataset='hpa'")
            images = self._resolve_hpa_images(perturbation, local_dir, max_images, resolution_ctx)
        else:
            raise ValueError(f"Unknown dataset: {dataset!r}")

        if not images:
            if verbose:
                # Print failure summary
                steps = resolution_ctx.steps
                attempted = ", ".join(steps) if steps else "direct lookup"
                print(f"[embpy] WARNING: {perturbation!r}: no images found (dataset={dataset}, tried: {attempted})")
            raise RuntimeError(
                f"No images found for perturbation={perturbation!r}, "
                f"dataset={dataset!r}, perturbation_type={perturbation_type!r}"
            )

        embeddings = self.embed_morphological_batch(
            images,
            model=model,
            pooling_strategy=pooling_strategy,
            **kwargs,
        )

        if verbose:
            # Print success summary
            source_desc = resolution_ctx.final_source or dataset
            n_images = len(images)
            print(f"[embpy] {perturbation!r}: embedded {n_images} image(s) (source: {source_desc})")

        if aggregate == "mean":
            return np.mean(embeddings, axis=0).astype(np.float32)
        return embeddings[0]

    def embed_perturbation_morphology_batch(
        self,
        perturbations: Sequence[str],
        perturbation_type: str = "genetic",
        dataset: str = "jump",
        source: str = "subcell",
        model: str = "subcell_mae_rybg",
        pooling_strategy: str = "attention_pool",
        local_dir: str | None = None,
        aggregate: str = "mean",
        max_images: int | None = 5,
        plate_type: str | None = None,
        jump_profiles_dir: str | None = None,
        verbose: bool = True,
        skip_failures: bool = True,
        n_workers: int = 8,
        **kwargs,
    ) -> tuple[np.ndarray, list[str]]:
        """Embed multiple perturbations from JUMP or HPA in batch.

        This is the batched version of :meth:`embed_perturbation_morphology`.
        Image resolution / download for all perturbations runs **in parallel**
        across a thread pool, and then a single GPU inference batch is run on
        the combined image stack.  This is dramatically faster than calling
        ``embed_perturbation_morphology`` in a Python loop, especially for
        HPA / JUMP CDN downloads which are network-bound.

        Parameters
        ----------
        perturbations
            Sequence of gene symbols, compound names, JCP2022 IDs, or Ensembl IDs.
        perturbation_type
            ``"genetic"`` (CRISPR/ORF) or ``"compound"``.
        dataset
            ``"jump"`` or ``"hpa"``.
        source
            ``"subcell"`` -- run SubCell inference on images.
            ``"precomputed"`` -- load pre-computed JUMP CellProfiler profiles
            (only valid with ``dataset="jump"``).
        model
            Morphology model key (used when ``source="subcell"``).
        pooling_strategy
            Pooling strategy for the SubCell model.
        local_dir
            Path to a local image directory.  When images are already
            present they are loaded from disk; when they are fetched
            from the CDN they are cached here for future re-use.
        aggregate
            ``"mean"`` averages embeddings across images per perturbation;
            ``"none"`` returns the first image embedding only.
        max_images
            Maximum number of images to download and embed per perturbation.
        plate_type
            JUMP plate type (``"crispr"``, ``"orf"``, ``"compound"``).
        jump_profiles_dir
            Directory containing JUMP profile parquets (for ``source="precomputed"``).
        verbose
            If ``True`` (default), prints progress and a summary at the end.
        skip_failures
            If ``True`` (default), perturbations that fail to resolve are
            skipped with a warning. If ``False``, raises on first failure.
        n_workers
            Number of parallel worker threads for resolving / downloading
            images across perturbations.  Defaults to ``8``.
        **kwargs
            Additional arguments passed to the embedding model.

        Returns
        -------
        tuple[np.ndarray, list[str]]
            A tuple of ``(embeddings, labels)`` where:

            - ``embeddings`` is a 2D array of shape ``(n_successful, embedding_dim)``
            - ``labels`` is a list of perturbation identifiers that succeeded,
              in the same order as the embedding rows.

        Examples
        --------
        >>> genes = ["TP53", "BRCA1", "EGFR", "UNKNOWN_GENE"]
        >>> embeddings, labels = embedder.embed_perturbation_morphology_batch(genes, dataset="hpa", max_images=3)
        >>> print(f"Embedded {len(labels)} / {len(genes)} genes")
        Embedded 3 / 4 genes
        >>> print(embeddings.shape)
        (3, 384)
        """
        import time
        from concurrent.futures import ThreadPoolExecutor, as_completed

        # ---- precomputed JUMP profiles: I/O-bound parquet reads only ----
        if source == "precomputed":
            return self._embed_perturbation_precomputed_batch(
                perturbations,
                perturbation_type=perturbation_type,
                plate_type=plate_type,
                jump_profiles_dir=jump_profiles_dir,
                aggregate=aggregate,
                verbose=verbose,
                skip_failures=skip_failures,
            )

        n_total = len(perturbations)
        t0 = time.time()
        hpa_catalog = None
        if dataset == "hpa":
            hpa_catalog = self._build_hpa_batch_catalog(
                perturbations,
                local_dir=local_dir,
                verbose=verbose,
            )

        # ── Stage 1: resolve images for ALL perturbations in parallel ──
        if verbose:
            print(
                f"[embpy] Stage 1/2: resolving images for {n_total} perturbations using {n_workers} parallel workers..."
            )

        def _resolve_one(pert: str) -> tuple[str, list[np.ndarray], str | None, Exception | None]:
            ctx = _ResolutionContext()
            try:
                if dataset == "jump":
                    imgs = self._resolve_jump_images(
                        pert,
                        perturbation_type,
                        local_dir,
                        plate_type,
                        max_images,
                        ctx,
                    )
                elif dataset == "hpa":
                    if perturbation_type == "compound":
                        raise ValueError("perturbation_type='compound' is not supported for dataset='hpa'")
                    imgs = self._resolve_hpa_images(
                        pert,
                        local_dir,
                        max_images,
                        ctx,
                        hpa_catalog=hpa_catalog,
                    )
                else:
                    raise ValueError(f"Unknown dataset: {dataset!r}")
                return pert, imgs, ctx.final_source, None
            except Exception as exc:  # noqa: BLE001
                return pert, [], ctx.final_source, exc

        per_gene_images: dict[str, list[np.ndarray]] = {}
        per_gene_source: dict[str, str | None] = {}
        failed_labels: list[str] = []
        first_exc: Exception | None = None

        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = {pool.submit(_resolve_one, p): p for p in perturbations}
            for i, fut in enumerate(as_completed(futures), start=1):
                pert, imgs, src, exc = fut.result()
                if exc is not None or not imgs:
                    failed_labels.append(pert)
                    if first_exc is None and exc is not None:
                        first_exc = exc
                    if verbose:
                        reason = str(exc) if exc else "no images found"
                        print(f"  [{i}/{n_total}] {pert!r:20s} FAILED ({reason})")
                else:
                    per_gene_images[pert] = imgs
                    per_gene_source[pert] = src
                    if verbose:
                        print(f"  [{i}/{n_total}] {pert!r:20s} {len(imgs)} image(s) from {src}")

        if not skip_failures and first_exc is not None:
            raise first_exc

        if not per_gene_images:
            if verbose:
                print("[embpy] No images resolved for any perturbation")
            return np.array([]).reshape(0, 0), []

        # ── Stage 2: stack all images and run a single GPU inference ──
        successful_labels = [p for p in perturbations if p in per_gene_images]
        all_images: list[np.ndarray] = []
        boundaries: list[int] = [0]
        for pert in successful_labels:
            imgs = per_gene_images[pert]
            all_images.extend(imgs)
            boundaries.append(boundaries[-1] + len(imgs))

        if verbose:
            t_resolve = time.time() - t0
            print(
                f"[embpy] Stage 2/2: running inference on "
                f"{len(all_images)} images (image resolution took "
                f"{t_resolve:.1f}s)..."
            )

        t1 = time.time()
        all_embeddings = self.embed_morphological_batch(
            all_images,
            model=model,
            pooling_strategy=pooling_strategy,
            **kwargs,
        )
        all_embeddings = np.asarray(all_embeddings)

        # Aggregate per gene using boundary slices
        per_gene_emb: list[np.ndarray] = []
        for j in range(len(successful_labels)):
            start, end = boundaries[j], boundaries[j + 1]
            sub = all_embeddings[start:end]
            if aggregate == "mean":
                per_gene_emb.append(np.mean(sub, axis=0).astype(np.float32))
            else:
                per_gene_emb.append(sub[0])

        if verbose:
            t_infer = time.time() - t1
            t_total = time.time() - t0
            print(
                f"[embpy] Done in {t_total:.1f}s "
                f"(resolve={t_resolve:.1f}s, inference={t_infer:.1f}s) -- "
                f"{len(successful_labels)}/{n_total} perturbations embedded"
            )
            if failed_labels:
                print(f"[embpy] Failed: {failed_labels}")

        return np.stack(per_gene_emb, axis=0), successful_labels

    def _embed_perturbation_precomputed_batch(
        self,
        perturbations: Sequence[str],
        perturbation_type: str,
        plate_type: str | None,
        jump_profiles_dir: str | None,
        aggregate: str,
        verbose: bool,
        skip_failures: bool,
    ) -> tuple[np.ndarray, list[str]]:
        """Batch-load pre-computed JUMP profiles for many perturbations.

        Reuses the parquet only once across all perturbations to avoid
        reloading it for every gene.
        """
        import pandas as pd

        from .resources.jump_metadata import get_jump_gene_mapper

        if plate_type is None:
            plate_type = "compound" if perturbation_type == "compound" else "crispr"

        pq_path = _resolve_jump_parquet_path(plate_type, jump_profiles_dir)
        df = pd.read_parquet(pq_path)
        feature_cols = [c for c in df.columns if not c.startswith("Metadata_")]
        jcp_col = "Metadata_JCP2022"

        mapper = get_jump_gene_mapper(plate_type) if perturbation_type != "compound" else {}
        inv_mapper = {v: k for k, v in mapper.items()}

        embeddings_list: list[np.ndarray] = []
        successful_labels: list[str] = []
        failed_labels: list[str] = []

        for pert in perturbations:
            if pert.startswith("JCP2022"):
                mask = df[jcp_col] == pert
            else:
                jcp_id = inv_mapper.get(pert)
                mask = df[jcp_col] == jcp_id if jcp_id else pd.Series(False, index=df.index)

            sub = df[mask]
            if len(sub) == 0:
                failed_labels.append(pert)
                if not skip_failures:
                    raise RuntimeError(f"No precomputed profile found for {pert!r}")
                continue

            features = sub[feature_cols].to_numpy(dtype=np.float32)
            emb = features.mean(axis=0) if aggregate == "mean" else features[0]
            embeddings_list.append(emb)
            successful_labels.append(pert)

        if verbose:
            print(
                f"[embpy] Loaded {len(successful_labels)}/{len(perturbations)} precomputed profiles from {pq_path.name}"
            )
            if failed_labels:
                print(f"[embpy] Failed: {failed_labels}")

        if not embeddings_list:
            return np.array([]).reshape(0, 0), []
        return np.stack(embeddings_list, axis=0), successful_labels

    # ------------------------------------------------------------------
    # Internal resolvers
    # ------------------------------------------------------------------

    @staticmethod
    def _hpa_shared_cache_root(local_dir: str | None) -> pathlib.Path | None:
        """Return a persistent shared HPA cache root when one is configured."""
        if os.environ.get("EMBPY_HPA_CACHE_DIR"):
            return pathlib.Path(os.environ["EMBPY_HPA_CACHE_DIR"])
        if local_dir is None:
            return None
        local_path = pathlib.Path(local_dir)
        if local_path.name in {"_hpa", "hpa"}:
            return local_path
        return local_path.parent / "_hpa"

    @staticmethod
    def _build_hpa_batch_catalog(
        perturbations: Sequence[str],
        *,
        local_dir: str | None,
        verbose: bool,
    ):
        """Build or load one HPA XML catalog slice for a batch of genes."""
        genes = sorted({str(p).strip().upper() for p in perturbations if str(p).strip()})
        if not genes:
            return None

        cache_root = BioEmbedder._hpa_shared_cache_root(local_dir)
        xml_source = None
        cache_path = None
        if cache_root is not None:
            cache_root.mkdir(parents=True, exist_ok=True)
            digest = sha1("\n".join(genes).encode("utf-8")).hexdigest()[:16]
            xml_source = cache_root / "proteinatlas.xml.gz"
            cache_path = cache_root / f"subcellular_catalog_{digest}.csv"

        from .resources.hpa_images import build_hpa_subcellular_catalog

        if verbose:
            cache_msg = f" (cache: {cache_path})" if cache_path is not None else ""
            print(f"[embpy] Building/loading one HPA XML catalog for {len(genes)} genes{cache_msg}...")
        try:
            return build_hpa_subcellular_catalog(
                xml_source=xml_source,
                cache_path=cache_path,
                genes=genes,
            )
        except Exception as exc:  # noqa: BLE001
            if verbose:
                print(
                    f"[embpy] WARNING: failed to build shared HPA XML catalog; falling back to per-gene lookup ({exc})"
                )
            return None

    @staticmethod
    def _hpa_catalog_rows_to_antibodies(catalog, perturbation: str) -> list[dict[str, Any]]:
        """Convert pre-filtered HPA catalog rows into fetchable image records."""
        if catalog is None or len(catalog) == 0:
            return []
        gene_upper = str(perturbation).upper()
        rows = catalog[catalog["gene"].astype(str).str.upper() == gene_upper]
        antibodies: list[dict[str, Any]] = []
        for _, row in rows.iterrows():
            antibodies.append(
                {
                    "id": row["antibody"],
                    "_plate": row["plate"],
                    "_position": row["position"],
                    "_sample": row["sample"],
                    "_url_prefix": row.get("image_url_prefix", ""),
                }
            )
        return antibodies

    @staticmethod
    def _resolve_jump_images(
        perturbation: str,
        perturbation_type: str,
        local_dir: str | None,
        plate_type: str | None,
        max_images: int | None,
        resolution_ctx: _ResolutionContext | None = None,
    ) -> list[np.ndarray]:
        """Resolve a JUMP perturbation to preprocessed SubCell-ready arrays.

        For compounds, uses ``DrugResolver`` to resolve alternate names
        when the initial lookup fails.  For genetic perturbations, uses
        ``GeneResolver`` to map symbols to canonical forms.
        """
        from .pp.morphology_preprocessing import (
            cell_painting_to_subcell,
            prepare_subcell_canvas,
        )
        from .resources.jump_metadata import (
            fetch_jump_fov,
            get_jump_item_location_metadata,
        )

        if plate_type is None:
            plate_type = "compound" if perturbation_type == "compound" else "crispr"

        is_jcp = perturbation.startswith("JCP2022")
        input_column = "JCP2022" if is_jcp else "standard_key"

        if resolution_ctx:
            resolution_ctx.add_step("JUMP direct lookup")

        rows = get_jump_item_location_metadata(
            perturbation,
            input_column=input_column,
        )

        used_fallback = False
        # Fallback resolution when initial lookup fails
        if not rows and not is_jcp:
            if resolution_ctx:
                resolution_ctx.add_step("JUMP fallback resolver")
            rows, fallback_source = _resolve_jump_perturbation_fallback(
                perturbation,
                perturbation_type,
                return_source=True,
            )
            if rows:
                used_fallback = True
                if resolution_ctx and fallback_source:
                    resolution_ctx.set_source(f"JUMP CDN via {fallback_source}")

        if not used_fallback and rows and resolution_ctx:
            resolution_ctx.set_source("JUMP CDN (direct)")

        if max_images is not None:
            rows = rows[:max_images]

        images: list[np.ndarray] = []
        failed_count = 0
        for row in rows:
            try:
                if local_dir is not None:
                    fov = _load_jump_fov_local(row, local_dir)
                else:
                    fov = fetch_jump_fov(row)
                subcell_4ch = cell_painting_to_subcell(fov)
                canvas = prepare_subcell_canvas(subcell_4ch, nm_per_pixel=325.0)
                images.append(canvas)
            except Exception:  # noqa: BLE001
                failed_count += 1
        if failed_count > 0 and resolution_ctx:
            resolution_ctx.add_step(f"{failed_count} image(s) failed to load")
        return images

    @staticmethod
    def _resolve_hpa_images(
        perturbation: str,
        local_dir: str | None,
        max_images: int | None,
        resolution_ctx: _ResolutionContext | None = None,
        hpa_catalog=None,
    ) -> list[np.ndarray]:
        """Resolve an HPA gene to SubCell-ready image arrays.

        When *local_dir* is given and already contains channel images for
        this gene the images are loaded from disk.  Otherwise images are
        fetched from the HPA CDN and -- if *local_dir* is provided --
        cached there as per-channel JPEGs for future re-use.

        Gene symbol resolution uses ``GeneResolver`` internally (via
        ``hpa_images._resolve_ensembl_id``).
        """
        import os
        import re
        from pathlib import Path as _Path

        from .resources.hpa_images import (
            fetch_hpa_if_image,
            fetch_hpa_if_image_by_prefix,
            get_hpa_antibodies_quiet,
            load_hpa_if_image,
        )

        # ── Try loading from local cache first ──────────────────────
        if local_dir is not None:
            gene_dir = _Path(local_dir) / perturbation
            if gene_dir.is_dir():
                if resolution_ctx:
                    resolution_ctx.add_step("local cache check")
                candidates = sorted(f for f in os.listdir(gene_dir) if f.endswith((".png", ".jpg")))
                prefixes: set[str] = set()
                for f in candidates:
                    if re.search(r"_blue\.\w+$", f):
                        prefixes.add(re.sub(r"_blue\.\w+$", "", f))
                if prefixes:
                    images: list[np.ndarray] = []
                    failed_local = 0
                    for prefix in sorted(prefixes):
                        try:
                            img = load_hpa_if_image(prefix=str(gene_dir / prefix))
                            images.append(img)
                        except Exception:  # noqa: BLE001
                            failed_local += 1
                    if max_images is not None:
                        images = images[:max_images]
                    if images:
                        if resolution_ctx:
                            resolution_ctx.set_source(f"HPA local cache ({gene_dir})")
                            if failed_local > 0:
                                resolution_ctx.add_step(f"{failed_local} local image(s) failed to load")
                        return images

        # ── Fetch from CDN ──────────────────────────────────────────
        gene_source = None
        if hpa_catalog is not None:
            if resolution_ctx:
                resolution_ctx.add_step("HPA shared XML catalog")
            antibodies = BioEmbedder._hpa_catalog_rows_to_antibodies(hpa_catalog, perturbation)
            if antibodies:
                gene_source = "HPA shared XML catalog"
            else:
                return []
        else:
            if resolution_ctx:
                resolution_ctx.add_step("HPA API lookup")
            antibodies, gene_source = get_hpa_antibodies_quiet(perturbation)
        if not antibodies:
            if resolution_ctx:
                resolution_ctx.add_step("HPA per-gene XML catalog fallback")
            try:
                from .resources.hpa_images import build_hpa_subcellular_catalog

                catalog = build_hpa_subcellular_catalog(genes=[perturbation])
                if len(catalog) == 0:
                    return []
                antibodies = BioEmbedder._hpa_catalog_rows_to_antibodies(catalog, perturbation)
                gene_source = "HPA XML catalog"
            except Exception:  # noqa: BLE001
                return []

        # Cap the antibody list *before* downloading so we only fetch
        # what the user actually requested.
        if max_images is not None:
            antibodies = antibodies[:max_images]

        cache_dir = None
        if local_dir is not None:
            cache_dir = _Path(local_dir) / perturbation
            cache_dir.mkdir(parents=True, exist_ok=True)

        def _unpack_ab(ab):
            if isinstance(ab, dict):
                return (
                    ab.get("id", ""),
                    ab.get("_plate", 1),
                    ab.get("_position", "A1"),
                    ab.get("_sample", 1),
                    ab.get("_url_prefix", ""),
                )
            return str(ab), 1, "A1", 1, ""

        failed_fetch = [0]  # Use list to allow mutation in nested function

        def _fetch_one(ab):
            ab_id, plate, position, sample, url_prefix = _unpack_ab(ab)
            try:
                # When we have a full URL prefix from the XML catalog, use
                # it directly -- it preserves cell-line subdirectories that
                # plate/position/sample alone cannot reconstruct.
                if url_prefix:
                    img = fetch_hpa_if_image_by_prefix(url_prefix)
                else:
                    img = fetch_hpa_if_image(ab_id, plate, position, sample)
                if cache_dir is not None:
                    _save_hpa_channels(
                        img,
                        cache_dir,
                        ab_id,
                        plate,
                        position,
                        sample,
                    )
                return img
            except Exception:  # noqa: BLE001
                failed_fetch[0] += 1
                return None

        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=min(8, len(antibodies))) as pool:
            results = list(pool.map(_fetch_one, antibodies))

        images = [r for r in results if r is not None]

        if resolution_ctx:
            if images:
                src = f"HPA CDN via {gene_source}" if gene_source else "HPA CDN"
                resolution_ctx.set_source(src)
            if failed_fetch[0] > 0:
                resolution_ctx.add_step(f"{failed_fetch[0]} CDN fetch(es) failed")

        return images

    @staticmethod
    def _load_jump_precomputed(
        perturbation: str,
        perturbation_type: str | None,
        plate_type: str | None,
        profiles_dir: str | None,
        aggregate: str,
    ) -> np.ndarray:
        """Load pre-computed JUMP profiles from parquet.

        When the parquet file is not found locally it is automatically
        downloaded from the Cell Painting Gallery S3 bucket into
        *profiles_dir* (or the default cache directory).
        """
        import pandas as pd

        from .resources.jump_metadata import get_jump_gene_mapper

        if plate_type is None:
            plate_type = "compound" if perturbation_type == "compound" else "crispr"

        pq_path = _resolve_jump_parquet_path(plate_type, profiles_dir)

        df = pd.read_parquet(pq_path)

        meta_cols = [c for c in df.columns if c.startswith("Metadata_")]
        jcp_col = "Metadata_JCP2022"
        feature_cols = [c for c in df.columns if not c.startswith("Metadata_")]

        if perturbation.startswith("JCP2022"):
            mask = df[jcp_col] == perturbation
        else:
            mapper = get_jump_gene_mapper(plate_type)
            inv_mapper = {v: k for k, v in mapper.items()}
            jcp_id = inv_mapper.get(perturbation)
            if jcp_id:
                mask = df[jcp_col] == jcp_id
            else:
                sk_col = next(
                    (c for c in meta_cols if c != jcp_col and "standard" in c.lower()),
                    None,
                )
                if sk_col and sk_col in df.columns:
                    mask = df[sk_col].astype(str).str.upper() == perturbation.upper()
                else:
                    jcp_to_name = pd.Series(mapper)
                    df["_mapped_name"] = df[jcp_col].map(jcp_to_name).fillna("")
                    mask = df["_mapped_name"].str.upper() == perturbation.upper()
                    df.drop(columns=["_mapped_name"], inplace=True)

        subset = df.loc[mask, feature_cols]
        if subset.empty:
            raise ValueError(f"Perturbation {perturbation!r} not found in JUMP {plate_type} profiles.")

        if aggregate == "mean":
            return subset.mean(axis=0).values.astype(np.float32)
        return subset.iloc[0].values.astype(np.float32)

    # ------------------------------------------------------------------
    # FASTA / FASTQ file embedding
    # ------------------------------------------------------------------

    _FASTA_EXTENSIONS: dict[str, str] = {
        ".fasta": "fasta",
        ".fa": "fasta",
        ".fna": "fasta",
        ".faa": "fasta",
        ".fastq": "fastq",
        ".fq": "fastq",
    }

    _DNA_CHARS = frozenset("ACGTNUacgtnu")

    def embed_fasta(
        self,
        path: str | os.PathLike[str],
        model: str,
        seq_type: Literal["dna", "protein"] | None = None,
        pooling_strategy: str = "mean",
        obsm_key: str | None = None,
        **kwargs: Any,
    ):
        """Embed all sequences from a FASTA or FASTQ file.

        Supports plain and gzip-compressed files.  The method
        auto-detects the file format from the extension and optionally
        infers whether sequences are DNA or protein when *seq_type*
        is not provided.

        Parameters
        ----------
        path
            Path to a FASTA/FASTQ file.  Recognised extensions:
            ``.fasta``, ``.fa``, ``.fna``, ``.faa`` (FASTA),
            ``.fastq``, ``.fq`` (FASTQ).  A trailing ``.gz`` is
            handled transparently.
        model
            Model name registered with this embedder.
        seq_type
            ``"dna"`` or ``"protein"``.  When ``None`` the type is
            resolved using a three-tier fallback:

            1. **Infer from model** -- if the model is a DNA or
               protein model the type is set accordingly.
            2. **Auto-detect from sequences** -- the first 100
               sequences are inspected; if they consist exclusively
               of ``{A, C, G, T, N, U}`` the type is set to
               ``"dna"``, otherwise ``"protein"``.
            3. **Fail** -- a ``ValueError`` is raised asking the
               caller to specify *seq_type* explicitly.
        pooling_strategy
            Passed to the underlying model.
        obsm_key
            Key used to store embeddings in ``adata.obsm``.
            Defaults to ``"X_{model}"``.
        **kwargs
            Forwarded to the model ``embed`` call.

        Returns
        -------
        anndata.AnnData
            An :class:`~anndata.AnnData` object with:

            * ``.obs`` -- ``sequence_id``, ``sequence_length``,
              ``description``, ``seq_type``
            * ``.obsm[obsm_key]`` -- embedding matrix
              (n_sequences x embedding_dim)
            * ``.uns`` -- ``model``, ``pooling_strategy``,
              ``source_file``, ``n_sequences``, ``n_failed``
        """
        import gzip
        from pathlib import Path

        import anndata as ad
        from Bio import SeqIO

        filepath = Path(path)

        # -- detect file format -----------------------------------------
        is_gz = filepath.suffix.lower() == ".gz"
        stem = filepath.with_suffix("") if is_gz else filepath
        fmt = self._FASTA_EXTENSIONS.get(stem.suffix.lower())
        if fmt is None:
            raise ValueError(
                f"Unrecognised file extension {stem.suffix!r}.  Supported: {', '.join(sorted(self._FASTA_EXTENSIONS))}"
            )

        # -- parse sequences --------------------------------------------
        handle = gzip.open(filepath, "rt") if is_gz else open(filepath)
        try:
            records = list(SeqIO.parse(handle, fmt))
        finally:
            handle.close()

        if not records:
            raise ValueError(f"No sequences found in {filepath}")

        seq_ids: list[str] = []
        descriptions: list[str] = []
        sequences: list[str] = []
        for rec in records:
            seq_ids.append(rec.id)
            descriptions.append(rec.description)
            sequences.append(str(rec.seq))

        # -- resolve seq_type -------------------------------------------
        resolved_seq_type = self._resolve_seq_type(
            seq_type,
            model,
            sequences,
        )

        # -- embed sequences --------------------------------------------
        inst = self._get_model(model)
        embeddings: list[np.ndarray | None] = []
        n_failed = 0
        for i, seq in enumerate(sequences):
            try:
                emb = inst.embed(
                    input=seq,
                    pooling_strategy=pooling_strategy,
                    **kwargs,
                )
                embeddings.append(emb)
            except Exception as e:  # noqa: BLE001
                logging.warning(
                    "Failed to embed sequence %s (%s): %s",
                    seq_ids[i],
                    seq[:30] + "...",
                    e,
                )
                embeddings.append(None)
                n_failed += 1

        # -- build AnnData ----------------------------------------------
        emb_dim = next(
            (e.shape[-1] for e in embeddings if e is not None),
            0,
        )
        emb_matrix = np.full(
            (len(sequences), emb_dim),
            np.nan,
            dtype=np.float32,
        )
        for i, emb in enumerate(embeddings):
            if emb is not None:
                emb_matrix[i] = emb

        import pandas as pd

        obs = pd.DataFrame(
            {
                "sequence_id": seq_ids,
                "sequence_length": [len(s) for s in sequences],
                "description": descriptions,
                "seq_type": resolved_seq_type,
            }
        )
        obs.index = obs["sequence_id"].astype(str)
        # ensure unique index
        if obs.index.duplicated().any():
            obs.index = pd.Index([f"{sid}_{i}" for i, sid in enumerate(obs.index)])

        key = obsm_key or f"X_{model}"
        adata = ad.AnnData(obs=obs)
        adata.obsm[key] = emb_matrix
        adata.uns["embed_fasta"] = {
            "model": model,
            "pooling_strategy": pooling_strategy,
            "source_file": str(filepath),
            "n_sequences": len(sequences),
            "n_failed": n_failed,
        }

        n_ok = len(sequences) - n_failed
        logging.info(
            "embed_fasta: embedded %d/%d sequences from %s (model=%s, seq_type=%s)",
            n_ok,
            len(sequences),
            filepath.name,
            model,
            resolved_seq_type,
        )
        return adata

    def _resolve_seq_type(
        self,
        explicit: Literal["dna", "protein"] | None,
        model: str,
        sequences: list[str],
    ) -> str:
        """Resolve sequence type with three-tier fallback."""
        if explicit is not None:
            logging.info("embed_fasta: using explicit seq_type=%r", explicit)
            return explicit

        # Tier 1: infer from model type
        try:
            inst = self._get_model(model)
            mtype = getattr(inst, "model_type", None)
            if mtype == "dna":
                logging.info(
                    "embed_fasta: inferred seq_type='dna' from model %r",
                    model,
                )
                return "dna"
            if mtype == "protein":
                logging.info(
                    "embed_fasta: inferred seq_type='protein' from model %r",
                    model,
                )
                return "protein"
        except Exception:  # noqa: BLE001
            pass

        # Tier 2: auto-detect from character set
        sample = sequences[:100]
        all_dna = all(set(seq.upper()).issubset(self._DNA_CHARS) for seq in sample if seq)
        if all_dna:
            logging.info(
                "embed_fasta: auto-detected seq_type='dna' from sequence character set",
            )
            return "dna"
        else:
            has_non_dna = any(not set(seq.upper()).issubset(self._DNA_CHARS) for seq in sample if seq)
            if has_non_dna:
                logging.info(
                    "embed_fasta: auto-detected seq_type='protein' from sequence character set",
                )
                return "protein"

        raise ValueError(
            "Could not determine sequence type automatically.  Please specify seq_type='dna' or seq_type='protein'."
        )


# ------------------------------------------------------------------
# Resolution context for tracking fallback steps without verbose warnings
# ------------------------------------------------------------------
class _ResolutionContext:
    """Track identifier resolution steps for consolidated reporting.

    Used by ``embed_perturbation_morphology`` to collect fallback attempts
    internally and report a single summary message at the end instead of
    printing warnings for each intermediate step.
    """

    def __init__(self) -> None:
        self.steps: list[str] = []
        self.final_source: str | None = None

    def add_step(self, description: str) -> None:
        """Record an attempted resolution step."""
        self.steps.append(description)

    def set_source(self, source: str) -> None:
        """Record the final successful data source."""
        self.final_source = source


# ------------------------------------------------------------------
# JUMP S3 URLs for pre-computed profile parquets (most processed)
# ------------------------------------------------------------------
_JUMP_S3_BASE = (
    "https://cellpainting-gallery.s3.amazonaws.com/cpg0016-jump-assembled/source_all/workspace/profiles_assembled"
)
_JUMP_PARQUET_URLS: dict[str, str] = {
    "crispr": (
        f"{_JUMP_S3_BASE}/CRISPR/v1.0a/"
        "profiles_wellpos_cc_var_mad_outlier_featselect_sphering_harmony_PCA_corrected.parquet"
    ),
    "orf": (f"{_JUMP_S3_BASE}/ORF/v1.0a/profiles_wellpos_cc_var_mad_outlier_featselect_sphering_harmony.parquet"),
    "compound": (f"{_JUMP_S3_BASE}/COMPOUND/v1.0/profiles_var_mad_int_featselect_harmony.parquet"),
}

_DEFAULT_MORPHOLOGY_EMBEDDINGS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    os.pardir,
    os.pardir,
    "data",
    "embeddings",
    "morphology_embeddings",
)


def _resolve_jump_parquet_path(
    plate_type: str,
    profiles_dir: str | None,
) -> pathlib.Path:
    """Locate or auto-download the JUMP pre-computed profile parquet.

    Resolution order:

    1. If *profiles_dir* is given, look for
       ``<profiles_dir>/<plate_type>/standard.parquet``.
    2. Otherwise look under the default
       ``data/embeddings/morphology_embeddings/JUMP/<plate_type>/standard.parquet``.
    3. If the file does not exist, download it from the Cell Painting
       Gallery S3 bucket (anonymous, no auth required).
    """
    import pathlib
    import urllib.request

    if profiles_dir is not None:
        base = pathlib.Path(profiles_dir)
    else:
        base = pathlib.Path(os.path.normpath(_DEFAULT_MORPHOLOGY_EMBEDDINGS_DIR)) / "JUMP"

    pq_path = base / plate_type / "standard.parquet"

    if pq_path.exists():
        return pq_path

    url = _JUMP_PARQUET_URLS.get(plate_type)
    if url is None:
        raise ValueError(f"Unknown JUMP plate type {plate_type!r}. Expected one of {list(_JUMP_PARQUET_URLS)}.")

    pq_path.parent.mkdir(parents=True, exist_ok=True)
    logging.info(
        "Downloading JUMP %s profiles from Cell Painting Gallery (this may take a while for large files)...",
        plate_type,
    )
    logging.info("  URL : %s", url)
    logging.info("  Dest: %s", pq_path)

    tmp_path = pq_path.with_suffix(".parquet.tmp")
    try:
        urllib.request.urlretrieve(url, str(tmp_path))
        tmp_path.rename(pq_path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise

    logging.info("Download complete: %s", pq_path)
    return pq_path


# ------------------------------------------------------------------
# Module-level helpers for morphological resolution
# ------------------------------------------------------------------


def _load_jump_fov_local(well_meta: dict, local_dir: str) -> np.ndarray:
    """Load a JUMP field-of-view from local files."""
    import os

    import numpy as np
    from PIL import Image

    from embpy.pp.morphology_preprocessing import CELL_PAINTING_CHANNELS

    def _get(row: dict, *keys: str):
        for k in keys:
            if k in row and row[k] is not None:
                return row[k]
        return None

    plate = _get(well_meta, "plate", "Metadata_Plate")
    well = _get(well_meta, "well", "Metadata_Well")
    site = _get(well_meta, "site", "Metadata_Site") or 1

    planes = []
    for ch in CELL_PAINTING_CHANNELS:
        pattern = f"{plate}_{well}_{site}_{ch}"
        found = None
        for f in os.listdir(local_dir):
            if pattern in f and os.path.isfile(os.path.join(local_dir, f)):
                found = os.path.join(local_dir, f)
                break
        if found is None:
            raise FileNotFoundError(f"Channel {ch} not found for {pattern} in {local_dir}")
        img = np.asarray(Image.open(found), dtype=np.float32)
        planes.append(img)
    return np.stack(planes, axis=0)


def _save_hpa_channels(
    img: np.ndarray,
    cache_dir,
    antibody: str,
    plate,
    position: str,
    sample,
) -> None:
    """Save a (4, H, W) HPA image as per-channel JPEGs in *cache_dir*."""
    from pathlib import Path as _Path

    from PIL import Image as _PILImage

    from embpy.resources.hpa_images import HPA_IF_CHANNELS

    cache_dir = _Path(cache_dir)
    for i, ch in enumerate(HPA_IF_CHANNELS):
        fname = f"{antibody}_{plate}_{position}_{sample}_{ch}.jpg"
        dest = cache_dir / fname
        if dest.exists():
            continue
        plane = img[i]
        _PILImage.fromarray(plane).save(dest)


def _resolve_jump_perturbation_fallback(
    perturbation: str,
    perturbation_type: str,
    return_source: bool = False,
) -> list[dict] | tuple[list[dict], str | None]:
    """Try resolving a perturbation using DrugResolver or GeneResolver.

    When the initial ``get_jump_item_location_metadata`` lookup fails for a
    given name, this function:

    - **Compounds**: uses ``DrugResolver`` to generate name variants
      (salt-stripping, SMILES resolution, synonym lookup) and retries.
    - **Genetic perturbations**: uses ``GeneResolver`` to canonicalise the
      gene symbol and retries.

    Parameters
    ----------
    perturbation
        The perturbation identifier (gene symbol or compound name).
    perturbation_type
        ``"genetic"`` or ``"compound"``.
    return_source
        If ``True``, returns a tuple ``(rows, source)`` where ``source``
        describes how the perturbation was resolved.

    Returns
    -------
    list[dict] | tuple[list[dict], str | None]
        If ``return_source=False``: list of metadata rows.
        If ``return_source=True``: tuple of (rows, source_description).
    """
    if perturbation_type == "compound":
        result = _resolve_compound_jump_fallback(perturbation, return_source=True)
    else:
        result = _resolve_gene_jump_fallback(perturbation, return_source=True)

    if return_source:
        return result
    return result[0]


def _resolve_compound_jump_fallback(
    compound_name: str, return_source: bool = False
) -> list[dict] | tuple[list[dict], str | None]:
    """Use DrugResolver to find alternate compound names for JUMP lookup."""
    from .resources.jump_metadata import get_jump_item_location_metadata

    def _ret(rows: list[dict], source: str | None):
        if return_source:
            return rows, source
        return rows

    try:
        from .resources.drug_resolver import DrugResolver

        resolver = DrugResolver(use_rdkit=False, sleep_sec=0.2)

        # 1) Try cleaned name variants (salt stripping, Greek letters, etc.)
        for variant in resolver._name_variants(compound_name):
            if variant == compound_name:
                continue
            rows = get_jump_item_location_metadata(variant, input_column="standard_key")
            if rows:
                return _ret(rows, f"DrugResolver name variant '{variant}'")

        # 2) Resolve to SMILES, then look up synonyms and try those
        smiles = resolver.name_to_smiles(compound_name)
        if smiles:
            synonyms = resolver.smiles_to_names(smiles, top_k=5)
            for synonym in synonyms:
                if synonym.upper() == compound_name.upper():
                    continue
                rows = get_jump_item_location_metadata(
                    synonym,
                    input_column="standard_key",
                )
                if rows:
                    return _ret(rows, f"DrugResolver synonym '{synonym}'")

        # 3) Try SMILES directly as standard_key (some datasets use InChIKey/SMILES)
        if smiles:
            rows = get_jump_item_location_metadata(smiles, input_column="standard_key")
            if rows:
                return _ret(rows, "DrugResolver SMILES")

    except Exception:  # noqa: BLE001
        pass

    return _ret([], None)


def _resolve_gene_jump_fallback(
    gene_symbol: str, return_source: bool = False
) -> list[dict] | tuple[list[dict], str | None]:
    """Use GeneResolver to canonicalise a gene symbol for JUMP lookup.

    Tries resolving via Ensembl ID -> canonical symbol, then looks up
    alternate symbols from the MyGene.info API.
    """
    from .resources.jump_metadata import get_jump_item_location_metadata

    def _ret(rows: list[dict], source: str | None):
        if return_source:
            return rows, source
        return rows

    try:
        from .resources.gene_resolver import GeneResolver

        resolver = GeneResolver(organism="human")
        ensembl_id = resolver.symbol_to_ensembl(gene_symbol)
        if ensembl_id:
            canonical = resolver.ensembl_to_symbol(ensembl_id)
            if canonical and canonical.upper() != gene_symbol.upper():
                rows = get_jump_item_location_metadata(
                    canonical,
                    input_column="standard_key",
                )
                if rows:
                    return _ret(rows, f"GeneResolver canonical '{canonical}'")

        # Try aliases from mygene
        try:
            import mygene

            mg = mygene.MyGeneInfo()
            result = mg.query(
                gene_symbol,
                scopes="symbol,alias",
                fields="symbol,alias",
                species="human",
            )
            for hit in result.get("hits", []):
                alt = hit.get("symbol", "")
                if alt and alt.upper() != gene_symbol.upper():
                    rows = get_jump_item_location_metadata(
                        alt,
                        input_column="standard_key",
                    )
                    if rows:
                        return _ret(rows, f"mygene alias '{alt}'")
        except Exception:  # noqa: BLE001
            pass

    except Exception:  # noqa: BLE001
        pass

    return _ret([], None)
