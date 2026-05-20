"""Custom exception hierarchy for ``embpy``.

All exceptions inherit from :class:`EmbpyError`, so users can catch
every package-level error with a single ``except EmbpyError`` if they
want a broad safety net, **or** catch specific subclasses for precise
control::

    from embpy.errors import EmbpyError, IdentifierError

    try:
        emb = embedder.embed_gene("FAKE", model="esm2_650M")
    except IdentifierError:
        print("Gene not found – skip")
    except EmbpyError:
        print("Something else went wrong in embpy")
"""

from __future__ import annotations

# ── Base ────────────────────────────────────────────────────────────


class EmbpyError(Exception):
    """Base class for all ``embpy`` exceptions.

    Two class-level attributes power downstream automation:

    * ``category`` -- a short stable string that identifies the failure
      mode for log aggregation. Subclasses override it to one of
      ``"config"``, ``"identifier"``, ``"resolver"``, ``"model_load"``,
      ``"context_overflow"``, ``"oom"``, ``"dependency"``,
      ``"model_inference"``, ``"data"``, etc. Custom subclasses defined
      outside the package should pick a unique string.
    * ``exit_code`` -- a numeric code that ``embed_perturbations.py`` and
      similar entry points use to give SLURM ``sacct`` a structured
      signal about *why* a job failed. ``0`` is reserved for success;
      ``1`` is the catch-all; categories use ``10..29``. Group these so
      "all OOMs failed" can be queried as ``ExitCode==10`` across the
      whole sweep without grepping any log files.
    """

    category: str = "embpy"
    exit_code: int = 1


# ── Configuration ───────────────────────────────────────────────────


class ConfigError(EmbpyError):
    """The package or a component was configured incorrectly.

    Common causes
    -------------
    * Passing an invalid ``device`` string (must be ``"auto"``,
      ``"cpu"``, ``"cuda"``, or ``"mps"``).
    * Using ``resolver_backend="local"`` without providing both
      ``mart_file`` and ``chromosome_folder``.

    How to fix
    ----------
    Check the arguments you passed to ``BioEmbedder(...)`` or the
    relevant class constructor.
    """

    category = "config"
    exit_code = 20

    def __init__(self, message: str = "Invalid configuration"):
        super().__init__(message)


# ── Identifiers & resolution ───────────────────────────────────────


class IdentifierError(EmbpyError):
    """A gene, protein, or molecule identifier could not be resolved.

    Common causes
    -------------
    * The gene symbol or Ensembl ID does not exist or is misspelled.
    * The Ensembl / UniProt / MyGene.info API is unreachable.
    * A DNA or protein sequence could not be fetched for the given
      identifier and organism.

    How to fix
    ----------
    * Double-check the identifier (e.g. ``"TP53"`` not ``"tp53"``
      — some APIs are case-sensitive).
    * Verify that the ``organism`` argument matches the identifier
      (e.g. don't pass a mouse gene with ``organism="human"``).
    * Try resolving manually with ``GeneResolver`` to see the raw
      API response.
    """

    category = "identifier"
    exit_code = 21

    def __init__(self, message: str = "Invalid or unresolvable identifier"):
        super().__init__(message)


class InvalidSMILESError(EmbpyError):
    """The provided SMILES string is not a valid molecular structure.

    Common causes
    -------------
    * Typo in the SMILES (e.g. unmatched parentheses or ring digits).
    * Passing a drug *name* (like ``"aspirin"``) where a SMILES string
      is expected.

    How to fix
    ----------
    * Validate the SMILES with ``rdkit.Chem.MolFromSmiles(smiles)``
      — if it returns ``None``, the SMILES is malformed.
    * Use ``DrugResolver.name_to_smiles("aspirin")`` to convert a
      drug name to a canonical SMILES first.
    """

    category = "identifier"
    exit_code = 21

    def __init__(self, smiles: str, message: str | None = None):
        self.smiles = smiles
        msg = message or (
            f"Invalid SMILES string: '{smiles}'. "
            "Use DrugResolver.name_to_smiles() to convert drug names, "
            "or validate with rdkit.Chem.MolFromSmiles()."
        )
        super().__init__(msg)


# ── Models ──────────────────────────────────────────────────────────


class ModelNotFoundError(EmbpyError):
    """The requested model name is not in the registry or on Hugging Face.

    Common causes
    -------------
    * Misspelled model key (e.g. ``"esm2_650m"`` instead of
      ``"esm2_650M"`` — keys are case-sensitive).
    * Using an Evo2 model without installing the optional dependency
      (``pip install embpy[evo2]``).

    How to fix
    ----------
    Call ``embedder.list_available_models()`` to see all registered
    model names.
    """

    category = "model_load"
    exit_code = 14

    def __init__(self, message: str = "Model not found"):
        super().__init__(message)


class ModelNotLoadedError(EmbpyError):
    """A model method was called before ``load()`` was invoked.

    Common causes
    -------------
    * Calling ``wrapper.embed(...)`` or ``wrapper.embed_batch(...)``
      without first calling ``wrapper.load(device)``.

    How to fix
    ----------
    Call ``wrapper.load(torch.device("cpu"))`` (or ``"cuda"``) before
    embedding. If you use ``BioEmbedder``, loading is automatic.
    """

    category = "model_load"
    exit_code = 14

    def __init__(self, model_name: str | None = None):
        name = f" '{model_name}'" if model_name else ""
        super().__init__(
            f"Model{name} is not loaded. Call load(device) before embedding."
        )


class InvalidPoolingError(EmbpyError):
    """The requested pooling strategy is not supported by this model.

    Common causes
    -------------
    * Requesting ``"cls"`` pooling on a model that only supports
      ``"mean"`` and ``"max"``.
    * Typo in the strategy name.

    How to fix
    ----------
    Check ``wrapper.available_pooling_strategies`` for the list of
    valid options for the model you are using.
    """

    category = "config"
    exit_code = 20

    def __init__(self, strategy: str, available: list[str]):
        self.strategy = strategy
        self.available = available
        super().__init__(
            f"Invalid pooling strategy '{strategy}'. "
            f"Choose from: {available}"
        )


class EmbeddingError(EmbpyError):
    """The model failed to produce an embedding for the given input.

    Common causes
    -------------
    * Input sequence is too long or too short for the model.
    * GPU out-of-memory during inference.
    * Unexpected model output shape.

    How to fix
    ----------
    * Check that the input meets the model's length requirements
      (e.g. Enformer needs ~196 608 bp of DNA).
    * Try a smaller model or reduce batch size.
    * If on GPU, try ``device="cpu"`` to rule out memory issues.
    """

    category = "model_inference"
    exit_code = 15

    def __init__(self, identifier: str, model: str, cause: str | None = None):
        self.identifier = identifier
        self.model_name = model
        detail = f": {cause}" if cause else ""
        super().__init__(
            f"Embedding failed for '{identifier}' with model '{model}'{detail}"
        )


# ── PPI / graph ─────────────────────────────────────────────────────


class GraphNotBuiltError(EmbpyError):
    """A PPI embedding operation was called before loading data.

    Common causes
    -------------
    * Calling ``embed()`` on a ``PrecomputedPPIWrapper`` without first
      calling ``load(device)``.

    How to fix
    ----------
    Load the embeddings first::

        wrapper = PrecomputedPPIWrapper(data_dir="...", species=9606)
        wrapper.load(torch.device("cpu"))
        emb = wrapper.embed("TP53")
    """

    category = "data"
    exit_code = 22

    def __init__(self, message: str | None = None):
        super().__init__(
            message or "PPI embeddings not loaded. Call load(device) first."
        )


class GeneNotInGraphError(EmbpyError):
    """The requested gene is not present in the PPI embeddings.

    Common causes
    -------------
    * The gene name was not resolved during the STRING API mapping
      at load time.
    * The species does not have a protein for this gene.

    How to fix
    ----------
    * Check ``wrapper.available_genes`` to see which genes are available.
    * Verify the correct ``species`` taxonomy ID was used.
    """

    category = "identifier"
    exit_code = 21

    def __init__(self, gene: str, num_nodes: int):
        self.gene = gene
        self.num_nodes = num_nodes
        super().__init__(
            f"Gene '{gene}' not found in the PPI graph "
            f"({num_nodes} nodes). "
            "Check wrapper.graph_genes or rebuild with this gene included."
        )


# ── Dependencies ────────────────────────────────────────────────────


class DependencyError(EmbpyError):
    """An optional dependency required for this feature is not installed.

    Common causes
    -------------
    * Using ``PrecomputedPPIWrapper`` without ``h5py``.
    * Using ``Evo2Wrapper`` without the ``evo2`` package.
    * Using ``ESMCWrapper`` without the ``esm`` SDK.

    How to fix
    ----------
    Install the missing package::

        pip install h5py              # for PPI embeddings
        pip install embpy[evo2]       # for Evo2
    """

    category = "dependency"
    exit_code = 12

    def __init__(self, package: str, feature: str | None = None):
        self.package = package
        self.feature = feature
        what = f" for {feature}" if feature else ""
        super().__init__(
            f"'{package}' is required{what} but not installed. "
            f"Install with: pip install {package}"
        )


# ── Data / AnnData ──────────────────────────────────────────────────


class DataError(EmbpyError):
    """Something is wrong with the input data structure.

    Common causes
    -------------
    * The specified ``.obs`` column does not exist in the AnnData.
    * The ``.obsm`` key holding embeddings is missing (e.g. calling
      ``reduce_embeddings`` before ``build_embedding_matrix``).
    * A BioMart CSV is missing required columns.

    How to fix
    ----------
    * Print ``adata.obs.columns`` or ``adata.obsm.keys()`` to check
      what is actually available.
    * Make sure you ran the embedding step before reduction or
      filtering.
    """

    category = "data"
    exit_code = 22

    def __init__(self, message: str = "Invalid or missing input data"):
        super().__init__(message)


# ── New: structured failure modes for the embedder catch-sites ─────
#
# These were added after a sweep-debugging session where several
# distinct root causes -- CUDA OOM, model context overflow, missing
# Python package, gene-resolver returning empty -- all surfaced as the
# same misleading "BioEmbedder returned no embeddings for any of N
# symbols" upstream ValueError. The new typed exceptions below let the
# embedder classify the failure *at the source*, so the operator (or
# automation) can distinguish them without grepping tracebacks. See
# ``world_model.scripts.embed_perturbations`` for the top-level handler
# that converts these into category-specific SLURM exit codes.


class ModelOOMError(EmbpyError):
    """The model's forward pass ran out of GPU memory.

    Distinct from a generic ``RuntimeError`` so the caller can decide
    whether to retry with a smaller ``batch_size``, switch to an 80 GB
    GPU, or escalate. Reraised from ``torch.OutOfMemoryError`` and
    similar CUDA OOM signatures.

    Carries enough structured fields (``model_name``, ``batch_size``,
    ``device``, ``attempted_bytes``) to be useful in a JSON sidecar
    summary later (Layer 2 of the error system, not yet implemented).
    """

    category = "oom"
    exit_code = 10

    def __init__(
        self,
        model_name: str,
        batch_size: int | None = None,
        device: str | None = None,
        attempted_bytes: int | None = None,
        message: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device
        self.attempted_bytes = attempted_bytes
        if message is None:
            parts = [f"CUDA OOM during forward pass of '{model_name}'"]
            if batch_size is not None:
                parts.append(f"batch_size={batch_size}")
            if device:
                parts.append(f"device={device}")
            if attempted_bytes:
                parts.append(f"attempted={attempted_bytes / 1e9:.1f} GB")
            message = ". ".join(parts) + (
                ". Lower batch_size, shorten inputs, or use a larger GPU."
            )
        super().__init__(message)


class ContextOverflowError(EmbpyError):
    """A tokenized input exceeds the model's positional embedding capacity.

    Raised when chunk-and-pool cannot or will not be applied (e.g.
    pooling_strategy='none' on a chunked input). The default path in
    ``_hf_batched_embed`` chunks transparently, so callers normally do
    not see this; it surfaces for the strict-output strategies that
    cannot be reconstructed from per-chunk results.
    """

    category = "context_overflow"
    exit_code = 11

    def __init__(
        self,
        model_name: str,
        input_length: int,
        context_window: int,
        message: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.input_length = input_length
        self.context_window = context_window
        super().__init__(
            message
            or (
                f"Input of length {input_length} exceeds context window "
                f"{context_window} for model '{model_name}', and the "
                f"requested pooling strategy cannot be reconstructed "
                f"from chunked outputs."
            )
        )


class ResolverError(EmbpyError):
    """Gene / protein / text resolver returned no usable data.

    Raised when *every* requested identifier was looked up successfully
    by the resolver layer but came back empty (e.g. the NCBI gene
    description API returned no text for any symbol). This is distinct
    from an embedder crash -- the resolver succeeded operationally; the
    payload is just unusable.

    Note: when the resolver itself raised, ``IdentifierError`` is the
    right class (the lookup never completed). ``ResolverError`` is for
    the "lookup completed, output empty" case.
    """

    category = "resolver"
    exit_code = 13

    def __init__(
        self,
        backend: str,
        organism: str,
        n_requested: int,
        n_resolved: int = 0,
        model_name: str | None = None,
        message: str | None = None,
    ) -> None:
        self.backend = backend
        self.organism = organism
        self.n_requested = n_requested
        self.n_resolved = n_resolved
        self.model_name = model_name
        if message is None:
            who = f" for model '{model_name}'" if model_name else ""
            message = (
                f"Resolver '{backend}' returned only {n_resolved}/{n_requested} "
                f"usable entries{who} (organism={organism!r}). "
                f"This typically means the resolver API is up but yields empty "
                f"payloads for the requested symbols. For text-description models "
                f"(e.g. MiniLM) it indicates the gene-summary API is misconfigured "
                f"or rate-limited."
            )
        super().__init__(message)


class ModelLoadError(EmbpyError):
    """A model wrapper's ``load()`` failed for a reason other than missing deps.

    Examples: HuggingFace download failed, weights file is corrupted,
    ``trust_remote_code=True`` module raised, CUDA kernel could not be
    linked. Reserved for *load-time* failures; runtime failures during
    a forward pass are ``ModelOOMError`` or ``EmbeddingError``.
    """

    category = "model_load"
    exit_code = 14

    def __init__(
        self,
        model_name: str,
        cause: str | None = None,
        message: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.cause = cause
        detail = f": {cause}" if cause else ""
        super().__init__(message or f"Failed to load model '{model_name}'{detail}")
