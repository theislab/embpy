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
    """Base class for all ``embpy`` exceptions."""


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

    def __init__(self, message: str = "Invalid or missing input data"):
        super().__init__(message)
