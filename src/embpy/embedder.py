import logging
import traceback
from collections.abc import Sequence
from typing import Any, Literal

import numpy as np
import torch
from rdkit import Chem

from .errors import ConfigError, IdentifierError, ModelNotFoundError
from .models.base import BaseModelWrapper

# Import all potential wrappers - handle ImportErrors later if deps are missing
from .models.dna_models import BorzoiWrapper, EnformerWrapper
from .models.molecule_models import (
    ChembertaWrapper,
    MHGGNNWrapper,
    MiniMolWrapper,
    MolEWrapper,
    MolformerWrapper,
    RDKitWrapper,
)
from .models.protein_models import ESM2Wrapper, ESMCWrapper, ProtT5Wrapper
from .models.text_models import TextLLMWrapper
from .resources.gene_resolver import GeneResolver

# Evo (v1/v1.5) is an optional dependency - import conditionally
try:
    from .models.dna_models import EvoWrapper

    _HAVE_EVO = True
except ImportError:
    _HAVE_EVO = False
    EvoWrapper = None  # type: ignore

# Evo2 is an optional dependency - import conditionally
try:
    from .models.dna_models import Evo2Wrapper

    _HAVE_EVO2 = True
except ImportError:
    _HAVE_EVO2 = False
    Evo2Wrapper = None  # type: ignore


# Helper function (can be moved to utils later)
def get_device() -> torch.device:
    """Selects the best available device: CUDA, MPS (for Apple Silicon), or CPU."""
    if torch.cuda.is_available():
        logging.info("CUDA device found, using GPU.")
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        logging.info("MPS device found, using Apple Silicon GPU.")
        return torch.device("mps")
    else:
        logging.info("No GPU found, using CPU.")
        return torch.device("cpu")


# Define the model registry mapping user-facing names to Wrapper classes and model paths
# This could potentially be loaded from a config file or use entry points for extensibility
MODEL_REGISTRY: dict[str, tuple[type[BaseModelWrapper] | None, str | None]] = {
    # User-facing name: (WrapperClass, HuggingFace_or_Path_Identifier)
    # --- DNA Models ---
    "enformer_human_rough": (EnformerWrapper, "EleutherAI/enformer-official-rough"),
    "borzoi_v0": (BorzoiWrapper, "johahi/borzoi-replicate-0"),
    "borzoi_v1": (BorzoiWrapper, "johani/borzoi-replicate-1"),
    # Evo models (requires optional `evo-model` dependency: pip install embpy[evo])
    "evo1_8k": (EvoWrapper if _HAVE_EVO else None, "evo-1-8k-base"),
    "evo1_131k": (EvoWrapper if _HAVE_EVO else None, "evo-1-131k-base"),
    "evo1.5_8k": (EvoWrapper if _HAVE_EVO else None, "evo-1.5-8k-base"),
    "evo1_crispr": (EvoWrapper if _HAVE_EVO else None, "evo-1-8k-crispr"),
    "evo1_transposon": (EvoWrapper if _HAVE_EVO else None, "evo-1-8k-transposon"),
    # Evo2 models (requires optional `evo2` dependency: pip install embpy[evo2])
    "evo2_7b": (Evo2Wrapper if _HAVE_EVO2 else None, "evo2_7b"),
    "evo2_40b": (Evo2Wrapper if _HAVE_EVO2 else None, "evo2_40b"),
    "evo2_7b_base": (Evo2Wrapper if _HAVE_EVO2 else None, "evo2_7b_base"),
    "evo2_1b_base": (Evo2Wrapper if _HAVE_EVO2 else None, "evo2_1b_base"),
    # --- Protein Models ---
    "esm2_8M": (ESM2Wrapper, "facebook/esm2_t6_8M_UR50D"),
    "esm2_35M": (ESM2Wrapper, "facebook/esm2_t12_35M_UR50D"),
    "esm2_150M": (ESM2Wrapper, "facebook/esm2_t30_150M_UR50D"),
    "esm2_650M": (ESM2Wrapper, "facebook/esm2_t33_650M_UR50D"),
    "esm2_3B": (ESM2Wrapper, "facebook/esm2_t36_3B_UR50D"),
    # ESMC Models
    "esmc_300m": (ESMCWrapper, "esmc_300m"),
    "esmc_600m": (ESMCWrapper, "esmc_600m"),
    # ProtT5 Models (ProtTrans)
    "prot_t5_xl": (ProtT5Wrapper, "Rostlab/prot_t5_xl_uniref50"),
    "prot_t5_xl_half": (ProtT5Wrapper, "Rostlab/prot_t5_xl_half_uniref50-enc"),
    # --- Molecule Models ---
    "chemberta2MTR": (ChembertaWrapper, "DeepChem/ChemBERTa-77M-MTR"),
    "chemberta2MLM": (ChembertaWrapper, "DeepChem/ChemBERTa-100M-MLM"),
    "molformer_base": (MolformerWrapper, "ibm/MoLFormer-XL-both-10pct"),
    # RDKit Fingerprints (CPU-only, no download needed)
    "rdkit_fp": (RDKitWrapper, "rdkit"),
    "morgan_fp": (RDKitWrapper, "morgan"),
    "morgan_count_fp": (RDKitWrapper, "morgan_count"),
    "maccs_fp": (RDKitWrapper, "maccs"),
    "atom_pair_fp": (RDKitWrapper, "atom_pair"),
    "atom_pair_count_fp": (RDKitWrapper, "atom_pair_count"),
    "torsion_fp": (RDKitWrapper, "topological_torsion"),
    "torsion_count_fp": (RDKitWrapper, "topological_torsion_count"),
    # GNN-based molecule models (optional dependencies)
    "minimol": (MiniMolWrapper, "minimol"),
    "mhg_gnn": (MHGGNNWrapper, "ibm-research/materials.mhg-ged"),
    "mole": (MolEWrapper, "mole"),
    # --- Text Models ---
    "minilm_l6_v2": (TextLLMWrapper, "sentence-transformers/all-MiniLM-L6-v2"),
    "bert_base_uncased": (TextLLMWrapper, "bert-base-uncased"),
}


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
        device: str | torch.device | None = "auto",
        resolver_backend: Literal["api", "local"] = "api",
        mart_file: str | None = None,
        chromosome_folder: str | None = None,
    ):
        """
        Initializes the BioEmbedder.

        Args:
            device: 'auto', 'cuda', 'mps', or 'cpu', or torch.device.
            resolver_backend: 'api' to use online APIs, 'local' to use local FASTAs.
            mart_file: path to Mart CSV (required if resolver_backend='local').
            chromosome_folder: path to folder with chr*.fa files (required if 'local').
        """
        # Device setup
        if isinstance(device, str):
            if device == "auto":
                self.device = get_device()
            else:
                self.device = torch.device(device)
        elif isinstance(device, torch.device):
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
            # API mode; mart/chrom args ignored
            self.gene_resolver = GeneResolver()

        # Model cache and discovery
        self.model_cache: dict[str, BaseModelWrapper] = {}
        self._available_models = self._discover_models()

        logging.info(f"BioEmbedder initialized on device {self.device}, backend={self.resolver_backend}")
        logging.info(f"Available models: {self.list_available_models()}")

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
        # Return from cache if already loaded
        if model_name in self.model_cache:
            return self.model_cache[model_name]

        # Check if it's in the registry first
        if model_name in self._available_models:
            logging.info(f"Loading registered model '{model_name}' onto device '{self.device}'...")
            WrapperClass, model_path_or_name = self._available_models[model_name]
            try:
                model_instance = WrapperClass(model_path_or_name=model_path_or_name)
                model_instance.load(self.device)
                self.model_cache[model_name] = model_instance
                logging.info(f"Model '{model_name}' loaded successfully.")
                return model_instance
            except Exception as e:
                logging.error(
                    f"Failed to load model '{model_name}' using wrapper {WrapperClass.__name__} and path '{model_path_or_name}': {e}"
                )
                raise RuntimeError(f"Could not load model '{model_name}'.") from e

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

    def embed_gene(
        self,
        identifier: str,
        model: str,
        id_type: Literal["symbol", "ensembl_id", "uniprot_id"] = "symbol",
        organism: str = "human",
        pooling_strategy: str = "mean",
        gene_description_format: str = "Gene: {identifier}. Type: {id_type}. Organism: {organism}.",
        **kwargs: Any,
    ) -> np.ndarray:
        """
        Generates an embedding for a single gene using the specified model.

        Determines whether to use sequence (DNA/protein) or text description
        based on the selected model's type.

        Args:
            identifier (str): The gene identifier (e.g., "TP53", "ENSG00000141510").
            model (str): The user-facing name of the model to use (e.g., "enformer_human_rough",
                        "esm2_650M", "minilm_l6_v2"). Must be a key in MODEL_REGISTRY.
            id_type (Literal): Type of the identifier provided. Defaults to "symbol".
            organism (str): Organism name (e.g., "human", "mouse"). Used for sequence/description lookup.
                            Defaults to "human".
            pooling_strategy (str): Pooling strategy if the model outputs per-token/residue embeddings.
                                    Defaults to "mean". Check model's `available_pooling_strategies`.
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
        # Fetch input data
        if mtype == "dna":
            if id_type not in ("symbol", "ensembl_id"):
                raise IdentifierError(
                    f"DNA models require id_type 'symbol' or 'ensembl_id', got '{id_type}'."
                )
            dna_id_type: Literal["symbol", "ensembl_id"] = id_type  # type: ignore[assignment]
            if self.resolver_backend == "local":
                seq = self.gene_resolver.get_local_dna_sequence(identifier, dna_id_type)
            else:
                seq = self.gene_resolver.get_dna_sequence(identifier, dna_id_type, organism)
            if not seq:
                raise IdentifierError(f"DNA not found for {id_type}='{identifier}'")
            input_data = seq
        elif mtype == "protein":
            prot = self.gene_resolver.get_protein_sequence(identifier, id_type, organism)
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
            # PPI models take the gene identifier directly (no sequence needed)
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

    def embed_genes_batch(
        self,
        model: str,
        identifiers: Sequence[str] | None = None,  # Changed: Now optional
        id_type: Literal["symbol", "ensembl_id", "uniprot_id"] = "symbol",
        organism: str = "human",
        pooling_strategy: str = "mean",
        gene_description_format: str | None = None,
        fetch_all_dna: bool = False,
        biotype: str = "protein_coding",
        **kwargs: Any,
    ) -> list[np.ndarray | None]:
        """
        Generates embeddings for a batch of genes.

        If `identifiers` is None and `fetch_all_dna` is True, it will automatically
        fetch ALL genes of the specified `biotype` and embed them.
        """
        inst = self._get_model(model)
        mtype = inst.model_type

        # Dictionary to hold pre-fetched sequences (if any)
        prefetched_data: dict[str, str] = {}

        # --- LOGIC CHANGE: Discovery Mode ---
        # If identifiers is None OR we explicitly want to pre-fetch
        if fetch_all_dna or identifiers is None:
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
            return []

        input_data_list: list[str | None] = []
        logging.info(f"Batch: {len(identifiers)} items for '{model}' ({mtype})...")

        # Narrow id_type for DNA resolver methods (only accept symbol/ensembl_id)
        dna_id: Literal["symbol", "ensembl_id"] | None = (
            id_type if id_type in ("symbol", "ensembl_id") else None  # type: ignore[assignment]
        )

        for ident in identifiers:
            data = None
            try:
                if mtype == "dna":
                    # Try pre-fetched first
                    if prefetched_data:
                        if id_type == "ensembl_id":
                            data = prefetched_data.get(ident)

                    # Fallback
                    if data is None:
                        if dna_id is None:
                            logging.warning(
                                "DNA models require id_type 'symbol' or 'ensembl_id', "
                                "got '%s'; skipping %s.",
                                id_type,
                                ident,
                            )
                        elif self.resolver_backend == "local":
                            data = self.gene_resolver.get_local_dna_sequence(ident, dna_id)
                        else:
                            data = self.gene_resolver.get_dna_sequence(ident, dna_id, organism)

                elif mtype == "protein":
                    # ESMC: Resolves Ensembl ID -> Protein Sequence via API
                    data = self.gene_resolver.get_protein_sequence(ident, id_type, organism)

                elif mtype == "text":
                    fmt = gene_description_format or "Gene: {identifier}..."
                    data = self.gene_resolver.get_gene_description(ident, id_type, organism, format_string=fmt)

                if data is None:
                    logging.warning(f"No data for {ident}; skipping.")
            except Exception as e:
                logging.warning(f"Error fetching {ident}: {e}")
            input_data_list.append(data)

        # ... (Rest of filtering and embedding logic remains the same) ...
        valid_inputs = [d for d in input_data_list if d is not None]
        valid_indices = [i for i, d in enumerate(input_data_list) if d is not None]

        if not valid_inputs:
            return [None] * len(identifiers)

        try:
            batch_results = inst.embed_batch(inputs=valid_inputs, pooling_strategy=pooling_strategy, **kwargs)
        except Exception as e:
            logging.error(f"Batch embed failed: {e}")
            traceback.print_exc()
            return [None] * len(identifiers)

        results: list[np.ndarray | None] = [None] * len(identifiers)
        for idx, emb in zip(valid_indices, batch_results, strict=False):
            results[idx] = emb

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
                inputs=valid_inputs, pooling_strategy=pooling_strategy, batch_size=batch_size, **kwargs
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

    def list_available_models(self) -> list[str]:
        """Returns a list of available model names."""
        return sorted(self._available_models.keys())
