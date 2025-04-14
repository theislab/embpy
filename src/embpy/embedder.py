import logging
from collections.abc import Sequence
from typing import Any, Literal

import numpy as np
import torch

# Import all potential wrappers - handle ImportErrors later if deps are missing
try:
    from .models.DNA_models.enformer import EnformerWrapper
except ImportError:
    EnformerWrapper = None  # type: ignore
# try:
#     from .models.dna_models import BorzoiWrapper
# except ImportError:
#     BorzoiWrapper = None
try:
    from .models.protein_models import ESMWrapper
except ImportError:
    ESMWrapper = None  # type: ignore
try:
    from .models.molecule_models import ChembertaWrapper
except ImportError:
    ChembertaWrapper = None  # type: ignore
# try:
#     from .models.molecule_models import MolformerWrapper
# except ImportError:
#     MolformerWrapper = None
try:
    from .models.text_models import TextLLMWrapper
except ImportError:
    TextLLMWrapper = None  # type: ignore

from .errors import ConfigError, IdentifierError, ModelNotFoundError
from .models.base import BaseModelWrapper
from .resources.gene_resolver import GeneResolver


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
    # Use specific, descriptive names users will provide
    "enformer_human_rough": (EnformerWrapper, "EleutherAI/enformer-official-rough"),
    # "borzoi_human_v1": (BorzoiWrapper, "calico/borzoi-human-v1"), # Hypothetical Borzoi
    # --- Protein Models ---
    # Add specific ESM models you want to support by default
    "esm2_8M": (ESMWrapper, "facebook/esm2_t6_8M_UR50D"),  # iam not sure about this, TODO: check original esm repo
    "esm2_35M": (ESMWrapper, "facebook/esm2_t12_35M_UR50D"),  # Corrected name likely
    "esm2_150M": (ESMWrapper, "facebook/esm2_t30_150M_UR50D"),
    "esm2_650M": (ESMWrapper, "facebook/esm2_t33_650M_UR50D"),
    "esm2_3B": (ESMWrapper, "facebook/esm2_t36_3B_UR50D"),
    # --- Molecule Models ---
    "chemberta_zinc_v1": (ChembertaWrapper, "seyonec/ChemBERTa-zinc-base-v1"),
    # "molformer_base": (MolformerWrapper, "ibm/MoLFormer-Base"), # Hypothetical Molformer
    # --- Text Models ---
    "minilm_l6_v2": (TextLLMWrapper, "sentence-transformers/all-MiniLM-L6-v2"),
    "bert_base_uncased": (TextLLMWrapper, "bert-base-uncased"),  # Example standard HF model
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
        # Add options for gene resolver backend, cache paths etc. later
    ):
        """
        Initializes the BioEmbedder.

        Args:
            device (Optional[Union[str, torch.device]]): The device to use ('cuda', 'mps', 'cpu', 'auto').
                Defaults to "auto".
            # Add other config args
        """
        if isinstance(device, str):
            if device == "auto":
                self.device = get_device()
            else:
                self.device = torch.device(device)
        elif isinstance(device, torch.device):
            self.device = device
        else:
            raise ConfigError("Invalid device specified. Use 'auto', 'cpu', 'cuda', 'mps', or a torch.device object.")

        self.model_cache: dict[str, BaseModelWrapper] = {}
        self.gene_resolver = GeneResolver()  # Initialize the resolver
        # Filter registry based on available wrappers (handles optional dependencies)
        self._available_models = self._discover_models()

        logging.info(f"BioEmbedder initialized on device: {self.device}")
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
        """Loads a model or retrieves it from the cache using the registry."""
        if model_name not in self._available_models:
            # Provide helpful error message, suggesting available models
            available_model_names = self.list_available_models()
            message = f"Model '{model_name}' is not available or supported."
            if available_model_names:
                message += f" Available models: {available_model_names}"
            else:
                message += " No models are currently available (check dependencies and registry)."
            raise ModelNotFoundError(message)

        if model_name not in self.model_cache:
            logging.info(f"Loading model '{model_name}' onto device '{self.device}'...")
            WrapperClass, model_path_or_name = self._available_models[model_name]

            # Instantiate the specific wrapper with the correct model path/name from the registry
            try:
                # Pass the specific HF path/name to the wrapper instance
                model_instance = WrapperClass(model_path_or_name=model_path_or_name)
                model_instance.load(self.device)
                self.model_cache[model_name] = model_instance
                logging.info(f"Model '{model_name}' loaded successfully.")
            except Exception as e:
                logging.error(
                    f"Failed to load model '{model_name}' using wrapper {WrapperClass.__name__} and path '{model_path_or_name}': {e}"
                )
                # Chain the exception for better debugging
                raise RuntimeError(f"Could not load model '{model_name}'.") from e

        return self.model_cache[model_name]

    def embed_gene(
        self,
        identifier: str,
        model: str,  # User provides name like "enformer_human_rough" or "esm2_650M"
        id_type: Literal["symbol", "ensembl_id", "uniprot_id"] = "symbol",
        organism: str = "human",
        # sequence_type: Optional[Literal["dna", "protein"]] = None, # Keep auto-detection
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
        model_instance = self._get_model(model)  # Gets the specific loaded model instance
        input_data: str | None = None  # To hold sequence or text description

        # --- Determine input type based on model ---
        if model_instance.model_type in ["dna", "protein"]:
            sequence_type = model_instance.model_type
            logging.debug(
                f"Model '{model}' requires '{sequence_type}' sequence for {id_type} '{identifier}'. Fetching..."
            )
            try:
                if sequence_type == "dna":
                    input_data = self.gene_resolver.get_dna_sequence(identifier, id_type, organism)
                else:  # protein
                    input_data = self.gene_resolver.get_protein_sequence(identifier, id_type, organism)
            except (KeyError, ValueError) as e:
                # Catch specific errors during sequence fetching
                raise IdentifierError(
                    f"Failed to fetch {sequence_type} sequence for {id_type} '{identifier}': {e}"
                ) from e

            if not input_data:
                raise IdentifierError(f"No {sequence_type} sequence found for {id_type} '{identifier}' ({organism}).")
            logging.debug(f"Sequence fetched successfully (length: {len(input_data)}). Embedding...")

        elif model_instance.model_type == "text":
            logging.debug(f"Model '{model}' requires text description for {id_type} '{identifier}'. Constructing...")
            try:
                input_data = self.gene_resolver.get_gene_description(
                    identifier, id_type, organism, format_string=gene_description_format
                )
            except Exception as e:
                raise IdentifierError(f"Failed to construct description for {id_type} '{identifier}': {e}") from e

            if not input_data:
                raise IdentifierError(f"Could not construct description for {id_type} '{identifier}' ({organism}).")
            logging.debug(f"Description constructed: '{input_data[:100]}...'. Embedding...")

        else:
            # Should not happen if registry/discovery is correct, but as safeguard
            raise ValueError(
                f"Unsupported model type '{model_instance.model_type}' for model '{model}'. Cannot embed gene."
            )

        # --- Embed the input data (sequence or text) ---
        logging.debug(
            f"Embedding gene '{identifier}' using model '{model}' with input type '{model_instance.model_type}'"
        )
        try:
            embedding = model_instance.embed(input=input_data, pooling_strategy=pooling_strategy, **kwargs)
            logging.debug(f"Gene embedding generated with shape: {embedding.shape}")
            return embedding
        except Exception as e:
            logging.error(f"Error during embedding generation for '{identifier}' with model '{model}': {e}")
            raise RuntimeError(f"Embedding failed for identifier '{identifier}'.") from e

    def embed_genes_batch(
        self,
        identifiers: Sequence[str],
        model: str,  # User provides name like "enformer_human_rough" or "esm2_650M"
        id_type: Literal["symbol", "ensembl_id", "uniprot_id"] = "symbol",
        organism: str = "human",
        pooling_strategy: str = "mean",
        gene_description_format: str | None = None,
        **kwargs: Any,
    ) -> list[np.ndarray | None]:  # Return None for errors
        """
        Generates embeddings for a batch of genes (sequence or text based).

        Args:
            identifiers (Sequence[str]): List of gene identifiers.
            model (str): The user-facing name of the model to use.
            id_type (Literal): Type of the identifiers. Defaults to "symbol".
            organism (str): Organism name. Defaults to "human".
            pooling_strategy (str): Pooling strategy. Defaults to "mean".
            gene_description_format (str): Format string for text models.
            **kwargs: Additional arguments for the model's embed_batch method.

        Returns
        -------
            list[Optional[np.ndarray]]: List of embeddings, with None for identifiers that failed.
        """
        model_instance = self._get_model(model)
        input_data_list: list[str | None] = []  # Holds sequences or descriptions

        logging.info(
            f"Preparing batch of {len(identifiers)} identifiers for model '{model}' ({model_instance.model_type} input)..."
        )

        # --- Prepare batch input data ---
        if model_instance.model_type in ["dna", "protein"]:
            sequence_type = model_instance.model_type
            for identifier in identifiers:
                seq = None
                try:
                    if sequence_type == "dna":
                        seq = self.gene_resolver.get_dna_sequence(identifier, id_type, organism)
                    else:
                        seq = self.gene_resolver.get_protein_sequence(identifier, id_type, organism)
                    if seq is None:
                        logging.warning(
                            f"No {sequence_type} sequence found for {id_type} '{identifier}' ({organism}). Skipping."
                        )
                except (KeyError, ValueError) as e:
                    logging.warning(
                        f"Failed to fetch {sequence_type} sequence for {id_type} '{identifier}': {e}. Skipping."
                    )
                input_data_list.append(seq)  # Append sequence or None

        elif model_instance.model_type == "text":
            for identifier in identifiers:
                desc = None
                try:
                    desc = self.gene_resolver.get_gene_description(
                        identifier, id_type, organism, format_string=gene_description_format
                    )
                    if desc is None:
                        logging.warning(
                            f"Could not construct description for {id_type} '{identifier}' ({organism}). Skipping."
                        )
                except (KeyError, ValueError) as e:
                    logging.warning(f"Failed to construct description for {id_type} '{identifier}': {e}. Skipping.")
                input_data_list.append(desc)  # Append description or None
        else:
            raise ValueError(f"Unsupported model type '{model_instance.model_type}' for batch gene embedding.")

        # Filter out None entries before passing to batch embedding
        valid_inputs = [item for item in input_data_list if item is not None]
        valid_indices = [i for i, item in enumerate(input_data_list) if item is not None]

        if not valid_inputs:
            logging.warning("No valid inputs could be generated for the batch.")
            return [None] * len(identifiers)

        logging.info(f"Embedding batch of {len(valid_inputs)} valid inputs using model '{model}'...")

        # --- Embed the batch ---
        batch_results: list[np.ndarray] = []
        try:
            # Assume embed_batch takes list[str] and returns list[np.ndarray]
            batch_results = model_instance.embed_batch(inputs=valid_inputs, pooling_strategy=pooling_strategy, **kwargs)
            if len(batch_results) != len(valid_inputs):
                logging.error(
                    f"Batch embedding returned {len(batch_results)} results for {len(valid_inputs)} inputs. Mismatch!"
                )
                # Handle mismatch - maybe return all None?
                return [None] * len(identifiers)
            logging.info("Batch gene embedding successful.")
        except (KeyError, ValueError) as e:
            logging.warning(f"Failed to construct description for {id_type} '{identifier}': {e}. Skipping.")
            # If batch fails entirely, return None for all original identifiers
            return [None] * len(identifiers)

        # --- Reconstruct the full results list including Nones ---
        final_results: list[np.ndarray | None] = [None] * len(identifiers)
        for i, result_embedding in enumerate(batch_results):
            original_index = valid_indices[i]
            final_results[original_index] = result_embedding

        return final_results

    def embed_molecule(
        self,
        identifier: str,  # Expecting SMILES string
        model: str,  # User provides name like "chemberta_zinc_v1"
        pooling_strategy: str = "mean",
        **kwargs: Any,
    ) -> np.ndarray:
        """
        Generates an embedding for a single small molecule from its SMILES string.

        Args:
            identifier (str): The SMILES string representation of the molecule.
            model (str): The user-facing name of the molecule embedding model (e.g., "chemberta_zinc_v1").
            pooling_strategy (str): Pooling strategy. Defaults to "mean".
            **kwargs: Additional arguments for the model's embed method.

        Returns
        -------
            np.ndarray: The computed molecule embedding.

        Raises
        ------
            ModelNotFoundError: If the requested model name is not registered or available.
            ValueError: If the model is not of type 'molecule'.
            RuntimeError: If model loading or inference fails.
        """
        model_instance = self._get_model(model)
        if model_instance.model_type != "molecule":
            raise ValueError(f"Model '{model}' is not a molecule embedder.")

        smiles = identifier
        logging.debug(f"Embedding molecule SMILES: {smiles[:30]}... using model '{model}'")

        try:
            embedding = model_instance.embed(input=smiles, pooling_strategy=pooling_strategy, **kwargs)
            logging.debug(f"Molecule embedding generated with shape: {embedding.shape}")
            return embedding
        except Exception as e:
            logging.error(f"Error during embedding generation for SMILES '{smiles[:30]}...' with model '{model}': {e}")
            raise RuntimeError(f"Embedding failed for SMILES '{smiles[:30]}...'.") from e

    def embed_molecules_batch(
        self,
        identifiers: Sequence[str],  # Expecting list of SMILES
        model: str,  # User provides name like "chemberta_zinc_v1"
        pooling_strategy: str = "mean",
        **kwargs: Any,
    ) -> list[np.ndarray | None]:  # Return None for errors
        """
        Generates embeddings for a batch of small molecules from their SMILES strings.

        Args:
            identifiers (Sequence[str]): A list or tuple of SMILES strings.
            model (str): The name of the molecule embedding model.
            pooling_strategy (str): Pooling strategy. Defaults to "mean".
            **kwargs: Additional arguments for the model's embed_batch method.

        Returns
        -------
            list[Optional[np.ndarray]]: List of embeddings, with None for SMILES that failed.
        """
        model_instance = self._get_model(model)
        if model_instance.model_type != "molecule":
            raise ValueError(f"Model '{model}' is not a molecule embedder.")

        # Basic validation could be added here (e.g., check if strings look like SMILES)
        valid_inputs = list(identifiers)  # Assume all are valid for now
        logging.info(f"Embedding batch of {len(valid_inputs)} SMILES using model '{model}'...")

        try:
            # Assuming embed_batch takes list[str] and returns list[np.ndarray]
            batch_results = model_instance.embed_batch(inputs=valid_inputs, pooling_strategy=pooling_strategy, **kwargs)
            if len(batch_results) != len(valid_inputs):
                logging.error(
                    f"Batch embedding returned {len(batch_results)} results for {len(valid_inputs)} inputs. Mismatch!"
                )
                return [None] * len(identifiers)  # Or handle differently
            logging.info("Batch molecule embedding successful.")
            # Assuming embed_batch didn't produce Nones internally for errors
            results: list[np.ndarray | None] = batch_results

        except (KeyError, ValueError) as e:
            logging.error(f"Error during batch molecule embedding generation for model '{model}': {e}")
            results = [None] * len(identifiers)  # Return None for all if batch fails

        # TODO: Add per-item error handling if the wrapper's embed_batch can return None or raise partially.
        # For now, assume batch success or complete failure.

        return results

    def embed_text(
        self,
        text: str,
        model: str,  # User provides name like "minilm_l6_v2"
        pooling_strategy: str = "mean",
        **kwargs: Any,
    ) -> np.ndarray:
        """
        Generates an embedding for an arbitrary text string using a text model.

        Args:
            text (str): The input text string.
            model (str): The name of the text embedding model (e.g., "minilm_l6_v2", "bert_base_uncased").
            pooling_strategy (str): Pooling strategy. Defaults to "mean".
            **kwargs: Additional arguments for the text model's embed method.

        Returns
        -------
            np.ndarray: The computed text embedding.

        Raises
        ------
            ModelNotFoundError: If the requested model name is not registered or available.
            ValueError: If the model is not of type 'text'.
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
        model: str,  # User provides name like "minilm_l6_v2"
        pooling_strategy: str = "mean",
        **kwargs: Any,
    ) -> list[np.ndarray | None]:  # Return None for errors
        """
        Generates embeddings for a batch of arbitrary text strings.

        Args:
            texts (Sequence[str]): A list or tuple of text strings.
            model (str): The name of the text embedding model.
            pooling_strategy (str): Pooling strategy. Defaults to "mean".
            **kwargs: Additional arguments for the model's embed_batch method.

        Returns
        -------
            list[Optional[np.ndarray]]: List of embeddings, with None for texts that failed.
        """
        model_instance = self._get_model(model)
        if model_instance.model_type != "text":
            raise ValueError(f"Model '{model}' is not a text embedder.")

        valid_inputs = list(texts)  # Assume all valid
        logging.info(f"Embedding batch of {len(valid_inputs)} texts using model '{model}'...")

        try:
            # Assuming embed_batch takes list[str] and returns list[np.ndarray]
            batch_results = model_instance.embed_batch(inputs=valid_inputs, pooling_strategy=pooling_strategy, **kwargs)
            if len(batch_results) != len(valid_inputs):
                logging.error(
                    f"Batch embedding returned {len(batch_results)} results for {len(valid_inputs)} inputs. Mismatch!"
                )
                return [None] * len(texts)
            logging.info("Batch text embedding successful.")
            results: list[np.ndarray | None] = batch_results

        except (ValueError, KeyError) as e:
            logging.error(f"Error during batch text embedding generation for model '{model}': {e}")
            results = [None] * len(texts)

        # TODO: Add per-item error handling if the wrapper's embed_batch can return None or raise partially.

        return results

    # Add methods for direct sequence embedding if desired
    # def embed_dna_sequence(self, sequence: str, model: str, ...) -> np.ndarray: ...
    # def embed_protein_sequence(self, sequence: str, model: str, ...) -> np.ndarray: ...

    def list_available_models(self) -> list[str]:
        """Returns a list of available model names."""
        return sorted(self._available_models.keys())
