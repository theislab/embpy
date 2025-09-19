import logging
from collections.abc import Sequence
from typing import Any, Literal

import numpy as np
import torch
from rdkit import Chem

from .errors import ConfigError, IdentifierError, ModelNotFoundError
from .models.base import BaseModelWrapper

# Import all potential wrappers - handle ImportErrors later if deps are missing
from .models.dna_models import BorzoiWrapper, EnformerWrapper
from .models.molecule_models import ChembertaWrapper, MolformerWrapper
from .models.protein_models import ESM2Wrapper, ESMCWrapper
from .models.text_models import TextLLMWrapper
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
    "borzoi_v0": (BorzoiWrapper, "johahi/borzoi-replicate-0"),
    "borzoi_v1": (BorzoiWrapper, "johani/borzoi-replicate-1"),
    # consider adding flashzoi
    # --- Protein Models ---
    # Add specific ESM models you want to support by default
    "esm2_8M": (ESM2Wrapper, "facebook/esm2_t6_8M_UR50D"),
    "esm2_35M": (ESM2Wrapper, "facebook/esm2_t12_35M_UR50D"),
    "esm2_150M": (ESM2Wrapper, "facebook/esm2_t30_150M_UR50D"),
    "esm2_650M": (ESM2Wrapper, "facebook/esm2_t33_650M_UR50D"),
    "esm2_3B": (ESM2Wrapper, "facebook/esm2_t36_3B_UR50D"),
    # ESMC Models
    "esmc_300m": (ESMCWrapper, "esmc_300m"),
    "esmc_600m": (ESMCWrapper, "esmc_600m"),
    # --- Molecule Models ---
    "chemberta2MTR": (ChembertaWrapper, "DeepChem/ChemBERTa-77M-MTR"),
    "chemberta2MLM": (ChembertaWrapper, "DeepChem/ChemBERTa-100M-MLM"),
    "molformer_base": (MolformerWrapper, "ibm/MoLFormer-XL-both-10pct"),  # Hypothetical Molformer
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
            print(self._available_models)
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
            if self.resolver_backend == "local":
                seq = self.gene_resolver.get_local_dna_sequence(identifier, id_type)
            else:
                seq = self.gene_resolver.get_dna_sequence(identifier, id_type, organism)
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
        identifiers: Sequence[str],
        model: str,
        id_type: Literal["symbol", "ensembl_id", "uniprot_id"] = "symbol",
        organism: str = "human",
        pooling_strategy: str = "mean",
        gene_description_format: str | None = None,
        **kwargs: Any,
    ) -> list[np.ndarray | None]:
        """
        Generates embeddings in batch.

        Generates embeddings for a batch of genes using DNA, protein, text, or molecule models.
        Honors the resolver_backend for DNA lookups.

        Returns list of embeddings or None, preserving original order.
        """
        inst = self._get_model(model)
        mtype = inst.model_type
        input_data_list: list[str | None] = []
        logging.info(f"Batch: {len(identifiers)} items for '{model}' ({mtype})...")
        for ident in identifiers:
            data = None
            try:
                if mtype == "dna":
                    data = (
                        self.gene_resolver.get_local_dna_sequence(ident, id_type)
                        if self.resolver_backend == "local"
                        else self.gene_resolver.get_dna_sequence(ident, id_type, organism)
                    )
                elif mtype == "protein":
                    data = self.gene_resolver.get_protein_sequence(ident, id_type, organism)
                elif mtype == "text":
                    fmt = gene_description_format or "Gene: {identifier}. Type: {id_type}. Organism: {organism}."
                    data = self.gene_resolver.get_gene_description(ident, id_type, organism, format_string=fmt)
                elif mtype == "molecule":
                    data = ident
                else:
                    logging.error(f"Unsupported model type '{mtype}' in batch.")
                if data is None:
                    logging.warning(f"No data for {ident}; skipping.")
            except Exception as e:  # noqa: BLE001
                logging.warning(f"Error fetching {ident}: {e}")
            input_data_list.append(data)
        # Filter and embed
        valid_inputs = [d for d in input_data_list if d is not None]
        valid_indices = [i for i, d in enumerate(input_data_list) if d is not None]
        if not valid_inputs:
            return [None] * len(identifiers)
        try:
            batch_results = inst.embed_batch(inputs=valid_inputs, pooling_strategy=pooling_strategy, **kwargs)
        except Exception as e:  # noqa: BLE001
            logging.error(f"Batch embed failed: {e}")
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
