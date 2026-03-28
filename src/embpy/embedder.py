from __future__ import annotations

import logging
import traceback
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import torch
from rdkit import Chem

from .errors import ConfigError, IdentifierError, ModelNotFoundError
from .models.base import BaseModelWrapper
from .models.dna_models import (
    BorzoiWrapper,
    CaduceusWrapper,
    EnformerWrapper,
    GENALMWrapper,
    HyenaDNAWrapper,
    NucleotideTransformerV3Wrapper,
    NucleotideTransformerWrapper,
)

if TYPE_CHECKING:
    pass
from .models.molecule_models import (
    ChembertaWrapper,
    MHGGNNWrapper,
    MiniMolWrapper,
    MolEWrapper,
    MolformerWrapper,
    RDKitWrapper,
)
from .models.protein_models import ESM2Wrapper, ESM3Wrapper, ESMCWrapper, ProtT5Wrapper
from .models.text_models import TextLLMWrapper
from .resources.gene_resolver import GeneResolver
from .resources.protein_resolver import ProteinResolver

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


# Define the model registry mapping user-facing names to Wrapper classes and model paths
# This could potentially be loaded from a config file or use entry points for extensibility
MODEL_REGISTRY: dict[str, tuple[type[BaseModelWrapper] | None, str | None]] = {
    # User-facing name: (WrapperClass, HuggingFace_or_Path_Identifier)
    # --- DNA Models ---
    "enformer_human_rough": (EnformerWrapper, "EleutherAI/enformer-official-rough"),
    # Borzoi (johahi/borzoi-pytorch, 4 replicates + mouse variants)
    "borzoi_v0": (BorzoiWrapper, "johahi/borzoi-replicate-0"),
    "borzoi_v1": (BorzoiWrapper, "johahi/borzoi-replicate-1"),
    "borzoi_v2": (BorzoiWrapper, "johahi/borzoi-replicate-2"),
    "borzoi_v3": (BorzoiWrapper, "johahi/borzoi-replicate-3"),
    "borzoi_v0_mouse": (BorzoiWrapper, "johahi/borzoi-replicate-0-mouse"),
    "borzoi_v1_mouse": (BorzoiWrapper, "johahi/borzoi-replicate-1-mouse"),
    "borzoi_v2_mouse": (BorzoiWrapper, "johahi/borzoi-replicate-2-mouse"),
    "borzoi_v3_mouse": (BorzoiWrapper, "johahi/borzoi-replicate-3-mouse"),
    # Flashzoi (3x faster Borzoi with FlashAttention-2)
    "flashzoi_v0": (BorzoiWrapper, "johahi/flashzoi-replicate-0"),
    "flashzoi_v1": (BorzoiWrapper, "johahi/flashzoi-replicate-1"),
    "flashzoi_v2": (BorzoiWrapper, "johahi/flashzoi-replicate-2"),
    "flashzoi_v3": (BorzoiWrapper, "johahi/flashzoi-replicate-3"),
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
    # ESM-1b (Meta AI, 650M params, HuggingFace)
    "esm1b": (ESM2Wrapper, "facebook/esm-1b"),
    # ESM-1v (Meta AI, 650M params, 5 random seeds, HuggingFace)
    "esm1v_1": (ESM2Wrapper, "facebook/esm1v_t33_650M_UR90S_1"),
    "esm1v_2": (ESM2Wrapper, "facebook/esm1v_t33_650M_UR90S_2"),
    "esm1v_3": (ESM2Wrapper, "facebook/esm1v_t33_650M_UR90S_3"),
    "esm1v_4": (ESM2Wrapper, "facebook/esm1v_t33_650M_UR90S_4"),
    "esm1v_5": (ESM2Wrapper, "facebook/esm1v_t33_650M_UR90S_5"),
    # ESM-2 (Meta AI, HuggingFace Transformers)
    "esm2_8M": (ESM2Wrapper, "facebook/esm2_t6_8M_UR50D"),
    "esm2_35M": (ESM2Wrapper, "facebook/esm2_t12_35M_UR50D"),
    "esm2_150M": (ESM2Wrapper, "facebook/esm2_t30_150M_UR50D"),
    "esm2_650M": (ESM2Wrapper, "facebook/esm2_t33_650M_UR50D"),
    "esm2_3B": (ESM2Wrapper, "facebook/esm2_t36_3B_UR50D"),
    "esm2_15B": (ESM2Wrapper, "facebook/esm2_t48_15B_UR50D"),
    # ESM-C (EvolutionaryScale SDK)
    "esmc_300m": (ESMCWrapper, "esmc_300m"),
    "esmc_600m": (ESMCWrapper, "esmc_600m"),
    "esmc_6b": (ESMCWrapper, "esmc-6b-2024-12"),
    # ESM3 (EvolutionaryScale SDK -- open weights or Forge API)
    "esm3_small": (ESM3Wrapper, "esm3-small-2024-08"),
    "esm3_medium": (ESM3Wrapper, "esm3-medium-2024-08"),
    "esm3_large": (ESM3Wrapper, "esm3-large-2024-03"),
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
    # GENA-LM (AIRI-Institute) — pip install transformers
    "gena_lm_bert_base": (GENALMWrapper, "AIRI-Institute/gena-lm-bert-base-t2t"),
    "gena_lm_bert_large": (GENALMWrapper, "AIRI-Institute/gena-lm-bert-large-t2t"),
    "gena_lm_bert_base_multi": (GENALMWrapper, "AIRI-Institute/gena-lm-bert-base-t2t-multi"),
    "gena_lm_bigbird_base": (GENALMWrapper, "AIRI-Institute/gena-lm-bigbird-base-t2t"),
    # Nucleotide Transformer v1/v2 (InstaDeep) — pip install transformers
    "nt_500m_human_ref": (NucleotideTransformerWrapper, "InstaDeepAI/nucleotide-transformer-500m-human-ref"),
    "nt_500m_1000g": (NucleotideTransformerWrapper, "InstaDeepAI/nucleotide-transformer-500m-1000g"),
    "nt_2b5_1000g": (NucleotideTransformerWrapper, "InstaDeepAI/nucleotide-transformer-2.5b-1000g"),
    "nt_2b5_multi": (NucleotideTransformerWrapper, "InstaDeepAI/nucleotide-transformer-2.5b-multi-species"),
    "nt_v2_50m": (NucleotideTransformerWrapper, "InstaDeepAI/nucleotide-transformer-v2-50m-multi-species"),
    "nt_v2_100m": (NucleotideTransformerWrapper, "InstaDeepAI/nucleotide-transformer-v2-100m-multi-species"),
    "nt_v2_250m": (NucleotideTransformerWrapper, "InstaDeepAI/nucleotide-transformer-v2-250m-multi-species"),
    "nt_v2_500m": (NucleotideTransformerWrapper, "InstaDeepAI/nucleotide-transformer-v2-500m-multi-species"),
    # Nucleotide Transformer v3 (InstaDeep) — pip install transformers
    "ntv3_8m_pre": (NucleotideTransformerV3Wrapper, "InstaDeepAI/NTv3_8M_pre"),
    "ntv3_100m_pre": (NucleotideTransformerV3Wrapper, "InstaDeepAI/NTv3_100M_pre"),
    "ntv3_100m_pos": (NucleotideTransformerV3Wrapper, "InstaDeepAI/NTv3_100M_pos"),
    "ntv3_650m_pre": (NucleotideTransformerV3Wrapper, "InstaDeepAI/NTv3_650M_pre"),
    "ntv3_650m_pos": (NucleotideTransformerV3Wrapper, "InstaDeepAI/NTv3_650M_pos"),
    # HyenaDNA (HazyResearch) — pip install transformers
    "hyenadna_tiny_1k": (HyenaDNAWrapper, "LongSafari/hyenadna-tiny-1k-seqlen-hf"),
    "hyenadna_small_32k": (HyenaDNAWrapper, "LongSafari/hyenadna-small-32k-seqlen-hf"),
    "hyenadna_medium_160k": (HyenaDNAWrapper, "LongSafari/hyenadna-medium-160k-seqlen-hf"),
    "hyenadna_medium_450k": (HyenaDNAWrapper, "LongSafari/hyenadna-medium-450k-seqlen-hf"),
    "hyenadna_large_1m": (HyenaDNAWrapper, "LongSafari/hyenadna-large-1m-seqlen-hf"),
    # Caduceus (kuleshov-group) — pip install embpy[caduceus]
    "caduceus_ph_131k": (CaduceusWrapper, "kuleshov-group/caduceus-ph_seqlen-131k_d_model-256_n_layer-16"),
    "caduceus_ps_131k": (CaduceusWrapper, "kuleshov-group/caduceus-ps_seqlen-131k_d_model-256_n_layer-16"),
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
        device: str | torch.device | None = "auto",  # type: ignore[name-defined]
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
            # API mode; mart/chrom args ignored
            self.gene_resolver = GeneResolver()

        # Protein resolver
        self.protein_resolver = ProteinResolver(organism="human")

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
        id_type: Literal["symbol", "ensembl_id", "uniprot_id"] = "symbol",
        organism: str = "human",
        pooling_strategy: str = "mean",
        isoform: Literal["canonical", "all"] = "canonical",
        **kwargs: Any,
    ) -> dict[str, np.ndarray] | dict[str, dict[str, np.ndarray]]:
        """Embed proteins for a batch of gene identifiers.

        Parameters
        ----------
        identifiers
            List of gene identifiers.
        model
            Protein model name.
        id_type
            Type of identifiers.
        organism
            Organism name.
        pooling_strategy
            Pooling strategy.
        isoform
            ``"canonical"`` or ``"all"``.
        **kwargs
            Forwarded to the model.

        Returns
        -------
        dict[str, np.ndarray]
            When ``isoform="canonical"`` -- maps identifier to embedding.
        dict[str, dict[str, np.ndarray]]
            When ``isoform="all"`` -- maps identifier to
            ``{isoform_accession: embedding}``.
        """
        inst = self._get_model(model)
        if inst.model_type != "protein":
            raise ValueError(f"Model '{model}' is not a protein model.")

        if isoform == "canonical":
            seqs = self.protein_resolver.get_canonical_sequences_batch(
                identifiers,
                id_type,
                organism,
            )
            results: dict[str, np.ndarray] = {}
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
                if id_type == "sequence":
                    data = ident
                elif mtype == "dna":
                    if region in ("exons", "introns") and dna_id is not None:
                        data = self.gene_resolver.get_gene_region_sequence(
                            ident,
                            id_type=dna_id,
                            organism=organism,
                            region=region,
                        )
                    else:
                        if prefetched_data:
                            if id_type == "ensembl_id":
                                data = prefetched_data.get(ident)

                        if data is None:
                            if dna_id is None:
                                logging.warning(
                                    "DNA models require id_type 'symbol', 'ensembl_id', or 'sequence', "
                                    "got '%s'; skipping %s.",
                                    id_type,
                                    ident,
                                )
                            elif self.resolver_backend == "local":
                                data = self.gene_resolver.get_local_dna_sequence(ident, dna_id)
                            else:
                                data = self.gene_resolver.get_dna_sequence(ident, dna_id, organism)

                elif mtype == "protein":
                    data = self.protein_resolver.get_canonical_sequence(ident, id_type, organism)

                elif mtype == "text":
                    fmt = gene_description_format or "Gene: {identifier}..."
                    data = self.gene_resolver.get_gene_description(ident, id_type, organism, format_string=fmt)

                if data is None:
                    logging.warning(f"No data for {ident}; skipping.")
            except Exception as e:  # noqa: BLE001
                logging.warning(f"Error fetching {ident}: {e}")
            input_data_list.append(data)

        # ... (Rest of filtering and embedding logic remains the same) ...
        valid_inputs = [d for d in input_data_list if d is not None]
        valid_indices = [i for i, d in enumerate(input_data_list) if d is not None]

        if not valid_inputs:
            return [None] * len(identifiers)

        try:
            batch_results = inst.embed_batch(inputs=valid_inputs, pooling_strategy=pooling_strategy, **kwargs)
        except Exception as e:  # noqa: BLE001
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
            get_singlecell_wrapper,
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

                else:
                    wrapper = get_singlecell_wrapper(
                        model_key,
                        batch_size=batch_size,
                    )
                    wrapper.load(device_str)
                    embs = wrapper.embed_cells(adata)

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
            - ``.obsm["{prefix}{pert_model}"]`` = perturbation embeddings
              (mapped per cell)
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
        >>> result.obsm["X_esm2_650M"].shape  # perturbation embeddings
        (5000, 1280)
        """
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

                    # Map embeddings back to cells
                    n_cells = adata.n_obs
                    cell_matrix = np.zeros(
                        (n_cells, emb_dim),
                        dtype=np.float32,
                    )
                    n_mapped = 0
                    for i, pert in enumerate(pert_ids):
                        emb = pert_embs.get(pert)
                        if emb is not None:
                            cell_matrix[i] = emb
                            n_mapped += 1

                    adata.obsm[obsm_key] = cell_matrix
                    n_ok = sum(1 for v in pert_embs.values() if v is not None)
                    metadata[model_key] = {
                        "obsm_key": obsm_key,
                        "embedding_dim": emb_dim,
                        "n_cells": n_cells,
                        "n_perturbations_total": len(unique_perts),
                        "n_perturbations_embedded": n_ok,
                        "n_cells_mapped": n_mapped,
                        "type": "perturbation",
                        "perturbation_column": perturbation_column,
                    }
                    logging.info(
                        "  -> stored in .obsm['%s'], shape %s (%d/%d perturbations embedded, %d/%d cells mapped)",
                        obsm_key,
                        cell_matrix.shape,
                        n_ok,
                        len(unique_perts),
                        n_mapped,
                        n_cells,
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
            "single_cell",
        ] = "all",
    ) -> list[str]:
        """Return available model names, optionally filtered by category.

        Parameters
        ----------
        category
            ``"all"`` returns every model (sequence + single-cell).
            ``"dna"``, ``"protein"``, ``"molecule"``, ``"text"`` filter
            the sequence/structure model registry.
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
