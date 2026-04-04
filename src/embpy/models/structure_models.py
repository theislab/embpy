"""Structure prediction model wrappers for embedding extraction.

Currently supports Boltz-2 trunk embeddings (single and pairwise
representations from the pairformer module).

Boltz-2 is a biomolecular interaction prediction model that produces:
- ``s``: per-token single representation (N_tokens, token_s)
- ``z``: pairwise representation (N_tokens, N_tokens, token_z)

These representations capture rich structural and evolutionary
information that can be used as protein embeddings.

Requires: ``pip install boltz[cuda]``
"""

from __future__ import annotations

import logging
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch

from .base import BaseModelWrapper

logger = logging.getLogger(__name__)


class Boltz2Wrapper(BaseModelWrapper):
    """Extract trunk embeddings from Boltz-2.

    Boltz-2 is a structure prediction model (similar to AlphaFold3)
    whose internal pairformer trunk produces rich per-residue (``s``)
    and pairwise (``z``) representations. This wrapper runs a forward
    pass through the trunk and extracts these representations as
    embeddings.

    Parameters
    ----------
    model_path_or_name
        Model identifier. Use ``"boltz2"`` for the standard checkpoint.
    output_type
        Which representation to extract:

        - ``"single"`` -- per-token representation pooled to a vector
        - ``"pairwise"`` -- pairwise representation pooled to a vector
        - ``"both"`` -- concatenation of single + pairwise
    use_msa
        Whether to generate MSA features (slower but more informative).
        Default ``False`` for speed.
    recycling_steps
        Number of recycling iterations in the trunk.
    """

    model_type = "protein"
    available_pooling_strategies = ["mean", "max", "cls", "none"]

    def __init__(
        self,
        model_path_or_name: str = "boltz2",
        output_type: Literal["single", "pairwise", "both"] = "single",
        use_msa: bool = False,
        recycling_steps: int = 3,
        **kwargs: Any,
    ):
        super().__init__(model_path_or_name, **kwargs)
        self.output_type = output_type
        self.use_msa = use_msa
        self.recycling_steps = recycling_steps
        self._boltz_model = None
        self._cache_dir: Path | None = None
        self._ccd_path: Path | None = None

    def load(self, device: torch.device) -> None:
        """Download and load the Boltz-2 model."""
        if self._boltz_model is not None:
            return

        try:
            from boltz.main import download_boltz2, get_cache_path
            from boltz.model.models.boltz2 import Boltz2
        except ImportError as e:
            raise ImportError(
                "Boltz-2 is not installed. Install with:\n"
                "  pip install boltz[cuda]\n"
                "See: https://github.com/jwohlwend/boltz"
            ) from e

        self._cache_dir = Path(get_cache_path())
        logger.info("Downloading Boltz-2 data to %s (if needed)...", self._cache_dir)
        download_boltz2(self._cache_dir)

        checkpoint = self._cache_dir / "boltz2_conf.ckpt"
        if not checkpoint.exists():
            raise FileNotFoundError(
                f"Boltz-2 checkpoint not found at {checkpoint}. "
                "Run 'boltz predict --help' to trigger download."
            )

        self._ccd_path = self._cache_dir / "ccd.pkl"

        from dataclasses import asdict
        from boltz.main import Boltz2DiffusionParams, PairformerArgsV2, MSAModuleArgs

        diffusion_params = Boltz2DiffusionParams()
        pairformer_args = PairformerArgsV2()
        msa_args = MSAModuleArgs(
            subsample_msa=True,
            num_subsampled_msa=512,
            use_paired_feature=True,
        )

        predict_args = {
            "recycling_steps": self.recycling_steps,
            "sampling_steps": 50,
            "diffusion_samples": 1,
            "max_parallel_samples": 1,
            "write_confidence_summary": False,
            "write_full_pae": False,
            "write_full_pde": False,
        }

        logger.info("Loading Boltz-2 from checkpoint...")
        self._boltz_model = Boltz2.load_from_checkpoint(
            str(checkpoint),
            strict=True,
            predict_args=predict_args,
            map_location="cpu",
            diffusion_process_args=asdict(diffusion_params),
            ema=False,
            pairformer_args=asdict(pairformer_args),
            msa_args=asdict(msa_args),
            run_trunk_and_structure=False,
        )
        self._boltz_model = self._boltz_model.to(device)
        self._boltz_model.eval()
        self.device = device
        self.model = self._boltz_model
        logger.info("Boltz-2 loaded successfully (trunk-only mode).")

    def _prepare_input(self, sequence: str) -> dict[str, torch.Tensor]:
        """Prepare featurized input for Boltz-2 from a protein sequence.

        Uses Boltz's internal data processing pipeline to create the
        required feature tensors from a single protein chain.
        """
        try:
            from boltz.data.parse.yaml import parse_yaml
            from boltz.data.tokenize import tokenize_structure
        except ImportError:
            pass

        yaml_content = (
            f"sequences:\n"
            f"  - protein:\n"
            f"      id: A\n"
            f"      sequence: {sequence}\n"
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            yaml_path = tmpdir / "input.yaml"
            yaml_path.write_text(yaml_content)

            try:
                from boltz.main import process_inputs
                from boltz.data.types import Manifest

                out_dir = tmpdir / "output"
                out_dir.mkdir()

                mol_dir = self._cache_dir / "mols" if self._cache_dir else tmpdir / "mols"
                mol_dir.mkdir(parents=True, exist_ok=True)

                process_inputs(
                    data=[yaml_path],
                    out_dir=out_dir,
                    ccd_path=self._ccd_path,
                    mol_dir=mol_dir,
                    use_msa_server=self.use_msa,
                    msa_server_url="https://api.colabfold.com",
                    msa_pairing_strategy="greedy",
                    boltz2=True,
                )

                manifest = Manifest.load(out_dir / "processed" / "manifest.json")

                from boltz.data.module import Boltz2InferenceDataModule

                data_module = Boltz2InferenceDataModule(
                    manifest=manifest,
                    target_dir=out_dir / "processed" / "structures",
                    msa_dir=out_dir / "processed" / "msa",
                    mol_dir=mol_dir,
                    num_workers=0,
                )
                data_module.setup("predict")
                dl = data_module.predict_dataloader()
                batch = next(iter(dl))

                feats = {}
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        feats[k] = v.to(self.device)
                    else:
                        feats[k] = v
                return feats

            except Exception as e:
                logger.error("Failed to prepare Boltz-2 input: %s", e)
                raise RuntimeError(
                    f"Boltz-2 input preparation failed for sequence "
                    f"(length {len(sequence)}): {e}"
                ) from e

    def _extract_embeddings(
        self,
        feats: dict[str, torch.Tensor],
        pooling_strategy: str,
        output_type: str | None = None,
    ) -> np.ndarray:
        """Run forward pass and extract trunk embeddings."""
        ot = output_type or self.output_type

        with torch.no_grad():
            dict_out = self._boltz_model(
                feats,
                recycling_steps=self.recycling_steps,
            )

        s = dict_out["s"]
        z = dict_out["z"]

        if s.dim() == 3:
            s = s[0]
        if z.dim() == 4:
            z = z[0]

        mask = feats.get("token_pad_mask")
        if mask is not None:
            if mask.dim() == 2:
                mask = mask[0]
            mask_bool = mask.bool()
        else:
            mask_bool = None

        if ot in ("single", "both"):
            if mask_bool is not None:
                s_masked = s[mask_bool]
            else:
                s_masked = s
            s_pooled = self._pool_single(s_masked, pooling_strategy)

        if ot in ("pairwise", "both"):
            if mask_bool is not None:
                z_masked = z[mask_bool][:, mask_bool]
            else:
                z_masked = z
            z_pooled = z_masked.mean(dim=(0, 1)).cpu().numpy()

        if ot == "single":
            return s_pooled
        elif ot == "pairwise":
            return z_pooled
        else:
            return np.concatenate([s_pooled, z_pooled])

    def _pool_single(self, s: torch.Tensor, strategy: str) -> np.ndarray:
        """Pool the single representation over the token dimension."""
        if strategy == "none":
            return s.cpu().numpy()
        elif strategy == "mean":
            return s.mean(dim=0).cpu().numpy()
        elif strategy == "max":
            return s.max(dim=0).values.cpu().numpy()
        elif strategy == "cls":
            return s[0].cpu().numpy()
        else:
            raise ValueError(f"Unknown pooling strategy '{strategy}'")

    def embed(
        self,
        input: str,
        pooling_strategy: str = "mean",
        output_type: str | None = None,
        **kwargs: Any,
    ) -> np.ndarray:
        """Compute embedding for a single protein sequence.

        Parameters
        ----------
        input
            Protein amino acid sequence.
        pooling_strategy
            How to pool per-residue representations: ``"mean"``,
            ``"max"``, or ``"cls"`` (first token).
        output_type
            Override the default output type: ``"single"`` (per-token
            representation, ~384 dims), ``"pairwise"`` (~128 dims),
            or ``"both"`` (concatenated, ~512 dims).
        **kwargs
            Ignored (for API compatibility).

        Returns
        -------
        np.ndarray
            Embedding vector.
        """
        if self._boltz_model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        feats = self._prepare_input(input)
        emb = self._extract_embeddings(feats, pooling_strategy, output_type)

        if self.device and self.device.type == "cuda":
            torch.cuda.empty_cache()

        return emb.astype(np.float32)

    def embed_batch(
        self,
        inputs: Sequence[str],
        pooling_strategy: str = "mean",
        output_type: str | None = None,
        **kwargs: Any,
    ) -> list[np.ndarray]:
        """Compute embeddings for a batch of protein sequences.

        Boltz-2 processes one complex at a time, so this iterates
        over sequences sequentially.

        Parameters
        ----------
        inputs
            List of protein amino acid sequences.
        pooling_strategy
            Pooling strategy for per-residue representations.
        output_type
            Override the default output type.
        **kwargs
            Ignored.

        Returns
        -------
        list of np.ndarray
            One embedding per input sequence.
        """
        results = []
        for i, seq in enumerate(inputs):
            try:
                emb = self.embed(seq, pooling_strategy=pooling_strategy,
                                 output_type=output_type)
                results.append(emb)
            except Exception as e:  # noqa: BLE001
                logger.warning("Boltz-2 embedding failed for sequence %d: %s", i, e)
                results.append(np.zeros(1, dtype=np.float32))

            if self.device and self.device.type == "cuda":
                torch.cuda.empty_cache()

        return results
