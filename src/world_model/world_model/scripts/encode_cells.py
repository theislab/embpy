"""CLI smoke target: encode an AnnData through STATE / STACK and persist NPZ.

Usage
-----
::

    python -m world_model.scripts.encode_cells \\
        --kind state \\
        --adata /path/to/cells.h5ad \\
        --state-checkpoint /path/to/SE-600M/se600m_epoch15.ckpt \\
        --state-model-folder /path/to/SE-600M \\
        --output runs/_cache/state_backbone/state/<hash>/<ds>.npz

    python -m world_model.scripts.encode_cells \\
        --kind stack \\
        --adata /path/to/cells.h5ad \\
        --stack-checkpoint /path/to/bc_large.ckpt \\
        --stack-genelist /path/to/basecount_1000per_15000max.pkl \\
        --output runs/_cache/state_backbone/stack/<hash>/<ds>.npz

This is the canonical pre-flight check: if it fails, training will
fail too -- the same code path is reused inside
:func:`world_model.data.build_dataloaders`.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np

from world_model.configs import StateBackboneConfig
from world_model.models.encoders.backbones import build_backbone
from world_model.utils import setup_logging

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Encode an AnnData with a state backbone.")
    p.add_argument("--kind", choices=["state", "stack"], required=True)
    p.add_argument("--adata", required=True, help="Path to .h5ad")
    p.add_argument("--output", required=True, help="Path to write the NPZ to")
    p.add_argument("--device", default="auto", help="auto | cuda | cpu")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--cache-dir", default="", help="Optional persistent cache root.")
    p.add_argument("--state-checkpoint", default="")
    p.add_argument("--state-model-folder", default=None)
    p.add_argument("--state-protein-embeddings", default=None)
    p.add_argument("--state-config", default=None)
    p.add_argument("--stack-checkpoint", default="")
    p.add_argument("--stack-genelist", default="")
    p.add_argument("--stack-gene-name-col", default=None)
    return p.parse_args(argv)


def _peak_gpu_mb() -> float:
    try:
        import torch  # noqa: PLC0415

        if not torch.cuda.is_available():
            return 0.0
        return float(torch.cuda.max_memory_allocated()) / (1024 * 1024)
    except Exception:
        return 0.0


def main(argv: list[str] | None = None) -> int:
    setup_logging()
    args = parse_args(argv)

    # STACK / STATE wrappers persist the in-memory AnnData to a tempfile
    # before handing it off to their path-based CLI APIs. On a SLURM
    # compute node /tmp is typically a small local disk (a few GB) and
    # cannot fit the 10 GB Replogle dump. Redirect TMPDIR to a Lustre
    # path co-located with the rest of our outputs unless the caller
    # already set it. Use setdefault so explicit user choices win.
    tmp_root = Path("runs/_tmp").resolve()
    tmp_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TMPDIR", str(tmp_root))
    logger.info("TMPDIR=%s (used for wrapper tempfiles)", os.environ["TMPDIR"])

    cfg = StateBackboneConfig(
        kind=args.kind,
        state_checkpoint=args.state_checkpoint,
        state_model_folder=args.state_model_folder,
        state_protein_embeddings=args.state_protein_embeddings,
        state_config=args.state_config,
        stack_checkpoint=args.stack_checkpoint,
        stack_genelist=args.stack_genelist,
        stack_gene_name_col=args.stack_gene_name_col,
        device=args.device,
        freeze=True,
        batch_size=int(args.batch_size),
        cache_dir=args.cache_dir or "runs/_cache/state_backbone",
    )
    provider = build_backbone(cfg)

    import anndata as ad  # noqa: PLC0415

    logger.info("Loading AnnData from %s ...", args.adata)
    adata = ad.read_h5ad(args.adata)
    logger.info("AnnData: %d cells x %d genes", adata.n_obs, adata.n_vars)

    try:
        import torch  # noqa: PLC0415

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass

    t0 = time.time()
    embeddings = provider.encode(adata)
    wall = time.time() - t0

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, embeddings=np.asarray(embeddings, dtype=np.float32))
    logger.info(
        "Wrote %s -- shape=%s wall=%.2fs gpu_peak=%.1f MB",
        out, tuple(embeddings.shape), wall, _peak_gpu_mb(),
    )
    print(
        f"embedding_dim={embeddings.shape[1]} "
        f"n_cells={embeddings.shape[0]} "
        f"wall_s={wall:.2f} "
        f"gpu_peak_mb={_peak_gpu_mb():.1f} "
        f"output={out}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
