"""Train/test splits for perturbation transcriptomics.

Two split policies:

* ``"perturbation"`` -- partition the *set of perturbation labels* into
  train/test. Every cell with a held-out perturbation goes to the test
  side. This is the scientifically meaningful out-of-distribution
  setting (we measure generalisation to *unseen perturbations*).
* ``"cell"`` -- random per-cell holdout. Sanity check only -- a model
  can trivially memorise per-cell context here, so do not report
  headline numbers in this mode.

Splits are deterministic given a seed. They are persisted as a single
``.npz`` file so the world model and every baseline see byte-identical
train/test indices.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SplitArtifact:
    """Materialised train/test split.

    Attributes
    ----------
    split_by
        Either ``"perturbation"`` or ``"cell"``.
    train_indices, test_indices
        Cell indices into the parent expression matrix.
    train_perturbations, test_perturbations
        Sorted unique perturbation labels in each side. ``None`` for
        cell-level splits.
    seed
        Seed used to build the split (recorded for reproducibility).
    """

    split_by: str
    train_indices: np.ndarray
    test_indices: np.ndarray
    train_perturbations: list[str] | None
    test_perturbations: list[str] | None
    seed: int

    def summary(self) -> str:
        n_pert = (
            f", train_perts={len(self.train_perturbations)}, test_perts={len(self.test_perturbations)}"
            if self.train_perturbations is not None
            else ""
        )
        return (
            f"SplitArtifact(split_by={self.split_by}, "
            f"train_cells={self.train_indices.size}, test_cells={self.test_indices.size}"
            f"{n_pert}, seed={self.seed})"
        )


def make_split(
    perturbation_labels: np.ndarray,
    *,
    control_label: str,
    split_by: str = "perturbation",
    train_fraction: float = 0.8,
    seed: int = 0,
    keep_control_in_test: bool = True,
) -> SplitArtifact:
    """Build a :class:`SplitArtifact` for a flat collection of cells.

    Parameters
    ----------
    perturbation_labels
        ``(n_cells,)`` string array of perturbation labels.
    control_label
        Value flagging control cells. Controls are always assigned to
        the train side (the model needs them to define the basal
        state). When ``keep_control_in_test=True`` they are also
        accessible to the test side.
    split_by
        ``"perturbation"`` or ``"cell"``.
    train_fraction
        Fraction of perturbations (or cells) to keep in train.
    seed
        Numpy RNG seed.
    keep_control_in_test
        If True, controls are copied into the test indices so baselines
        and predictions can sample from them.
    """
    if not 0.0 < train_fraction < 1.0:
        raise ValueError(f"train_fraction must be in (0, 1), got {train_fraction}")
    if split_by not in {"perturbation", "cell"}:
        raise ValueError(f"split_by must be 'perturbation' or 'cell', got {split_by!r}")

    rng = np.random.default_rng(seed)
    labels = np.asarray(perturbation_labels)
    is_control = labels == control_label
    control_idx = np.flatnonzero(is_control)
    perturbed_idx = np.flatnonzero(~is_control)

    if split_by == "perturbation":
        unique_perts = sorted(set(labels[perturbed_idx].tolist()))
        n_train = max(1, int(round(train_fraction * len(unique_perts))))
        order = rng.permutation(len(unique_perts))
        train_perts = sorted(unique_perts[i] for i in order[:n_train])
        test_perts = sorted(unique_perts[i] for i in order[n_train:])

        train_set = set(train_perts)
        test_set = set(test_perts)
        train_pert_mask = np.array([lbl in train_set for lbl in labels])
        test_pert_mask = np.array([lbl in test_set for lbl in labels])

        train_indices = np.concatenate([control_idx, np.flatnonzero(train_pert_mask)])
        test_indices = np.flatnonzero(test_pert_mask)
        if keep_control_in_test:
            test_indices = np.concatenate([control_idx, test_indices])

        train_indices = np.unique(train_indices)
        test_indices = np.unique(test_indices)
        artifact = SplitArtifact(
            split_by="perturbation",
            train_indices=train_indices,
            test_indices=test_indices,
            train_perturbations=train_perts,
            test_perturbations=test_perts,
            seed=seed,
        )
    else:
        n = labels.size
        n_train = max(1, int(round(train_fraction * n)))
        order = rng.permutation(n)
        train_indices = np.sort(order[:n_train])
        test_indices = np.sort(order[n_train:])
        artifact = SplitArtifact(
            split_by="cell",
            train_indices=train_indices,
            test_indices=test_indices,
            train_perturbations=None,
            test_perturbations=None,
            seed=seed,
        )

    logger.info("Built split: %s", artifact.summary())
    return artifact


def save_split(spec: SplitArtifact, path: str | Path) -> None:
    """Persist a :class:`SplitArtifact` as ``.npz``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "split_by": np.asarray(spec.split_by),
        "train_indices": spec.train_indices.astype(np.int64),
        "test_indices": spec.test_indices.astype(np.int64),
        "seed": np.asarray(spec.seed, dtype=np.int64),
    }
    if spec.train_perturbations is not None:
        payload["train_perturbations"] = np.asarray(spec.train_perturbations, dtype=object)
        payload["test_perturbations"] = np.asarray(spec.test_perturbations, dtype=object)
    np.savez(path, **payload)
    logger.info("Saved split to %s", path)


def load_split(path: str | Path) -> SplitArtifact:
    """Load a :class:`SplitArtifact` from ``.npz``."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No split file at {path}")
    archive = np.load(path, allow_pickle=True)
    train_perts = archive["train_perturbations"].tolist() if "train_perturbations" in archive else None
    test_perts = archive["test_perturbations"].tolist() if "test_perturbations" in archive else None
    return SplitArtifact(
        split_by=str(archive["split_by"]),
        train_indices=np.asarray(archive["train_indices"], dtype=np.int64),
        test_indices=np.asarray(archive["test_indices"], dtype=np.int64),
        train_perturbations=train_perts,
        test_perturbations=test_perts,
        seed=int(archive["seed"]),
    )


def split_or_load(
    perturbation_labels: np.ndarray,
    *,
    cache_path: str | Path | None,
    control_label: str,
    split_by: str = "perturbation",
    train_fraction: float = 0.8,
    seed: int = 0,
    keep_control_in_test: bool = True,
    force: bool = False,
) -> SplitArtifact:
    """Load a cached split if ``cache_path`` exists, otherwise build and save one."""
    if cache_path is not None and not force:
        path = Path(cache_path)
        if path.exists():
            spec = load_split(path)
            if spec.split_by != split_by or spec.seed != seed:
                logger.warning(
                    "Cached split at %s does not match requested split_by/seed (%s/%d vs %s/%d). Rebuilding.",
                    path, spec.split_by, spec.seed, split_by, seed,
                )
            else:
                return spec
    spec = make_split(
        perturbation_labels,
        control_label=control_label,
        split_by=split_by,
        train_fraction=train_fraction,
        seed=seed,
        keep_control_in_test=keep_control_in_test,
    )
    if cache_path is not None:
        save_split(spec, cache_path)
    return spec


def subsample_train_perturbations(
    spec: SplitArtifact,
    perturbation_labels: np.ndarray,
    *,
    fraction: float,
    seed: int,
    control_label: str = "non-targeting",
    keep_controls: bool = True,
) -> SplitArtifact:
    """Reduce a perturbation-level split's train side to ``fraction`` of its perturbations.

    Used for the transfer setup's "fine-tune on p% of Replogle".
    """
    if spec.split_by != "perturbation":
        raise ValueError("subsample_train_perturbations requires a perturbation-level split")
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")
    rng = np.random.default_rng(seed)
    train_perts = list(spec.train_perturbations or [])
    if not train_perts:
        raise ValueError("Split has no train perturbations to subsample.")
    n_keep = max(1, int(round(fraction * len(train_perts))))
    keep = sorted(rng.choice(train_perts, size=n_keep, replace=False).tolist())

    keep_set = set(keep)
    labels = np.asarray(perturbation_labels)
    keep_mask = np.array([lbl in keep_set for lbl in labels])
    train_indices = np.flatnonzero(keep_mask)
    if keep_controls:
        control_idx = np.flatnonzero(labels == control_label)
        train_indices = np.unique(np.concatenate([train_indices, control_idx]))
    return SplitArtifact(
        split_by="perturbation",
        train_indices=train_indices,
        test_indices=spec.test_indices,
        train_perturbations=keep,
        test_perturbations=spec.test_perturbations,
        seed=seed,
    )


__all__ = [
    "SplitArtifact",
    "load_split",
    "make_split",
    "save_split",
    "split_or_load",
    "subsample_train_perturbations",
]
