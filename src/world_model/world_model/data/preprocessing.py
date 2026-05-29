"""Dataloader helpers for already-materialised world-model arrays."""

from __future__ import annotations

from typing import Any


def sequence_collate_fn(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Stack a list of dataset samples into a batch dict.

    The dataset classes already produce torch tensors, so this is just a
    concatenation along a new leading dim. Pulled out into its own
    function so the dataloader builder can pass it explicitly.
    """
    import torch  # noqa: PLC0415

    batch: dict[str, Any] = {}
    keys = samples[0].keys()
    for k in keys:
        v0 = samples[0][k]
        if isinstance(v0, torch.Tensor):
            batch[k] = torch.stack([s[k] for s in samples], dim=0)
        else:
            batch[k] = [s[k] for s in samples]
    return batch


__all__ = ["sequence_collate_fn"]
