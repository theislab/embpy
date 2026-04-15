"""Cell Painting image visualisation.

Provides :func:`plot_cell_painting` which renders the standard 5-channel
Cell Painting channels as individually pseudo-coloured panels alongside an
additive RGB composite -- the visualisation that every Cell Painting user
re-implements from scratch.
"""

from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure

from ...pp.morphology_preprocessing import (
    CELL_PAINTING_COLORS,
    composite_cell_painting,
    normalize_channels,
)


def _mono_cmap(color: tuple[float, float, float]) -> LinearSegmentedColormap:
    """Black-to-*color* colormap for a single fluorescence channel."""
    return LinearSegmentedColormap.from_list("", [(0, 0, 0), color])


def plot_cell_painting(
    images: dict[str, np.ndarray] | np.ndarray,
    channels: Sequence[str] | None = None,
    *,
    colors: dict[str, tuple[float, float, float]] | None = None,
    clip_percentile: float = 0.0,
    title: str = "",
    show_composite: bool = True,
    figsize: tuple[float, float] | None = None,
    show: bool = True,
) -> Figure:
    """Plot Cell Painting channels with canonical pseudo-colours.

    Each channel is shown in its own panel with a black-to-colour
    colormap, and an optional additive RGB composite is appended.

    Parameters
    ----------
    images
        Either a ``{channel_name: 2-D array}`` dict or a ``(C, H, W)``
        array.  When an array, *channels* lists the names in order.
    channels
        Channel names for the array form.  Ignored for dicts.
    colors
        Per-channel ``(R, G, B)`` colour tuples.  Defaults to
        :data:`~embpy.pp.CELL_PAINTING_COLORS`.
    clip_percentile
        Forwarded to :func:`~embpy.pp.normalize_channels`.
    title
        Figure super-title.
    show_composite
        Whether to append a merged composite panel (default ``True``).
    figsize
        Explicit ``(width, height)``.  Computed automatically if ``None``.
    show
        Call ``plt.show()`` at the end (default ``True``).  Set to
        ``False`` when embedding in a larger figure or saving to disk.

    Returns
    -------
    :class:`matplotlib.figure.Figure`
    """
    if colors is None:
        colors = CELL_PAINTING_COLORS

    normed = normalize_channels(images, channels, clip_percentile=clip_percentile)

    ch_names = list(normed.keys())
    n_panels = len(ch_names) + (1 if show_composite else 0)

    if figsize is None:
        figsize = (4.0 * n_panels, 4.0)

    fig, axes = plt.subplots(1, n_panels, figsize=figsize)
    if n_panels == 1:
        axes = [axes]

    for ax, ch in zip(axes, ch_names):
        col = colors.get(ch, (1.0, 1.0, 1.0))
        ax.imshow(normed[ch], cmap=_mono_cmap(col))
        ax.set_title(ch, fontsize=11)
        ax.axis("off")

    if show_composite:
        comp = composite_cell_painting(normed, colors=colors, clip_percentile=0.0)
        axes[-1].imshow(comp)
        axes[-1].set_title("Composite", fontsize=11)
        axes[-1].axis("off")

    if title:
        fig.suptitle(title, fontsize=14, y=1.02)

    fig.tight_layout()
    if show:
        plt.show()

    return fig
