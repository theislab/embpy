"""Preprocessing utilities for single-cell morphology images.

SubCell inference expects 2D single-cell crops saved as one grayscale PNG per
channel on a 640x640 pixel canvas at 80.0885 nm effective pixel size.
``SubCellWrapper`` will further resize to 448x448 internally.

This module provides every step *except* cell segmentation (use Cellpose,
StarDist, CellProfiler, etc.) and then feed masks / bounding boxes to the
crop helpers here.

Typical pipeline::

    volume       = tifffile.imread("field_of_view.tif")      # (C, Z, Y, X)
    img_2d       = max_projection_z_multichannel(volume)      # (C, Y, X)
    cell_crop    = crop_to_mask(img_2d, single_cell_mask)     # tight bbox
    canvas       = prepare_subcell_canvas(cell_crop, nm_per_pixel=108.3)
    png_paths    = save_channels_as_pngs(canvas, "out/", "cell_001")

Reference
---------
`SubCellPortable data requirements
<https://github.com/CellProfiling/SubCellPortable>`_
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal, Sequence

import numpy as np
from scipy import ndimage

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants -- SubCell / CZI recommended acquisition targets
# ---------------------------------------------------------------------------

SUBCELL_TARGET_NM_PER_PIXEL: float = 80.0885
"""Physical pixel size in nanometres for the 640x640 SubCell canvas."""

SUBCELL_CANVAS_HEIGHT: int = 640
"""Canvas height (pixels) of the standardised cell crop."""

SUBCELL_CANVAS_WIDTH: int = 640
"""Canvas width (pixels) of the standardised cell crop."""

# ---------------------------------------------------------------------------
# Standard Cell Painting channel definitions
# ---------------------------------------------------------------------------

CELL_PAINTING_CHANNELS: tuple[str, ...] = ("DNA", "ER", "RNA", "AGP", "Mito")
"""Standard 5-channel Cell Painting order (Bray 2016 / Cimini 2023).

- **DNA** -- Hoechst 33342 (nucleus)
- **ER**  -- Concanavalin A / Alexa Fluor 488 (endoplasmic reticulum)
- **RNA** -- SYTO 14 (nucleoli and cytoplasmic RNA)
- **AGP** -- Phalloidin + WGA / Alexa Fluor 594 (actin, Golgi, plasma membrane)
- **Mito** -- MitoTracker Deep Red (mitochondria)
"""

SUBCELL_CHANNELS: tuple[str, ...] = ("Mito", "ER", "DNA", "AGP")
"""SubCell 4-channel RYBG order.

- index 0 (Red)    = Mito
- index 1 (Yellow) = ER
- index 2 (Blue)   = DNA
- index 3 (Green)  = AGP (used as the protein channel)

The RNA channel from Cell Painting is dropped because SubCell was
trained on HPA images which lack a dedicated RNA stain.
"""

_CP_TO_SUBCELL_INDICES: tuple[int, ...] = (4, 1, 0, 3)
"""Indices into CELL_PAINTING_CHANNELS that produce SUBCELL_CHANNELS order."""

# ---------------------------------------------------------------------------
# Canonical Cell Painting fluorescence colours (RGB, 0-1)
# ---------------------------------------------------------------------------

CELL_PAINTING_COLORS: dict[str, tuple[float, float, float]] = {
    "DNA":  (0.0, 0.4, 1.0),
    "ER":   (0.0, 1.0, 0.0),
    "RNA":  (1.0, 1.0, 0.0),
    "AGP":  (1.0, 0.0, 0.0),
    "Mito": (1.0, 0.0, 1.0),
}
"""Canonical pseudo-colour mapping for the 5 Cell Painting channels.

Matches the fluorophore emission profiles used in the standard
protocol (Bray et al. 2016, Cimini et al. 2023 *Nature Protocols*):

=====  =====================  =========
Chan   Dye                    Colour
=====  =====================  =========
DNA    Hoechst 33342          Blue
ER     Concanavalin A / 488   Green
RNA    SYTO 14                Yellow
AGP    Phalloidin + WGA / 594 Red
Mito   MitoTracker Deep Red   Magenta
=====  =====================  =========
"""

# ---------------------------------------------------------------------------
# Composite image helpers
# ---------------------------------------------------------------------------


def normalize_channels(
    images: dict[str, np.ndarray] | np.ndarray,
    channels: Sequence[str] | None = None,
    *,
    clip_percentile: float = 0.0,
) -> dict[str, np.ndarray]:
    """Min-max normalise each channel independently to [0, 1].

    Parameters
    ----------
    images
        Either a ``{channel_name: 2-D array}`` dict or a single array of
        shape ``(C, H, W)`` (channel-first).  When an array is given,
        *channels* must list the channel names in order.
    channels
        Channel names corresponding to axis-0 of *images* when it is an
        array.  Ignored when *images* is already a dict.
    clip_percentile
        If > 0, clip each channel at this and ``100 - this`` percentile
        before normalising.  Useful for suppressing hot pixels.

    Returns
    -------
    Dict mapping channel name to a ``float64`` array in [0, 1].
    """
    if isinstance(images, np.ndarray):
        if channels is None:
            raise ValueError(
                "channels must be provided when images is an ndarray"
            )
        if images.ndim != 3 or images.shape[0] != len(channels):
            raise ValueError(
                f"Expected array of shape ({len(channels)}, H, W), "
                f"got {images.shape}"
            )
        images = {ch: images[i] for i, ch in enumerate(channels)}

    result: dict[str, np.ndarray] = {}
    for ch, img in images.items():
        arr = img.astype(np.float64)
        if clip_percentile > 0:
            lo = np.percentile(arr, clip_percentile)
            hi = np.percentile(arr, 100.0 - clip_percentile)
            arr = np.clip(arr, lo, hi)
        else:
            lo, hi = arr.min(), arr.max()
        if hi > lo:
            arr = (arr - lo) / (hi - lo)
        else:
            arr = np.zeros_like(arr)
        result[ch] = arr
    return result


def composite_cell_painting(
    images: dict[str, np.ndarray] | np.ndarray,
    channels: Sequence[str] | None = None,
    *,
    colors: dict[str, tuple[float, float, float]] | None = None,
    clip_percentile: float = 0.0,
) -> np.ndarray:
    """Build an additive RGB composite from Cell Painting channels.

    Parameters
    ----------
    images
        Either a ``{channel_name: 2-D array}`` dict or a ``(C, H, W)``
        array.  When an array, *channels* lists the names in order.
    channels
        Channel names for the array form.  Ignored for dicts.
    colors
        Per-channel ``(R, G, B)`` colour tuples.  Defaults to
        :data:`CELL_PAINTING_COLORS`.
    clip_percentile
        Forwarded to :func:`normalize_channels`.

    Returns
    -------
    ``(H, W, 3)`` float64 array in [0, 1] suitable for ``plt.imshow``.
    """
    if colors is None:
        colors = CELL_PAINTING_COLORS

    normed = normalize_channels(images, channels, clip_percentile=clip_percentile)

    composite: np.ndarray | None = None
    for ch, img in normed.items():
        r, g, b = colors.get(ch, (1.0, 1.0, 1.0))
        plane = np.stack([img * r, img * g, img * b], axis=-1)
        if composite is None:
            composite = plane
        else:
            composite = composite + plane

    if composite is None:
        raise ValueError("No channels provided")
    return np.clip(composite, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Cell Painting -> SubCell channel remapping
# ---------------------------------------------------------------------------

def cell_painting_to_subcell(
    image: np.ndarray,
    *,
    channel_axis: Literal["first", "last"] = "first",
) -> np.ndarray:
    """Remap a 5-channel Cell Painting image to SubCell's 4-channel RYBG order.

    The standard Cell Painting assay produces 5 channels in the order
    ``(DNA, ER, RNA, AGP, Mito)``.  SubCell expects 4 channels:
    ``(Mito, ER, DNA, AGP)`` -- known as **RYBG** (Red, Yellow, Blue, Green).
    The RNA channel is dropped.

    Parameters
    ----------
    image
        ``(5, H, W)`` when ``channel_axis="first"`` or ``(H, W, 5)``.
    channel_axis
        ``"first"`` or ``"last"``.

    Returns
    -------
    np.ndarray
        ``(4, H, W)`` or ``(H, W, 4)`` in SubCell RYBG order.

    Examples
    --------
    >>> fov = np.random.rand(5, 1080, 1080).astype(np.float32)
    >>> subcell = cell_painting_to_subcell(fov)
    >>> subcell.shape
    (4, 1080, 1080)
    """
    if channel_axis == "first":
        if image.ndim != 3 or image.shape[0] != 5:
            raise ValueError(
                f"Expected (5, H, W) with channel_axis='first', got {image.shape}"
            )
        return image[list(_CP_TO_SUBCELL_INDICES)]
    if image.ndim != 3 or image.shape[-1] != 5:
        raise ValueError(
            f"Expected (H, W, 5) with channel_axis='last', got {image.shape}"
        )
    return image[..., list(_CP_TO_SUBCELL_INDICES)]


# ---------------------------------------------------------------------------
# Internal channel-axis helpers
# ---------------------------------------------------------------------------

ChannelPosition = Literal["first", "last"]


def _to_chw(image: np.ndarray, channel_axis: ChannelPosition) -> np.ndarray:
    """Normalise a 3-D image to (C, H, W) layout."""
    if image.ndim != 3:
        raise ValueError(
            f"Expected a 3-D array (C,H,W) or (H,W,C), got shape {image.shape}"
        )
    if channel_axis == "last":
        return np.moveaxis(image, -1, 0)
    return image


def _from_chw(image: np.ndarray, channel_axis: ChannelPosition) -> np.ndarray:
    """Convert (C, H, W) back to the caller's preferred layout."""
    if channel_axis == "last":
        return np.moveaxis(image, 0, -1)
    return image


# ---------------------------------------------------------------------------
# Z-projection
# ---------------------------------------------------------------------------

_REDUCERS = {
    "max": np.max,
    "mean": np.mean,
    "sum": np.sum,
}


def max_projection_z(
    volume: np.ndarray,
    axis: int = 0,
    *,
    projection: Literal["max", "mean", "sum"] = "max",
) -> np.ndarray:
    """Collapse a 3-D stack along *axis* to produce a 2-D image.

    Parameters
    ----------
    volume
        ``(Z, Y, X)`` array (when ``axis=0``).
    axis
        Axis to project away.  ``0`` is the most common choice (Z-first).
    projection
        Reduction function: ``"max"`` (default), ``"mean"``, or ``"sum"``.

    Returns
    -------
    np.ndarray
        2-D array of shape ``(Y, X)``.
    """
    if volume.ndim != 3:
        raise ValueError(f"Expected a 3-D volume, got shape {volume.shape}")
    fn = _REDUCERS.get(projection)
    if fn is None:
        raise ValueError(
            f"Unknown projection {projection!r}; choose from {list(_REDUCERS)}"
        )
    return fn(volume, axis=axis)


def max_projection_z_multichannel(
    volume: np.ndarray,
    z_axis: int = 1,
    *,
    projection: Literal["max", "mean", "sum"] = "max",
) -> np.ndarray:
    """Project a 4-D ``(C, Z, Y, X)`` stack along the Z axis.

    Parameters
    ----------
    volume
        4-D array, typically ``(C, Z, Y, X)``.
    z_axis
        Which axis is Z (default ``1`` for C-first layout).
    projection
        ``"max"`` (default), ``"mean"``, or ``"sum"``.

    Returns
    -------
    np.ndarray
        3-D array ``(C, Y, X)`` after collapsing Z.
    """
    if volume.ndim != 4:
        raise ValueError(f"Expected a 4-D volume, got shape {volume.shape}")
    fn = _REDUCERS.get(projection)
    if fn is None:
        raise ValueError(
            f"Unknown projection {projection!r}; choose from {list(_REDUCERS)}"
        )
    return fn(volume, axis=z_axis)


# ---------------------------------------------------------------------------
# Physical pixel-size rescaling
# ---------------------------------------------------------------------------

def rescale_to_target_nm_per_pixel(
    image: np.ndarray,
    nm_per_pixel: float | tuple[float, float],
    *,
    target_nm_per_pixel: float = SUBCELL_TARGET_NM_PER_PIXEL,
    channel_axis: ChannelPosition = "first",
    order: int = 1,
) -> np.ndarray:
    """Rescale spatial dimensions so the effective pixel size matches *target_nm_per_pixel*.

    The zoom factor along each spatial axis is ``current_nm / target_nm``,
    preserving the physical field of view while changing the sampling density.

    Parameters
    ----------
    image
        ``(C, H, W)`` when ``channel_axis="first"`` or ``(H, W, C)``.
    nm_per_pixel
        Current pixel size in nm -- a single float for isotropic pixels or a
        ``(nm_y, nm_x)`` tuple for anisotropic ones.
    target_nm_per_pixel
        Desired pixel size.  Defaults to :data:`SUBCELL_TARGET_NM_PER_PIXEL`.
    channel_axis
        ``"first"`` or ``"last"``.
    order
        Spline interpolation order for :func:`scipy.ndimage.zoom`
        (``0`` = nearest, ``1`` = bilinear).

    Returns
    -------
    np.ndarray
        Rescaled image in the same dtype and channel layout as the input.
    """
    chw = _to_chw(image, channel_axis)
    if isinstance(nm_per_pixel, tuple):
        zoom_y = float(nm_per_pixel[0]) / target_nm_per_pixel
        zoom_x = float(nm_per_pixel[1]) / target_nm_per_pixel
    else:
        zoom_y = zoom_x = float(nm_per_pixel) / target_nm_per_pixel
    zoomed = ndimage.zoom(chw, (1.0, zoom_y, zoom_x), order=order)
    return _from_chw(zoomed.astype(chw.dtype, copy=False), channel_axis)


# ---------------------------------------------------------------------------
# Canvas resizing (zoom + centre crop/pad for off-by-one)
# ---------------------------------------------------------------------------

def resize_to_canvas(
    image: np.ndarray,
    height: int = SUBCELL_CANVAS_HEIGHT,
    width: int = SUBCELL_CANVAS_WIDTH,
    *,
    channel_axis: ChannelPosition = "first",
    order: int = 1,
) -> np.ndarray:
    """Resize spatial dimensions to exactly *height* x *width*.

    Uses :func:`scipy.ndimage.zoom` for the bulk scaling.  If the result is
    off by one pixel (a known rounding artefact), a centre crop or zero-pad
    corrects it.

    Parameters
    ----------
    image
        Multi-channel image, ``(C, H, W)`` or ``(H, W, C)``.
    height, width
        Target canvas size (default 640 x 640).
    channel_axis
        ``"first"`` or ``"last"``.
    order
        Spline interpolation order.

    Returns
    -------
    np.ndarray
        Image with spatial shape ``(height, width)``.
    """
    chw = _to_chw(image, channel_axis)
    _, h, w = chw.shape
    if h == height and w == width:
        return _from_chw(chw, channel_axis)
    zoomed = ndimage.zoom(chw, (1.0, height / h, width / w), order=order)
    if zoomed.shape[1] != height or zoomed.shape[2] != width:
        zoomed = _centre_crop_or_pad(zoomed, height, width)
    return _from_chw(zoomed, channel_axis)


def _centre_crop_or_pad(
    chw: np.ndarray, target_h: int, target_w: int,
) -> np.ndarray:
    """Deterministically centre-crop or zero-pad to exact (target_h, target_w)."""
    _, h, w = chw.shape
    canvas = np.zeros((chw.shape[0], target_h, target_w), dtype=chw.dtype)

    # How many rows/cols of actual data to transfer.
    copy_h = min(h, target_h)
    copy_w = min(w, target_w)

    # Source offsets (crop centre if source is larger).
    sy = (h - copy_h) // 2
    sx = (w - copy_w) // 2

    # Destination offsets (pad centre if target is larger).
    dy = (target_h - copy_h) // 2
    dx = (target_w - copy_w) // 2

    canvas[:, dy : dy + copy_h, dx : dx + copy_w] = chw[
        :, sy : sy + copy_h, sx : sx + copy_w
    ]
    return canvas


# ---------------------------------------------------------------------------
# Bounding-box / mask helpers
# ---------------------------------------------------------------------------

def bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int]:
    """Compute the tight bounding box of a 2-D boolean mask.

    Returns
    -------
    (row_min, row_max_exclusive, col_min, col_max_exclusive)
        Half-open intervals suitable for direct slicing.
    """
    if mask.ndim != 2:
        raise ValueError(f"Expected a 2-D mask, got shape {mask.shape}")
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any():
        raise ValueError("Empty mask -- no foreground pixels")
    rr = np.where(rows)[0]
    cc = np.where(cols)[0]
    return int(rr[0]), int(rr[-1]) + 1, int(cc[0]), int(cc[-1]) + 1


def crop_spatial(
    image: np.ndarray,
    row_range: slice | tuple[int, int],
    col_range: slice | tuple[int, int],
    *,
    channel_axis: ChannelPosition = "first",
) -> np.ndarray:
    """Crop spatial dimensions ``[r0:r1, c0:c1]`` across all channels.

    Parameters
    ----------
    image
        ``(C, H, W)`` or ``(H, W, C)``.
    row_range, col_range
        A ``slice`` or ``(start, stop)`` tuple (half-open).
    channel_axis
        ``"first"`` or ``"last"``.
    """
    chw = _to_chw(image, channel_axis)
    if isinstance(row_range, tuple):
        row_range = slice(row_range[0], row_range[1])
    if isinstance(col_range, tuple):
        col_range = slice(col_range[0], col_range[1])
    return _from_chw(chw[:, row_range, col_range], channel_axis)


def crop_to_mask(
    image: np.ndarray,
    mask: np.ndarray,
    *,
    channel_axis: ChannelPosition = "first",
    pad: int = 0,
) -> np.ndarray:
    """Tight-crop an image to the bounding box of a binary mask.

    Parameters
    ----------
    image
        Multi-channel image whose spatial axes match *mask*.
    mask
        2-D boolean (or castable) array.
    channel_axis
        ``"first"`` or ``"last"``.
    pad
        Extra pixels of context to keep around the bounding box (clamped to
        the image boundary).
    """
    r0, r1, c0, c1 = bbox_from_mask(np.asarray(mask, dtype=bool))
    if pad > 0:
        h, w = mask.shape[:2]
        r0 = max(0, r0 - pad)
        c0 = max(0, c0 - pad)
        r1 = min(h, r1 + pad)
        c1 = min(w, c1 + pad)
    return crop_spatial(image, (r0, r1), (c0, c1), channel_axis=channel_axis)


# ---------------------------------------------------------------------------
# Full pipeline helper
# ---------------------------------------------------------------------------

def prepare_subcell_canvas(
    single_cell_crop: np.ndarray,
    nm_per_pixel: float | tuple[float, float],
    *,
    channel_axis: ChannelPosition = "first",
    height: int = SUBCELL_CANVAS_HEIGHT,
    width: int = SUBCELL_CANVAS_WIDTH,
    target_nm_per_pixel: float = SUBCELL_TARGET_NM_PER_PIXEL,
    order: int = 1,
) -> np.ndarray:
    """Rescale to target nm/px then resize to the SubCell canvas.

    This is a convenience wrapper that chains :func:`rescale_to_target_nm_per_pixel`
    and :func:`resize_to_canvas`.  Call it on an already-cropped single-cell
    image (all channels, spatially aligned).

    Parameters
    ----------
    single_cell_crop
        ``(C, H, W)`` or ``(H, W, C)`` crop of one cell.
    nm_per_pixel
        Current pixel size of *single_cell_crop*.
    channel_axis, height, width, target_nm_per_pixel, order
        Forwarded to the two underlying functions.

    Returns
    -------
    np.ndarray
        Image of shape ``(C, 640, 640)`` or ``(640, 640, C)``.
    """
    rescaled = rescale_to_target_nm_per_pixel(
        single_cell_crop,
        nm_per_pixel,
        target_nm_per_pixel=target_nm_per_pixel,
        channel_axis=channel_axis,
        order=order,
    )
    return resize_to_canvas(
        rescaled, height, width, channel_axis=channel_axis, order=order,
    )


# ---------------------------------------------------------------------------
# PNG I/O  (requires Pillow -- optional dependency)
# ---------------------------------------------------------------------------

def _require_pillow():
    try:
        import PIL  # noqa: F401
    except ImportError:
        raise ImportError(
            "PNG channel I/O requires Pillow.  Install it with:\n"
            "  pip install 'embpy[morphology]'   # or: pip install pillow"
        ) from None


def _normalise_to_uint8(plane: np.ndarray) -> np.ndarray:
    """Min-max normalise a single 2-D plane to ``[0, 255]`` uint8."""
    p = plane.astype(np.float64)
    lo, hi = float(p.min()), float(p.max())
    if hi <= lo:
        return np.zeros(plane.shape, dtype=np.uint8)
    return ((p - lo) / (hi - lo) * 255.0).clip(0, 255).astype(np.uint8)


def save_channels_as_pngs(
    cell_image: np.ndarray,
    out_dir: str | Path,
    base_name: str,
    *,
    channel_axis: ChannelPosition = "first",
    channel_names: Sequence[str] | None = None,
    normalize_per_channel: bool = True,
) -> list[Path]:
    """Save each channel of a cell crop as a separate grayscale PNG.

    SubCell pipelines expect one file per channel per cell.

    Parameters
    ----------
    cell_image
        ``(C, H, W)`` or ``(H, W, C)`` single-cell crop.
    out_dir
        Directory to write into (created if necessary).
    base_name
        Stem shared by all output files (e.g. ``"cell_001"``).
    channel_axis
        ``"first"`` or ``"last"``.
    channel_names
        Optional human-readable suffixes.  When provided, the *i*-th file is
        named ``{base_name}_{channel_names[i]}.png``; otherwise
        ``{base_name}_ch00.png``, etc.
    normalize_per_channel
        If ``True`` (default), each channel is independently min-max scaled to
        ``[0, 255]``.  Set to ``False`` if the data is already in uint8 range.

    Returns
    -------
    list[Path]
        Paths to the written PNG files, one per channel.
    """
    _require_pillow()
    from PIL import Image

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    chw = _to_chw(cell_image, channel_axis)
    n_channels = chw.shape[0]
    paths: list[Path] = []

    for i in range(n_channels):
        plane = chw[i]
        if normalize_per_channel:
            arr = _normalise_to_uint8(plane)
        else:
            arr = np.clip(plane, 0, 255).astype(np.uint8)

        if channel_names is not None and i < len(channel_names):
            fname = f"{base_name}_{channel_names[i]}.png"
        else:
            fname = f"{base_name}_ch{i:02d}.png"

        path = out_dir / fname
        Image.fromarray(arr, mode="L").save(path)
        paths.append(path)

    logger.debug("Wrote %d channel PNGs to %s", n_channels, out_dir)
    return paths


def load_channels_from_pngs(
    paths: Sequence[str | Path],
    *,
    channel_axis: ChannelPosition = "first",
) -> np.ndarray:
    """Load grayscale PNGs and stack them into a multi-channel float32 array.

    Parameters
    ----------
    paths
        One path per channel, in the desired channel order.
    channel_axis
        ``"first"`` -> ``(C, H, W)``; ``"last"`` -> ``(H, W, C)``.

    Returns
    -------
    np.ndarray
        Float32 array with pixel values in ``[0, 1]``.
    """
    _require_pillow()
    from PIL import Image

    planes = [
        np.asarray(Image.open(p).convert("L"), dtype=np.float32) / 255.0
        for p in paths
    ]
    chw = np.stack(planes, axis=0)
    return _from_chw(chw, channel_axis)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "CELL_PAINTING_CHANNELS",
    "CELL_PAINTING_COLORS",
    "ChannelPosition",
    "SUBCELL_CANVAS_HEIGHT",
    "SUBCELL_CANVAS_WIDTH",
    "SUBCELL_CHANNELS",
    "SUBCELL_TARGET_NM_PER_PIXEL",
    "bbox_from_mask",
    "cell_painting_to_subcell",
    "composite_cell_painting",
    "crop_spatial",
    "crop_to_mask",
    "load_channels_from_pngs",
    "max_projection_z",
    "max_projection_z_multichannel",
    "normalize_channels",
    "prepare_subcell_canvas",
    "rescale_to_target_nm_per_pixel",
    "resize_to_canvas",
    "save_channels_as_pngs",
]
