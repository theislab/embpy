"""Tests for embpy.pp.morphology_preprocessing.

Importing embpy eagerly loads embpy.models which requires torch, so we gate
the entire module behind pytest.importorskip("torch").
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from embpy.pp.morphology_preprocessing import (
    CELL_PAINTING_CHANNELS,
    SUBCELL_CANVAS_HEIGHT,
    SUBCELL_CANVAS_WIDTH,
    SUBCELL_CHANNELS,
    SUBCELL_TARGET_NM_PER_PIXEL,
    bbox_from_mask,
    cell_painting_to_subcell,
    crop_spatial,
    crop_to_mask,
    max_projection_z,
    max_projection_z_multichannel,
    prepare_subcell_canvas,
    rescale_to_target_nm_per_pixel,
    resize_to_canvas,
)

RNG = np.random.default_rng(42)


# ---- Z-projection -----------------------------------------------------------

class TestMaxProjectionZ:
    def test_max(self):
        vol = RNG.standard_normal((5, 12, 16))
        proj = max_projection_z(vol, axis=0, projection="max")
        assert proj.shape == (12, 16)
        np.testing.assert_array_equal(proj, np.max(vol, axis=0))

    def test_mean(self):
        vol = RNG.standard_normal((4, 10, 10))
        proj = max_projection_z(vol, axis=0, projection="mean")
        np.testing.assert_allclose(proj, np.mean(vol, axis=0))

    def test_sum(self):
        vol = RNG.standard_normal((3, 8, 8))
        proj = max_projection_z(vol, axis=0, projection="sum")
        np.testing.assert_allclose(proj, np.sum(vol, axis=0))

    def test_rejects_2d(self):
        with pytest.raises(ValueError, match="3-D"):
            max_projection_z(np.zeros((8, 8)))

    def test_rejects_bad_projection(self):
        with pytest.raises(ValueError, match="Unknown projection"):
            max_projection_z(np.zeros((2, 3, 4)), projection="median")


class TestMaxProjectionZMultichannel:
    def test_default_axes(self):
        vol = RNG.standard_normal((3, 5, 12, 16))
        proj = max_projection_z_multichannel(vol, z_axis=1)
        assert proj.shape == (3, 12, 16)
        np.testing.assert_array_equal(proj, np.max(vol, axis=1))

    def test_rejects_3d(self):
        with pytest.raises(ValueError, match="4-D"):
            max_projection_z_multichannel(np.zeros((3, 8, 8)))


# ---- Cell Painting -> SubCell channel remapping -----------------------------

class TestCellPaintingToSubcell:
    def test_channel_first(self):
        cp = RNG.random((5, 32, 64)).astype(np.float32)
        sc = cell_painting_to_subcell(cp, channel_axis="first")
        assert sc.shape == (4, 32, 64)
        # Mito=idx4, ER=idx1, DNA=idx0, AGP=idx3
        np.testing.assert_array_equal(sc[0], cp[4])
        np.testing.assert_array_equal(sc[1], cp[1])
        np.testing.assert_array_equal(sc[2], cp[0])
        np.testing.assert_array_equal(sc[3], cp[3])

    def test_channel_last(self):
        cp = RNG.random((32, 64, 5)).astype(np.float32)
        sc = cell_painting_to_subcell(cp, channel_axis="last")
        assert sc.shape == (32, 64, 4)
        np.testing.assert_array_equal(sc[..., 0], cp[..., 4])
        np.testing.assert_array_equal(sc[..., 2], cp[..., 0])

    def test_rejects_wrong_channel_count(self):
        with pytest.raises(ValueError, match="5"):
            cell_painting_to_subcell(np.zeros((4, 10, 10)))

    def test_constants(self):
        assert len(CELL_PAINTING_CHANNELS) == 5
        assert CELL_PAINTING_CHANNELS == ("DNA", "ER", "RNA", "AGP", "Mito")
        assert len(SUBCELL_CHANNELS) == 4
        assert SUBCELL_CHANNELS == ("Mito", "ER", "DNA", "AGP")


# ---- Pixel-size rescaling ---------------------------------------------------

class TestRescaleToTargetNmPerPixel:
    def test_2x_nm_doubles_spatial(self):
        img = RNG.random((2, 10, 20))
        out = rescale_to_target_nm_per_pixel(
            img,
            nm_per_pixel=SUBCELL_TARGET_NM_PER_PIXEL * 2,
            channel_axis="first",
            order=0,
        )
        assert out.shape[0] == 2
        assert out.shape[1] == pytest.approx(20, abs=1)
        assert out.shape[2] == pytest.approx(40, abs=1)

    def test_anisotropic_nm(self):
        img = RNG.random((1, 20, 20))
        out = rescale_to_target_nm_per_pixel(
            img,
            nm_per_pixel=(SUBCELL_TARGET_NM_PER_PIXEL * 2, SUBCELL_TARGET_NM_PER_PIXEL),
            channel_axis="first",
            order=0,
        )
        assert out.shape[0] == 1
        assert out.shape[1] == pytest.approx(40, abs=1)
        assert out.shape[2] == pytest.approx(20, abs=1)

    def test_preserves_dtype(self):
        img = RNG.random((1, 10, 10)).astype(np.float32)
        out = rescale_to_target_nm_per_pixel(img, nm_per_pixel=100.0)
        assert out.dtype == np.float32

    def test_channel_last(self):
        img = RNG.random((10, 20, 2))
        out = rescale_to_target_nm_per_pixel(
            img,
            nm_per_pixel=SUBCELL_TARGET_NM_PER_PIXEL * 2,
            channel_axis="last",
            order=0,
        )
        assert out.shape[2] == 2
        assert out.shape[0] == pytest.approx(20, abs=1)


# ---- Canvas resizing --------------------------------------------------------

class TestResizeToCanvas:
    def test_output_shape(self):
        img = RNG.random((3, 100, 50))
        out = resize_to_canvas(img, SUBCELL_CANVAS_HEIGHT, SUBCELL_CANVAS_WIDTH)
        assert out.shape == (3, SUBCELL_CANVAS_HEIGHT, SUBCELL_CANVAS_WIDTH)

    def test_noop_when_already_correct(self):
        img = RNG.random((2, 640, 640))
        out = resize_to_canvas(img)
        np.testing.assert_array_equal(out, img)

    def test_channel_last(self):
        img = RNG.random((80, 120, 4))
        out = resize_to_canvas(img, 640, 640, channel_axis="last")
        assert out.shape == (640, 640, 4)


# ---- Bounding box / crop helpers --------------------------------------------

class TestBboxFromMask:
    def test_basic(self):
        mask = np.zeros((20, 30), dtype=bool)
        mask[5:15, 8:22] = True
        assert bbox_from_mask(mask) == (5, 15, 8, 22)

    def test_single_pixel(self):
        mask = np.zeros((10, 10), dtype=bool)
        mask[3, 7] = True
        assert bbox_from_mask(mask) == (3, 4, 7, 8)

    def test_empty_mask_raises(self):
        with pytest.raises(ValueError, match="[Ee]mpty mask"):
            bbox_from_mask(np.zeros((5, 5), dtype=bool))

    def test_3d_raises(self):
        with pytest.raises(ValueError, match="2-D"):
            bbox_from_mask(np.zeros((3, 5, 5), dtype=bool))


class TestCropSpatial:
    def test_tuple_range(self):
        img = np.arange(3 * 20 * 30, dtype=float).reshape(3, 20, 30)
        out = crop_spatial(img, (2, 10), (5, 15))
        assert out.shape == (3, 8, 10)
        np.testing.assert_array_equal(out, img[:, 2:10, 5:15])

    def test_slice_range(self):
        img = np.arange(3 * 20 * 30, dtype=float).reshape(3, 20, 30)
        out = crop_spatial(img, slice(2, 10), slice(5, 15))
        assert out.shape == (3, 8, 10)

    def test_channel_last(self):
        img = np.zeros((20, 30, 2))
        img[3:8, 4:9, :] = 1.0
        out = crop_spatial(img, (3, 8), (4, 9), channel_axis="last")
        assert out.shape == (5, 5, 2)
        assert out.sum() == pytest.approx(50.0)


class TestCropToMask:
    def test_tight_crop(self):
        mask = np.zeros((20, 30), dtype=bool)
        mask[5:15, 8:22] = True
        img = RNG.random((3, 20, 30))
        out = crop_to_mask(img, mask)
        assert out.shape == (3, 10, 14)

    def test_with_padding(self):
        mask = np.zeros((20, 30), dtype=bool)
        mask[5:15, 8:22] = True
        img = RNG.random((3, 20, 30))
        out = crop_to_mask(img, mask, pad=3)
        assert out.shape == (3, 16, 20)

    def test_padding_clamped_to_boundary(self):
        mask = np.zeros((10, 10), dtype=bool)
        mask[0:3, 0:4] = True
        img = RNG.random((1, 10, 10))
        out = crop_to_mask(img, mask, pad=5)
        assert out.shape[1] <= 10
        assert out.shape[2] <= 10


# ---- Full pipeline -----------------------------------------------------------

class TestPrepareSubcellCanvas:
    def test_output_shape(self):
        crop = RNG.random((4, 32, 48))
        out = prepare_subcell_canvas(crop, nm_per_pixel=100.0)
        assert out.shape == (4, SUBCELL_CANVAS_HEIGHT, SUBCELL_CANVAS_WIDTH)

    def test_channel_last(self):
        crop = RNG.random((32, 48, 2))
        out = prepare_subcell_canvas(crop, nm_per_pixel=100.0, channel_axis="last")
        assert out.shape == (SUBCELL_CANVAS_HEIGHT, SUBCELL_CANVAS_WIDTH, 2)


# ---- PNG round-trip ----------------------------------------------------------

class TestPngIO:
    def test_roundtrip(self, tmp_path):
        PIL = pytest.importorskip("PIL")
        from embpy.pp.morphology_preprocessing import (
            load_channels_from_pngs,
            save_channels_as_pngs,
        )

        cell = RNG.random((3, 64, 64))
        paths = save_channels_as_pngs(cell, tmp_path, "cell01")
        assert len(paths) == 3
        for p in paths:
            assert p.exists()
            assert p.suffix == ".png"

        loaded = load_channels_from_pngs(paths)
        assert loaded.shape == (3, 64, 64)
        assert loaded.dtype == np.float32
        assert loaded.min() >= 0.0
        assert loaded.max() <= 1.0

    def test_custom_channel_names(self, tmp_path):
        pytest.importorskip("PIL")
        from embpy.pp.morphology_preprocessing import save_channels_as_pngs

        cell = RNG.random((2, 32, 32))
        paths = save_channels_as_pngs(
            cell, tmp_path, "test", channel_names=["DAPI", "GFP"],
        )
        assert paths[0].name == "test_DAPI.png"
        assert paths[1].name == "test_GFP.png"

    def test_channel_last_roundtrip(self, tmp_path):
        pytest.importorskip("PIL")
        from embpy.pp.morphology_preprocessing import (
            load_channels_from_pngs,
            save_channels_as_pngs,
        )

        cell = RNG.random((64, 64, 2))
        paths = save_channels_as_pngs(cell, tmp_path, "hwc", channel_axis="last")
        assert len(paths) == 2
        loaded = load_channels_from_pngs(paths, channel_axis="last")
        assert loaded.shape == (64, 64, 2)
