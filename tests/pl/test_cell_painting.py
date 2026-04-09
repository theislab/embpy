"""Tests for embpy.pl.cell_painting."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
matplotlib = pytest.importorskip("matplotlib")

import matplotlib
matplotlib.use("Agg")

from embpy.pl.cell_painting import _mono_cmap, plot_cell_painting
from embpy.pp.morphology_preprocessing import CELL_PAINTING_CHANNELS

RNG = np.random.default_rng(99)


def _make_images(h: int = 32, w: int = 32) -> dict[str, np.ndarray]:
    return {ch: RNG.random((h, w)) * 500 for ch in CELL_PAINTING_CHANNELS}


class TestMonoCmap:
    def test_returns_colormap(self):
        from matplotlib.colors import LinearSegmentedColormap
        cmap = _mono_cmap((1.0, 0.0, 0.0))
        assert isinstance(cmap, LinearSegmentedColormap)

    def test_black_at_zero(self):
        cmap = _mono_cmap((0.0, 1.0, 0.0))
        rgba = cmap(0.0)
        assert rgba[0] == pytest.approx(0.0)
        assert rgba[1] == pytest.approx(0.0)
        assert rgba[2] == pytest.approx(0.0)

    def test_color_at_one(self):
        cmap = _mono_cmap((1.0, 0.0, 1.0))
        rgba = cmap(1.0)
        assert rgba[0] == pytest.approx(1.0)
        assert rgba[1] == pytest.approx(0.0)
        assert rgba[2] == pytest.approx(1.0)


class TestPlotCellPainting:
    def test_returns_figure(self):
        fig = plot_cell_painting(_make_images(), show=False)
        from matplotlib.figure import Figure
        assert isinstance(fig, Figure)
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_panel_count_with_composite(self):
        fig = plot_cell_painting(_make_images(), show=False)
        axes = fig.axes
        assert len(axes) == len(CELL_PAINTING_CHANNELS) + 1
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_panel_count_without_composite(self):
        fig = plot_cell_painting(
            _make_images(), show_composite=False, show=False,
        )
        assert len(fig.axes) == len(CELL_PAINTING_CHANNELS)
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_title(self):
        fig = plot_cell_painting(_make_images(), title="Test", show=False)
        assert fig._suptitle is not None
        assert "Test" in fig._suptitle.get_text()
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_array_input(self):
        arr = RNG.random((5, 16, 16)).astype(np.float32) * 1000
        fig = plot_cell_painting(
            arr, list(CELL_PAINTING_CHANNELS), show=False,
        )
        assert len(fig.axes) == 6
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_custom_figsize(self):
        fig = plot_cell_painting(
            _make_images(), figsize=(30, 5), show=False,
        )
        w, h = fig.get_size_inches()
        assert w == pytest.approx(30.0)
        assert h == pytest.approx(5.0)
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_subset_channels(self):
        imgs = {"DNA": RNG.random((8, 8)), "ER": RNG.random((8, 8))}
        fig = plot_cell_painting(imgs, show=False)
        assert len(fig.axes) == 3  # 2 channels + composite
        import matplotlib.pyplot as plt
        plt.close(fig)
