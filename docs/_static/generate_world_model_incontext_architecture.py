"""Generate a Keynote-ready architecture figure for the in-context world model.

The figure is intentionally specific to the current stability experiments
described by ``submit_incontext_stability_experiments.sh`` and the baseline run
inspected on 2026-06-02:

* state_obsm_key = X_stack, state_backbone.kind = stack
* action_obsm_key = X_pert_borzoi_v0
* data.context_mode = incontext_set
* dynamics.kind = incontext_set
* d_model = 256, support size = 16

It also shows the stabilization knobs now present in the implementation:
latent normalization and residual-delta prediction.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


OUT_DIR = Path(__file__).resolve().parent
OUT_BASE = OUT_DIR / "world_model_incontext_architecture"

FIG_W = 16
FIG_H = 9
DPI = 240

COLORS = {
    "bg": "#fbfcfe",
    "ink": "#172033",
    "muted": "#5b667a",
    "line": "#9aa6b8",
    "data": "#e8f1ff",
    "data_edge": "#7ca9e6",
    "frozen": "#e9eef5",
    "frozen_edge": "#7890ad",
    "train": "#fff0e8",
    "train_edge": "#e07a3f",
    "train_dark": "#b95324",
    "loss": "#fff0f2",
    "loss_edge": "#d85772",
    "loss_dark": "#aa2547",
    "target": "#edf9f2",
    "target_edge": "#56a878",
    "panel": "#ffffff",
}


def _box(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    *,
    title: str,
    lines: list[str] | None = None,
    fill: str,
    edge: str,
    title_color: str | None = None,
    lw: float = 1.4,
    fs_title: float = 9.4,
    fs_body: float = 6.7,
    z: int = 2,
) -> None:
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.035,rounding_size=0.08",
        fc=fill,
        ec=edge,
        lw=lw,
        zorder=z,
    )
    ax.add_patch(patch)
    ax.text(
        x + 0.16,
        y + h - 0.22,
        title,
        ha="left",
        va="top",
        fontsize=fs_title,
        fontweight="bold",
        color=title_color or COLORS["ink"],
        zorder=z + 1,
    )
    if lines:
        yy = y + h - 0.50
        for line in lines:
            ax.text(
                x + 0.16,
                yy,
                line,
                ha="left",
                va="top",
                fontsize=fs_body,
                color=COLORS["muted"],
                zorder=z + 1,
            )
            yy -= 0.22


def _label(ax: plt.Axes, x: float, y: float, text: str, *, size: float = 8.0, color: str | None = None) -> None:
    ax.text(x, y, text, ha="center", va="center", fontsize=size, color=color or COLORS["muted"])


def _arrow(
    ax: plt.Axes,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    *,
    color: str | None = None,
    lw: float = 1.4,
    dashed: bool = False,
    rad: float = 0.0,
    z: int = 1,
) -> None:
    arrow = FancyArrowPatch(
        (x1, y1),
        (x2, y2),
        arrowstyle="-|>",
        mutation_scale=11,
        lw=lw,
        color=color or COLORS["line"],
        linestyle=(0, (4, 3)) if dashed else "solid",
        connectionstyle=f"arc3,rad={rad}",
        shrinkA=3,
        shrinkB=3,
        zorder=z,
    )
    ax.add_patch(arrow)


def _section(ax: plt.Axes, x: float, y: float, w: float, h: float, title: str) -> None:
    ax.add_patch(Rectangle((x, y), w, h, fc=COLORS["panel"], ec="#e3e8ef", lw=0.8, zorder=0))
    ax.text(x + 0.12, y + h - 0.16, title, ha="left", va="top", fontsize=8.5, color=COLORS["muted"], fontweight="bold")


def main() -> None:
    fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=DPI)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, FIG_W)
    ax.set_ylim(0, FIG_H)
    ax.axis("off")
    fig.patch.set_facecolor(COLORS["bg"])

    ax.text(
        0.6,
        8.55,
        "In-context perturbation world model",
        ha="left",
        va="center",
        fontsize=20,
        fontweight="bold",
        color=COLORS["ink"],
    )
    ax.text(
        0.6,
        8.17,
        "Current stability setup: STACK state embeddings + Borzoi action embeddings -> set transformer -> perturbed state",
        ha="left",
        va="center",
        fontsize=9.5,
        color=COLORS["muted"],
    )

    _section(ax, 0.45, 0.7, 3.05, 7.45, "Inputs and task construction")
    _section(ax, 3.75, 0.7, 4.25, 7.45, "Encoders")
    _section(ax, 8.25, 0.7, 3.9, 7.45, "InContextSetDynamics")
    _section(ax, 12.4, 0.7, 3.15, 7.45, "Prediction and loss")

    # Inputs.
    _box(
        ax,
        0.75,
        6.55,
        2.45,
        1.12,
        title="AnnData .obsm states",
        lines=["X_stack, cells x 1600", "cached STACK embeddings", "upstream model frozen/offline"],
        fill=COLORS["frozen"],
        edge=COLORS["frozen_edge"],
        title_color="#405875",
    )
    _box(
        ax,
        0.75,
        5.15,
        2.45,
        1.1,
        title="AnnData .obsm actions",
        lines=["X_pert_borzoi_v0", "gene/action vectors, dim=1536", "lookup table is frozen"],
        fill=COLORS["frozen"],
        edge=COLORS["frozen_edge"],
        title_color="#405875",
    )
    _box(
        ax,
        0.7,
        3.35,
        2.55,
        1.35,
        title="In-context sampler",
        lines=["support: M=16 triplets", "query: one held-out perturbation", "bucket: cell_line + batch"],
        fill=COLORS["data"],
        edge=COLORS["data_edge"],
        title_color="#285b9f",
    )
    _box(
        ax,
        0.7,
        1.4,
        2.55,
        1.35,
        title="Batch tensors",
        lines=["support_obs, support_next, support_act", "query_obs, query_act", "query_next target"],
        fill=COLORS["data"],
        edge=COLORS["data_edge"],
        title_color="#285b9f",
    )

    _arrow(ax, 1.95, 6.55, 1.95, 4.7)
    _arrow(ax, 1.95, 5.15, 1.95, 4.7)
    _arrow(ax, 1.95, 3.35, 1.95, 2.75)

    # State encoder path.
    _box(
        ax,
        4.05,
        6.15,
        3.25,
        1.18,
        title="State head: ForeignBackboneHead",
        lines=["mean-pool stack K=4", "Linear 1600 -> 256", "trainable params: 409,856"],
        fill=COLORS["train"],
        edge=COLORS["train_edge"],
        title_color=COLORS["train_dark"],
    )
    _box(
        ax,
        4.05,
        4.88,
        3.25,
        0.82,
        title="Latent normalization",
        lines=["none | LayerNorm | l2", "applied before dynamics/loss"],
        fill=COLORS["train"],
        edge=COLORS["train_edge"],
        title_color=COLORS["train_dark"],
        fs_title=9.2,
        fs_body=7.1,
    )
    _box(
        ax,
        4.05,
        3.75,
        3.25,
        0.72,
        title="State tokens",
        lines=["support_s, support_sp, query_s"],
        fill=COLORS["target"],
        edge=COLORS["target_edge"],
        title_color="#2c7650",
        fs_title=9.2,
    )

    # Action encoder path.
    _box(
        ax,
        4.05,
        2.32,
        3.25,
        1.12,
        title="GeneEmbeddingAction",
        lines=["frozen embedding lookup", "ActionAdapter Linear 1536 -> 256", "masked mean over n_pert=2"],
        fill=COLORS["train"],
        edge=COLORS["train_edge"],
        title_color=COLORS["train_dark"],
    )
    _box(
        ax,
        4.05,
        1.2,
        3.25,
        0.72,
        title="Action tokens",
        lines=["support_a, query_a"],
        fill=COLORS["target"],
        edge=COLORS["target_edge"],
        title_color="#2c7650",
        fs_title=9.2,
    )

    _arrow(ax, 3.25, 6.9, 4.05, 6.82)
    _arrow(ax, 3.25, 2.05, 4.05, 2.88)
    _arrow(ax, 5.68, 6.15, 5.68, 5.7)
    _arrow(ax, 5.68, 4.88, 5.68, 4.47)
    _arrow(ax, 5.68, 2.32, 5.68, 1.92)

    # Dynamics.
    _box(
        ax,
        8.55,
        6.2,
        3.0,
        1.05,
        title="Triplet fusion",
        lines=["support: concat(s, a, s')", "query: concat(query_s, query_a, MASK)", "Linear + GELU + LayerNorm"],
        fill=COLORS["train"],
        edge=COLORS["train_edge"],
        title_color=COLORS["train_dark"],
    )
    _box(
        ax,
        8.55,
        4.38,
        3.0,
        1.35,
        title="Bidirectional set transformer",
        lines=["6 TransformerBlock layers, 8 heads", "bidirectional; no positional embeddings", "role embedding marks query token"],
        fill=COLORS["train"],
        edge=COLORS["train_edge"],
        title_color=COLORS["train_dark"],
    )
    _box(
        ax,
        8.55,
        2.85,
        3.0,
        0.92,
        title="Query token head",
        lines=["LayerNorm -> Linear(256, 256)", "trainable params: 5,003,008"],
        fill=COLORS["train"],
        edge=COLORS["train_edge"],
        title_color=COLORS["train_dark"],
    )
    _box(
        ax,
        8.55,
        1.25,
        3.0,
        0.98,
        title="Support selection",
        lines=["random or action_similarity", "same-bucket constraints retained"],
        fill=COLORS["data"],
        edge=COLORS["data_edge"],
        title_color="#285b9f",
    )

    _arrow(ax, 7.3, 4.08, 8.55, 6.68, rad=0.08)
    _arrow(ax, 7.3, 1.56, 8.55, 6.52, rad=-0.04)
    _arrow(ax, 10.05, 6.2, 10.05, 5.73)
    _arrow(ax, 10.05, 4.38, 10.05, 3.77)
    _arrow(ax, 2.0, 3.35, 8.55, 1.74, color="#b8c1cf", dashed=True, rad=-0.08)

    # Prediction / losses.
    _box(
        ax,
        12.7,
        6.1,
        2.75,
        1.22,
        title="Prediction mode",
        lines=["absolute: raw = s_hat", "residual_delta: raw = delta_hat", "then s_hat = query_s + delta_hat"],
        fill=COLORS["train"],
        edge=COLORS["train_edge"],
        title_color=COLORS["train_dark"],
    )
    _box(
        ax,
        12.7,
        4.65,
        2.75,
        0.98,
        title="ExpressionDecoder",
        lines=["MLP 256 -> 512 -> 1024 -> 1600", "outputs x_hat"],
        fill=COLORS["train"],
        edge=COLORS["train_edge"],
        title_color=COLORS["train_dark"],
    )
    _box(
        ax,
        12.7,
        2.42,
        2.75,
        1.72,
        title="Training losses",
        lines=[
            "latent objective: MSE",
            "residual target: s_target - query_s",
            "decoder_mse on query_next_expression",
            "InfoNCE on deltas",
            "action-counterfactual CE",
        ],
        fill=COLORS["loss"],
        edge=COLORS["loss_edge"],
        title_color=COLORS["loss_dark"],
    )
    _box(
        ax,
        12.7,
        1.1,
        2.75,
        0.82,
        title="Post-fit evaluation",
        lines=["CPU cell_eval profile=full", "baselines: identity, mean, additive, linear"],
        fill=COLORS["data"],
        edge=COLORS["data_edge"],
        title_color="#285b9f",
        fs_title=9.2,
    )

    _arrow(ax, 11.55, 3.31, 12.7, 6.72, rad=0.05)
    _arrow(ax, 14.08, 6.1, 14.08, 5.63)
    _arrow(ax, 14.08, 4.65, 14.08, 4.14)
    _arrow(ax, 7.3, 4.08, 12.7, 3.45, color="#72aa84", dashed=True, rad=-0.04)
    _label(ax, 10.1, 3.86, "s_target is detached", size=6.8, color="#47805f")

    # Legend.
    legend_x = 0.63
    legend_y = 0.25
    legend_items = [
        ("frozen/cached", COLORS["frozen"], COLORS["frozen_edge"]),
        ("trainable", COLORS["train"], COLORS["train_edge"]),
        ("tensor/data", COLORS["data"], COLORS["data_edge"]),
        ("loss/target", COLORS["loss"], COLORS["loss_edge"]),
        ("dashed = target/control flow", "#ffffff", "#9aa6b8"),
    ]
    xx = legend_x
    for text, fill, edge in legend_items:
        ax.add_patch(Rectangle((xx, legend_y), 0.22, 0.16, fc=fill, ec=edge, lw=1.2))
        ax.text(xx + 0.28, legend_y + 0.08, text, va="center", ha="left", fontsize=7.2, color=COLORS["muted"])
        xx += 2.55

    ax.text(
        15.48,
        0.25,
        "Source: incontext_world_model.py, dynamics/incontext_set.py, train.log",
        ha="right",
        va="center",
        fontsize=6.8,
        color="#8793a4",
    )

    for ext in ("svg", "pdf", "png"):
        path = OUT_BASE.with_suffix(f".{ext}")
        if ext == "png":
            fig.savefig(path, dpi=DPI, facecolor=COLORS["bg"])
        else:
            fig.savefig(path, facecolor=COLORS["bg"])
        print(path)


if __name__ == "__main__":
    main()
