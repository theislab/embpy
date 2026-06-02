"""Generate Keynote-ready step figures for the in-context world model.

These figures decompose the current explicit-token in-context objective into
four separate 16:9 panels:

1. Query/task construction.
2. Encoders.
3. Autoregressive triplet-token dynamics.
4. Prediction, decoding, and losses.

The content mirrors the current implementation in:

* ``world_model.models.incontext_world_model.InContextWorldModel``
* ``world_model.models.dynamics.incontext_tokens.InContextTokensDynamics``
* ``world_model.models.world_model.build_world_model``
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Circle, FancyArrowPatch, FancyBboxPatch, Rectangle


OUT_DIR = Path(__file__).resolve().parent
FIG_W = 16
FIG_H = 9
DPI = 240

COLORS = {
    "bg": "#fbfcfe",
    "ink": "#162033",
    "muted": "#5d687a",
    "soft": "#8793a6",
    "line": "#9aa6b8",
    "panel": "#ffffff",
    "panel_edge": "#a8b0bd",
    "state": "#e9f2ff",
    "state_edge": "#4f86d9",
    "state_dark": "#225aa8",
    "action": "#edf8e9",
    "action_edge": "#62a65f",
    "action_dark": "#28743f",
    "next": "#fff4db",
    "next_edge": "#db9b31",
    "next_dark": "#a4630b",
    "mask": "#fff1f4",
    "mask_edge": "#d85875",
    "mask_dark": "#a72648",
    "train": "#fff0e8",
    "train_edge": "#df7a42",
    "train_dark": "#ad4d22",
    "frozen": "#e9eef5",
    "frozen_edge": "#7890ad",
    "frozen_dark": "#425b78",
    "loss": "#fff0f2",
    "loss_edge": "#d85772",
    "loss_dark": "#aa2547",
    "target": "#eefaf3",
    "target_edge": "#58aa7a",
    "target_dark": "#24724c",
}


def _new_panel(number: int, title: str, subtitle: str) -> tuple[plt.Figure, plt.Axes]:
    fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=DPI)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, FIG_W)
    ax.set_ylim(0, FIG_H)
    ax.axis("off")
    fig.patch.set_facecolor(COLORS["bg"])

    border = FancyBboxPatch(
        (0.35, 0.35),
        15.3,
        8.1,
        boxstyle="round,pad=0.04,rounding_size=0.18",
        fc=COLORS["panel"],
        ec=COLORS["panel_edge"],
        lw=1.3,
        linestyle=(0, (4, 4)),
        zorder=0,
    )
    ax.add_patch(border)
    ax.add_patch(Circle((0.92, 7.92), 0.25, fc="#05070b", ec="#05070b", zorder=5))
    ax.text(0.92, 7.92, str(number), ha="center", va="center", fontsize=15, color="white", fontweight="bold", zorder=6)
    ax.text(1.35, 8.06, title, ha="left", va="center", fontsize=21, fontweight="bold", color=COLORS["ink"])
    ax.text(1.36, 7.62, subtitle, ha="left", va="center", fontsize=10.5, color=COLORS["muted"])
    return fig, ax


def _save(fig: plt.Figure, stem: str) -> list[Path]:
    paths: list[Path] = []
    for ext in ("svg", "pdf", "png"):
        path = OUT_DIR / f"{stem}.{ext}"
        if ext == "png":
            fig.savefig(path, dpi=DPI, facecolor=COLORS["bg"])
        else:
            fig.savefig(path, facecolor=COLORS["bg"])
        paths.append(path)
    plt.close(fig)
    return paths


def _arrow(
    ax: plt.Axes,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    *,
    color: str | None = None,
    lw: float = 1.7,
    dashed: bool = False,
    rad: float = 0.0,
    mutation_scale: float = 13,
    z: int = 3,
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="-|>",
            mutation_scale=mutation_scale,
            lw=lw,
            color=color or COLORS["line"],
            linestyle=(0, (5, 4)) if dashed else "solid",
            connectionstyle=f"arc3,rad={rad}",
            shrinkA=5,
            shrinkB=5,
            zorder=z,
        )
    )


def _box(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    lines: list[str] | None = None,
    *,
    fill: str = "#ffffff",
    edge: str = "#b8c0cc",
    title_color: str | None = None,
    fs_title: float = 12,
    fs_body: float = 8.3,
    lw: float = 1.45,
    tag: str | None = None,
) -> None:
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.035,rounding_size=0.11",
            fc=fill,
            ec=edge,
            lw=lw,
            zorder=2,
        )
    )
    ax.text(
        x + 0.16,
        y + h - 0.22,
        title,
        ha="left",
        va="top",
        fontsize=fs_title,
        fontweight="bold",
        color=title_color or COLORS["ink"],
        zorder=3,
    )
    if tag:
        tag_w = 0.18 + 0.085 * len(tag)
        ax.add_patch(
            FancyBboxPatch(
                (x + w - tag_w - 0.12, y + h - 0.36),
                tag_w,
                0.24,
                boxstyle="round,pad=0.02,rounding_size=0.06",
                fc="#ffffff",
                ec=edge,
                lw=0.9,
                zorder=3,
            )
        )
        ax.text(x + w - tag_w / 2 - 0.12, y + h - 0.24, tag, ha="center", va="center", fontsize=6.9, color=COLORS["muted"], zorder=4)
    if lines:
        yy = y + h - 0.56
        for line in lines:
            ax.text(x + 0.17, yy, line, ha="left", va="top", fontsize=fs_body, color=COLORS["muted"], zorder=3)
            yy -= 0.25


def _token(
    ax: plt.Axes,
    x: float,
    y: float,
    text: str,
    *,
    kind: str = "state",
    w: float = 0.82,
    h: float = 0.48,
    fs: float = 12,
    dashed: bool = False,
) -> None:
    fill = COLORS[kind]
    edge = COLORS[f"{kind}_edge"]
    color = COLORS.get(f"{kind}_dark", COLORS["ink"])
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.025,rounding_size=0.12",
            fc=fill,
            ec=edge,
            lw=1.45,
            linestyle=(0, (4, 3)) if dashed else "solid",
            zorder=4,
        )
    )
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=color, fontweight="bold", zorder=5)


def _cell_cluster(ax: plt.Axes, cx: float, cy: float, scale: float = 1.0, color: str = "state") -> None:
    fill = COLORS[color]
    edge = COLORS[f"{color}_edge"]
    dark = COLORS.get(f"{color}_dark", COLORS["ink"])
    offsets = [(-0.18, -0.02), (0.18, -0.04), (0.0, 0.24)]
    for dx, dy in offsets:
        ax.add_patch(Circle((cx + dx * scale, cy + dy * scale), 0.18 * scale, fc=fill, ec=edge, lw=1.4, zorder=3))
        ax.add_patch(Circle((cx + dx * scale, cy + dy * scale), 0.06 * scale, fc=dark, ec=dark, lw=0.8, alpha=0.75, zorder=4))


def _gene_icon(ax: plt.Axes, cx: float, cy: float, scale: float = 1.0) -> None:
    edge = COLORS["action_edge"]
    xs = [cx - 0.22 * scale, cx - 0.1 * scale, cx + 0.02 * scale, cx + 0.14 * scale, cx + 0.26 * scale]
    ys1 = [cy - 0.22 * scale, cy - 0.08 * scale, cy + 0.1 * scale, cy + 0.2 * scale, cy + 0.08 * scale]
    ys2 = [cy + 0.22 * scale, cy + 0.08 * scale, cy - 0.1 * scale, cy - 0.2 * scale, cy - 0.08 * scale]
    ax.plot(xs, ys1, color=edge, lw=2.0, zorder=4)
    ax.plot(xs, ys2, color=edge, lw=2.0, zorder=4)
    for x, y1, y2 in zip(xs, ys1, ys2, strict=True):
        ax.plot([x, x], [y1, y2], color=edge, lw=1.2, zorder=4)


def _protein_icon(ax: plt.Axes, cx: float, cy: float, scale: float = 1.0) -> None:
    edge = COLORS["action_dark"]
    xs = [cx - 0.28 * scale, cx - 0.15 * scale, cx - 0.18 * scale, cx, cx + 0.12 * scale, cx + 0.06 * scale, cx + 0.25 * scale]
    ys = [cy - 0.10 * scale, cy + 0.20 * scale, cy - 0.22 * scale, cy + 0.18 * scale, cy + 0.02 * scale, cy - 0.18 * scale, cy + 0.13 * scale]
    ax.plot(xs, ys, color=edge, lw=4.0, solid_capstyle="round", solid_joinstyle="round", zorder=4)
    ax.plot(xs, ys, color="#dff0da", lw=1.2, solid_capstyle="round", solid_joinstyle="round", zorder=5)


def _decoder_bars(ax: plt.Axes, x: float, y: float, scale: float = 1.0) -> None:
    heights = [0.5, 0.78, 1.08, 1.42]
    for i, h in enumerate(heights):
        ax.add_patch(
            FancyBboxPatch(
                (x + 0.32 * i * scale, y - h / 2 * scale),
                0.18 * scale,
                h * scale,
                boxstyle="round,pad=0.02,rounding_size=0.06",
                fc="#ffe7c2",
                ec=COLORS["next_edge"],
                lw=1.3,
                zorder=4,
            )
        )


def _transformer_icon(ax: plt.Axes, x: float, y: float, w: float, h: float) -> None:
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.035,rounding_size=0.13",
            fc="#f3efff",
            ec="#7c63c7",
            lw=1.5,
            zorder=3,
        )
    )
    pts = [
        (x + 0.32 * w, y + 0.25 * h),
        (x + 0.32 * w, y + 0.72 * h),
        (x + 0.68 * w, y + 0.25 * h),
        (x + 0.68 * w, y + 0.72 * h),
    ]
    for i, (x1, y1) in enumerate(pts):
        for x2, y2 in pts[i + 1 :]:
            ax.plot([x1, x2], [y1, y2], color="#8a78ca", lw=1.1, zorder=4, alpha=0.9)
    for px, py in pts:
        ax.add_patch(Circle((px, py), 0.075 * w, fc="#9b85d6", ec="#6753af", lw=1.0, zorder=5))


def _legend(ax: plt.Axes, x: float, y: float) -> None:
    items = [
        ("state token", "state"),
        ("action token", "action"),
        ("next/target", "next"),
        ("masked target", "mask"),
        ("trainable", "train"),
        ("frozen/cached", "frozen"),
    ]
    xx = x
    for label, key in items:
        ax.add_patch(Rectangle((xx, y), 0.22, 0.16, fc=COLORS[key], ec=COLORS[f"{key}_edge"], lw=1.0, zorder=3))
        ax.text(xx + 0.28, y + 0.08, label, ha="left", va="center", fontsize=7.3, color=COLORS["muted"], zorder=3)
        xx += 1.95


def figure_query() -> list[Path]:
    fig, ax = _new_panel(
        1,
        "Query and Task Construction",
        "One training item is an in-context perturbation task: support triplets plus one query triplet with hidden next state.",
    )

    _box(
        ax,
        0.85,
        5.35,
        3.25,
        1.55,
        "Same context bucket",
        ["cell type / cell line", "batch / donor / sample", "keeps support and query comparable"],
        fill=COLORS["state"],
        edge=COLORS["state_edge"],
        title_color=COLORS["state_dark"],
    )
    _cell_cluster(ax, 1.45, 4.7, scale=1.65)
    ax.text(1.45, 3.88, "control stack", ha="center", va="center", fontsize=12, color=COLORS["state_dark"], fontweight="bold")
    ax.text(1.45, 3.55, r"$x_t \rightarrow s_t$", ha="center", va="center", fontsize=10.0, color=COLORS["muted"])

    _gene_icon(ax, 3.25, 4.72, scale=1.3)
    ax.text(3.25, 3.88, "query action", ha="center", va="center", fontsize=12, color=COLORS["action_dark"], fontweight="bold")
    ax.text(3.25, 3.55, r"$a_t$ from perturbation ids", ha="center", va="center", fontsize=9.5, color=COLORS["muted"])

    _arrow(ax, 4.05, 4.62, 5.05, 4.62)
    _box(
        ax,
        5.15,
        5.25,
        4.3,
        1.38,
        "Support set",
        ["M visible examples from the same bucket", "each support item has its true perturbed next state"],
        fill="#ffffff",
        edge="#d5dbe5",
        fs_title=13,
    )
    y_rows = [4.78, 4.08, 3.38]
    labels = ["i", "j", "k"]
    for y, lab in zip(y_rows, labels, strict=True):
        _token(ax, 5.25, y, rf"$s_t^{{({lab})}}$", kind="state")
        _arrow(ax, 6.05, y + 0.24, 6.55, y + 0.24, lw=1.2, mutation_scale=10)
        _token(ax, 6.6, y, rf"$a_t^{{({lab})}}$", kind="action")
        _arrow(ax, 7.4, y + 0.24, 7.9, y + 0.24, lw=1.2, mutation_scale=10)
        _token(ax, 7.95, y, rf"$s_{{t+1}}^{{({lab})}}$", kind="next")
    ax.text(6.95, 2.92, "visible support triplets", ha="center", va="center", fontsize=9.2, color=COLORS["muted"])

    _box(
        ax,
        10.35,
        5.25,
        4.35,
        1.38,
        "Query triplet",
        ["same local structure, but the third token is masked", "query_next is target during loss; absent at inference"],
        fill=COLORS["mask"],
        edge=COLORS["mask_edge"],
        title_color=COLORS["mask_dark"],
        fs_title=13,
    )
    _token(ax, 10.85, 4.08, r"$s_t^{q}$", kind="state", w=0.95)
    _arrow(ax, 11.8, 4.32, 12.3, 4.32, lw=1.2, mutation_scale=10)
    _token(ax, 12.35, 4.08, r"$a_t^{q}$", kind="action", w=0.95)
    _arrow(ax, 13.3, 4.32, 13.8, 4.32, lw=1.2, mutation_scale=10)
    _token(ax, 13.85, 4.08, "MASK", kind="mask", w=1.0)
    _arrow(ax, 13.95, 3.78, 13.95, 2.72, color=COLORS["mask_edge"], dashed=True, rad=-0.03)
    _token(ax, 13.45, 2.2, r"$s_{t+1}^{q}$", kind="next", w=1.05)
    ax.text(13.98, 1.86, "hidden target", ha="center", va="center", fontsize=9.0, color=COLORS["muted"])

    _box(
        ax,
        1.0,
        1.05,
        12.65,
        0.72,
        "Data keys used by the current code",
        ["support_obs, support_act, support_next, query_obs, query_act, query_next, query_next_expression"],
        fill="#ffffff",
        edge="#d5dbe5",
        fs_title=11.2,
        fs_body=8.8,
    )
    _legend(ax, 1.0, 0.66)
    ax.text(15.1, 0.63, "Source: DataConfig.context_mode='incontext_set'", ha="right", va="center", fontsize=7.3, color=COLORS["soft"])
    return _save(fig, "world_model_incontext_step1_query")


def figure_encoders() -> list[Path]:
    fig, ax = _new_panel(
        2,
        "Encoders",
        "The batch is converted into latent state tokens and action tokens before the dynamics module sees it.",
    )

    _box(
        ax,
        0.85,
        5.86,
        2.75,
        1.28,
        "State tensors",
        ["support_obs", "support_next", "query_obs"],
        fill=COLORS["state"],
        edge=COLORS["state_edge"],
        title_color=COLORS["state_dark"],
    )
    _box(
        ax,
        0.85,
        2.05,
        2.75,
        1.28,
        "Action tensors",
        ["support_act", "query_act", "gene ids / perturbation ids"],
        fill=COLORS["action"],
        edge=COLORS["action_edge"],
        title_color=COLORS["action_dark"],
    )

    _box(
        ax,
        4.25,
        5.46,
        3.95,
        1.82,
        "State encoder",
        ["_encode_stacks / _encode_one_stack", "StateStackEncoder or ForeignBackboneHead", "optional latent normalization"],
        fill=COLORS["train"],
        edge=COLORS["train_edge"],
        title_color=COLORS["train_dark"],
        tag="TRAINABLE HEAD",
    )
    _transformer_icon(ax, 4.75, 4.18, 1.05, 0.85)
    ax.text(6.9, 4.62, "maps cell stacks to d_model", ha="center", va="center", fontsize=10.2, color=COLORS["muted"])
    _arrow(ax, 3.6, 6.48, 4.25, 6.42)
    _arrow(ax, 8.2, 6.36, 9.0, 6.36)

    _box(
        ax,
        9.0,
        5.56,
        4.2,
        1.56,
        "State latent tokens",
        [r"$support\_s,\ support\_sp \in R^{B \times M \times d}$", r"$query\_s \in R^{B \times d}$", "targets are detached in the loss"],
        fill=COLORS["target"],
        edge=COLORS["target_edge"],
        title_color=COLORS["target_dark"],
    )
    _token(ax, 13.45, 6.4, r"$s$", kind="state", w=0.65, h=0.42)
    _token(ax, 14.2, 6.4, r"$s'$", kind="next", w=0.65, h=0.42)
    _token(ax, 13.82, 5.85, r"$s_q$", kind="state", w=0.75, h=0.42)

    _box(
        ax,
        4.25,
        2.45,
        3.95,
        1.78,
        "Action encoder",
        ["GeneEmbeddingAction", "frozen lookup from AnnData .obsm", "adapter to d_model + masked mean"],
        fill=COLORS["train"],
        edge=COLORS["train_edge"],
        title_color=COLORS["train_dark"],
        tag="FROZEN TABLE",
    )
    ax.text(6.25, 1.98, "support and query action encoders may be separate", ha="center", va="center", fontsize=9.7, color=COLORS["muted"])
    _arrow(ax, 3.6, 2.72, 4.25, 3.05)
    _arrow(ax, 8.2, 3.25, 9.0, 3.25)

    _box(
        ax,
        9.0,
        2.6,
        4.2,
        1.25,
        "Action latent tokens",
        [r"$support\_a \in R^{B \times M \times d}$", r"$query\_a \in R^{B \times d}$"],
        fill=COLORS["action"],
        edge=COLORS["action_edge"],
        title_color=COLORS["action_dark"],
    )
    _token(ax, 13.45, 3.15, r"$a$", kind="action", w=0.65, h=0.42)
    _token(ax, 14.2, 3.15, r"$a_q$", kind="action", w=0.75, h=0.42)

    _box(
        ax,
        1.0,
        0.88,
        13.75,
        0.76,
        "What the dynamics receives",
        ["support_s, support_a, support_sp, query_s, query_a, and query_sp only during teacher-forced training"],
        fill="#ffffff",
        edge="#d5dbe5",
        fs_title=11.4,
        fs_body=9.0,
    )
    _legend(ax, 1.0, 0.52)
    ax.text(15.1, 0.52, "Source: InContextWorldModel._encode_context_batch", ha="right", va="center", fontsize=7.3, color=COLORS["soft"])
    return _save(fig, "world_model_incontext_step2_encoders")


def figure_triplet_dynamics() -> list[Path]:
    fig, ax = _new_panel(
        3,
        "Triplet Context and Dynamics",
        "Training uses teacher-forced progressive prediction: reveal previous triplets, mask the current next-state token.",
    )

    ax.text(1.0, 6.92, "Autoregressive training views inside one batch item", ha="left", va="center", fontsize=13.5, fontweight="bold", color=COLORS["ink"])
    row_y = [6.05, 5.12, 4.19, 3.26]
    row_labels = ["target 0", "target 1", "...", "query"]
    for idx, (y, lab) in enumerate(zip(row_y, row_labels, strict=True)):
        ax.text(0.95, y + 0.22, lab, ha="left", va="center", fontsize=9.5, color=COLORS["muted"])
        x = 2.05
        if idx == 0:
            _token(ax, x, y, r"$s_0$", kind="state", w=0.65)
            _token(ax, x + 0.75, y, r"$a_0$", kind="action", w=0.65)
            _token(ax, x + 1.5, y, "MASK", kind="mask", w=0.92, fs=9.5)
            ax.text(x + 2.65, y + 0.22, r"$\rightarrow\ \hat{s}'_0$", ha="left", va="center", fontsize=12.5, color=COLORS["target_dark"], fontweight="bold")
        elif idx == 1:
            _token(ax, x, y, r"$s_0$", kind="state", w=0.55)
            _token(ax, x + 0.62, y, r"$a_0$", kind="action", w=0.55)
            _token(ax, x + 1.24, y, r"$s'_0$", kind="next", w=0.6)
            ax.text(x + 1.95, y + 0.22, "+", ha="center", va="center", fontsize=13, color=COLORS["muted"])
            _token(ax, x + 2.18, y, r"$s_1$", kind="state", w=0.55)
            _token(ax, x + 2.8, y, r"$a_1$", kind="action", w=0.55)
            _token(ax, x + 3.42, y, "MASK", kind="mask", w=0.82, fs=8.7)
            ax.text(x + 4.48, y + 0.22, r"$\rightarrow\ \hat{s}'_1$", ha="left", va="center", fontsize=12.5, color=COLORS["target_dark"], fontweight="bold")
        elif idx == 2:
            _token(ax, x, y, r"$s_0,a_0,s'_0$", kind="next", w=1.45, fs=9.4)
            ax.text(x + 1.72, y + 0.22, "...", ha="center", va="center", fontsize=13, color=COLORS["muted"])
            _token(ax, x + 2.05, y, r"$s_j$", kind="state", w=0.58)
            _token(ax, x + 2.7, y, r"$a_j$", kind="action", w=0.58)
            _token(ax, x + 3.35, y, "MASK", kind="mask", w=0.82, fs=8.7)
            ax.text(x + 4.38, y + 0.22, r"$\rightarrow\ \hat{s}'_j$", ha="left", va="center", fontsize=12.5, color=COLORS["target_dark"], fontweight="bold")
        else:
            _token(ax, x, y, "all support triplets", kind="next", w=2.25, fs=9.2)
            ax.text(x + 2.52, y + 0.22, "+", ha="center", va="center", fontsize=13, color=COLORS["muted"])
            _token(ax, x + 2.75, y, r"$s_q$", kind="state", w=0.58)
            _token(ax, x + 3.4, y, r"$a_q$", kind="action", w=0.58)
            _token(ax, x + 4.05, y, "MASK", kind="mask", w=0.82, fs=8.7)
            ax.text(x + 5.1, y + 0.22, r"$\rightarrow\ \hat{s}'_q$", ha="left", va="center", fontsize=12.5, color=COLORS["target_dark"], fontweight="bold")

    _box(
        ax,
        0.95,
        1.45,
        6.55,
        1.05,
        "Local token construction",
        ["Every triplet contributes three tokens: [state, action, next_state].",
         "For the current prediction target, next_state is replaced by the learned MASK token."],
        fill="#ffffff",
        edge="#d5dbe5",
        fs_title=12.2,
        fs_body=8.7,
    )

    _arrow(ax, 7.8, 5.25, 9.05, 5.25)
    _box(
        ax,
        9.05,
        5.72,
        4.55,
        1.42,
        "Token embeddings added",
        ["type_embed: state / action / next", "role_embed: support or query", "group_embed: random triplet-binding tag"],
        fill=COLORS["train"],
        edge=COLORS["train_edge"],
        title_color=COLORS["train_dark"],
        fs_title=12.2,
        fs_body=8.3,
        tag="TRAINABLE",
    )
    _box(
        ax,
        9.05,
        4.12,
        4.55,
        1.36,
        "Valid-token attention mask",
        ["future triplets are absent", "current [s, a, MASK] is visible", "previous triplets are fully revealed"],
        fill=COLORS["mask"],
        edge=COLORS["mask_edge"],
        title_color=COLORS["mask_dark"],
        fs_title=12.2,
        fs_body=8.3,
    )
    _transformer_icon(ax, 9.05, 2.48, 1.35, 1.1)
    _box(
        ax,
        10.75,
        2.42,
        2.85,
        1.3,
        "InContextTokensDynamics",
        ["TransformerBlock stack", "masked bidirectional prefixes", "LayerNorm -> Linear head"],
        fill=COLORS["train"],
        edge=COLORS["train_edge"],
        title_color=COLORS["train_dark"],
        fs_title=11.4,
        fs_body=7.4,
    )
    _arrow(ax, 11.35, 5.72, 11.35, 5.48)
    _arrow(ax, 11.35, 4.12, 11.35, 3.72)
    _arrow(ax, 13.6, 3.15, 14.55, 3.15)
    _box(
        ax,
        14.55,
        2.68,
        1.0,
        0.92,
        "raw_all",
        [r"$B \times (M+1) \times d$"],
        fill=COLORS["target"],
        edge=COLORS["target_edge"],
        title_color=COLORS["target_dark"],
        fs_title=10.0,
        fs_body=7.4,
    )

    _legend(ax, 1.0, 0.67)
    ax.text(15.1, 0.67, "Source: InContextTokensDynamics.forward_autoregressive", ha="right", va="center", fontsize=7.3, color=COLORS["soft"])
    return _save(fig, "world_model_incontext_step3_triplet_dynamics")


def figure_prediction_loss() -> list[Path]:
    fig, ax = _new_panel(
        4,
        "Predict, Decode, and Compute Loss",
        "The loss is computed on all autoregressive targets; expression decoding is applied to the final query prediction.",
    )

    _box(
        ax,
        0.95,
        5.85,
        3.15,
        1.2,
        "Dynamics output",
        [r"$raw\_all \in R^{B \times (M+1) \times d}$", "one predicted next state per support target", "last index is the query prediction"],
        fill=COLORS["target"],
        edge=COLORS["target_edge"],
        title_color=COLORS["target_dark"],
        fs_title=12.2,
        fs_body=8.2,
    )
    _arrow(ax, 4.1, 6.38, 5.15, 6.38)
    _box(
        ax,
        5.15,
        5.84,
        3.25,
        1.38,
        "Prediction mode",
        ["absolute: raw_all = s_hat_all", "residual_delta: raw_all = delta_hat_all", "s_hat_all = states_all + delta_hat_all"],
        fill=COLORS["train"],
        edge=COLORS["train_edge"],
        title_color=COLORS["train_dark"],
        fs_title=12.0,
        fs_body=8.0,
    )
    _arrow(ax, 8.4, 6.38, 9.45, 6.38)
    _token(ax, 9.45, 6.13, r"$\hat{s}'_q$", kind="target", w=0.95)
    ax.text(9.95, 5.83, "final query state", ha="center", va="center", fontsize=8.5, color=COLORS["muted"])
    _arrow(ax, 10.42, 6.38, 11.25, 6.38)
    _decoder_bars(ax, 11.45, 6.35, scale=0.8)
    ax.text(11.9, 5.68, "Expression Decoder", ha="center", va="center", fontsize=9.3, color=COLORS["next_dark"])
    _arrow(ax, 12.85, 6.38, 13.45, 6.38)
    ax.add_patch(Circle((14.0, 6.38), 0.48, fc="#fff4db", ec=COLORS["next_edge"], lw=1.5, zorder=3))
    ax.text(14.0, 6.38, r"$\hat{x}_{q}$", ha="center", va="center", fontsize=14, color=COLORS["next_dark"], fontweight="bold")
    ax.text(14.0, 5.75, "reconstructed expression", ha="center", va="center", fontsize=8.5, color=COLORS["muted"])

    _box(
        ax,
        1.0,
        3.55,
        6.45,
        1.18,
        "Latent objective",
        [
            r"$L_{lat}=MSE(\hat{s}_{all}, sg(s'_{all}))$",
            r"$L_{\Delta}=MSE(\hat{\Delta}_{all}, sg(s'_{all}-s_{all}))$ when residual_delta is enabled",
        ],
        fill=COLORS["loss"],
        edge=COLORS["loss_edge"],
        title_color=COLORS["loss_dark"],
        fs_title=12.4,
        fs_body=9.4,
    )
    _box(
        ax,
        8.0,
        3.55,
        6.45,
        1.18,
        "Decoder objective",
        [
            r"$L_{dec}=MSE(\hat{x}_{q}, x_{q})$",
            "computed only when query_next_expression is present and decoder weight > 0",
        ],
        fill=COLORS["loss"],
        edge=COLORS["loss_edge"],
        title_color=COLORS["loss_dark"],
        fs_title=12.4,
        fs_body=9.4,
    )
    _box(
        ax,
        1.0,
        1.62,
        6.45,
        1.28,
        "Optional contrastive losses",
        [
            r"$L_{nce}=InfoNCE(\hat{\Delta}_{all}, \Delta_{all})$",
            "counterfactual action CE: real query action should beat a permuted action",
        ],
        fill=COLORS["loss"],
        edge=COLORS["loss_edge"],
        title_color=COLORS["loss_dark"],
        fs_title=12.4,
        fs_body=9.0,
    )
    _box(
        ax,
        8.0,
        1.62,
        6.45,
        1.28,
        "Total loss and diagnostics",
        [
            r"$L=\lambda_{lat}L_{obj}+\lambda_{dec}L_{dec}+\lambda_{nce}L_{nce}+\lambda_{cf}L_{cf}$",
            "logs: latent_mse, delta_mse, delta_norm, s_hat_dim_var, pos_minus_neg",
        ],
        fill=COLORS["loss"],
        edge=COLORS["loss_edge"],
        title_color=COLORS["loss_dark"],
        fs_title=12.4,
        fs_body=8.8,
    )

    _arrow(ax, 2.5, 5.85, 2.5, 4.73, color=COLORS["loss_edge"], dashed=True)
    _arrow(ax, 13.8, 5.85, 12.0, 4.73, color=COLORS["loss_edge"], dashed=True, rad=-0.1)
    _arrow(ax, 2.5, 3.55, 2.5, 2.9, color=COLORS["loss_edge"], dashed=True)
    _arrow(ax, 10.8, 3.55, 10.8, 2.9, color=COLORS["loss_edge"], dashed=True)

    _legend(ax, 1.0, 0.67)
    ax.text(15.1, 0.67, "Source: InContextWorldModel.loss", ha="right", va="center", fontsize=7.3, color=COLORS["soft"])
    return _save(fig, "world_model_incontext_step4_prediction_loss")


def main() -> None:
    paths = []
    paths.extend(figure_query())
    paths.extend(figure_encoders())
    paths.extend(figure_triplet_dynamics())
    paths.extend(figure_prediction_loss())
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
