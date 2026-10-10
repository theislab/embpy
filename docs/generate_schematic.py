"""Generate the embpy architecture schematic as a self-contained SVG.

Replaces the older matplotlib PNG generator. The SVG paints its own light panel
so it renders identically on GitHub's light and dark themes and in the docs.
Run: ``python docs/generate_schematic.py`` -> writes ``docs/embpy_architecture.svg``.

The diagram depicts the actual data flow, not a component inventory:
biological entities -> resolve/preprocess -> BioEmbedder.embed() -> model
registry -> typed AnnData slots, with annotation feeding in and tl/pl reading out.
"""
from __future__ import annotations

import html
import os

# ── palette (reads on the self-contained light panel) ────────────────────────
INK, MID, SUB = "#1e293b", "#475569", "#64748b"
PANEL, CARD, LINE = "#fbfcfd", "#ffffff", "#e2e8f0"
STAGES = {
    "entity":   {"hdr": "#5a6b78", "bg": "#f1f5f7", "bdr": "#cbd5e1"},
    "prep":     {"hdr": "#0e7490", "bg": "#ecfeff", "bdr": "#a5d8e6"},
    "hub":      {"hdr": "#0f7d8c", "bg": "#0f7d8c", "bdr": "#0b5c68"},
    "registry": {"hdr": "#6d28d9", "bg": "#f5f3ff", "bdr": "#c4b5fd"},
    "output":   {"hdr": "#b45309", "bg": "#fffbeb", "bdr": "#fcd34d"},
    "annotate": {"hdr": "#047857", "bg": "#ecfdf5", "bdr": "#6ee7b7"},
    "analysis": {"hdr": "#b91c1c", "bg": "#fef2f2", "bdr": "#fca5a5"},
}

W, H = 1320, 744
_parts: list[str] = []


def esc(s: str) -> str:
    return html.escape(s, quote=True)


def rrect(x, y, w, h, fill, stroke, sw=1.5, rx=10, opacity=1.0):
    _parts.append(
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" ry="{rx}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}" opacity="{opacity}"/>'
    )


def text(x, y, s, size=13, fill=INK, anchor="start", weight="400", spacing=None, mono=False):
    fam = "ui-monospace, SFMono-Regular, Menlo, monospace" if mono else \
          "system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif"
    ls = f' letter-spacing="{spacing}"' if spacing else ""
    _parts.append(
        f'<text x="{x}" y="{y}" font-family="{fam}" font-size="{size}" '
        f'fill="{fill}" text-anchor="{anchor}" font-weight="{weight}"{ls}>{esc(s)}</text>'
    )


def card(x, y, w, h, stage, title, chips, sub=None, title_size=14):
    s = STAGES[stage]
    rrect(x, y, w, h, s["bg"], s["bdr"], sw=1.6)
    rrect(x, y, w, 34, s["hdr"], s["hdr"], sw=0, rx=10)
    rrect(x, y + 22, w, 12, s["bg"], s["bg"], sw=0, rx=0)  # square off header's lower corners
    text(x + 14, y + 22, title, size=title_size, fill="#ffffff", weight="650")
    cy = y + 58
    for c in chips:
        rrect(x + 12, cy - 15, w - 24, 24, CARD, s["bdr"], sw=1)
        text(x + 22, cy + 1, c, size=12, fill=MID, mono=("·" in c or "→" in c or "." in c))
        cy += 31
    if sub:
        text(x + w / 2, y + h - 12, sub, size=11, fill=SUB, anchor="middle")


def arrow(x1, y1, x2, y2, label=None, color=MID, label_dy=-8):
    _parts.append(
        f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" '
        f'stroke-width="2" marker-end="url(#ah)"/>'
    )
    if label:
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        text(mx, my + label_dy, label, size=11, fill=color, anchor="middle", weight="600")


# ── background + defs ─────────────────────────────────────────────────────────
_parts.append(
    f'<defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" '
    f'markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" '
    f'fill="{MID}"/></marker></defs>'
)
rrect(0, 0, W, H, PANEL, PANEL, sw=0, rx=0)

# ── title ─────────────────────────────────────────────────────────────────────
text(40, 50, "embpy", size=30, fill=INK, weight="700")
text(148, 50, "one embedding API across every biological modality", size=16, fill=MID)
text(40, 74, "BioEmbedder.embed(entity)  →  resolve  →  model  →  typed AnnData",
     size=12.5, fill=SUB, mono=True)

# ── main pipeline (left → right) ───────────────────────────────────────────────
Y, HGT = 232, 300
xs = {"entity": 30, "prep": 288, "registry": 780, "output": 1058}
WID = {"entity": 218, "prep": 210, "registry": 258, "output": 232}

card(xs["entity"], Y, WID["entity"], HGT, "entity", "Biological entities",
     ["genes", "proteins", "molecules", "DNA", "single cells", "morphology",
      "text · perturbations"], sub="passed as AnnData or lists")

card(xs["prep"], Y, WID["prep"], HGT, "prep", "Resolve + preprocess",
     ["symbol → sequence", "name → SMILES", "counts · log-norm · HVG",
      "rank / bin tokenize", "morphology canvas"], sub="9 resolvers · embpy.pp")

# hub — emphasized, vertically centred
HUBX, HUBW, HUBH = 522, 210, 176
HUBY = Y + (HGT - HUBH) / 2
rrect(HUBX, HUBY, HUBW, HUBH, STAGES["hub"]["bg"], STAGES["hub"]["bdr"], sw=2)
text(HUBX + HUBW / 2, HUBY + 40, "BioEmbedder", size=18, fill="#ffffff",
     anchor="middle", weight="700")
text(HUBX + HUBW / 2, HUBY + 62, ".embed(entity, model=…)", size=12.5,
     fill="#d7f0f3", anchor="middle", mono=True)
for i, line in enumerate(["one entry point", "device auto · batching",
                          "caching · provenance"]):
    text(HUBX + HUBW / 2, HUBY + 92 + i * 22, line, size=12, fill="#eafafb",
         anchor="middle")

card(xs["registry"], Y, WID["registry"], HGT, "registry", "Model registry · ~150",
     ["Protein — ESM2 · ESMC · ProtT5", "DNA — Enformer · Borzoi · NT · Evo",
      "Molecule — ChemBERTa · RDKit", "Single-cell — scGPT · Geneformer · scVI",
      "Morphology — SubCell · Structure"], sub="one key selects the backend")

card(xs["output"], Y, WID["output"], HGT, "output", "AnnData output contract",
     [".obsm — row-aligned", ".varm — feature-aligned", ".uns — payload + provenance"],
     sub=".X stays raw counts")

# pipeline arrows
midY = Y + HGT / 2
arrow(xs["entity"] + WID["entity"], midY, xs["prep"], midY, "resolve")
arrow(xs["prep"] + WID["prep"], midY, HUBX, HUBY + HUBH / 2, "dispatch")
arrow(HUBX + HUBW, HUBY + HUBH / 2, xs["registry"], midY, "select")
arrow(xs["registry"] + WID["registry"], midY, xs["output"], midY, "write")

# ── annotate (feeds AnnData from above) ────────────────────────────────────────
AY = 78
card(xs["output"], AY, WID["output"], 128, "annotate", "Annotate",
     ["pathways · GO · domains", "diseases · interactions", "cell-line metadata"])
arrow(xs["output"] + WID["output"] / 2, AY + 128,
      xs["output"] + WID["output"] / 2, Y, "enrich → .obs/.uns", label_dy=-6)

# ── analysis (reads AnnData from below) ────────────────────────────────────────
LY = Y + HGT + 40
card(xs["output"], LY, WID["output"], 128, "analysis", "Analyze — tl · pl",
     ["metrics · similarity · scIB", "benchmark · clustering", "embedding-space · KNN plots"])
arrow(xs["output"] + WID["output"] / 2, Y + HGT,
      xs["output"] + WID["output"] / 2, LY, "analyze")

# ── caption strip ──────────────────────────────────────────────────────────────
text(40, H - 22,
     "A single call resolves an identifier, routes it to the right model among seven "
     "families, and writes embeddings into the correct AnnData slot with provenance.",
     size=12, fill=SUB)

svg = (
    f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
    f'role="img" aria-label="embpy architecture: BioEmbedder.embed routes any '
    f'biological entity through resolve and preprocess to one of ~150 models across '
    f'seven modality families, writing embeddings into typed AnnData slots, with '
    f'annotation feeding in and tl/pl analysis reading out." '
    f'style="max-width:100%;height:auto">\n' + "\n".join(_parts) + "\n</svg>\n"
)

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "embpy_architecture.svg")
with open(out, "w") as fh:
    fh.write(svg)
print(f"wrote {out} ({len(svg)} bytes)")
