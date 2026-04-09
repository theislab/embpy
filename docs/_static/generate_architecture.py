"""Generate a publication-quality embpy architecture diagram."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# ── Colour palette ──────────────────────────────────────────────────────
C = {
    "input":      "#4A90D9",
    "resolver":   "#7B68EE",
    "annotator":  "#9B59B6",
    "model_dna":  "#E74C3C",
    "model_prot": "#E67E22",
    "model_mol":  "#F1C40F",
    "model_sc":   "#2ECC71",
    "model_morph":"#1ABC9C",
    "model_text": "#3498DB",
    "strategy":   "#95A5A6",
    "output":     "#34495E",
    "analysis_tl":"#D4E6F1",
    "analysis_pl":"#D5F5E3",
    "analysis_pp":"#FDEBD0",
    "bg":         "#FFFFFF",
    "header":     "#2C3E50",
    "arrow":      "#7F8C8D",
    "morph_res":  "#16A085",
}

# ── Text colour for each bg ─────────────────────────────────────────────
def text_col(bg):
    r, g, b = matplotlib.colors.to_rgb(bg)
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    return "white" if lum < 0.55 else "#2C3E50"

# ── Figure setup ────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(32, 18), dpi=250)
ax.set_xlim(0, 32)
ax.set_ylim(0, 18)
ax.set_aspect("equal")
ax.axis("off")
fig.patch.set_facecolor(C["bg"])

# ── Drawing helpers ─────────────────────────────────────────────────────
def draw_box(x, y, w, h, colour, label, fontsize=7, alpha=0.92, bold=False,
             sublabel=None, sublabel_size=5.5, radius=0.15):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.05,rounding_size={radius}",
        facecolor=colour, edgecolor="white", linewidth=0.6, alpha=alpha,
        transform=ax.transData, zorder=2,
    )
    ax.add_patch(box)
    weight = "bold" if bold else "normal"
    tc = text_col(colour)
    if sublabel:
        ax.text(x + w / 2, y + h * 0.62, label, ha="center", va="center",
                fontsize=fontsize, fontweight=weight, color=tc, zorder=3)
        ax.text(x + w / 2, y + h * 0.30, sublabel, ha="center", va="center",
                fontsize=sublabel_size, color=tc, alpha=0.85, zorder=3,
                fontstyle="italic")
    else:
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
                fontsize=fontsize, fontweight=weight, color=tc, zorder=3)
    return (x, y, w, h)

def draw_column_header(x, y, w, label, colour=C["header"], fontsize=9):
    ax.text(x + w / 2, y, label, ha="center", va="bottom",
            fontsize=fontsize, fontweight="bold", color=colour, zorder=4)

def draw_section_bg(x, y, w, h, colour, alpha=0.12, label=None, label_size=6.5):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.08,rounding_size=0.2",
        facecolor=colour, edgecolor=matplotlib.colors.to_rgba(colour, 0.35),
        linewidth=0.8, alpha=alpha, zorder=0,
    )
    ax.add_patch(box)
    if label:
        ax.text(x + 0.12, y + h - 0.15, label, fontsize=label_size,
                fontweight="bold", color=matplotlib.colors.to_rgba(colour, 0.9),
                va="top", zorder=1)

def arrow(x1, y1, x2, y2, colour=C["arrow"], lw=0.7, style="-|>"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=colour, lw=lw,
                                connectionstyle="arc3,rad=0.0"),
                zorder=1)

def arrow_curve(x1, y1, x2, y2, colour=C["arrow"], lw=0.7, rad=0.15):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=colour, lw=lw,
                                connectionstyle=f"arc3,rad={rad}"),
                zorder=1)

# ── Layout constants ────────────────────────────────────────────────────
bw = 3.0    # box width
bh = 0.55   # box height
gap = 0.18  # vertical gap between boxes
col_gap = 0.55  # horizontal gap between columns

# Column x positions
cx = {
    "input":    0.5,
    "resolve":  4.2,
    "models":   9.0,
    "strategy": 15.0,
    "output":   19.0,
    "analysis": 22.5,
}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# COLUMN 1: INPUTS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
x = cx["input"]
top = 16.5
draw_column_header(x, top + 0.15, bw, "Input Modalities")

inputs_data = [
    ("Genetic Perturbations", "Gene Symbol | Ensembl ID | DNA Seq"),
    ("Protein Targets", "UniProt ID | Isoforms"),
    ("Chemical Perturbations", "SMILES | Drug Name | PubChem CID"),
    ("Single-Cell Data", "AnnData | Raw / Log-normalized"),
    ("Morphology Data", "JUMP Cell Painting | HPA ICC-IF"),
    ("Multi-Species Support", "human | mouse | rat | fly | worm | ..."),
]
input_boxes = []
for i, (lab, sub) in enumerate(inputs_data):
    by = top - (i + 1) * (bh + gap)
    b = draw_box(x, by, bw, bh, C["input"], lab, fontsize=6.5, bold=True,
                 sublabel=sub, sublabel_size=5)
    input_boxes.append(b)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# COLUMN 2: RESOLUTION & ANNOTATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
x = cx["resolve"]
draw_column_header(x, top + 0.15, bw, "Resolution & Annotation")

# Sequence resolvers
res_top = top - 0.1
draw_section_bg(x - 0.1, res_top - 4 * (bh + gap) - 0.1, bw + 0.2,
                4 * (bh + gap) + 0.15, C["resolver"], alpha=0.08,
                label="Sequence Resolution")

resolvers_data = [
    ("GeneResolver", "pyensembl | MyGene | Ensembl REST"),
    ("ProteinResolver", "UniProt REST | MyGene.info"),
    ("DrugResolver", "PubChem | NIH Cactus | CIRpy | RDKit"),
    ("TextResolver", "MyGene | NCBI | UniProt | Wikipedia"),
]
resolver_boxes = []
for i, (lab, sub) in enumerate(resolvers_data):
    by = res_top - (i + 1) * (bh + gap)
    b = draw_box(x, by, bw, bh, C["resolver"], lab, fontsize=6.5, bold=True,
                 sublabel=sub, sublabel_size=5)
    resolver_boxes.append(b)

# Morphology resolvers
morph_res_top = res_top - 4 * (bh + gap) - 0.35
draw_section_bg(x - 0.1, morph_res_top - 3 * (bh + gap) - 0.1, bw + 0.2,
                3 * (bh + gap) + 0.15, C["morph_res"], alpha=0.08,
                label="Morphology Resolution")

morph_res_data = [
    ("JUMP Resolver", "broad_babel | jump_portrait | DuckDB"),
    ("HPA Resolver", "proteinatlas.xml | HPA JSON API"),
    ("Morph. Preprocessing", "cell_painting_to_subcell | canvas prep"),
]
morph_res_boxes = []
for i, (lab, sub) in enumerate(morph_res_data):
    by = morph_res_top - (i + 1) * (bh + gap)
    b = draw_box(x, by, bw, bh, C["morph_res"], lab, fontsize=6.5, bold=True,
                 sublabel=sub, sublabel_size=5)
    morph_res_boxes.append(b)

# Annotators
ann_top = morph_res_top - 3 * (bh + gap) - 0.35
draw_section_bg(x - 0.1, ann_top - 4 * (bh + gap) - 0.1, bw + 0.2,
                4 * (bh + gap) + 0.15, C["annotator"], alpha=0.08,
                label="Annotation Sources")

annotators_data = [
    ("MoleculeAnnotator", "RDKit | ChEMBL | ChEBI | KEGG"),
    ("GeneAnnotator", "GTEx | STRING-DB | Open Targets | GWAS"),
    ("ProteinAnnotator", "UniProt | InterPro | GO Terms"),
    ("CellLineAnnotator", "Cellosaurus | DepMap | Passports"),
]
annotator_boxes = []
for i, (lab, sub) in enumerate(annotators_data):
    by = ann_top - (i + 1) * (bh + gap)
    b = draw_box(x, by, bw, bh, C["annotator"], lab, fontsize=6.5, bold=True,
                 sublabel=sub, sublabel_size=5)
    annotator_boxes.append(b)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# COLUMN 3: FOUNDATION MODELS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
x = cx["models"]
mw = 5.2  # wider column for models
draw_column_header(x, top + 0.15, mw, "Foundation Models (60+)")

model_sections = [
    ("DNA Models", C["model_dna"], [
        "Enformer 250M", "Borzoi v0-v3 200M", "Flashzoi v0-v3 200M",
        "Evo 1 / 1.5 / 2  (7B-40B)", "Nucleotide Transformer v1-v3  (50M-2.5B)",
        "HyenaDNA  (1.6M-6.6M)", "GENA-LM  (110M-336M)", "Caduceus 16M",
    ]),
    ("Protein Models", C["model_prot"], [
        "ESM-1b / 1v 650M", "ESM-2  (8M-15B)", "ESM-C  (300M-6B)",
        "ESM3  (1.4B-98B)", "ProtT5 3B", "Boltz-2 Trunk",
    ]),
    ("Molecule Models", C["model_mol"], [
        "ChemBERTa  (77M-100M)", "MolFormer XL",
        "RDKit FP  (Morgan | MACCS)", "MiniMol 10M", "MHG-GNN", "MolE",
    ]),
    ("Single-Cell Models", C["model_sc"], [
        "scGPT 51M", "Geneformer v1/v2  (10M-316M)", "UCE 1.3B",
        "TranscriptFormer  (368M-542M)", "Tahoe-x1  (70M-3B)",
        "Cell2Sentence  (2B-27B)", "PCA", "scVI | scANVI | totalVI",
    ]),
    ("Morphology Models", C["model_morph"], [
        "SubCell MAE  (4 channel configs)", "SubCell ViT  (4 channel configs)",
        "JUMP Pre-computed CellProfiler  (259-dim)",
    ]),
    ("Text Models", C["model_text"], [
        "MiniLM-L6", "BERT",
    ]),
]

model_y = top - 0.1
model_section_mids = []
for sec_name, sec_col, items in model_sections:
    sec_h = len(items) * 0.32 + 0.35
    draw_section_bg(x - 0.1, model_y - sec_h, mw + 0.2, sec_h,
                    sec_col, alpha=0.10, label=sec_name, label_size=6)
    mid_y_acc = 0
    for j, item in enumerate(items):
        iy = model_y - 0.35 - j * 0.32
        ax.text(x + 0.2, iy, item, fontsize=5.5, color="#2C3E50",
                va="center", zorder=3)
        mid_y_acc += iy
    section_mid = mid_y_acc / len(items) if items else model_y - sec_h / 2
    model_section_mids.append((x, section_mid, sec_col))
    model_y -= sec_h + 0.15

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# COLUMN 4: EMBEDDING STRATEGIES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
x = cx["strategy"]
sw = 2.8
draw_column_header(x, top + 0.15, sw, "Embedding Strategies")

strategies_data = [
    ("Standard Pooling", "mean | max | cls"),
    ("Attention Pool", "gated, 1536-dim"),
    ("TPM-Weighted", "isoform average"),
    ("Annotation-Weighted", "residue pooling"),
    ("Expression-Context", "concatenation"),
    ("Region-Specific", "full | exons | introns"),
    ("Perturbation Agg.", "mean across images/wells"),
]
strategy_boxes = []
for i, (lab, sub) in enumerate(strategies_data):
    by = top - (i + 1) * (bh + gap)
    b = draw_box(x, by, sw, bh, C["strategy"], lab, fontsize=6.5, bold=True,
                 sublabel=sub, sublabel_size=5)
    strategy_boxes.append(b)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# COLUMN 5: OUTPUT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
x = cx["output"]
ow = 2.5
draw_column_header(x, top + 0.15, ow, "Output (AnnData)")

output_data = [
    (".obsm", "embedding matrices"),
    (".obs", "annotations & metadata"),
    (".uns", "model metadata"),
    (".npz", "standalone matrices"),
]
output_boxes = []
for i, (lab, sub) in enumerate(output_data):
    by = top - (i + 1) * (bh + gap)
    b = draw_box(x, by, ow, bh, C["output"], lab, fontsize=7, bold=True,
                 sublabel=sub, sublabel_size=5)
    output_boxes.append(b)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# COLUMN 6: ANALYSIS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
x = cx["analysis"]
aw = 8.8
draw_column_header(x, top + 0.15, aw, "Analysis & Visualization")

# embpy.tl
tl_top = top - 0.1
tl_items = [
    "compute_similarity  (cosine | pearson | spearman)",
    "compute_distance_matrix  (euclidean | cosine | correlation)",
    "compute_knn_overlap  (neighbourhood agreement)",
    "rank_perturbations  (by distance or similarity)",
    "pseudobulk_embeddings  (aggregate by group via scanpy)",
    "leiden | cluster_embeddings  (k-means | spectral)",
    "compute_umap | compute_tsne  (CPU & GPU via rapids)",
    "phenotypic_activity  (mAP, chunked cosine, CPU & GPU)",
    "benchmark_embeddings  (cross-validated regression)",
    "compute_metrics | cell_eval | phenocopy_score",
    "annotate_molecules | annotate_gene_perturbations",
    "embed_vcf  (SNP context extraction & embedding)",
]
tl_h = len(tl_items) * 0.28 + 0.35
draw_section_bg(x - 0.1, tl_top - tl_h, aw + 0.2, tl_h,
                "#2980B9", alpha=0.08, label="embpy.tl  --  Analysis Tools", label_size=6.5)
for j, item in enumerate(tl_items):
    iy = tl_top - 0.38 - j * 0.28
    ax.text(x + 0.2, iy, item, fontsize=5.5, color="#2C3E50", va="center", zorder=3)

# embpy.pl
pl_top = tl_top - tl_h - 0.2
pl_items = [
    "plot_similarity_heatmap | distance_heatmap | correlation_matrix",
    "embedding_clustermap | cross_embedding_correlation | cross_model_similarity",
    "plot_embedding_space | all_embeddings | umap_feature_panel",
    "leiden_overview | plot_cluster_composition | dendrogram",
    "embedding_distributions | embedding_norms | plot_perturbation_ranking",
    "parallel_coordinates | radar_chart | star_coordinates",
    "plot_cell_painting  (per-channel fluorescence)",
    "plot_benchmark | plot_benchmark_comparison",
]
pl_h = len(pl_items) * 0.28 + 0.35
draw_section_bg(x - 0.1, pl_top - pl_h, aw + 0.2, pl_h,
                "#27AE60", alpha=0.08, label="embpy.pl  --  Visualization (20+ functions)", label_size=6.5)
for j, item in enumerate(pl_items):
    iy = pl_top - 0.38 - j * 0.28
    ax.text(x + 0.2, iy, item, fontsize=5.5, color="#2C3E50", va="center", zorder=3)

# embpy.pp
pp_top = pl_top - pl_h - 0.2
pp_items = [
    "preprocess_counts  (normalize | log1p | HVG | scale, CPU & GPU)",
    "cell_painting_to_subcell | prepare_subcell_canvas | max_projection_z",
    "reduce_embeddings  (construct perturbation embedding matrices)",
    "load_depmap  (DepMap / CCLE datasets)",
]
pp_h = len(pp_items) * 0.28 + 0.35
draw_section_bg(x - 0.1, pp_top - pp_h, aw + 0.2, pp_h,
                "#E67E22", alpha=0.08, label="embpy.pp  --  Preprocessing", label_size=6.5)
for j, item in enumerate(pp_items):
    iy = pp_top - 0.38 - j * 0.28
    ax.text(x + 0.2, iy, item, fontsize=5.5, color="#2C3E50", va="center", zorder=3)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ARROWS: Input -> Resolution
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def box_right(b):
    return (b[0] + b[2], b[1] + b[3] / 2)

def box_left(b):
    return (b[0], b[1] + b[3] / 2)

# Gene -> GeneResolver, ProteinResolver
arrow(*box_right(input_boxes[0]), *box_left(resolver_boxes[0]), colour=C["input"], lw=0.8)
arrow_curve(*box_right(input_boxes[0]), *box_left(resolver_boxes[1]), colour=C["input"], lw=0.6, rad=0.1)

# Protein -> ProteinResolver
arrow(*box_right(input_boxes[1]), *box_left(resolver_boxes[1]), colour=C["input"], lw=0.8)

# Chemical -> DrugResolver
arrow(*box_right(input_boxes[2]), *box_left(resolver_boxes[2]), colour=C["input"], lw=0.8)

# Morphology -> Morph resolvers
arrow(*box_right(input_boxes[4]), *box_left(morph_res_boxes[0]), colour=C["input"], lw=0.8)
arrow_curve(*box_right(input_boxes[4]), *box_left(morph_res_boxes[1]), colour=C["input"], lw=0.6, rad=0.1)

# Gene -> GeneAnnotator, TextResolver
arrow_curve(*box_right(input_boxes[0]), *box_left(annotator_boxes[1]), colour=C["input"], lw=0.5, rad=0.2)
arrow_curve(*box_right(input_boxes[0]), *box_left(resolver_boxes[3]), colour=C["input"], lw=0.5, rad=0.15)

# Chemical -> MolAnnotator
arrow_curve(*box_right(input_boxes[2]), *box_left(annotator_boxes[0]), colour=C["input"], lw=0.5, rad=0.2)

# GeneResolver -> Morph resolvers (gene symbol resolution)
arrow_curve(*box_right(resolver_boxes[0]), *box_left(morph_res_boxes[0]),
            colour=C["resolver"], lw=0.6, rad=0.15)
# DrugResolver -> Morph resolvers (compound name resolution)
arrow_curve(*box_right(resolver_boxes[2]), *box_left(morph_res_boxes[0]),
            colour=C["resolver"], lw=0.6, rad=0.15)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ARROWS: Resolution -> Models (broad arrows)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
models_x = cx["models"]

# GeneResolver -> DNA models
arrow(cx["resolve"] + bw, resolver_boxes[0][1] + bh / 2,
      models_x, model_section_mids[0][1],
      colour=C["model_dna"], lw=1.0)

# ProteinResolver -> Protein models
arrow(cx["resolve"] + bw, resolver_boxes[1][1] + bh / 2,
      models_x, model_section_mids[1][1],
      colour=C["model_prot"], lw=1.0)

# DrugResolver -> Molecule models
arrow(cx["resolve"] + bw, resolver_boxes[2][1] + bh / 2,
      models_x, model_section_mids[2][1],
      colour=C["model_mol"], lw=1.0)

# Single-Cell (direct from input)
arrow(cx["input"] + bw, input_boxes[3][1] + bh / 2,
      models_x, model_section_mids[3][1],
      colour=C["model_sc"], lw=1.0)

# Morph preprocessing -> Morphology models
arrow(cx["resolve"] + bw, morph_res_boxes[2][1] + bh / 2,
      models_x, model_section_mids[4][1],
      colour=C["model_morph"], lw=1.0)

# TextResolver -> Text models
arrow(cx["resolve"] + bw, resolver_boxes[3][1] + bh / 2,
      models_x, model_section_mids[5][1],
      colour=C["model_text"], lw=1.0)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ARROWS: Models -> Strategies -> Output -> Analysis
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Broad arrow: Models -> Strategies
strat_mid_y = sum(b[1] + bh / 2 for b in strategy_boxes) / len(strategy_boxes)
for _, my, mc in model_section_mids:
    arrow(models_x + mw, my, cx["strategy"], strat_mid_y,
          colour=C["arrow"], lw=0.6)

# Strategies -> Output
out_mid_y = sum(b[1] + bh / 2 for b in output_boxes) / len(output_boxes)
for b in strategy_boxes:
    arrow(cx["strategy"] + sw, b[1] + bh / 2,
          cx["output"], out_mid_y,
          colour=C["arrow"], lw=0.5)

# Annotators -> .obs
arrow(cx["resolve"] + bw, annotator_boxes[1][1] + bh / 2,
      cx["output"], output_boxes[1][1] + bh / 2,
      colour=C["annotator"], lw=0.7)

# Output -> Analysis
analysis_mid_y = (tl_top + pp_top - pp_h) / 2
for b in output_boxes:
    arrow(cx["output"] + ow, b[1] + bh / 2,
          cx["analysis"] - 0.1, analysis_mid_y,
          colour=C["output"], lw=0.6)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Title
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ax.text(16, 17.6, "embpy", fontsize=20, fontweight="bold", ha="center",
        color=C["header"], fontstyle="italic")
ax.text(16, 17.2, "Unified Biological Perturbation Embedding Framework",
        fontsize=10, ha="center", color="#7F8C8D")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Save
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
fig.savefig("docs/_static/embpy_architecture_detailed.png",
            dpi=300, bbox_inches="tight", facecolor="white", pad_inches=0.3)
fig.savefig("docs/_static/embpy_architecture_detailed.svg",
            bbox_inches="tight", facecolor="white", pad_inches=0.3)
fig.savefig("docs/_static/embpy_architecture_detailed.pdf",
            bbox_inches="tight", facecolor="white", pad_inches=0.3)
print("Saved PNG, SVG, and PDF")
