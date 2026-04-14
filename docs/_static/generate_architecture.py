"""Generate a publication-quality embpy architecture diagram."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import os

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_W, FIG_H = 30, 18
DPI = 180

# ── Colour palette ──────────────────────────────────────────────────
PAL = {
    "input":    {"hdr": "#2563eb", "bg": "#eff6ff", "accent": "#1d4ed8", "bdr": "#93c5fd"},
    "resolve":  {"hdr": "#059669", "bg": "#ecfdf5", "accent": "#047857", "bdr": "#6ee7b7"},
    "model":    {"hdr": "#7c3aed", "bg": "#f5f3ff", "accent": "#6d28d9", "bdr": "#c4b5fd"},
    "output":   {"hdr": "#d97706", "bg": "#fffbeb", "accent": "#b45309", "bdr": "#fcd34d"},
    "analysis": {"hdr": "#dc2626", "bg": "#fef2f2", "accent": "#b91c1c", "bdr": "#fca5a5"},
    "pp":       {"hdr": "#0891b2", "bg": "#ecfeff", "accent": "#0e7490", "bdr": "#67e8f9"},
}
TXT = {"dark": "#1e293b", "mid": "#475569", "sub": "#64748b", "muted": "#94a3b8"}


def main():
    fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=DPI)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, FIG_W)
    ax.set_ylim(0, FIG_H)
    ax.axis("off")
    fig.patch.set_facecolor("#fafbfc")

    # ── Helpers ──────────────────────────────────────────────────
    def rbox(x, y, w, h, fc="white", ec="#e2e8f0", lw=1.5, zorder=2, alpha=1):
        b = FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.12",
            fc=fc, ec=ec, lw=lw, zorder=zorder, alpha=alpha,
        )
        ax.add_patch(b)

    def hbar(x, y, w, h, color, label, fs=13):
        rbox(x, y, w, h, fc=color, ec="none", zorder=3)
        ax.text(
            x + w / 2, y + h / 2, label, ha="center", va="center",
            fontsize=fs, fontweight="bold", color="white", zorder=4,
        )

    def draw_arrow(x1, y, x2):
        ax.annotate(
            "", xy=(x2, y), xytext=(x1, y),
            arrowprops=dict(
                arrowstyle="-|>", color=TXT["muted"], lw=2.8,
                mutation_scale=20,
            ),
            zorder=1,
        )

    def divider(x, y, w, color, alpha=0.25):
        ax.plot([x, x + w], [y, y], color=color, lw=1, alpha=alpha, zorder=3)

    # ── Title banner ─────────────────────────────────────────────
    rbox(0.3, FIG_H - 1.7, FIG_W - 0.6, 1.5, fc="#1e293b", ec="#334155", lw=2, zorder=5)
    ax.text(
        FIG_W / 2, FIG_H - 0.65, "embpy",
        fontsize=42, fontweight="bold", color="white",
        ha="center", va="center", zorder=6, fontstyle="italic",
    )
    ax.text(
        FIG_W / 2, FIG_H - 1.25,
        "Unified Biological Embedding Framework  --  60+ models, 20+ databases, scanpy-style API",
        fontsize=14, color="#94a3b8", ha="center", va="center", zorder=6,
    )

    # ── Layout constants ─────────────────────────────────────────
    top = FIG_H - 2.1
    bot = 0.5
    card_h = top - bot
    aw = 0.7  # arrow zone width

    c1x, c1w = 0.4, 3.4
    a1 = c1x + c1w + 0.05
    c2x, c2w = a1 + aw + 0.05, 4.6
    a2 = c2x + c2w + 0.05
    c3x, c3w = a2 + aw + 0.05, 8.6
    a3 = c3x + c3w + 0.05
    c4x, c4w = a3 + aw + 0.05, 3.0
    a4 = c4x + c4w + 0.05
    c5x, c5w = a4 + aw + 0.05, FIG_W - (a4 + aw + 0.05) - 0.4

    mid_y = bot + card_h / 2
    for ax_start in (a1, a2, a3, a4):
        draw_arrow(ax_start, mid_y, ax_start + aw)

    # =================================================================
    #  COLUMN 1 -- INPUT MODALITIES
    # =================================================================
    p = PAL["input"]
    rbox(c1x, bot, c1w, card_h, fc=p["bg"], ec=p["bdr"], lw=2)
    hbar(c1x, bot + card_h - 0.65, c1w, 0.65, p["hdr"], "INPUT", fs=13)

    inputs = [
        ("Gene symbols", "PLK1, TP53, BRCA1"),
        ("Protein seqs", "MTEYKLVVVGA..."),
        ("SMILES", "CC(=O)Oc1ccc..."),
        ("DNA sequences", "ATCGATCG..."),
        ("Free text", "Tumor suppressor p53..."),
        ("Single-cell", "AnnData objects"),
        ("Images", "Cell Painting / HPA IF"),
    ]
    yy = bot + card_h - 1.25
    for label, ex in inputs:
        ax.text(c1x + 0.25, yy, label, fontsize=10, fontweight="bold",
                color=p["accent"], va="top", zorder=4)
        ax.text(c1x + 0.25, yy - 0.4, ex, fontsize=7.5,
                color=TXT["sub"], va="top", zorder=4, fontstyle="italic")
        yy -= 1.25
        if yy > bot + 0.5:
            divider(c1x + 0.2, yy + 0.35, c1w - 0.4, p["bdr"])

    # Small BioEmbedder badge
    badge_y = bot + 0.15
    rbox(c1x + 0.3, badge_y, c1w - 0.6, 0.55, fc="#1e293b", ec="#334155", lw=1, zorder=5)
    ax.text(c1x + c1w / 2, badge_y + 0.27, "BioEmbedder",
            ha="center", va="center", fontsize=9, fontweight="bold",
            color="white", zorder=6, fontstyle="italic")

    # =================================================================
    #  COLUMN 2 -- RESOLUTION & RESOURCES
    # =================================================================
    p = PAL["resolve"]
    rbox(c2x, bot, c2w, card_h, fc=p["bg"], ec=p["bdr"], lw=2)
    hbar(c2x, bot + card_h - 0.65, c2w, 0.65, p["hdr"], "RESOLUTION & RESOURCES", fs=12)

    yy = bot + card_h - 1.2

    # -- Resolvers
    ax.text(c2x + 0.2, yy, "Resolvers", fontsize=11, fontweight="bold",
            color=p["accent"], va="top", zorder=4)
    yy -= 0.45
    resolvers = [
        ("GeneResolver", "pyensembl, MyGene, Ensembl REST"),
        ("ProteinResolver", "UniProt REST, isoform mapping"),
        ("DrugResolver", "PubChem, name-to-SMILES"),
        ("TextResolver", "NCBI, Wikipedia, UniProt"),
    ]
    for name, detail in resolvers:
        ax.text(c2x + 0.35, yy, name, fontsize=9.5, fontweight="bold",
                color=TXT["dark"], va="top", zorder=4)
        ax.text(c2x + 0.35, yy - 0.32, detail, fontsize=7.5,
                color=TXT["sub"], va="top", zorder=4)
        yy -= 0.78

    yy -= 0.15
    divider(c2x + 0.2, yy + 0.1, c2w - 0.4, p["hdr"], alpha=0.4)
    yy -= 0.2

    # -- Annotators
    ax.text(c2x + 0.2, yy, "Annotators", fontsize=11, fontweight="bold",
            color=p["accent"], va="top", zorder=4)
    yy -= 0.4
    annotators = [
        "GeneAnnotator  (pathways, PPI, expression)",
        "MoleculeAnnotator  (ChEMBL, KEGG)",
        "ProteinAnnotator  (InterPro, GO, domains)",
        "CellLineAnnotator  (DepMap, Cellosaurus)",
    ]
    for a in annotators:
        ax.text(c2x + 0.35, yy, a, fontsize=8.5, color=TXT["dark"],
                va="top", zorder=4)
        yy -= 0.5

    yy -= 0.15
    divider(c2x + 0.2, yy + 0.1, c2w - 0.4, p["hdr"], alpha=0.4)
    yy -= 0.2

    # -- Data sources
    ax.text(c2x + 0.2, yy, "Data Sources", fontsize=11, fontweight="bold",
            color=p["accent"], va="top", zorder=4)
    yy -= 0.4
    sources = [
        ("HPA", "ICC-IF microscopy images"),
        ("JUMP Cell Painting", "CellProfiler profiles (S3)"),
        ("STRING 12.0", "PPI network embeddings"),
        ("GTEx", "Tissue expression"),
        ("Open Targets", "Disease associations"),
        ("DepMap / Lamin", "Dataset loaders"),
        ("Ensembl 109", "Genome annotations"),
        ("UniProt", "Proteome sequences"),
        ("PubChem", "Compound metadata"),
    ]
    for name, detail in sources:
        ax.text(c2x + 0.35, yy, name, fontsize=8.5, fontweight="bold",
                color=TXT["dark"], va="top", zorder=4)
        ax.text(c2x + 1.9, yy, detail, fontsize=7.5,
                color=TXT["sub"], va="top", zorder=4)
        yy -= 0.45

    # =================================================================
    #  COLUMN 3 -- EMBEDDING MODELS (2-col grid)
    # =================================================================
    p = PAL["model"]
    rbox(c3x, bot, c3w, card_h, fc=p["bg"], ec=p["bdr"], lw=2)
    hbar(c3x, bot + card_h - 0.65, c3w, 0.65, p["hdr"], "EMBEDDING MODELS  (60+)", fs=13)

    sub_w = (c3w - 0.7) / 2
    lcol = c3x + 0.2
    rcol = c3x + 0.2 + sub_w + 0.3

    model_rows = [
        (
            ("DNA / Genomics", [
                "Enformer", "Borzoi / Flashzoi", "Evo 1 & Evo 2",
                "GENA-LM (BERT)", "Nucleotide Transformer v1-v3",
                "HyenaDNA", "Caduceus",
            ]),
            ("Protein", [
                "ESM-1b / ESM-1v", "ESM-2 (8M -- 15B)", "ESM-C (300M -- 6B)",
                "ESM-3", "ProtT5-XL", "Boltz-2 (structure)",
            ]),
        ),
        (
            ("Molecule", [
                "ChemBERTa-2", "MoLFormer", "RDKit fingerprints",
                "MolE", "MiniMol", "MHG-GNN",
            ]),
            ("Text / LLM", [
                "MiniLM", "BERT", "LLaMA 3.x",
                "OpenAI ada/large", "Cohere v3",
                "Voyage 3", "Google embedding",
            ]),
        ),
        (
            ("Single-cell", [
                "scGPT", "Geneformer v1 / v2", "UCE",
                "TranscriptFormer", "Tahoe (70M -- 3B)",
                "Cell2Sentence", "scVI / scanVI / totalVI",
            ]),
            ("Morphology & PPI", [
                "SubCell ViT (contrast.)", "SubCell MAE (reconstruct.)",
                "4ch / 3ch / 2ch variants",
                "CLS / mean / attention pool",
                "---",
                "STRING node2vec", "SPACE functional emb.",
            ]),
        ),
    ]

    yy = bot + card_h - 1.15
    for left_cat, right_cat in model_rows:
        left_name, left_items = left_cat
        right_name, right_items = right_cat
        n = max(len(left_items), len(right_items))
        sub_h = 0.55 + n * 0.35

        for col_x, (cat_name, items) in [(lcol, left_cat), (rcol, right_cat)]:
            rbox(col_x, yy - sub_h, sub_w, sub_h, fc="white", ec=p["bdr"], lw=1, zorder=3)
            ax.text(col_x + 0.15, yy - 0.15, cat_name,
                    fontsize=10, fontweight="bold", color=p["accent"],
                    va="top", zorder=4)
            ty = yy - 0.5
            for item in items:
                if item == "---":
                    divider(col_x + 0.15, ty + 0.12, sub_w - 0.3, p["bdr"], alpha=0.5)
                    ty -= 0.15
                    continue
                ax.text(col_x + 0.25, ty, item, fontsize=8.5,
                        color=TXT["dark"], va="top", zorder=4)
                ty -= 0.35

        yy -= sub_h + 0.25

    # =================================================================
    #  COLUMN 4 -- OUTPUT
    # =================================================================
    p = PAL["output"]
    rbox(c4x, bot, c4w, card_h, fc=p["bg"], ec=p["bdr"], lw=2)
    hbar(c4x, bot + card_h - 0.65, c4w, 0.65, p["hdr"], "OUTPUT", fs=13)

    yy = bot + card_h - 1.3
    output_items = [
        ("Embeddings", [
            "Unified dense vectors",
            "768d -- 15360d",
            "float32 / float16",
        ]),
        ("Storage", [
            "AnnData .obsm",
            "NumPy arrays",
            "Parquet / CSV",
        ]),
        ("Multi-modal", [
            "Gene + Protein + Mol",
            "DNA + Text + Image",
            "Cross-model similarity",
        ]),
        ("Strategies", [
            "CLS token pooling",
            "Mean pooling",
            "Attention pooling",
            "Per-token / per-patch",
        ]),
    ]
    for section, items in output_items:
        ax.text(c4x + 0.2, yy, section, fontsize=10.5, fontweight="bold",
                color=p["accent"], va="top", zorder=4)
        yy -= 0.42
        for item in items:
            ax.text(c4x + 0.3, yy, item, fontsize=8.5,
                    color=TXT["dark"], va="top", zorder=4)
            yy -= 0.35
        yy -= 0.3
        if yy > bot + 1:
            divider(c4x + 0.15, yy + 0.2, c4w - 0.3, p["bdr"])
            yy -= 0.1

    # =================================================================
    #  COLUMN 5 -- ANALYSIS
    # =================================================================
    p = PAL["analysis"]
    rbox(c5x, bot, c5w, card_h, fc=p["bg"], ec=p["bdr"], lw=2)
    hbar(c5x, bot + card_h - 0.65, c5w, 0.65, p["hdr"], "ANALYSIS", fs=13)

    yy = bot + card_h - 1.2

    # -- Tools (tl)
    ax.text(c5x + 0.2, yy, "Tools  (embpy.tl)", fontsize=11, fontweight="bold",
            color=p["accent"], va="top", zorder=4)
    yy -= 0.45
    tl_items = [
        "compute_similarity", "compute_distance_matrix",
        "rank_perturbations", "phenotypic_activity",
        "compute_umap / compute_tsne",
        "cluster_embeddings / leiden",
        "find_nearest_neighbors",
        "pseudobulk_embeddings",
        "benchmark_embeddings",
        "annotate_genes / annotate_drugs",
        "embed_vcf (SNP embeddings)",
    ]
    for item in tl_items:
        ax.text(c5x + 0.3, yy, item, fontsize=8.5,
                color=TXT["dark"], va="top", zorder=4, family="monospace")
        yy -= 0.38

    yy -= 0.15
    divider(c5x + 0.15, yy + 0.1, c5w - 0.3, p["hdr"], alpha=0.4)
    yy -= 0.25

    # -- Plotting (pl)
    ax.text(c5x + 0.2, yy, "Plotting  (embpy.pl)", fontsize=11, fontweight="bold",
            color=p["accent"], va="top", zorder=4)
    yy -= 0.45
    pl_items = [
        "plot_similarity_heatmap",
        "plot_embedding_space / umap",
        "dendrogram / cluster_property",
        "correlation_matrix",
        "cross_model_similarity",
        "radar_chart / parallel_coords",
        "plot_cell_painting",
        "plot_benchmark",
    ]
    for item in pl_items:
        ax.text(c5x + 0.3, yy, item, fontsize=8.5,
                color=TXT["dark"], va="top", zorder=4, family="monospace")
        yy -= 0.38

    yy -= 0.15
    divider(c5x + 0.15, yy + 0.1, c5w - 0.3, p["hdr"], alpha=0.4)
    yy -= 0.25

    # -- Preprocessing (pp)
    ax.text(c5x + 0.2, yy, "Preprocessing  (embpy.pp)", fontsize=11, fontweight="bold",
            color=p["accent"], va="top", zorder=4)
    yy -= 0.45
    pp_items = [
        "Cell Painting -> SubCell remap",
        "Max Z-projection",
        "Rescale / resize / crop",
        "PerturbationProcessor",
        "load_depmap / load_lamin",
    ]
    for item in pp_items:
        ax.text(c5x + 0.3, yy, item, fontsize=8.5,
                color=TXT["dark"], va="top", zorder=4, family="monospace")
        yy -= 0.38

    # ── Save ──────────────────────────────────────────────────────
    for ext in ("png", "svg", "pdf"):
        path = os.path.join(OUT_DIR, f"embpy_architecture.{ext}")
        fig.savefig(path, dpi=DPI, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"Saved: {path}")
    plt.close()


if __name__ == "__main__":
    main()
