"""Shared dataset and action-embedding catalog for world-model launchers."""

from __future__ import annotations

import argparse
from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetSpec:
    """Dataset defaults shared by local and cluster launchers."""

    name: str
    config_path: str
    h5ad_path: str
    control_label: str
    cell_type_key: str | None


@dataclass(frozen=True)
class ActionEmbeddingSpec:
    """Launcher-facing action embedding metadata."""

    key: str
    kind: str
    target: str
    description: str
    pixi_env: str = "gpu"
    mps_safe: bool = True


DATASETS: dict[str, DatasetSpec] = {
    "nadig": DatasetSpec(
        name="nadig",
        config_path="src/world_model/world_model/configs/experiments/single_nadig.yaml",
        h5ad_path="data/crispr_datasets/nadig/NadigOConner2024_jurkat.h5ad",
        control_label="control",
        cell_type_key=None,
    ),
    "replogle": DatasetSpec(
        name="replogle",
        config_path="src/world_model/world_model/configs/experiments/single_replogle.yaml",
        h5ad_path="data/crispr_datasets/replogle/replogle_2022_k562_essential.h5ad",
        control_label="control",
        cell_type_key="cell_type",
    ),
}


GE = "data/embeddings/gene_embeddings"

ACTION_EMBEDDINGS: dict[str, ActionEmbeddingSpec] = {
    "genept": ActionEmbeddingSpec(
        "genept",
        "precomputed",
        f"{GE}/genept/embeddings_3072.csv",
        "GenePT GPT-3.5 text embedding, 3072d",
    ),
    "genept_scaled": ActionEmbeddingSpec(
        "genept_scaled",
        "precomputed",
        f"{GE}/genept/scaled/embeddings_3072.csv",
        "GenePT z-scored, 3072d",
    ),
    "gene2vec": ActionEmbeddingSpec(
        "gene2vec",
        "precomputed",
        f"{GE}/gene2vec/embeddings_d200.csv",
        "Gene2Vec co-expression, 200d",
    ),
    "wikicrow": ActionEmbeddingSpec(
        "wikicrow",
        "precomputed",
        f"{GE}/wikicrow/scaled/embeddings_4096.csv",
        "WikiCrow text embedding, 4096d",
    ),
    "ccle": ActionEmbeddingSpec(
        "ccle",
        "precomputed",
        f"{GE}/ccle/expression_1270_symbol.csv",
        "CCLE expression, 1270d",
    ),
    "ccle_ensembl": ActionEmbeddingSpec(
        "ccle_ensembl",
        "precomputed",
        f"{GE}/ccle/expression_300_ensemblid.csv",
        "CCLE expression, 300d Ensembl-keyed",
    ),
    "crispr_gene_effect": ActionEmbeddingSpec(
        "crispr_gene_effect",
        "precomputed",
        f"{GE}/crispr_gene_effect/gene_effect.csv",
        "DepMap CRISPR gene effect, full",
    ),
    "crispr_gene_effect_1178": ActionEmbeddingSpec(
        "crispr_gene_effect_1178",
        "precomputed",
        f"{GE}/crispr_gene_effect/scaled/gene_effect_1178.csv",
        "DepMap CRISPR gene effect, 1178d",
    ),
    "crispr_gene_effect_205": ActionEmbeddingSpec(
        "crispr_gene_effect_205",
        "precomputed",
        f"{GE}/crispr_gene_effect/scaled/gene_effect_205.csv",
        "DepMap CRISPR gene effect, 205d",
    ),
    "borzoi_v0": ActionEmbeddingSpec("borzoi_v0", "bio", "borzoi_v0", "Borzoi rep-0 DNA"),
    "borzoi_v1": ActionEmbeddingSpec("borzoi_v1", "bio", "borzoi_v1", "Borzoi rep-1 DNA"),
    "enformer_human_rough": ActionEmbeddingSpec(
        "enformer_human_rough",
        "bio",
        "enformer_human_rough",
        "Enformer human rough",
    ),
    "gena_lm_bert_base": ActionEmbeddingSpec(
        "gena_lm_bert_base",
        "bio",
        "gena_lm_bert_base",
        "GENA-LM BERT base",
    ),
    "gena_lm_bert_large": ActionEmbeddingSpec(
        "gena_lm_bert_large",
        "bio",
        "gena_lm_bert_large",
        "GENA-LM BERT large",
    ),
    "gena_lm_bigbird_base": ActionEmbeddingSpec(
        "gena_lm_bigbird_base",
        "bio",
        "gena_lm_bigbird_base",
        "GENA-LM BigBird base",
    ),
    "hyenadna_large_1m": ActionEmbeddingSpec(
        "hyenadna_large_1m",
        "bio",
        "hyenadna_large_1m",
        "HyenaDNA large 1m",
    ),
    "nt_v2_500m": ActionEmbeddingSpec("nt_v2_500m", "bio", "nt_v2_500m", "Nucleotide Transformer v2 500m"),
    "esm2_650M": ActionEmbeddingSpec("esm2_650M", "bio", "esm2_650M", "ESM-2 650M protein"),
    "esm2_3B": ActionEmbeddingSpec("esm2_3B", "bio", "esm2_3B", "ESM-2 3B protein"),
    "esmc_600m": ActionEmbeddingSpec("esmc_600m", "bio", "esmc_600m", "ESM-C 600M protein"),
    "minilm_l6_v2": ActionEmbeddingSpec("minilm_l6_v2", "bio", "minilm_l6_v2", "MiniLM-L6-v2 text"),
    "caduceus_ph_131k": ActionEmbeddingSpec(
        "caduceus_ph_131k",
        "bio",
        "caduceus_ph_131k",
        "Caduceus-PH 131k",
        pixi_env="caduceus",
        mps_safe=False,
    ),
    "caduceus_ps_131k": ActionEmbeddingSpec(
        "caduceus_ps_131k",
        "bio",
        "caduceus_ps_131k",
        "Caduceus-PS 131k",
        pixi_env="caduceus",
        mps_safe=False,
    ),
    "evo2_7b": ActionEmbeddingSpec(
        "evo2_7b",
        "bio",
        "evo2_7b",
        "Evo2 7B",
        pixi_env="evo2",
        mps_safe=False,
    ),
    "omics": ActionEmbeddingSpec(
        "omics",
        "convert",
        f"{GE}/omics/embeddings_d256.tsv",
        "omics 256d, Ensembl-keyed TSV; convert before training",
    ),
    "pops": ActionEmbeddingSpec(
        "pops",
        "convert",
        f"{GE}/pops/features_d256.tsv",
        "PoPS 256d, Ensembl-keyed TSV; convert before training",
    ),
    "string_functional": ActionEmbeddingSpec(
        "string_functional",
        "convert",
        "data/embeddings/precomputed_embeddings_string/functional_embeddings/functional_emb",
        "STRING functional per-anchor HDF5; reduce before training",
    ),
    "string_node2vec": ActionEmbeddingSpec(
        "string_node2vec",
        "convert",
        "data/embeddings/precomputed_embeddings_string/node2vec/node2vec",
        "STRING node2vec per-anchor HDF5; reduce before training",
    ),
}

DEFAULT_MPS_EMBEDDINGS = (
    "genept",
    "genept_scaled",
    "gene2vec",
    "wikicrow",
    "ccle",
    "crispr_gene_effect_1178",
    "borzoi_v0",
    "enformer_human_rough",
    "nt_v2_500m",
    "esm2_650M",
    "minilm_l6_v2",
)
DEFAULT_SLURM_TRANSFER_EMBEDDINGS = (
    "borzoi_v0",
    "enformer_human_rough",
    "nt_v2_500m",
    "esm2_650M",
    "minilm_l6_v2",
)
SUPPORTED_SETUPS = ("single", "finetune", "zeroshot")


def dataset(name: str) -> DatasetSpec:
    """Return defaults for a named perturb-seq dataset."""
    try:
        return DATASETS[name]
    except KeyError as exc:
        raise KeyError(f"Unknown world-model dataset {name!r}; supported: {sorted(DATASETS)}") from exc


def embedding(name: str) -> ActionEmbeddingSpec:
    """Return a catalog entry, falling back to a custom BioEmbedder model key."""
    return ACTION_EMBEDDINGS.get(
        name,
        ActionEmbeddingSpec(name, "bio", name, "custom BioEmbedder MODEL_REGISTRY key"),
    )


def default_embeddings(target: str) -> tuple[str, ...]:
    """Return the default action-embedding keys for a launcher target."""
    if target == "mps":
        return DEFAULT_MPS_EMBEDDINGS
    if target == "slurm-transfer":
        return DEFAULT_SLURM_TRANSFER_EMBEDDINGS
    if target == "slurm":
        return tuple(k for k, spec in ACTION_EMBEDDINGS.items() if spec.kind != "convert")
    if target == "all":
        return tuple(ACTION_EMBEDDINGS)
    raise KeyError(f"Unknown embedding target {target!r}; expected mps, slurm, slurm-transfer, or all.")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect the world-model run matrix.")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("base-config")
    p.add_argument("dataset")
    p = sub.add_parser("h5ad")
    p.add_argument("dataset")
    p = sub.add_parser("control-label")
    p.add_argument("dataset")
    p = sub.add_parser("embedding-spec")
    p.add_argument("embedding")
    p.add_argument("--target", default="slurm")
    p = sub.add_parser("pixi-env")
    p.add_argument("embedding")
    p = sub.add_parser("default-embeddings")
    p.add_argument("--target", default="mps")
    p = sub.add_parser("catalog")
    p.add_argument("--target", default="slurm")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the small command-line inspector used by shell launchers."""
    args = _build_parser().parse_args(argv)
    if args.command == "base-config":
        print(dataset(args.dataset).config_path)
    elif args.command == "h5ad":
        print(dataset(args.dataset).h5ad_path)
    elif args.command == "control-label":
        print(dataset(args.dataset).control_label)
    elif args.command == "embedding-spec":
        spec = embedding(args.embedding)
        if args.target == "mps" and not spec.mps_safe:
            print(f"unsupported|{spec.key} requires a CUDA-specific pixi env, not local MPS")
        else:
            print(f"{spec.kind}|{spec.target}")
    elif args.command == "pixi-env":
        print(embedding(args.embedding).pixi_env)
    elif args.command == "default-embeddings":
        print(" ".join(default_embeddings(args.target)))
    elif args.command == "catalog":
        for key in default_embeddings(args.target):
            spec = embedding(key)
            print(f"{spec.key}|{spec.kind}|{spec.target}|{spec.description}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
