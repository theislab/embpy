---
name: embpy-variant-effects
description: Score single-nucleotide variants with DNA language models in embpy - fetch the genomic context around a SNP, embed reference and alternate alleles with HyenaDNA, Nucleotide Transformer, Caduceus or Enformer, and use the embedding change (delta norm) as a variant-effect signal; also batch variants from a VCF. Use when the user asks for the effect of a SNP or mutation on DNA embeddings, wants to compare variants, embed a VCF, or prioritize variants with a genomic foundation model.
---

# Variant-effect embeddings

The idea: embed the sequence around a variant with the reference allele and with each
alternate allele. The size of the change (`delta_norms`) is a model-based, unsupervised
signal of how much the variant alters the sequence representation. It is not a validated
pathogenicity score; say so when reporting.

## 1. Load a DNA model and fetch the context

```python
import embpy as ep

wrapper = ep.BioEmbedder(device="cpu").get_model("hyenadna_tiny_1k")
provider = ep.tl.SequenceProvider()
window, offset = provider.get_window("17", 7_675_088, context=512)
print(len(window), offset, window[offset - 1])
```

- `get_model` returns the loaded model wrapper. Pick a context the model supports:
  `hyenadna_tiny_1k` (1 kb, CPU), `nt_v2_100m`, `caduceus_ph_131k`; Enformer and Borzoi need
  their full input length and a GPU.
- `SequenceProvider()` fetches only the window from the Ensembl REST API (GRCh38 by default;
  `genome_build="GRCh37"` for hg19 coordinates). Chromosome names are `"17"`, not `"chr17"`,
  for the REST API. With a local genome use `SequenceProvider(fasta_dir=...)` or `fasta_file=...`.
- `offset` is the 1-based position of the variant inside `window`.

## 2. Check the reference allele, then embed

```python
ref = window[offset - 1]
assert ref == "C", f"reference is {ref}: check the genome build and coordinates"
snp = ep.tl.SNPContext(position=offset, ref_allele="C", alt_alleles=["T"], context_window=512, chrom="17", variant_id="TP53_R175H")
result = ep.tl.SNPEmbedder(wrapper).embed_snp(snp, window)
print(result.delta_norms)
```

- Always compare the user's reference allele with the fetched sequence first. A mismatch means
  the wrong genome build, a 0-based coordinate, or the opposite strand; stop and ask.
- Positions are **1-based** (VCF convention). For genes on the minus strand, alleles are given
  on the forward strand in VCFs; pass `strand="-"` only if the alleles are on the minus strand.
- `result` also holds `ref_embedding`, `alt_embeddings` and `delta_embeddings` (alt - ref).

## 3. Many variants from a VCF

```python
# Requires: a local genome FASTA (whole chromosomes; positions are chromosome coordinates)
provider = ep.tl.SequenceProvider(fasta_file="/data/hg38.fa")
results = ep.tl.embed_vcf(
    "variants.vcf.gz",
    model_wrapper=wrapper,
    chromosome_sequences={"chr17": provider.get_chromosome("chr17")},
    context_window=512,
    max_variants=100,
)
```

Keys of `chromosome_sequences` must match the VCF `CHROM` column. Only SNPs and single-base
indels are embedded; multi-allelic records are split.

## Reporting

Give model, context length, genome build and the delta norm per variant. Compare variants
against each other (rank them) or against a set of known-benign variants embedded the same way;
an absolute delta norm has no fixed threshold.
