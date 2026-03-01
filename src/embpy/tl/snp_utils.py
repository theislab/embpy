from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from ..models.base import BaseModelWrapper


import os
import shutil
import subprocess
import urllib.request
from pathlib import Path


# UCSC hg38 — per-chromosome FASTAs (chromFa.tar.gz, ~938 MB compressed)
_UCSC_CHROMFA_URL = "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.chromFa.tar.gz"

# UCSC hg38 — single whole-genome FASTA (~938 MB compressed, ~3.1 GB uncompressed)
_UCSC_HG38_URL = "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz"

# Ensembl hg38 (GRCh38.p14) — single FASTA, primary assembly only (~850 MB compressed)
_ENSEMBL_HG38_URL = (
    "https://ftp.ensembl.org/pub/release-110/fasta/homo_sapiens/dna/"
    "Homo_sapiens.GRCh38.dna.primary_assembly.fa.gz"
)

# Standard autosomes + sex chromosomes kept by default
_PRIMARY_CHROMS = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY", "chrM"]


def _progress_hook(count: int, block_size: int, total_size: int) -> None:
    """Simple download progress reporter for urllib.request.urlretrieve."""
    if total_size <= 0:
        return
    pct = min(100.0, count * block_size * 100.0 / total_size)
    done = int(pct / 2)
    bar = "█" * done + "░" * (50 - done)
    print(f"\r  [{bar}] {pct:5.1f}%", end="", flush=True)


def _run(cmd: str) -> None:
    """Run a shell command, raising RuntimeError on failure."""
    ret = subprocess.run(cmd, shell=True)
    if ret.returncode != 0:
        raise RuntimeError(f"Command failed: {cmd}")


def download_hg38_per_chrom(
    output_dir: str | Path = "./hg38_chroms",
    chromosomes: list[str] | None = None,
    force: bool = False,
) -> Path:
    """Download hg38 as individual per-chromosome FASTA files (Option A).

    Downloads ``chromFa.tar.gz`` from UCSC (~938 MB), extracts it, keeps
    only the requested chromosomes, and renames files to the ``chr<N>.fa``
    convention expected by :class:`SequenceProvider`.

    Parameters
    ----------
    output_dir
        Directory where chromosome FASTA files will be saved.
        Created if it does not exist.
    chromosomes
        List of chromosome names to keep (e.g. ``["chr17", "chr1"]``).
        Defaults to the 24 primary chromosomes (chr1–chr22, chrX, chrY, chrM).
        Pass ``None`` to keep all chromosomes including unplaced contigs.
    force
        Re-download even if files already exist.

    Returns
    -------
    Path
        The ``output_dir`` path, ready to pass to
        ``SequenceProvider(fasta_dir=...)``.

    Examples
    --------
    >>> from embpy.snp_utils import download_hg38_per_chrom, SequenceProvider
    >>> fasta_dir = download_hg38_per_chrom("./hg38_chroms", chromosomes=["chr17"])
    >>> provider = SequenceProvider(fasta_dir=fasta_dir)
    >>> seq = provider.get_chromosome("chr17")
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    keep = set(chromosomes) if chromosomes is not None else None

    # Check if already complete
    if not force and keep:
        existing = {p.stem for p in output_dir.glob("chr*.fa")}
        needed = {c for c in keep}
        if needed.issubset(existing):
            logging.info(f"All requested chromosomes already present in {output_dir}. Skipping download.")
            return output_dir

    tarball = output_dir / "chromFa.tar.gz"
    if not tarball.exists() or force:
        logging.info(f"Downloading hg38 chromFa.tar.gz from UCSC (~938 MB) → {tarball}")
        print(f"Downloading {_UCSC_CHROMFA_URL}")
        urllib.request.urlretrieve(_UCSC_CHROMFA_URL, tarball, reporthook=_progress_hook)
        print()  # newline after progress bar

    # Extract
    extract_dir = output_dir / "_extracted"
    extract_dir.mkdir(exist_ok=True)
    logging.info(f"Extracting {tarball} …")
    print("Extracting archive …")
    _run(f"tar -xzf {tarball} -C {extract_dir}")

    # Move and rename files
    moved = 0
    for fa_path in extract_dir.rglob("*.fa"):
        chrom_name = fa_path.stem  # e.g. "chr17"
        if keep is not None and chrom_name not in keep:
            continue
        dest = output_dir / f"{chrom_name}.fa"
        shutil.move(str(fa_path), dest)
        logging.info(f"  {chrom_name}.fa → {dest}")
        moved += 1

    # Clean up
    shutil.rmtree(extract_dir, ignore_errors=True)
    if not force:
        tarball.unlink(missing_ok=True)

    logging.info(f"Done — {moved} chromosome FASTA files in {output_dir}.")
    print(f"Done — {moved} chromosome file(s) saved to {output_dir}")
    return output_dir


def download_hg38_single_fasta(
    output_path: str | Path = "./hg38.fa",
    source: str = "ucsc",
    force: bool = False,
) -> Path:
    """Download hg38 as a single whole-genome FASTA file (Option B).

    Downloads and decompresses hg38 from UCSC or Ensembl.  The resulting
    file is indexed on first use by :class:`SequenceProvider` (BioPython
    ``SeqIO.index``), so the full 3 GB is never loaded into RAM at once.

    Parameters
    ----------
    output_path
        Destination path for the uncompressed ``.fa`` file.
    source
        ``"ucsc"`` (default) or ``"ensembl"``.
        UCSC includes UCSC-style ``chr``-prefixed names (``chr17``).
        Ensembl uses numeric names (``17``) — :class:`SequenceProvider`
        handles both automatically.
    force
        Re-download even if the file already exists.

    Returns
    -------
    Path
        Path to the uncompressed FASTA file, ready to pass to
        ``SequenceProvider(fasta_file=...)``.

    Examples
    --------
    >>> from embpy.snp_utils import download_hg38_single_fasta, SequenceProvider
    >>> fa = download_hg38_single_fasta("./hg38.fa", source="ucsc")
    >>> provider = SequenceProvider(fasta_file=fa)
    >>> seq = provider.get_chromosome("chr17")
    """
    output_path = Path(output_path)
    if output_path.exists() and not force:
        logging.info(f"hg38 FASTA already present at {output_path}. Skipping download.")
        return output_path

    url = _UCSC_HG38_URL if source == "ucsc" else _ENSEMBL_HG38_URL
    gz_path = output_path.with_suffix(".fa.gz")

    print(f"Downloading hg38 from {source.upper()} (~938 MB compressed) → {gz_path}")
    logging.info(f"Downloading {url} → {gz_path}")
    urllib.request.urlretrieve(url, gz_path, reporthook=_progress_hook)
    print()

    print(f"Decompressing → {output_path}  (this may take a few minutes)")
    logging.info("Decompressing …")
    _run(f"gunzip -c {gz_path} > {output_path}")
    gz_path.unlink(missing_ok=True)

    logging.info(f"Done — hg38 FASTA at {output_path}.")
    print(f"Done — {output_path} ({output_path.stat().st_size / 1e9:.1f} GB)")
    return output_path


@dataclass
class SNPContext:
    """Describes a single-nucleotide variant and its genomic context window.

    Attributes
    ----------
    position : int
        1-based position of the variant on the chromosome (or within the
        provided sequence if ``chromosome_sequence`` is supplied directly).
    ref_allele : str
        Reference allele (must be a single nucleotide unless ``indel=True``).
    alt_alleles : list[str]
        One or more alternate alleles to evaluate.
    context_window : int
        Total context length centred on the variant.  For models with a fixed
        input requirement (e.g. Enformer: 196,608 bp) set this to match the
        model's expected input size.  For transformer models the sequence is
        truncated/padded by the tokeniser, so any reasonable window works.
    chrom : str, optional
        Chromosome identifier (informational only; not used for sequence
        retrieval within this dataclass).
    strand : {"+", "-"}
        Strand.  If ``"-"``, the reference and alternate sequences will be
        reverse-complemented before embedding.
    variant_id : str, optional
        A human-readable identifier for the variant (e.g. rsID).
    """

    position: int
    ref_allele: str
    alt_alleles: list[str]
    context_window: int = 512
    chrom: str = ""
    strand: Literal["+", "-"] = "+"
    variant_id: str = ""

    def __post_init__(self) -> None:
        self.ref_allele = self.ref_allele.upper()
        self.alt_alleles = [a.upper() for a in self.alt_alleles]


@dataclass
class SNPEmbeddingResult:
    """Container for SNP embedding outputs.

    Attributes
    ----------
    snp : SNPContext
        The input variant descriptor.
    ref_sequence : str
        Extracted reference context sequence (post strand-flip if applicable).
    alt_sequences : list[str]
        Alternate context sequences, one per alt allele.
    ref_embedding : np.ndarray
        Embedding of the reference sequence.
    alt_embeddings : list[np.ndarray]
        Embeddings of each alternate sequence.
    delta_embeddings : list[np.ndarray]
        ``alt_emb - ref_emb`` for each alternate allele.
    delta_norms : list[float]
        L2 norm of each delta embedding (scalar summary of effect size).
    cosine_similarities : list[float]
        Cosine similarity between reference and each alternate embedding.
    model_name : str
        Name of the model used.
    pooling_strategy : str
        Pooling strategy used.
    """

    snp: SNPContext
    ref_sequence: str
    alt_sequences: list[str]
    ref_embedding: np.ndarray
    alt_embeddings: list[np.ndarray]
    delta_embeddings: list[np.ndarray] = field(default_factory=list)
    delta_norms: list[float] = field(default_factory=list)
    cosine_similarities: list[float] = field(default_factory=list)
    model_name: str = ""
    pooling_strategy: str = "mean"

    def __post_init__(self) -> None:
        # Auto-compute deltas and summaries if not provided
        if not self.delta_embeddings:
            self.delta_embeddings = [a - self.ref_embedding for a in self.alt_embeddings]
        if not self.delta_norms:
            self.delta_norms = [float(np.linalg.norm(d)) for d in self.delta_embeddings]
        if not self.cosine_similarities:
            ref_norm = float(np.linalg.norm(self.ref_embedding))
            self.cosine_similarities = [
                float(
                    np.dot(self.ref_embedding, a)
                    / (ref_norm * float(np.linalg.norm(a)) + 1e-12)
                )
                for a in self.alt_embeddings
            ]

    def to_dict(self) -> dict[str, Any]:
        """Serialise summary statistics (not raw embeddings) to a dict."""
        rows = []
        for i, alt in enumerate(self.snp.alt_alleles):
            rows.append(
                {
                    "variant_id": self.snp.variant_id,
                    "chrom": self.snp.chrom,
                    "position": self.snp.position,
                    "ref": self.snp.ref_allele,
                    "alt": alt,
                    "delta_l2_norm": self.delta_norms[i],
                    "cosine_similarity": self.cosine_similarities[i],
                    "model": self.model_name,
                    "pooling": self.pooling_strategy,
                }
            )
        return rows


def _reverse_complement(seq: str) -> str:
    """Return the reverse complement of a DNA string."""
    comp = str.maketrans("ACGTNacgtn", "TGCANtgcan")
    return seq.translate(comp)[::-1]


def _extract_context(
    chromosome_sequence: str,
    position: int,
    context_window: int,
) -> tuple[int, int, int]:
    """Compute slice indices for a context window centred on ``position``.

    Parameters
    ----------
    chromosome_sequence
        Full (or partial) chromosome / contig sequence string.
    position
        1-based variant position.
    context_window
        Desired total window size.

    Returns
    -------
    (start_0based, end_0based, offset_in_window)
        Start/end as 0-based Python slice indices, and the 0-based offset of
        the variant position within the extracted window.
    """
    pos0 = position - 1  # convert to 0-based
    half = context_window // 2
    start = max(0, pos0 - half)
    end = min(len(chromosome_sequence), start + context_window)
    # Re-adjust start if we hit the end boundary
    start = max(0, end - context_window)
    offset = pos0 - start
    return start, end, offset


def _apply_snp(sequence: str, offset: int, ref: str, alt: str) -> str:
    """Return ``sequence`` with the allele at ``offset`` replaced by ``alt``.

    Raises
    ------
    ValueError
        If the nucleotide at ``offset`` does not match ``ref`` (case-insensitive).
    """
    ref_in_seq = sequence[offset : offset + len(ref)].upper()
    if ref_in_seq != ref.upper():
        raise ValueError(
            f"Reference mismatch at offset {offset}: "
            f"expected '{ref.upper()}', found '{ref_in_seq}'."
        )
    return sequence[:offset] + alt + sequence[offset + len(ref) :]


class SNPEmbedder:
    """Compute variant-effect embeddings for SNPs using any DNA/protein model.

    This class wraps a loaded ``BaseModelWrapper`` and provides a high-level
    interface for embedding reference and alternate allele contexts, then
    computing delta vectors.

    Parameters
    ----------
    model_wrapper
        An already-loaded ``BaseModelWrapper`` instance.  The model type should
        be ``"dna"`` for DNA SNP analysis or ``"protein"`` for amino-acid
        variant analysis (in which case ``chromosome_sequence`` should contain
        the protein sequence and ``position`` should be an amino-acid position).
    pooling_strategy
        Default pooling strategy forwarded to the wrapper's ``embed`` method.

    Examples
    --------
    >>> embedder = SNPEmbedder(model_wrapper=dnabert2_wrapper)
    >>> snp = SNPContext(position=500, ref_allele="C", alt_alleles=["T"],
    ...                  context_window=256, chrom="chr1", variant_id="rs123")
    >>> result = embedder.embed_snp(snp, chromosome_sequence=chr1_seq)
    >>> print(result.delta_norms)
    """

    def __init__(
        self,
        model_wrapper: BaseModelWrapper,
        pooling_strategy: str = "mean",
    ) -> None:
        self.wrapper = model_wrapper
        self.pooling_strategy = pooling_strategy
        if getattr(model_wrapper, "model", None) is None:
            logging.warning(
                "model_wrapper does not appear to be loaded (model=None). "
                "Call wrapper.load(device) before using SNPEmbedder."
            )

    def _build_sequences(
        self,
        snp: SNPContext,
        chromosome_sequence: str,
    ) -> tuple[str, list[str], int]:
        """Extract ref and alt context sequences from a chromosome string.

        Returns
        -------
        (ref_context, alt_contexts, offset_in_window)
        """
        start, end, offset = _extract_context(
            chromosome_sequence, snp.position, snp.context_window
        )
        ref_context = chromosome_sequence[start:end].upper()

        alt_contexts: list[str] = []
        for alt in snp.alt_alleles:
            try:
                alt_ctx = _apply_snp(ref_context, offset, snp.ref_allele, alt)
            except ValueError as exc:
                logging.warning(
                    f"SNP mismatch for alt '{alt}' at pos {snp.position} "
                    f"(offset {offset} in window): {exc}. Skipping ref-check."
                )

                alt_ctx = (
                    ref_context[:offset]
                    + alt
                    + ref_context[offset + len(snp.ref_allele) :]
                )
            alt_contexts.append(alt_ctx)

        if snp.strand == "-":
            ref_context = _reverse_complement(ref_context)
            alt_contexts = [_reverse_complement(a) for a in alt_contexts]

        return ref_context, alt_contexts, offset

    def embed_snp(
        self,
        snp: SNPContext,
        chromosome_sequence: str,
        pooling_strategy: str | None = None,
        **kwargs: Any,
    ) -> SNPEmbeddingResult:
        """Embed a single SNP and return full result.

        Parameters
        ----------
        snp
            Variant descriptor.
        chromosome_sequence
            The chromosome (or arbitrary genomic region) sequence as a plain
            string.  The ``snp.position`` is relative to the start of this
            string (1-based).  Pass the full chromosome for genome-wide use,
            or a pre-sliced region for efficiency.
        pooling_strategy
            Overrides ``self.pooling_strategy`` for this call only.
        **kwargs
            Additional arguments forwarded to the model's ``embed`` method
            (e.g. ``target_layer``, ``layer_name``).

        Returns
        -------
        SNPEmbeddingResult
        """
        pool = pooling_strategy or self.pooling_strategy

        ref_ctx, alt_ctxs, _ = self._build_sequences(snp, chromosome_sequence)

        logging.debug(
            f"Embedding SNP {snp.variant_id or snp.position} | "
            f"ref_len={len(ref_ctx)} | n_alts={len(alt_ctxs)}"
        )

        ref_emb = self.wrapper.embed(input=ref_ctx, pooling_strategy=pool, **kwargs)
        alt_embs: list[np.ndarray] = []
        for alt_ctx in alt_ctxs:
            alt_embs.append(
                self.wrapper.embed(input=alt_ctx, pooling_strategy=pool, **kwargs)
            )

        return SNPEmbeddingResult(
            snp=snp,
            ref_sequence=ref_ctx,
            alt_sequences=alt_ctxs,
            ref_embedding=np.asarray(ref_emb, dtype=np.float32),
            alt_embeddings=[np.asarray(e, dtype=np.float32) for e in alt_embs],
            model_name=getattr(self.wrapper, "model_name", ""),
            pooling_strategy=pool,
        )

    def embed_snps_batch(
        self,
        snps: list[SNPContext],
        chromosome_sequences: list[str] | dict[str, str],
        pooling_strategy: str | None = None,
        **kwargs: Any,
    ) -> list[SNPEmbeddingResult]:
        """Embed a list of SNPs.

        Parameters
        ----------
        snps
            List of ``SNPContext`` objects.
        chromosome_sequences
            Either a list of chromosome strings aligned 1-to-1 with ``snps``,
            or a ``dict`` mapping chromosome name (``snp.chrom``) to sequence
            string.
        pooling_strategy
            Override pooling strategy for all SNPs in this batch.
        **kwargs
            Forwarded to each ``embed_snp`` call.

        Returns
        -------
        list[SNPEmbeddingResult]
            One result per SNP.  Failed SNPs produce ``None`` entries in the
            log but are skipped (not returned).
        """
        results: list[SNPEmbeddingResult] = []
        for i, snp in enumerate(snps):
            if isinstance(chromosome_sequences, dict):
                chrom_seq = chromosome_sequences.get(snp.chrom, "")
                if not chrom_seq:
                    logging.warning(
                        f"Chromosome '{snp.chrom}' not found in chromosome_sequences dict. "
                        f"Skipping SNP {snp.variant_id or snp.position}."
                    )
                    continue
            else:
                if i >= len(chromosome_sequences):
                    logging.warning(
                        f"No chromosome sequence at index {i}. "
                        f"Skipping SNP {snp.variant_id or snp.position}."
                    )
                    continue
                chrom_seq = chromosome_sequences[i]

            try:
                res = self.embed_snp(
                    snp,
                    chromosome_sequence=chrom_seq,
                    pooling_strategy=pooling_strategy,
                    **kwargs,
                )
                results.append(res)
            except Exception as exc:
                logging.error(
                    f"Failed to embed SNP {snp.variant_id or snp.position}: {exc}"
                )

            if (i + 1) % 50 == 0:
                logging.info(f"SNP batch: processed {i + 1}/{len(snps)}.")

        return results

    def embed_snp_from_vcf_row(
        self,
        chrom: str,
        pos: int,
        ref: str,
        alt: str,
        chromosome_sequence: str,
        context_window: int = 512,
        variant_id: str = "",
        strand: Literal["+", "-"] = "+",
        pooling_strategy: str | None = None,
        **kwargs: Any,
    ) -> SNPEmbeddingResult:
        """Convenience wrapper that accepts VCF-style fields directly.

        Parameters
        ----------
        chrom, pos, ref, alt
            VCF CHROM, POS, REF, ALT columns (POS is 1-based).
        chromosome_sequence
            Full or windowed chromosome sequence.
        context_window
            Size of the context window extracted around the variant.
        variant_id
            Optional rsID or other identifier.
        strand
            Strand of the variant.
        pooling_strategy
            Override default pooling.
        **kwargs
            Forwarded to the embed method.
        """
        snp = SNPContext(
            chrom=chrom,
            position=pos,
            ref_allele=ref,
            alt_alleles=alt.split(","),  # multi-allelic support
            context_window=context_window,
            variant_id=variant_id,
            strand=strand,
        )
        return self.embed_snp(
            snp,
            chromosome_sequence=chromosome_sequence,
            pooling_strategy=pooling_strategy,
            **kwargs,
        )


def embed_vcf(
    vcf_path: str,
    model_wrapper: BaseModelWrapper,
    chromosome_sequences: dict[str, str],
    context_window: int = 512,
    pooling_strategy: str = "mean",
    max_variants: int | None = None,
    **kwargs: Any,
) -> list[SNPEmbeddingResult]:
    """Embed variants from a VCF file.

    Reads a (possibly gzipped) VCF and embeds each variant using
    ``SNPEmbedder``.  Only SNPs and small indels (single-base) are
    supported; multi-allelic records are split into individual alts.

    Parameters
    ----------
    vcf_path
        Path to a VCF (.vcf or .vcf.gz) file.
    model_wrapper
        Loaded ``BaseModelWrapper`` instance.
    chromosome_sequences
        Dict mapping chromosome names (matching the VCF CHROM column, e.g.
        ``"chr1"`` or ``"1"``) to genome sequences.
    context_window
        Context window size around each variant.
    pooling_strategy
        Pooling strategy forwarded to the model.
    max_variants
        Stop after this many variants (useful for testing).
    **kwargs
        Forwarded to the model's ``embed`` method.

    Returns
    -------
    list[SNPEmbeddingResult]
    """
    import gzip

    embedder = SNPEmbedder(model_wrapper, pooling_strategy=pooling_strategy)
    results: list[SNPEmbeddingResult] = []
    n = 0

    opener = gzip.open if vcf_path.endswith(".gz") else open
    with opener(vcf_path, "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            parts = line.rstrip().split("\t")
            if len(parts) < 5:
                continue
            chrom, pos_str, vid, ref, alts_str = parts[:5]
            pos = int(pos_str)
            chrom_seq = chromosome_sequences.get(chrom) or chromosome_sequences.get(
                chrom.lstrip("chr")
            ) or chromosome_sequences.get(f"chr{chrom}")
            if chrom_seq is None:
                logging.warning(f"Chromosome '{chrom}' not in chromosome_sequences. Skipping.")
                continue
            try:
                snp = SNPContext(
                    chrom=chrom,
                    position=pos,
                    ref_allele=ref,
                    alt_alleles=alts_str.split(","),
                    context_window=context_window,
                    variant_id=vid,
                )
                res = embedder.embed_snp(snp, chromosome_sequence=chrom_seq, **kwargs)
                results.append(res)
            except Exception as exc:
                logging.warning(f"Skipping variant {vid} at {chrom}:{pos}: {exc}")

            n += 1
            if max_variants is not None and n >= max_variants:
                break

    logging.info(f"Embedded {len(results)}/{n} variants from {vcf_path}.")
    return results


class SequenceProvider:
    """Fetch genomic sequences for use with :class:`SNPEmbedder`.

    Supports three backends, tried in preference order based on what you
    provide:

    1. **Local FASTA** — a directory of per-chromosome FASTA files
       (e.g. ``chr17.fa``), the same format used by ``GeneResolver``.
    2. **Single FASTA file** — one multi-record FASTA (e.g. the full hg38
       ``hg38.fa``), parsed with BioPython's indexed reader so the whole
       file is never loaded into memory at once.
    3. **Ensembl REST API** — fetches a slice on demand; no local files
       needed, but requires internet and is slow for large windows.

    Parameters
    ----------
    fasta_dir : str, optional
        Directory containing ``chr<N>.fa`` / ``chr<N>.fasta`` files.
        If provided, this is tried first.
    fasta_file : str, optional
        Path to a single multi-chromosome FASTA (e.g. full genome).
        Used when ``fasta_dir`` is not provided or the chromosome is not
        found there.
    genome_build : str
        Ensembl genome build used for the REST API fallback.
        ``"GRCh38"`` (default) or ``"GRCh37"``.
    species : str
        Species name for the Ensembl REST API (default ``"human"``).
    cache : bool
        If ``True``, keep fetched sequences in memory so repeated calls for
        the same chromosome don't re-read the file or hit the API.

    Examples
    --------
    **Option A — local per-chromosome FASTAs (fastest)**::

        provider = SequenceProvider(fasta_dir="/data/genome/hg38/")
        chr17_seq = provider.get_chromosome("chr17")

    **Option B — single genome FASTA**::

        provider = SequenceProvider(fasta_file="/data/hg38.fa")
        chr17_seq = provider.get_chromosome("chr17")

    **Option C — Ensembl REST (no files needed)**::

        provider = SequenceProvider()
        # Fetches only the 512 bp window you actually need — much faster
        window = provider.get_region("17", 7_674_092, 7_674_348)

    **Full SNP embedding workflow**::

        from embpy.snp_utils import SNPEmbedder, SequenceProvider

        provider = SequenceProvider(fasta_dir="/data/genome/hg38/")
        embedder = SNPEmbedder(model_wrapper=dnabert2_wrapper)

        result = embedder.embed_snp_from_vcf_row(
            chrom="chr17",
            pos=7_674_220,
            ref="C",
            alt="T",
            chromosome_sequence=provider.get_chromosome("chr17"),
            context_window=256,
            variant_id="rs28934578",
        )
        print(result.delta_norms)

    **Memory-efficient alternative — fetch only the window**::

        provider = SequenceProvider()   # Ensembl REST fallback
        window, snp_offset = provider.get_window("17", 7_674_220, context=256)

        from embpy.snp_utils import SNPContext, SNPEmbedder
        snp = SNPContext(
            chrom="17",
            position=snp_offset,   # position is now relative to the window
            ref_allele="C",
            alt_alleles=["T"],
            context_window=256,
            variant_id="rs28934578",
        )
        result = SNPEmbedder(dnabert2_wrapper).embed_snp(snp, window)
    """

    def __init__(
        self,
        fasta_dir: str | None = None,
        fasta_file: str | None = None,
        genome_build: str = "GRCh38",
        species: str = "human",
        cache: bool = True,
    ) -> None:
        self.fasta_dir = fasta_dir
        self.fasta_file = fasta_file
        self.genome_build = genome_build
        self.species = species
        self._cache: dict[str, str] = {} if cache else None  # type: ignore[assignment]
        self._fasta_index: Any = None  # BioPython SeqIO index, loaded lazily


    def get_chromosome(self, chrom: str) -> str:
        """Return the full sequence for a chromosome.

        Parameters
        ----------
        chrom
            Chromosome name, e.g. ``"chr17"``, ``"17"``, ``"chrX"``.
            The method tries both ``"chr17"`` and ``"17"`` spellings
            automatically.

        Returns
        -------
        str
            Upper-case nucleotide string for the entire chromosome.

        Raises
        ------
        FileNotFoundError
            If ``fasta_dir`` is set but no matching file is found.
        RuntimeError
            If no backend can provide the sequence.
        """
        key = chrom
        if self._cache is not None and key in self._cache:
            return self._cache[key]

        seq: str | None = None

        if self.fasta_dir:
            seq = self._from_fasta_dir(chrom)

        if seq is None and self.fasta_file:
            seq = self._from_fasta_file(chrom)

        if seq is None:
            raise RuntimeError(
                f"Could not load chromosome '{chrom}' from any backend. "
                "Set fasta_dir or fasta_file, or use get_region() for an "
                "Ensembl REST window fetch."
            )

        seq = seq.upper()
        if self._cache is not None:
            self._cache[key] = seq
        return seq

    def get_region(
        self,
        chrom: str,
        start: int,
        end: int,
    ) -> str:
        """Fetch a genomic region, using local files when available.

        Parameters
        ----------
        chrom
            Chromosome name (``"17"`` or ``"chr17"`` both work).
        start, end
            1-based inclusive coordinates (same as Ensembl REST convention).

        Returns
        -------
        str
            Upper-case nucleotide sequence for the requested region.
        """
        # If we already have the full chromosome cached/available, slice it
        if self.fasta_dir or self.fasta_file:
            try:
                full = self.get_chromosome(chrom)
                return full[start - 1 : end].upper()
            except Exception:
                pass  # fall through to REST

        return self._from_ensembl_rest(chrom, start, end)

    def get_window(
        self,
        chrom: str,
        position: int,
        context: int = 512,
    ) -> tuple[str, int]:
        """Fetch a context window centred on a variant position.

        This is the most memory-efficient approach when you only need a
        small region and don't want to load a whole chromosome.

        Parameters
        ----------
        chrom
            Chromosome name.
        position
            1-based position of the variant on the chromosome.
        context
            Total window size in base pairs.

        Returns
        -------
        (window_sequence, snp_offset_in_window)
            ``snp_offset_in_window`` is the **1-based** position of the
            variant *within* the returned window string — pass it as
            ``SNPContext.position`` when ``chromosome_sequence`` is the
            window rather than the full chromosome.
        """
        half = context // 2
        start = max(1, position - half)
        end = start + context - 1

        seq = self.get_region(chrom, start, end)
        # Re-derive offset in case we hit the chromosome boundary at start=1
        snp_offset_in_window = position - start + 1
        return seq, snp_offset_in_window


    def _normalise_chrom(self, chrom: str) -> tuple[str, str]:
        """Return (with_prefix, without_prefix) variants of a chrom name."""
        if chrom.lower().startswith("chr"):
            return chrom, chrom[3:]
        return f"chr{chrom}", chrom

    def _from_fasta_dir(self, chrom: str) -> str | None:
        """Read a single-chromosome FASTA file from ``self.fasta_dir``."""
        import os

        from Bio import SeqIO  # already a dep via biopython

        with_prefix, without_prefix = self._normalise_chrom(chrom)
        candidates = [
            f"{with_prefix}.fa",
            f"{with_prefix}.fasta",
            f"{without_prefix}.fa",
            f"{without_prefix}.fasta",
        ]
        for fname in candidates:
            path = os.path.join(self.fasta_dir, fname)  # type: ignore[arg-type]
            if os.path.isfile(path):
                logging.info(f"SequenceProvider: reading {path} …")
                rec = SeqIO.read(path, "fasta")
                return str(rec.seq)

        logging.warning(
            f"SequenceProvider: no FASTA file found for '{chrom}' in {self.fasta_dir}. "
            f"Tried: {candidates}"
        )
        return None

    def _from_fasta_file(self, chrom: str) -> str | None:
        """Look up a chromosome record in a multi-record FASTA index."""
        import os

        from Bio import SeqIO

        if self._fasta_index is None:
            logging.info(f"SequenceProvider: indexing {self.fasta_file} …")
            self._fasta_index = SeqIO.index(self.fasta_file, "fasta")

        with_prefix, without_prefix = self._normalise_chrom(chrom)
        for key in (with_prefix, without_prefix, chrom):
            if key in self._fasta_index:
                logging.info(f"SequenceProvider: fetching '{key}' from index …")
                return str(self._fasta_index[key].seq)

        logging.warning(
            f"SequenceProvider: '{chrom}' not found in {self.fasta_file}. "
            f"Available records (first 10): {list(self._fasta_index.keys())[:10]}"
        )
        return None

    def _from_ensembl_rest(self, chrom: str, start: int, end: int) -> str:
        """Fetch a sequence slice from the Ensembl REST API.

        Uses ``/sequence/region/{species}/{chrom}:{start}..{end}``
        which returns only the requested window — no full chromosome download.
        """
        import requests

        _, without_prefix = self._normalise_chrom(chrom)

        coord_system = "chromosome"
        if self.genome_build == "GRCh37":
            base = "https://grch37.rest.ensembl.org"
        else:
            base = "https://rest.ensembl.org"

        url = (
            f"{base}/sequence/region/{self.species}"
            f"/{without_prefix}:{start}..{end}:1"
            f"?content-type=text/plain"
        )
        logging.info(f"SequenceProvider: REST fetch {url} …")
        resp = requests.get(url, timeout=30)
        if not resp.ok:
            raise RuntimeError(
                f"Ensembl REST returned {resp.status_code} for "
                f"{chrom}:{start}-{end}: {resp.text[:200]}"
            )
        return resp.text.strip().upper()
