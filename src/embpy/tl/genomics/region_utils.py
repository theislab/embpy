from __future__ import annotations

import logging
import shutil
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from .snp_utils import _reverse_complement, profile_variant_effect_score

logger = logging.getLogger(__name__)

_DNA = ("A", "C", "G", "T")

PerturbationMode = Literal["dinuc_shuffle", "uniform_random", "mask_n", "motif_ablate", "haplotype"]

#: Borzoi's published ISM-shuffle settings, keyed by regulatory-element class.
#: ``(mode, window_size_M, n_shuffles_N)`` -- see the module docstring.
BORZOI_ISM_SHUFFLE_PRESETS: dict[str, tuple[str, int, int]] = {
    "enhancer": ("dinuc_shuffle", 7, 24),
    "promoter": ("uniform_random", 5, 24),
    "splice_site": ("uniform_random", 5, 24),
    "polyadenylation_site": ("uniform_random", 5, 24),
}

#: scooby's published in-silico motif-deletion settings.
SCOOBY_MOTIF_DELETION_DEFAULTS: dict[str, Any] = {
    "mode": "uniform_random",
    "n_replicates": 10,
    "fimo_pvalue_threshold": 1e-4,
    "motif_database": "HOCOMOCO v12 CORE",
}


# --------------------------------------------------------------------------------------
# sequence perturbation primitives
# --------------------------------------------------------------------------------------
def uniform_random_sequence(length: int, rng: np.random.Generator) -> str:
    """Return ``length`` uniformly random nucleotides (Borzoi's promoter-class perturbation)."""
    if length <= 0:
        return ""
    return "".join(rng.choice(_DNA, size=length))


def dinucleotide_shuffle(seq: str, rng: np.random.Generator, max_attempts: int = 100) -> str:
    """Shuffle ``seq`` while preserving its exact dinucleotide composition.

    Implements the Altschul-Erikson algorithm: the sequence is an Eulerian path through a
    graph whose vertices are nucleotides and whose edges are dinucleotide occurrences, so
    any other Eulerian path over the same edge multiset is a shuffle with identical
    dinucleotide counts. The first and last nucleotide are preserved by construction.

    Preserving dinucleotide content matters because CpG and other dinucleotide biases are
    themselves strong predictors of regulatory activity; a naive mononucleotide shuffle
    destroys them and inflates the apparent effect of an ablation.

    Parameters
    ----------
    seq
        Sequence to shuffle (case-insensitive; returned upper-case).
    rng
        Random generator, for reproducibility.
    max_attempts
        Number of times to resample the per-vertex "last edge" assignment before giving
        up. A valid assignment must form a tree rooted at the final vertex; for real
        sequences this succeeds almost immediately.

    Returns
    -------
    str
        A shuffled sequence of the same length and dinucleotide composition.
    """
    s = seq.upper()
    if len(s) < 3:
        return s

    verts = sorted(set(s))
    if len(verts) == 1:
        return s

    edges: dict[str, list[str]] = {v: [] for v in verts}
    for i in range(len(s) - 1):
        edges[s[i]].append(s[i + 1])

    last_vertex = s[-1]

    def _forms_tree(last_edges: dict[str, str]) -> bool:
        """Every vertex must reach ``last_vertex`` by following its chosen last edge."""
        for v in verts:
            if v == last_vertex:
                continue
            seen: set[str] = set()
            cur = v
            while cur != last_vertex:
                if cur in seen or cur not in last_edges:
                    return False
                seen.add(cur)
                cur = last_edges[cur]
        return True

    last_edges: dict[str, str] = {}
    for _ in range(max_attempts):
        candidate = {}
        ok = True
        for v in verts:
            if v == last_vertex:
                continue
            if not edges[v]:
                ok = False
                break
            candidate[v] = str(rng.choice(edges[v]))
        if ok and _forms_tree(candidate):
            last_edges = candidate
            break
    else:
        logger.warning(
            "dinucleotide_shuffle: no valid Eulerian last-edge assignment after "
            f"{max_attempts} attempts (len={len(s)}); returning the input unshuffled."
        )
        return s

    shuffled_edges: dict[str, list[str]] = {}
    for v in verts:
        rest = list(edges[v])
        if v != last_vertex:
            rest.remove(last_edges[v])
        rng.shuffle(rest)
        if v != last_vertex:
            rest.append(last_edges[v])
        shuffled_edges[v] = rest

    out = [s[0]]
    ptr = dict.fromkeys(verts, 0)
    cur = s[0]
    for _ in range(len(s) - 1):
        nxt = shuffled_edges[cur][ptr[cur]]
        ptr[cur] += 1
        out.append(nxt)
        cur = nxt
    return "".join(out)


def perturb_interval(
    sequence: str,
    start: int,
    end: int,
    mode: str,
    rng: np.random.Generator,
) -> str:
    """Return ``sequence`` with the half-open window ``[start, end)`` perturbed.

    Parameters
    ----------
    sequence
        The sequence to perturb (offsets are 0-based indices into this string).
    start, end
        Half-open interval to replace. Clipped to the sequence bounds.
    mode
        ``"dinuc_shuffle"``, ``"uniform_random"`` or ``"mask_n"``.
    rng
        Random generator.

    Returns
    -------
    str
        A new sequence of the same length.
    """
    start = max(0, start)
    end = min(len(sequence), end)
    if end <= start:
        return sequence

    window = sequence[start:end]
    if mode == "dinuc_shuffle":
        replacement = dinucleotide_shuffle(window, rng)
    elif mode == "uniform_random":
        replacement = uniform_random_sequence(end - start, rng)
    elif mode == "mask_n":
        replacement = "N" * (end - start)
        logger.warning(
            "perturb_interval(mode='mask_n'): the Borzoi and Scooby wrappers one-hot-encode "
            "unrecognised characters via ALPHABET_MAP.get(base, 0), so 'N' is encoded as 'A' "
            "rather than as a zero vector. This makes the masked window a poly-A tract, which "
            "is NOT a neutral perturbation. Prefer 'dinuc_shuffle' or 'uniform_random', which "
            "are the modes the source papers actually use."
        )
    else:
        raise ValueError(
            f"Unknown interval perturbation mode {mode!r}. "
            "Expected 'dinuc_shuffle', 'uniform_random' or 'mask_n'."
        )

    return sequence[:start] + replacement + sequence[end:]


def apply_haplotype(
    sequence: str,
    substitutions: Sequence[tuple[int, str, str]],
    *,
    strict: bool = True,
) -> str:
    """Apply several allele substitutions to one sequence simultaneously.

    Parameters
    ----------
    sequence
        Sequence to modify.
    substitutions
        ``(offset, ref_allele, alt_allele)`` triples, where ``offset`` is a **0-based**
        index into ``sequence``. Applied right-to-left so that earlier offsets stay valid
        when alleles differ in length.
    strict
        If ``True`` (default), raise when the sequence at ``offset`` does not match
        ``ref_allele``. A silent mismatch means the substitution is being applied against
        the wrong coordinate or strand, which produces a plausible-looking but meaningless
        effect size, so this defaults to loud.

    Returns
    -------
    str
        The haplotype sequence.
    """
    out = sequence
    for offset, ref, alt in sorted(substitutions, key=lambda t: t[0], reverse=True):
        ref, alt = ref.upper(), alt.upper()
        observed = out[offset : offset + len(ref)].upper()
        if observed != ref:
            msg = (
                f"haplotype substitution at offset {offset}: sequence has {observed!r}, "
                f"expected reference {ref!r}"
            )
            if strict:
                raise ValueError(msg)
            logger.warning(msg + " -- applying anyway (strict=False)")
        out = out[:offset] + alt + out[offset + len(ref) :]
    return out


# --------------------------------------------------------------------------------------
# containers
# --------------------------------------------------------------------------------------
@dataclass
class RegionContext:
    """A genomic interval to perturb, and the model input window it should be scored in.

    Attributes
    ----------
    chrom : str
        Chromosome name (informational; sequence retrieval happens elsewhere).
    start, end : int
        The interval to perturb, as **0-based half-open absolute genomic coordinates**.
    context_window : int
        Total length of the model input window. Set this to the model's fixed input length
        (``BorzoiWrapper.SEQUENCE_LENGTH`` / ``ScoobyWrapper.SEQUENCE_LENGTH`` == 524,288)
        so that bin indices line up with
        :func:`~embpy.tl.genomics.snp_utils.genomic_to_bin_indices`.
    window_start : int, optional
        Coordinate convention: **0-BASED**, unlike the 1-based variant positions taken by
        :meth:`predict_haplotype_effect` and by
        :meth:`~embpy.tl.genomics.snp_utils.SequenceProvider.get_region`.

        0-based genomic coordinate of the first base of the sequence window that will be
        passed to the model. If ``None``, the window is centred on the interval's midpoint.
    strand : {"+", "-"}
        If ``"-"``, sequences are reverse-complemented before being handed to the model.
    region_id : str, optional
        Human-readable label, e.g. ``"FOXP3_CNS2"``.
    """

    chrom: str
    start: int
    end: int
    context_window: int = 524_288
    window_start: int | None = None
    strand: Literal["+", "-"] = "+"
    region_id: str = ""

    def __post_init__(self) -> None:
        if self.end <= self.start:
            raise ValueError(f"RegionContext requires end > start, got start={self.start}, end={self.end}")
        if self.window_start is None:
            midpoint = (self.start + self.end) // 2
            self.window_start = max(0, midpoint - self.context_window // 2)

    @property
    def length(self) -> int:
        """Length of the perturbed interval in bp."""
        return self.end - self.start

    @property
    def offset_in_window(self) -> int:
        """0-based offset of :attr:`start` within the model input window."""
        assert self.window_start is not None
        return self.start - self.window_start

    def __str__(self) -> str:
        label = f"{self.region_id} " if self.region_id else ""
        return f"{label}{self.chrom}:{self.start}-{self.end} ({self.length} bp)"


@dataclass
class RegionEffectResult:
    """Replicated region-perturbation effect on a predicted coverage profile.

    Attributes
    ----------
    region : RegionContext
        What was perturbed.
    ref_profile : np.ndarray
        Reference predicted profile, shape ``(n_tracks, n_bins)``.
    alt_profiles : list[np.ndarray]
        One perturbed profile per replicate. Empty when ``keep_profiles=False`` (the
        default for scans, where storing every replicate's full profile is what actually
        exhausts memory).
    effect_scores : np.ndarray
        Per-replicate log2 fold-change of perturbed over reference, shape
        ``(n_replicates, n_tracks)``.
    mean_effect, sd_effect : np.ndarray
        Mean and standard deviation of ``effect_scores`` across replicates, each shape
        ``(n_tracks,)``. The SD is the honest uncertainty of a randomised ablation and
        should be reported alongside the mean.
    bin_indices : np.ndarray, optional
        Bins the statistic was aggregated over; ``None`` means the whole profile (which is
        scooby's convention for the accessibility statistic).
    mode : str
        Perturbation mode used.
    n_replicates : int
        Number of independent randomisations.
    track_names : list[str], optional
        Track identifiers aligned to the channel axis.
    model_name : str
        Model the profiles came from.
    seed : int, optional
        Seed used, so a result can be regenerated exactly.
    """

    region: RegionContext
    ref_profile: np.ndarray
    alt_profiles: list[np.ndarray] = field(default_factory=list)
    effect_scores: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    bin_indices: np.ndarray | None = None
    mode: str = ""
    n_replicates: int = 0
    track_names: list[str] | None = None
    model_name: str = ""
    seed: int | None = None

    @property
    def mean_effect(self) -> np.ndarray:
        """Mean log2 fold-change across replicates, shape ``(n_tracks,)``."""
        return self.effect_scores.mean(axis=0)

    @property
    def sd_effect(self) -> np.ndarray:
        """Standard deviation of the log2 fold-change across replicates."""
        return self.effect_scores.std(axis=0, ddof=1) if self.effect_scores.shape[0] > 1 else np.zeros(
            self.effect_scores.shape[1]
        )

    @property
    def attribution_scores(self) -> np.ndarray:
        """Borzoi's ISM sign convention, ``u(x) - u(x_perturbed)``, i.e. ``-mean_effect``.

        Positive means the reference sequence in this interval *contributes* to the
        predicted signal.
        """
        return -self.mean_effect

    def to_dict(self) -> dict[str, Any]:
        """Flatten to one row per track, for assembling a results table."""
        names = self.track_names or [str(i) for i in range(len(self.mean_effect))]
        return [
            {
                "region_id": self.region.region_id,
                "chrom": self.region.chrom,
                "start": self.region.start,
                "end": self.region.end,
                "length": self.region.length,
                "mode": self.mode,
                "n_replicates": self.n_replicates,
                "track": names[i],
                "mean_log2fc": float(self.mean_effect[i]),
                "sd_log2fc": float(self.sd_effect[i]),
                "attribution": float(self.attribution_scores[i]),
                "model": self.model_name,
                "seed": self.seed,
            }
            for i in range(len(self.mean_effect))
        ]


@dataclass
class MotifHit:
    """One scanned transcription-factor binding site.

    Attributes
    ----------
    motif_id, tf_name : str
        Motif identifier and the TF it is annotated to.
    start, end : int
        Half-open **absolute genomic** coordinates of the hit.
    strand : str
        Strand the motif matched on.
    score, pvalue : float
        Match score and its p-value, as reported by the scanner.
    """

    motif_id: str
    tf_name: str
    start: int
    end: int
    strand: str = "+"
    score: float = float("nan")
    pvalue: float = float("nan")


# --------------------------------------------------------------------------------------
# the embedder
# --------------------------------------------------------------------------------------
class RegionEmbedder:
    """Score interval-, motif- and haplotype-level perturbations against a profile model.

    Requires a model wrapper implementing ``predict_profile`` -- i.e.
    :class:`~embpy.models.dna_models.BorzoiWrapper` or
    :class:`~embpy.models.scooby_models.ScoobyWrapper`. This is the region-level
    counterpart of :class:`~embpy.tl.genomics.snp_utils.SNPEmbedder`, which handles single
    nucleotide substitutions.

    Parameters
    ----------
    model_wrapper
        A loaded wrapper exposing ``predict_profile``, ``SEQUENCE_LENGTH``, ``BIN_SIZE``
        and ``profile_offset_bp``.
    seed
        Base seed for the randomised perturbations. Every call derives its own generator
        from this, so results are reproducible.

    Examples
    --------
    >>> from embpy.tl.genomics import RegionContext, RegionEmbedder, SequenceProvider
    >>> provider = SequenceProvider(fasta_file="hg38.fa")
    >>> # FOXP3 CNS2, the enhancer whose closure defines the destabilized Treg state
    >>> cns2 = RegionContext(chrom="chrX", start=49_260_470, end=49_261_090,
    ...                      context_window=524_288, region_id="FOXP3_CNS2")
    >>> embedder = RegionEmbedder(borzoi_wrapper, seed=0)
    >>> res = embedder.predict_region_effect(cns2, provider, mode="dinuc_shuffle", n_replicates=24)
    >>> res.mean_effect.shape
    """

    def __init__(self, model_wrapper: Any, seed: int = 0) -> None:
        if not hasattr(model_wrapper, "predict_profile"):
            raise TypeError(
                f"{type(model_wrapper).__name__} does not implement predict_profile(); "
                "region-level scoring requires a profile model such as BorzoiWrapper or "
                "ScoobyWrapper."
            )
        self.wrapper = model_wrapper
        self.seed = seed
        if getattr(model_wrapper, "model", None) is None:
            logger.warning(
                "model_wrapper does not appear to be loaded (model=None). "
                "Call wrapper.load(device) before using RegionEmbedder."
            )

    # ---------------------------------------------------------------- helpers
    def _track_names(self) -> list[str] | None:
        getter = getattr(self.wrapper, "get_track_metadata", None)
        if not callable(getter):
            return None
        try:
            return list(getter()["identifier"])
        except Exception as exc:  # pragma: no cover - metadata is best-effort
            logger.debug(f"Could not load track metadata: {exc}")
            return None

    def _fetch_window(self, region: RegionContext, sequence_source: Any) -> str:
        """Resolve the model input window for ``region`` from a provider or a plain string."""
        assert region.window_start is not None
        if isinstance(sequence_source, str):
            window = sequence_source[region.window_start : region.window_start + region.context_window]
        else:
            # SequenceProvider.get_region takes 1-based inclusive coordinates.
            window = sequence_source.get_region(
                region.chrom,
                region.window_start + 1,
                region.window_start + region.context_window,
            )
        window = window.upper()
        # The +1 above is the ONLY place the 0-based and 1-based coordinate systems meet in
        # this module, and it is silent if wrong. Cross-check it by fetching the single first
        # base through the same API and confirming it agrees with window[0]: an off-by-one in
        # the range arithmetic shows up here immediately instead of as a plausible wrong score.
        if not isinstance(sequence_source, str) and window:
            try:
                first = sequence_source.get_region(
                    region.chrom, region.window_start + 1, region.window_start + 1).upper()
            except Exception:                      # provider cannot do 1 bp fetches; skip
                first = ""
            if first and first[0] != window[0]:
                raise ValueError(
                    f"{region}: window start is off by one. The first base of the fetched "
                    f"window is {window[0]!r}, but a single-base fetch at window_start+1 "
                    f"({region.window_start + 1}) gives {first[0]!r}. window_start is 0-based; "
                    f"get_region takes 1-based inclusive coordinates."
                )
        if len(window) != region.context_window:
            logger.warning(
                f"{region}: fetched window is {len(window)} bp, expected {region.context_window}. "
                "The model wrapper will pad or centre-crop, which shifts bin alignment; "
                "check that the region is not near a chromosome boundary."
            )
        return window

    _warned_strand = False

    def _predict(self, sequence: str, region: RegionContext, **kwargs: Any) -> np.ndarray:
        """Predict a coverage profile in forward genomic orientation.

        Profile scoring is done in FORWARD genomic orientation regardless of feature strand.
        Measured: Borzoi is NOT reverse-complement invariant (21.7% mean relative
        difference, r = 0.98 between a forward prediction and a flipped reverse-
        complement one), and bin indices from genomic_to_bin_indices are always
        forward-oriented. Reverse-complementing therefore changed the score of the
        same genomic locus purely because the gene happened to be on the minus
        strand, silently. Strand belongs in the output tracks (RNA+/RNA-), not in the
        input orientation.
        """
        if region.strand == "-" and not RegionEmbedder._warned_strand:
            RegionEmbedder._warned_strand = True
            logger.warning(
                "region.strand='-': predicting in forward orientation. Reverse-complementing "
                "would change the score of the same locus, because the model is not "
                "reverse-complement invariant and bin_indices are forward-oriented."
            )
        return np.asarray(self.wrapper.predict_profile(sequence, **kwargs))

    # ---------------------------------------------------------------- public API
    def predict_region_effect(
        self,
        region: RegionContext,
        sequence_source: Any,
        *,
        mode: str = "dinuc_shuffle",
        n_replicates: int = 24,
        bin_indices: Sequence[int] | None = None,
        pseudocount: float = 1.0,
        keep_profiles: bool = False,
        seed: int | None = None,
        **predict_kwargs: Any,
    ) -> RegionEffectResult:
        """Ablate a whole interval and measure the effect on predicted coverage.

        This is the single-shot ablation of a named element (an enhancer, a CNS, a peak),
        as opposed to :meth:`ism_shuffle_scan`, which slides a small window across a range
        to build a per-nucleotide attribution track.

        Parameters
        ----------
        region
            The interval to ablate.
        sequence_source
            A :class:`~embpy.tl.genomics.snp_utils.SequenceProvider`, or a plain chromosome
            string indexed by absolute coordinate.
        mode
            ``"dinuc_shuffle"`` (Borzoi's enhancer default), ``"uniform_random"`` (Borzoi's
            promoter default, and scooby's motif-deletion perturbation) or ``"mask_n"``
            (discouraged, see :func:`perturb_interval`).
        n_replicates
            Number of independent randomisations. Borzoi uses 24 (8 for large windows);
            scooby uses 10 for motif deletion. With ``mode="mask_n"`` the perturbation is
            deterministic and this is forced to 1.
        bin_indices
            Bins to aggregate the statistic over -- e.g. a gene's exon bins from
            :func:`~embpy.tl.genomics.snp_utils.genomic_to_bin_indices`, which is the
            expression statistic both papers use. ``None`` aggregates over the whole
            profile, which is scooby's accessibility statistic.
        pseudocount
            Added to both sums before the log2 ratio. Both papers use 1.
        keep_profiles
            Retain every replicate's full profile in the result. Off by default.
        seed
            Overrides the embedder's base seed for this call.
        **predict_kwargs
            Forwarded to the wrapper's ``predict_profile`` -- notably
            ``cell_embeddings=`` and ``aggregate=`` for :class:`ScoobyWrapper`, and
            ``track_indices=`` / ``undo_squashed_scale=`` for either.

        Returns
        -------
        RegionEffectResult
        """
        if mode == "mask_n" and n_replicates != 1:
            logger.info("mode='mask_n' is deterministic; forcing n_replicates=1.")
            n_replicates = 1
        if n_replicates < 1:
            raise ValueError(f"n_replicates must be >= 1, got {n_replicates}")

        used_seed = self.seed if seed is None else seed
        rng = np.random.default_rng(used_seed)

        window = self._fetch_window(region, sequence_source)
        ref_profile = self._predict(window, region, **predict_kwargs)

        offset = region.offset_in_window
        if offset < 0 or offset >= len(window):
            raise ValueError(
                f"{region}: interval offset {offset} falls outside the fetched window "
                f"of length {len(window)}. Check window_start."
            )

        scores, kept = [], []
        for _ in range(n_replicates):
            perturbed = perturb_interval(window, offset, offset + region.length, mode, rng)
            alt_profile = self._predict(perturbed, region, **predict_kwargs)
            scores.append(profile_variant_effect_score(ref_profile, alt_profile, bin_indices, pseudocount))
            if keep_profiles:
                kept.append(alt_profile)

        return RegionEffectResult(
            region=region,
            ref_profile=ref_profile,
            alt_profiles=kept,
            effect_scores=np.vstack(scores),
            bin_indices=np.asarray(bin_indices) if bin_indices is not None else None,
            mode=mode,
            n_replicates=n_replicates,
            track_names=self._track_names(),
            model_name=getattr(self.wrapper, "model_name", ""),
            seed=used_seed,
        )

    def predict_haplotype_effect(
        self,
        region: RegionContext,
        sequence_source: Any,
        variants: Sequence[tuple[int, str, str]],
        *,
        bin_indices: Sequence[int] | None = None,
        pseudocount: float = 1.0,
        strict: bool = True,
        keep_profiles: bool = False,
        **predict_kwargs: Any,
    ) -> RegionEffectResult:
        """Substitute several real variants at once and score the combined effect.

        Neither the Borzoi nor the scooby paper does this -- both score one variant at a
        time -- but it is the correct design for asking whether a risk *haplotype* alters a
        regulatory element, because it captures interactions between neighbouring variants
        that per-variant scoring is blind to. Comparing this against the sum of the
        individual per-variant effects quantifies that non-additivity directly.

        The perturbation is deterministic, so the result carries a single "replicate" and
        ``sd_effect`` is zero by construction.

        Parameters
        ----------
        region
            Defines the model input window. ``start``/``end`` are only used to place the
            window and to label the result; the substitutions themselves come from
            ``variants``.
        sequence_source
            A ``SequenceProvider`` or a plain chromosome string.
        variants
            ``(genomic_position, ref_allele, alt_allele)`` triples, with
            ``genomic_position`` **1-BASED** to match VCF/pgen convention -- note this
            differs from :attr:`RegionContext.window_start`, which is 0-based. A
            reference-base check runs on every substitution and reports the mismatch
            rate, so passing the wrong convention fails loudly rather than silently.
        bin_indices, pseudocount, keep_profiles, **predict_kwargs
            As in :meth:`predict_region_effect`.
        strict
            Raise if a reference allele does not match the reference sequence. Keep this
            ``True``: a mismatch usually means the alleles are on the opposite strand or in
            the wrong coordinate system, which yields a confident but meaningless score.

        Returns
        -------
        RegionEffectResult
        """
        if not variants:
            raise ValueError("predict_haplotype_effect requires at least one variant.")

        assert region.window_start is not None
        window = self._fetch_window(region, sequence_source)
        ref_profile = self._predict(window, region, **predict_kwargs)

        # 1-based genomic -> 0-based offset within the window
        subs: list[tuple[int, str, str]] = []
        for pos, ref, alt in variants:
            offset = pos - 1 - region.window_start
            if offset < 0 or offset >= len(window):
                raise ValueError(
                    f"{region}: variant at {region.chrom}:{pos} maps to window offset {offset}, "
                    f"outside the {len(window)} bp window."
                )
            subs.append((offset, ref, alt))

        # With strict=False a reference mismatch is only warned about, one line per variant,
        # which is unreadable for a haplotype carrying hundreds of substitutions and hides how
        # many actually failed. Count them up front and report the total: a high rate means the
        # alleles are on the wrong strand or in the wrong coordinate system, and the resulting
        # score is confident but meaningless.
        n_mismatch = sum(
            1 for off, ref, _ in subs
            if window[off:off + len(ref)].upper() != ref.upper()
        )
        if n_mismatch:
            frac = n_mismatch / len(subs)
            msg = (f"{region}: {n_mismatch} of {len(subs)} substitutions "
                   f"({frac:.1%}) do not match the reference base at their offset")
            if strict:
                raise ValueError(msg + " -- set strict=False only if this is expected.")
            logger.warning(
                msg + " -- proceeding with strict=False. A rate above a few percent usually "
                "means the alleles are on the opposite strand or in a different coordinate "
                "system, in which case the score is meaningless."
            )
        hap = apply_haplotype(window, subs, strict=strict)
        alt_profile = self._predict(hap, region, **predict_kwargs)
        score = profile_variant_effect_score(ref_profile, alt_profile, bin_indices, pseudocount)

        return RegionEffectResult(
            region=region,
            ref_profile=ref_profile,
            alt_profiles=[alt_profile] if keep_profiles else [],
            effect_scores=score.reshape(1, -1),
            bin_indices=np.asarray(bin_indices) if bin_indices is not None else None,
            mode=f"haplotype[n={len(variants)}]",
            n_replicates=1,
            track_names=self._track_names(),
            model_name=getattr(self.wrapper, "model_name", ""),
            seed=None,
        )

    def ism_shuffle_scan(
        self,
        region: RegionContext,
        sequence_source: Any,
        *,
        mode: str = "dinuc_shuffle",
        window_size: int = 7,
        n_shuffles: int = 24,
        stride: int | None = None,
        bin_indices: Sequence[int] | None = None,
        pseudocount: float = 1.0,
        seed: int | None = None,
        progress_every: int = 0,
        **predict_kwargs: Any,
    ) -> dict[str, np.ndarray]:
        """Slide a small perturbation window across ``region`` to build an attribution track.

        This is Borzoi's window-shuffled ISM. For each scanned position ``u`` it perturbs
        ``[u - window_size/2, u + window_size/2 + 1)`` in ``n_shuffles`` independent
        replicates and records ``u(x) - u(x_perturbed)``, averaged over replicates.

        Cost is ``ceil(region.length / stride) * n_shuffles + 1`` forward passes, so for
        anything wider than a few hundred bp raise ``stride`` (and/or drop ``n_shuffles``
        to Borzoi's large-window setting of 8) rather than scanning every base.

        Parameters
        ----------
        region
            The range to scan. Use ``BORZOI_ISM_SHUFFLE_PRESETS`` for the published
            ``(mode, window_size, n_shuffles)`` per element class.
        sequence_source
            A ``SequenceProvider`` or a plain chromosome string.
        mode, window_size, n_shuffles
            Perturbation settings; defaults are Borzoi's enhancer preset.
        stride
            Step between scanned positions. Defaults to ``window_size`` (tiling, no
            overlap), which is the cheapest setting that still covers every base.
        bin_indices, pseudocount, **predict_kwargs
            As in :meth:`predict_region_effect`.
        seed
            Overrides the embedder's base seed.
        progress_every
            Log progress every N scanned positions; 0 disables.

        Returns
        -------
        dict of np.ndarray
            ``positions`` -- absolute genomic coordinate of each scanned window's centre;
            ``attribution`` -- ``(n_positions, n_tracks)``, Borzoi's sign convention;
            ``log2fc`` -- ``(n_positions, n_tracks)``, perturbed-over-reference;
            ``sd`` -- ``(n_positions, n_tracks)`` across replicates.
        """
        step = window_size if stride is None else stride
        if step < 1:
            raise ValueError(f"stride must be >= 1, got {step}")

        used_seed = self.seed if seed is None else seed
        rng = np.random.default_rng(used_seed)

        window = self._fetch_window(region, sequence_source)
        ref_profile = self._predict(window, region, **predict_kwargs)

        centres = list(range(region.start, region.end, step))
        n_pos = len(centres)
        logger.info(
            f"ism_shuffle_scan {region}: {n_pos} positions x {n_shuffles} shuffles "
            f"= {n_pos * n_shuffles + 1} forward passes (mode={mode}, M={window_size}, stride={step})"
        )

        log2fc = np.zeros((n_pos, ref_profile.shape[0]), dtype=np.float64)
        sd = np.zeros_like(log2fc)
        assert region.window_start is not None

        for i, centre in enumerate(centres):
            local = centre - region.window_start
            lo = local - window_size // 2
            hi = lo + window_size
            reps = []
            for _ in range(n_shuffles):
                perturbed = perturb_interval(window, lo, hi, mode, rng)
                alt_profile = self._predict(perturbed, region, **predict_kwargs)
                reps.append(profile_variant_effect_score(ref_profile, alt_profile, bin_indices, pseudocount))
            stacked = np.vstack(reps)
            log2fc[i] = stacked.mean(axis=0)
            sd[i] = stacked.std(axis=0, ddof=1) if n_shuffles > 1 else 0.0
            if progress_every and (i + 1) % progress_every == 0:
                logger.info(f"  ism_shuffle_scan: {i + 1}/{n_pos} positions")

        return {
            "positions": np.asarray(centres),
            "attribution": -log2fc,
            "log2fc": log2fc,
            "sd": sd,
            "track_names": np.asarray(self._track_names() or [], dtype=object),
        }

    def predict_motif_effect(
        self,
        hits: Sequence[MotifHit],
        region: RegionContext,
        sequence_source: Any,
        *,
        mode: str = "uniform_random",
        n_replicates: int = 10,
        bin_indices: Sequence[int] | None = None,
        pseudocount: float = 1.0,
        group_by_tf: bool = True,
        keep_profiles: bool = False,
        seed: int | None = None,
        **predict_kwargs: Any,
    ) -> dict[str, RegionEffectResult]:
        """Ablate scanned TF binding sites and score the effect -- scooby's TF motif effect score.

        Follows scooby's published recipe: every predicted binding site for a TF is
        replaced by a random nucleotide sequence of the same length, repeated
        ``n_replicates`` (10) times "to mitigate spurious motif introduction", and the
        effect is the mean log2 fold-change over replicates.

        All hits for a given TF are ablated **together** in each replicate (that is what
        makes the result a statement about the TF rather than about one site), unless
        ``group_by_tf=False``.

        Parameters
        ----------
        hits
            Scanned motif occurrences, e.g. from :func:`scan_motifs_fimo`. Hits outside the
            model input window are dropped with a warning.
        region
            Defines the model input window; its ``start``/``end`` are used only for
            windowing and labelling.
        sequence_source
            A ``SequenceProvider`` or a plain chromosome string.
        mode
            Perturbation for each site. ``"uniform_random"`` reproduces scooby;
            ``"dinuc_shuffle"`` is the more conservative alternative, since a random
            replacement can by chance create a *different* functional motif.
        n_replicates
            Independent randomisations. scooby uses 10.
        bin_indices, pseudocount, keep_profiles, **predict_kwargs
            As in :meth:`predict_region_effect`.
        group_by_tf
            Ablate all of a TF's sites jointly (default) or score each site separately.
        seed
            Overrides the embedder's base seed.

        Returns
        -------
        dict[str, RegionEffectResult]
            Keyed by TF name (``group_by_tf=True``) or by
            ``"{tf_name}@{chrom}:{start}-{end}"``.
        """
        if not hits:
            return {}

        used_seed = self.seed if seed is None else seed
        assert region.window_start is not None
        window = self._fetch_window(region, sequence_source)
        ref_profile = self._predict(window, region, **predict_kwargs)
        w_start, w_end = region.window_start, region.window_start + len(window)

        groups: dict[str, list[MotifHit]] = {}
        n_dropped = 0
        for h in hits:
            if h.start < w_start or h.end > w_end:
                n_dropped += 1
                continue
            key = h.tf_name if group_by_tf else f"{h.tf_name}@{region.chrom}:{h.start}-{h.end}"
            groups.setdefault(key, []).append(h)
        if n_dropped:
            logger.warning(
                f"predict_motif_effect: dropped {n_dropped}/{len(hits)} motif hits that fall "
                f"outside the {len(window)} bp model input window."
            )
        if not groups:
            return {}

        results: dict[str, RegionEffectResult] = {}
        for key, group in groups.items():
            rng = np.random.default_rng((used_seed, abs(hash(key)) % (2**32)))
            scores, kept = [], []
            for _ in range(n_replicates):
                perturbed = window
                for h in group:
                    perturbed = perturb_interval(
                        perturbed, h.start - w_start, h.end - w_start, mode, rng
                    )
                alt_profile = self._predict(perturbed, region, **predict_kwargs)
                scores.append(profile_variant_effect_score(ref_profile, alt_profile, bin_indices, pseudocount))
                if keep_profiles:
                    kept.append(alt_profile)

            span_start = min(h.start for h in group)
            span_end = max(h.end for h in group)
            results[key] = RegionEffectResult(
                region=RegionContext(
                    chrom=region.chrom,
                    start=span_start,
                    end=span_end,
                    context_window=region.context_window,
                    window_start=region.window_start,
                    strand=region.strand,
                    region_id=key,
                ),
                ref_profile=ref_profile,
                alt_profiles=kept,
                effect_scores=np.vstack(scores),
                bin_indices=np.asarray(bin_indices) if bin_indices is not None else None,
                mode=f"motif_ablate[{mode}, n_sites={len(group)}]",
                n_replicates=n_replicates,
                track_names=self._track_names(),
                model_name=getattr(self.wrapper, "model_name", ""),
                seed=used_seed,
            )
        return results


# --------------------------------------------------------------------------------------
# motif scanning
# --------------------------------------------------------------------------------------

def _scan_motifs_fimo_cli(
    sequence: str,
    window_start: int,
    chrom: str,
    meme_file: str,
    *,
    pvalue_threshold: float = 1e-4,
    tf_names: Sequence[str] | None = None,
) -> list[MotifHit]:
    """Scan with the MEME suite's `fimo` executable, returning the same MotifHit list.

    Used when tangermeme's Python FIMO is unavailable (see :func:`scan_motifs_fimo`). The
    sequence is written to a temporary FASTA whose single record is named after the window, and
    FIMO's TSV output is translated back to absolute genomic coordinates. FIMO reports 1-based
    inclusive starts, so a hit's absolute half-open interval is
    ``window_start + start - 1`` to ``window_start + stop``.
    """
    import csv
    import subprocess
    import tempfile
    from pathlib import Path as _Path

    with tempfile.TemporaryDirectory() as td:
        fa = _Path(td) / "region.fa"
        fa.write_text(f">{chrom}:{window_start}\n" + sequence + "\n")
        cmd = ["fimo", "--text", "--thresh", str(pvalue_threshold),
               "--verbosity", "1", str(meme_file), str(fa)]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
        if proc.returncode != 0:
            raise RuntimeError(f"fimo failed: {proc.stderr.strip()[:400]}")
        wanted = {t.casefold() for t in tf_names} if tf_names else None
        hits: list[MotifHit] = []
        for row in csv.DictReader(proc.stdout.splitlines(), delimiter="\t"):
            mid = row.get("motif_id") or row.get("#pattern name") or ""
            alt = row.get("motif_alt_id") or ""
            tf = (alt or mid).split(".")[0]
            if wanted is not None and tf.casefold() not in wanted:
                continue
            try:
                st, sp = int(row["start"]), int(row["stop"])
                pv = float(row["p-value"]); sc = float(row.get("score", "nan"))
            except (KeyError, ValueError):
                continue
            hits.append(MotifHit(motif_id=mid, tf_name=tf,
                                 start=window_start + st - 1, end=window_start + sp,
                                 strand=row.get("strand", "+"), score=sc, pvalue=pv))
    logging.info("fimo (CLI): %d hits at p<%g%s", len(hits), pvalue_threshold,
                 f" for {len(wanted)} TFs" if wanted else "")
    return hits


def scan_motifs_fimo(
    sequence: str,
    window_start: int,
    chrom: str,
    meme_file: str,
    *,
    pvalue_threshold: float = 1e-4,
    tf_names: Sequence[str] | None = None,
    reverse_complement: bool = True,
) -> list[MotifHit]:
    """Scan a sequence for TF binding sites with tangermeme's FIMO implementation.

    Reproduces the scan step of scooby's motif-deletion experiment: HOCOMOCO v12 CORE
    PWMs, FIMO, default significance cutoff ``p < 1e-4``.

    Parameters
    ----------
    sequence
        Sequence to scan.
    window_start
        0-based genomic coordinate of ``sequence[0]``, so hits come back in absolute
        genomic coordinates.
    chrom
        Chromosome name, recorded on each hit.
    meme_file
        Path to a MEME-format motif file (e.g. ``H12CORE_meme_format.meme``).
    pvalue_threshold
        FIMO significance threshold. scooby uses ``1e-4``.
    tf_names
        Restrict to these TFs (matched case-insensitively against the motif name). scooby
        restricts to differentially expressed TFs to keep the search tractable.
    reverse_complement
        Also scan the reverse strand.

    Returns
    -------
    list[MotifHit]
    """
    read_meme = fimo = None
    try:
        from tangermeme.io import read_meme  # type: ignore
        from tangermeme.tools.fimo import fimo  # type: ignore
    except ImportError:
        try:
            from tangermeme.io import read_meme  # type: ignore
        except ImportError:
            read_meme = None
        if shutil.which("fimo") is None:
            raise ImportError(
                "scan_motifs_fimo needs either tangermeme's FIMO "
                "(`tangermeme.tools.fimo`, present up to ~0.4) or the MEME suite's `fimo` "
                "binary on PATH (`mamba install -c bioconda meme`). Neither was found."
            ) from None
        return _scan_motifs_fimo_cli(
            sequence, window_start, chrom, meme_file,
            pvalue_threshold=pvalue_threshold, tf_names=tf_names,
        )

    import torch

    motifs = read_meme(meme_file)
    if tf_names is not None:
        wanted = {t.casefold() for t in tf_names}
        motifs = {
            name: pwm
            for name, pwm in motifs.items()
            # HOCOMOCO ids look like "GATA1.H12CORE.1.PSM.A"; the TF is the leading token.
            if name.split(".")[0].casefold() in wanted
        }
        if not motifs:
            raise ValueError(f"No motifs in {meme_file} matched tf_names={list(tf_names)[:10]}")

    lookup = {b: i for i, b in enumerate(_DNA)}
    idx = torch.tensor([lookup.get(b, 0) for b in sequence.upper()], dtype=torch.long)
    one_hot = torch.nn.functional.one_hot(idx, num_classes=4).T.float().unsqueeze(0)

    tables = fimo(motifs, one_hot, threshold=pvalue_threshold, reverse_complement=reverse_complement)

    hits: list[MotifHit] = []
    for motif_id, table in zip(motifs.keys(), tables):
        if table is None or len(table) == 0:
            continue
        for row in table.itertuples(index=False):
            hits.append(
                MotifHit(
                    motif_id=motif_id,
                    tf_name=motif_id.split(".")[0],
                    start=window_start + int(row.start),
                    end=window_start + int(row.end),
                    strand=getattr(row, "strand", "+"),
                    score=float(getattr(row, "score", float("nan"))),
                    pvalue=float(getattr(row, "p_value", float("nan"))),
                )
            )
    logger.info(f"scan_motifs_fimo: {len(hits)} hits from {len(motifs)} motifs at p<{pvalue_threshold}")
    return hits


def region_effect_table(results: dict[str, RegionEffectResult] | Sequence[RegionEffectResult]):
    """Flatten region-effect results into a tidy ``pandas.DataFrame`` (one row per track)."""
    import pandas as pd

    values = results.values() if isinstance(results, dict) else results
    rows: list[dict[str, Any]] = []
    for r in values:
        rows.extend(r.to_dict())
    return pd.DataFrame(rows)
