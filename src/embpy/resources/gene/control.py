"""First-class control / non-targeting policy.

The perturbation pipeline historically conflated three very different
cases:

1. Genuine control / non-targeting guide.
2. Real gene whose Ensembl lookup failed for technical reasons (e.g.
   alias drift like ``AARS`` -> ``AARS1``, transient 4xx from Ensembl).
3. Real gene with a deliberately zero embedding row.

All three ended up as the same zero-filled row downstream, which makes
silent bugs trivial to introduce: an AnnData with thousands of unresolved
symbols is indistinguishable from one with thousands of true controls.

This module makes "is this label a control?" a typed, regex-driven
decision instead of a single hard-coded literal string match. Dataset
authors can either accept the curated defaults or override the patterns
/ extra labels from YAML.

The module is intentionally dependency-free: it must be importable in
environments without ``pyensembl``, ``boltz``, ``arc-state``, etc., so
that the world-model dataloader can classify labels before deciding
whether to call into any heavy embedder.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Literal

__all__ = [
    "DEFAULT_CONTROL_PATTERNS",
    "ControlClassification",
    "ControlPolicy",
]


# Curated regex patterns covering the control / non-targeting / safe-harbor
# conventions seen across Perturb-seq, CROP-seq, and Genome-wide CRISPR
# screens. All patterns are matched case-insensitively against the *whole*
# label after stripping whitespace.
#
# What these intentionally do NOT match (regression-tested in
# tests/test_control_policy.py):
#   * NTRK1, NT5C2  -- real genes that happen to start with "NT"
#   * CTRL2         -- a real gene symbol
#   * CONTROL_GENE_X-- a hypothetical real gene
#   * AAVS1B        -- AAVS1 is a safe-harbor locus; AAVS1B is not a
#                      real gene we have seen but the policy errs on the
#                      side of matching it as control because the literal
#                      "AAVS1" prefix is a strong signal in this corpus.
DEFAULT_CONTROL_PATTERNS: tuple[str, ...] = (
    r"^non[-_]?targeting(_.+)?$",
    r"^NTC([-_].*)?$",
    r"^NT[0-9]+$",
    r"^control([-_].*)?$",
    r"^ctrl([-_].*)?$",
    r"^safe[-_]?harbor([-_].*)?$",
    r"^AAVS1([-_].*)?$",
    r"^scramble[d]?([-_].*)?$",
    r"^empty[-_]?vector([-_].*)?$",
)


@dataclass(frozen=True)
class ControlClassification:
    """Outcome of :meth:`ControlPolicy.classify` for one label.

    ``kind`` is one of ``"control"``, ``"gene"``, ``"mixed"``:

    * ``"control"`` -- every component (after :meth:`split_combo`) is a
      control.
    * ``"gene"``    -- no component is a control; the label should be
      passed to the gene embedder as-is.
    * ``"mixed"``   -- at least one control AND at least one non-control
      component. The world-model contract treats this as "drop the
      controls, keep the genes" but flags it so dataset curators can fix
      ambiguous metadata (this is rare; in practice it happens when a
      sgRNA library co-targets a real gene with a non-targeting guide).
    """

    label: str
    kind: Literal["control", "gene", "mixed"]
    components: tuple[str, ...]
    control_components: tuple[str, ...]
    gene_components: tuple[str, ...]


@dataclass
class ControlPolicy:
    """Pattern-based classifier for perturbation labels.

    Parameters
    ----------
    patterns
        Regex patterns identifying control labels. Defaults to
        :data:`DEFAULT_CONTROL_PATTERNS`. Matching is case-insensitive.
        Pass ``()`` to disable pattern matching entirely (only the
        ``extra_labels`` allow-list will then be consulted).
    extra_labels
        Exact (case-insensitive) labels that must always be treated as
        control. Useful for dataset-specific conventions a regex cannot
        reasonably capture, e.g. a study using the sentinel
        ``"GFP_only"`` for a non-targeting guide.
    strict
        If ``True``, :meth:`classify` raises on a "mixed" label instead
        of returning ``kind="mixed"``. Off by default to avoid breaking
        unusual datasets; turn it on in CI / regression tests.
    """

    patterns: tuple[str, ...] = field(default_factory=lambda: DEFAULT_CONTROL_PATTERNS)
    extra_labels: tuple[str, ...] = field(default_factory=tuple)
    strict: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.patterns, tuple):
            object.__setattr__(self, "patterns", tuple(self.patterns))
        if not isinstance(self.extra_labels, tuple):
            object.__setattr__(self, "extra_labels", tuple(self.extra_labels))
        # Pre-compile patterns once; case-insensitive by contract.
        self._compiled: tuple[re.Pattern[str], ...] = tuple(
            re.compile(p, re.IGNORECASE) for p in self.patterns
        )
        self._extra_lower: frozenset[str] = frozenset(
            x.strip().lower() for x in self.extra_labels if x and x.strip()
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_control(self, label: str | None) -> bool:
        """Return ``True`` iff ``label`` matches any control rule.

        ``None`` and empty strings are treated as control: the dataloader
        emits ``None`` for cells with no recorded perturbation, which is
        operationally identical to "non-targeting".
        """
        if label is None:
            return True
        s = str(label).strip()
        if not s:
            return True
        if s.lower() in self._extra_lower:
            return True
        return any(p.fullmatch(s) for p in self._compiled)

    def split_combo(self, label: str | None) -> list[str]:
        """Split a (possibly multi-gene) label into component symbols.

        Handles the three conventions seen in the wild:

        * ``"TP53,MYC"`` (Replogle / Norman)
        * ``"TP53+MYC"`` (Nadig / Adamson)
        * ``"TP53|MYC"`` (older scperturb exports)

        Whitespace inside components is stripped. Empty components are
        dropped. The order of the returned list reflects the input order
        (the world-model action encoder is order-invariant so this only
        matters for human-readable logging).
        """
        if label is None:
            return []
        s = str(label).strip()
        if not s:
            return []
        parts = re.split(r"[,+|]", s)
        return [p.strip() for p in parts if p and p.strip()]

    def classify(self, label: str | None) -> ControlClassification:
        """Classify ``label`` into control / gene / mixed.

        See :class:`ControlClassification` for semantics. If ``strict``
        is set and the label is mixed, raises :class:`ValueError`.
        """
        components = self.split_combo(label)
        raw = "" if label is None else str(label)

        if not components:
            return ControlClassification(
                label=raw,
                kind="control",
                components=(),
                control_components=(),
                gene_components=(),
            )

        control_components: list[str] = []
        gene_components: list[str] = []
        for c in components:
            if self.is_control(c):
                control_components.append(c)
            else:
                gene_components.append(c)

        if control_components and not gene_components:
            kind: Literal["control", "gene", "mixed"] = "control"
        elif gene_components and not control_components:
            kind = "gene"
        else:
            if self.strict:
                raise ValueError(
                    f"ControlPolicy(strict=True): label {raw!r} mixes "
                    f"control ({control_components}) and gene "
                    f"({gene_components}) components."
                )
            kind = "mixed"

        return ControlClassification(
            label=raw,
            kind=kind,
            components=tuple(components),
            control_components=tuple(control_components),
            gene_components=tuple(gene_components),
        )

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def default(cls) -> ControlPolicy:
        """Return a policy with the curated default patterns and no extras."""
        return cls()

    @classmethod
    def from_iterable(
        cls,
        extra_labels: Iterable[str] = (),
        *,
        patterns: Sequence[str] | None = None,
        strict: bool = False,
    ) -> ControlPolicy:
        """Build a policy from an iterable of extra labels.

        ``patterns`` defaults to :data:`DEFAULT_CONTROL_PATTERNS`; pass
        ``[]`` to disable regex matching entirely.
        """
        pats = (
            DEFAULT_CONTROL_PATTERNS if patterns is None else tuple(patterns)
        )
        return cls(
            patterns=pats,
            extra_labels=tuple(extra_labels),
            strict=strict,
        )
