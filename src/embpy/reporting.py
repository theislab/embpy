"""Per-identifier resolution reports for batch embedding runs (Layer 2).

Motivation
----------
Layer 1 (typed exceptions + structured exit codes) answers the question
"which *category* of failure killed this job?" That's enough for SLURM
sacct triage but it loses information about *which specific inputs*
failed and *why*, when a batch run finishes with partial success.

A concrete example: the Borzoi run on Replogle resolves 2041 / 2057
perturbations, leaving 16 in the UNRESOLVED bucket. Today the only way
to find those 16 symbols is to grep the SLURM log for "No gene
information found" lines, which is fragile and slow. Worse, you cannot
tell whether a symbol failed because of:

* genuine HGNC drift (e.g. ``MARS`` -> ``MARS1``)
* a transient MyGene 503 (already retried -- see Layer 3)
* the embedder's ``--max-length`` cut applied to the wrong sequence
* a partial MyGene hit with missing ``summary`` field

Layer 2 fixes that by accumulating a ``ResolutionRecord`` per input as
the resolver / embedder runs, and serialising the collection as a JSON
sidecar next to the embedding NPZ. Downstream tooling (analysis
notebooks, the audit log emitted by ``embed_perturbations.py``, custom
QA scripts) can then answer questions like "which symbols failed for
which reason?" with ``jq`` instead of regex.

Design choices
--------------
* **Records are append-only and minimal.** Each record is one input's
  fate; aggregation happens at JSON-emission time. No locking, no
  rolling counters -- the surrounding loop is single-threaded in
  embpy.
* **Records can be merged.** ``ResolutionReport.merge(other)`` is the
  primitive used when a higher layer composes multiple sub-resolves
  (e.g. ``BioEmbedderProvider`` resolves the unique gene list, but
  also probes a fallback gene during dimensionality probing -- both
  passes contribute records and the merged report is what reaches the
  sidecar).
* **The status vocabulary mirrors Layer 1.** ``status`` is a short
  enum string -- ``"resolved"``, ``"unresolved"``, ``"cached"``,
  ``"control"`` -- not free text. ``reason`` is free text, used only
  when ``status != "resolved"``.
* **No required dependencies beyond stdlib.** ``json``, ``dataclasses``,
  ``time``. This keeps the reporting module importable from any pixi
  environment without dragging the heavy embedder dependency tree.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

logger = logging.getLogger(__name__)


ResolutionStatus = Literal[
    "resolved",
    "unresolved",
    "cached",
    "control",
    "skipped",
]


@dataclass(slots=True)
class ResolutionRecord:
    """One input's fate during a batch embedding pass.

    Attributes
    ----------
    identifier
        The input identifier as passed by the caller (gene symbol,
        Ensembl ID, SMILES, ...).
    status
        Outcome bucket. ``"resolved"`` means a vector was produced;
        ``"unresolved"`` means the resolver returned nothing usable;
        ``"cached"`` means the vector came from disk cache rather than
        a forward pass; ``"control"`` means the input was classified
        as a non-targeting / control label and routed to the sentinel;
        ``"skipped"`` is reserved for explicit dataset-author opt-outs.
    source
        What produced the answer. For resolved inputs this is the name
        of the data source ("mygene", "ensembl", "uniprot_fallback",
        "disk_cache", "model_forward"). For unresolved inputs it is
        the *last* source tried.
    reason
        Free-text explanation. Set when ``status != "resolved"``.
        Convention: ``"<source>:<short_code>"`` -- e.g.
        ``"mygene:no_hit"``, ``"mygene:empty_summary"``,
        ``"resolver:http_503_after_retries"``,
        ``"model:context_overflow"``. The short code is meant to be
        machine-greppable; the human-readable expansion lives in the
        log line emitted at the call site.
    latency_ms
        End-to-end latency for this input from the caller's
        perspective (typically dominated by the resolver HTTP roundtrip
        and the model forward pass). Recorded so post-hoc audits can
        spot slow upstream APIs without rerunning the job.
    attempted_sources
        Ordered list of every source consulted for this input. Useful
        for tracing fallback chains: if a gene is resolved by
        ``ensembl`` only after ``mygene`` returned no_hit, both names
        appear here in the order tried. Empty for cache hits.
    metadata
        Opaque per-record bag for extensions (e.g. ``{"n_chunks": 4,
        "mean_pool": true}`` recorded by the chunking helper). Kept
        loose on purpose so callers can attach context without
        bumping the schema.
    """

    identifier: str
    status: ResolutionStatus
    source: str | None = None
    reason: str | None = None
    latency_ms: float = 0.0
    attempted_sources: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Stable JSON shape -- safe to embed in sidecars."""
        d = asdict(self)
        # Round latency to microsecond precision; full float precision
        # is noise here.
        d["latency_ms"] = round(float(self.latency_ms), 3)
        return d


@dataclass(slots=True)
class ResolutionReport:
    """Accumulator for ``ResolutionRecord`` instances.

    Construct one per batch, call :meth:`record` once per input, and
    serialise via :meth:`to_dict` / :meth:`write_sidecar` when the
    batch finishes.
    """

    model_name: str | None = None
    organism: str | None = None
    started_at: float = field(default_factory=time.time)
    records: list[ResolutionRecord] = field(default_factory=list)

    # ----- accumulation primitives -----

    def record(
        self,
        identifier: str,
        status: ResolutionStatus,
        *,
        source: str | None = None,
        reason: str | None = None,
        latency_ms: float = 0.0,
        attempted_sources: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ResolutionRecord:
        """Append one record and return it (useful for chaining logs)."""
        rec = ResolutionRecord(
            identifier=str(identifier),
            status=status,
            source=source,
            reason=reason,
            latency_ms=float(latency_ms),
            attempted_sources=list(attempted_sources or []),
            metadata=dict(metadata or {}),
        )
        self.records.append(rec)
        return rec

    def merge(self, other: ResolutionReport) -> None:
        """In-place merge: append records from ``other`` after our own."""
        if other is self:
            return
        self.records.extend(other.records)
        if self.model_name is None:
            self.model_name = other.model_name
        if self.organism is None:
            self.organism = other.organism

    # ----- query / aggregation -----

    def count_by_status(self) -> dict[str, int]:
        """Histogram of ``status`` values across all records."""
        out: dict[str, int] = {}
        for r in self.records:
            out[r.status] = out.get(r.status, 0) + 1
        return out

    def count_by_source(self) -> dict[str, int]:
        """Histogram of ``source`` values, restricted to ``resolved`` records."""
        out: dict[str, int] = {}
        for r in self.records:
            if r.status not in ("resolved", "cached"):
                continue
            key = r.source or "unknown"
            out[key] = out.get(key, 0) + 1
        return out

    def count_by_reason(self) -> dict[str, int]:
        """Histogram of ``reason`` codes, restricted to non-resolved records."""
        out: dict[str, int] = {}
        for r in self.records:
            if r.status == "resolved":
                continue
            key = r.reason or "unspecified"
            out[key] = out.get(key, 0) + 1
        return out

    def unresolved(self) -> list[ResolutionRecord]:
        return [r for r in self.records if r.status == "unresolved"]

    def latency_summary(self) -> dict[str, float]:
        """Compact p50/p95/max latency stats in ms.

        ``records`` may be empty; an empty dict is returned in that
        case rather than throwing, so callers can dump unconditionally.
        """
        if not self.records:
            return {}
        latencies = sorted(r.latency_ms for r in self.records)
        n = len(latencies)

        def pct(p: float) -> float:
            # Nearest-rank percentile; good enough for n in [50, 5000].
            idx = max(0, min(n - 1, int(round((p / 100.0) * (n - 1)))))
            return latencies[idx]

        return {
            "n": n,
            "p50_ms": round(pct(50), 3),
            "p95_ms": round(pct(95), 3),
            "max_ms": round(latencies[-1], 3),
            "total_ms": round(sum(latencies), 3),
        }

    # ----- serialisation -----

    def to_dict(self, *, include_records: bool = True) -> dict[str, Any]:
        """Stable JSON shape for sidecar files.

        The aggregates (``counts``, ``by_source``, ``by_reason``,
        ``latency``) are always included because they're cheap and
        the most-frequently inspected fields. Set
        ``include_records=False`` to drop the per-input array if
        sidecar size matters.
        """
        finished_at = time.time()
        return {
            "model_name": self.model_name,
            "organism": self.organism,
            "started_at": self.started_at,
            "finished_at": finished_at,
            "wallclock_s": round(finished_at - self.started_at, 3),
            "n_total": len(self.records),
            "counts": self.count_by_status(),
            "by_source": self.count_by_source(),
            "by_reason": self.count_by_reason(),
            "latency": self.latency_summary(),
            **(
                {"records": [r.to_dict() for r in self.records]}
                if include_records else {}
            ),
        }

    def write_sidecar(
        self,
        npz_path: str | Path,
        *,
        suffix: str = ".resolution.json",
        include_records: bool = True,
    ) -> Path:
        """Write the report next to an NPZ artifact.

        Returns the sidecar path. The default suffix is
        ``.resolution.json`` (distinct from the existing
        ``.status.json`` written by
        ``world_model.scripts.embed_perturbations`` so the two can
        co-exist; the former is per-identifier, the latter is the
        rolled-up status summary).
        """
        path = Path(str(npz_path) + suffix)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(
            self.to_dict(include_records=include_records),
            indent=2, default=str,
        ))
        return path


__all__ = ["ResolutionRecord", "ResolutionReport", "ResolutionStatus"]
