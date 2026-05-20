"""List training runs as a compact table.

Scans a ``runs`` directory (recursively, one level deep by default) for
``run_info.json`` files written by ``train.py`` and prints a sorted
table with status, jobid, key hyperparameters, and final metrics.

Usage:

    python -m world_model.scripts.list_runs
    python -m world_model.scripts.list_runs --root runs/world_model
    python -m world_model.scripts.list_runs --columns run_name,status,slurm.job_id,key_hyperparams.optim.lr,final_metrics.final_val_loss
    python -m world_model.scripts.list_runs --csv runs_summary.csv
    python -m world_model.scripts.list_runs --filter 'key_hyperparams.data.sequence_length=32'

Notes:

* Only runs that have a ``run_info.json`` are listed. Older runs from
  before run-tracking was introduced can be backfilled by adding a
  minimal stub ``run_info.json`` to their directory.
* ``--columns`` accepts dotted paths into the JSON (e.g.
  ``slurm.job_id`` or ``final_metrics.eval.r2``).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any


DEFAULT_COLUMNS: list[str] = [
    "status",
    "started_at",
    "run_name",
    "slurm.job_id",
    "key_hyperparams.data.dataset",
    "key_hyperparams.data.sequence_length",
    "key_hyperparams.data.batch_size",
    "key_hyperparams.data.sequence_bucket_key",
    "key_hyperparams.optim.lr",
    "key_hyperparams.loss.info_nce",
    "key_hyperparams.loss.action_counterfactual",
    "final_metrics.final_epoch",
    "final_metrics.final_train_loss",
    "final_metrics.final_val_loss",
    "git.short_sha",
    "git.dirty",
    "output_dir",
]


def _dig(obj: Any, dotted: str) -> Any:
    """Walk a dotted path into a nested dict/list. Returns None on miss."""
    cur: Any = obj
    for part in dotted.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return None
    return cur


def _find_run_info(root: Path) -> list[Path]:
    """Return all ``run_info.json`` files anywhere under ``root``."""
    if not root.exists():
        return []
    return sorted(root.rglob("run_info.json"))


def _short(val: Any, width: int = 40) -> str:
    """Stringify and truncate so the table stays readable."""
    if val is None:
        return "-"
    if isinstance(val, float):
        s = f"{val:.4g}"
    else:
        s = str(val)
    if len(s) > width:
        s = s[: width - 1] + "..."
    return s


def _format_table(rows: list[dict[str, str]], columns: list[str]) -> str:
    if not rows:
        return "(no runs found)"
    widths = {c: max(len(c), *(len(r.get(c, "-")) for r in rows)) for c in columns}
    header = " | ".join(c.ljust(widths[c]) for c in columns)
    sep = "-+-".join("-" * widths[c] for c in columns)
    body = "\n".join(
        " | ".join(r.get(c, "-").ljust(widths[c]) for c in columns) for r in rows
    )
    return f"{header}\n{sep}\n{body}"


def _parse_filter(expr: str) -> tuple[str, str]:
    """Parse a single ``dotted.path=value`` filter."""
    if "=" not in expr:
        raise SystemExit(f"--filter must be 'path=value', got: {expr!r}")
    key, _, val = expr.partition("=")
    return key.strip(), val.strip()


def _matches(info: dict[str, Any], filters: list[tuple[str, str]]) -> bool:
    for key, expected in filters:
        actual = _dig(info, key)
        if actual is None:
            return False
        if str(actual) != expected:
            return False
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path("runs"),
        help="Root directory to scan for run_info.json (default: runs).",
    )
    parser.add_argument(
        "--columns", type=str, default=None,
        help="Comma-separated dotted column paths to display.",
    )
    parser.add_argument(
        "--csv", type=Path, default=None,
        help="Also write the table to this CSV file.",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Emit the full list as a JSON array (ignores --columns / --csv).",
    )
    parser.add_argument(
        "--filter", action="append", default=[],
        help="Filter rows; repeat for AND. Example: --filter status=completed.",
    )
    parser.add_argument(
        "--sort", type=str, default="started_at",
        help="Dotted path to sort by (default: started_at).",
    )
    parser.add_argument(
        "--reverse", action="store_true",
        help="Reverse the sort order (newest first when sorting by time).",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Show only the top N rows after sorting.",
    )
    args = parser.parse_args(argv)

    columns = (
        [c.strip() for c in args.columns.split(",") if c.strip()]
        if args.columns else DEFAULT_COLUMNS
    )
    filters = [_parse_filter(e) for e in args.filter]

    infos: list[dict[str, Any]] = []
    for path in _find_run_info(args.root):
        try:
            data = json.loads(path.read_text())
        except Exception as exc:  # noqa: BLE001
            print(f"warning: failed to read {path}: {exc}", file=sys.stderr)
            continue
        data.setdefault("_path", str(path))
        if _matches(data, filters):
            infos.append(data)

    infos.sort(key=lambda d: (_dig(d, args.sort) is None, _dig(d, args.sort) or ""))
    if args.reverse:
        infos.reverse()
    if args.limit is not None:
        infos = infos[: args.limit]

    if args.json:
        print(json.dumps(infos, indent=2, default=str))
        return 0

    rows = [{c: _short(_dig(info, c)) for c in columns} for info in infos]
    print(_format_table(rows, columns))
    print(f"\n{len(rows)} run(s) under {args.root}")

    if args.csv:
        with open(args.csv, "w", newline="") as fp:
            writer = csv.DictWriter(fp, fieldnames=columns)
            writer.writeheader()
            for r in rows:
                writer.writerow(r)
        print(f"wrote {args.csv}")

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
