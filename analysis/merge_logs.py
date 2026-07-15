#!/usr/bin/env python3
"""Merge per-process MP-path access logs (and/or single-process fallback
logs) into one globally-ordered trace.

Milestone 2.4 of .claude/plans/2026-07-14-rung0-multiworker-headroom-gate.md.

Two source scenarios:

  1. MP path (run_sweep_mp.sh, LMCACHE_ACCESS_LOG=1): up to 4 files per N=2
     run (1 lmcache-server CPU-tier file + 1 API-server routing file + 1
     scheduler/GPU-eviction file per DP worker), each already
     globally-ordered on its own (single writer per file, monotonic
     per-file seq). Every event already carries its own rank field —
     "kv_rank" (CPU-tier chunk + offload events; a TP/PP shard index, NOT a
     DP identity — see lmcache's access_log.py module docstring) or
     "dp_rank" (vLLM-tier route/sched/gpu_* events; the real DP replica
     index). These are NOT comparable to each other and are passed through
     as-is — no synthetic unified "worker_id" is invented for these events.
  2. Fallback (N independent single-GPU run_sweep.sh runs, one per worker):
     N files, each single-process LMCACHE_ACCESS_LOG output
     (_AccessLogger schema — store/hit/evict only, no rank field at all,
     seq restarts at 0 per file). Each such file must be paired with an
     explicit worker id on the command line (see --help), injected as a
     "worker_id" field — the *only* place this merge tool adds that field,
     since it's the only schema with no in-band rank of its own.

Merge algorithm: read every event from every input, sort ALL of them by
wall-clock "t" (the only field comparable across independently-started
processes), then re-assign a fresh globally-monotonic "seq" in that order.

Sanity checks (see --skip-checks to disable):
  - per-input-file seq is monotonically non-decreasing as written (catches a
    truncated/corrupted/reordered source file before it pollutes the merge).
  - no orphan "serve" demand event without a preceding, unmatched "wait" for
    the same req_id within its source file (catches a source file that was
    started/stopped mid-request); only applies to the fallback schema, which
    is the only one that emits wait/serve.

Usage:
    # MP path: pass every per-process file from one run, no ":worker_id" needed
    python3 analysis/merge_logs.py \\
        mp/cpu_access.20260715_060814.5737.jsonl \\
        mp/cpu_access.20260715_060814.6001.jsonl \\
        mp/cpu_access.20260715_060814.6002.jsonl \\
        -o merged.jsonl

    # Fallback: N single-process files, one worker id per file
    python3 analysis/merge_logs.py \\
        worker0/cpu_access.jsonl:0 worker1/cpu_access.jsonl:1 \\
        -o merged.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Ops the single-process fallback schema (_AccessLogger) always emits and
# that never carry any rank field of their own — these need the CLI-supplied
# default_worker_id. "store"/"hit"/"evict" are ambiguous (both schemas use
# these op names): disambiguated by "kv_rank" presence instead, since the
# MP-path versions of those three ops always carry it. Every other op
# (lookup, offload_*, prefetch_*, route, sched_*, gpu_*) is MP-path/vLLM-tier
# only and never needs a synthesized worker_id, regardless of which fields
# it happens to carry (e.g. "lookup"/"prefetch_lookup" have no rank field at
# all, but are not fallback events).
_ALWAYS_FALLBACK_OPS = frozenset({"wait", "serve"})
_AMBIGUOUS_CHUNK_OPS = frozenset({"store", "hit", "evict"})


def _needs_worker_id(ev: dict[str, Any]) -> bool:
    """Whether *ev* is single-process fallback schema and needs a synthesized
    "worker_id" (see module docstring's two source scenarios)."""
    op = ev["op"]
    if op in _ALWAYS_FALLBACK_OPS:
        return True
    if op in _AMBIGUOUS_CHUNK_OPS:
        return "kv_rank" not in ev
    return False


def parse_source_arg(arg: str) -> tuple[Path, int | None]:
    """Parse a "path" or "path:worker_id" positional argument.

    Args:
        arg: Either a bare file path, or "path:worker_id" (worker_id must be
            an integer). The colon form is only needed for source files that
            do not already carry a "worker_id" field per event.

    Returns:
        (path, worker_id) — worker_id is None when not supplied.

    Raises:
        ValueError: worker_id is present but not an integer.
    """
    if ":" in arg:
        raw_path, _, raw_worker_id = arg.rpartition(":")
        try:
            return Path(raw_path), int(raw_worker_id)
        except ValueError:
            raise ValueError(
                f"'{arg}': worker_id suffix must be an integer, got {raw_worker_id!r}"
            ) from None
    return Path(arg), None


def load_source(path: Path, default_worker_id: int | None) -> list[dict[str, Any]]:
    """Load one JSONL access-log file, checking per-file seq monotonicity.

    Args:
        path: Path to a cpu_access*.jsonl file.
        default_worker_id: Injected as "worker_id" into events identified as
            single-process fallback schema by ``_needs_worker_id`` (see
            module docstring). MP-path/vLLM-tier events already carry their
            own correct rank field and are passed through unmodified —
            "worker_id" is never synthesized for them.

    Returns:
        The file's events, each tagged with "_source" (str(path), for error
        messages) in insertion (original seq) order.

    Raises:
        ValueError: seq is not monotonically non-decreasing within the file,
            or a rank-less (fallback-schema) event was found with no
            default_worker_id given.
    """
    events: list[dict[str, Any]] = []
    prev_seq = -1
    with open(path) as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            ev = json.loads(line)
            if ev["seq"] < prev_seq:
                raise ValueError(
                    f"{path}:{lineno}: seq went backwards "
                    f"({ev['seq']} after {prev_seq}) — corrupted or "
                    f"out-of-order source file"
                )
            prev_seq = ev["seq"]
            if _needs_worker_id(ev):
                if default_worker_id is None:
                    raise ValueError(
                        f"{path}: op={ev['op']!r} event is single-process "
                        f"fallback schema and no default worker id was "
                        f"given — pass 'path:worker_id' for these logs"
                    )
                ev["worker_id"] = default_worker_id
            ev["_source"] = str(path)
            events.append(ev)
    return events


def check_no_orphan_serve(events: list[dict[str, Any]], source: str) -> None:
    """Verify every "serve" demand event has a preceding unmatched "wait".

    Args:
        events: One source file's events, in original (seq) order.
        source: The file path, for the error message.

    Raises:
        ValueError: a "serve" event's req_id has no open "wait".
    """
    open_waits: set[str] = set()
    for ev in events:
        if ev["op"] == "wait":
            open_waits.add(ev["req_id"])
        elif ev["op"] == "serve":
            if ev["req_id"] not in open_waits:
                raise ValueError(
                    f"{source}: orphan 'serve' for req_id={ev['req_id']!r} "
                    f"(seq={ev['seq']}) with no preceding unmatched 'wait'"
                )
            open_waits.discard(ev["req_id"])


def merge(
    sources: list[tuple[Path, int | None]], skip_checks: bool
) -> list[dict[str, Any]]:
    """Load, validate, and merge all sources into one seq-renumbered trace.

    Args:
        sources: (path, default_worker_id) pairs, as returned by
            parse_source_arg.
        skip_checks: When True, skip the per-file seq-monotonicity and
            orphan-serve checks (they still run per-file; this only disables
            raising on failure — a warning is printed instead).

    Returns:
        All events from all sources, sorted by wall-clock "t", each with a
        freshly assigned globally-monotonic "seq" (old per-file seq is kept
        under "_orig_seq" for traceability) and "_source" dropped.
    """
    all_events: list[dict[str, Any]] = []
    for path, default_worker_id in sources:
        try:
            events = load_source(path, default_worker_id)
        except ValueError as e:
            if skip_checks:
                print(f"WARNING: {e}", file=sys.stderr)
                # Best-effort: reload without the seq check by treating every
                # line independently; still requires worker_id resolvable for
                # rank-less (fallback-schema) events.
                events = []
                with open(path) as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        ev = json.loads(line)
                        if _needs_worker_id(ev):
                            ev.setdefault("worker_id", default_worker_id)
                        ev["_source"] = str(path)
                        events.append(ev)
            else:
                raise

        if any(ev["op"] in ("wait", "serve") for ev in events):
            try:
                check_no_orphan_serve(events, str(path))
            except ValueError as e:
                if skip_checks:
                    print(f"WARNING: {e}", file=sys.stderr)
                else:
                    raise

        all_events.extend(events)

    all_events.sort(key=lambda e: e["t"])
    for new_seq, ev in enumerate(all_events):
        ev["_orig_seq"] = ev.pop("seq")
        ev["seq"] = new_seq
        ev.pop("_source", None)
    return all_events


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "sources",
        nargs="+",
        help="One or more 'path' or 'path:worker_id' cpu_access.jsonl inputs. "
        "The ':worker_id' suffix is required only for single-process "
        "fallback logs (events with no 'kv_rank'/'dp_rank' field); MP-path "
        "and vLLM-tier files carry their own rank and don't need it.",
    )
    parser.add_argument(
        "-o", "--output", required=True, help="Path to write the merged JSONL to."
    )
    parser.add_argument(
        "--skip-checks",
        action="store_true",
        help="Warn instead of aborting on a failed sanity check.",
    )
    args = parser.parse_args()

    try:
        sources = [parse_source_arg(s) for s in args.sources]
    except ValueError as e:
        parser.error(str(e))

    try:
        merged = merge(sources, args.skip_checks)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for ev in merged:
            f.write(json.dumps(ev) + "\n")

    n_files = len(sources)
    ops = sorted({ev["op"] for ev in merged})
    dp_ranks = sorted({ev["dp_rank"] for ev in merged if "dp_rank" in ev})
    fallback_workers = sorted({ev["worker_id"] for ev in merged if "worker_id" in ev})
    print(
        f"Merged {n_files} file(s), {len(merged)} events, ops={ops}, "
        f"dp_ranks={dp_ranks}"
        + (f", fallback_worker_ids={fallback_workers}" if fallback_workers else "")
        + f" -> {out_path}"
    )


if __name__ == "__main__":
    main()
