#!/usr/bin/env python3
"""Offline frequency-signal bake-off for LMCache CPU access logs.

Step 1 of docs/plans/2026-07-04-reuse-scored-cpu-eviction.md: before touching any
serving code, ask whether an *observed-frequency* signal (on its own) already
closes the LRU->OPT gap that QSLRU leaves behind. Replays frequency-family CPU
eviction policies on the demand sequence of an existing cpu_access.jsonl, next to
the LRU baseline and the OPT ceiling that belady_sim already computes.

This does NOT modify belady_sim.py — it imports and reuses its loaders and its
LRU/OPT simulators, and adds the extra policies + a "% of the LRU->OPT gap closed"
column.

Policies replayed here (each a pure function of the policy-independent demand
sequence, so it can run on any run's log regardless of the policy that produced it):
    LRU     recency only (belady_sim.simulate_lru)              -- baseline
    LFU     frequency only, oldest-first on ties                -- "pure frequency"
    LFUDA   LFU with dynamic aging (recency x frequency blend)  -- does aging help?
    OPT     Belady MIN (belady_sim.simulate_opt)                -- ceiling

Fan-out / prefix-structure is intentionally NOT here: the current log has only
chunk_hash (no parent linkage), so true fan-out needs the log enrichment in Step 2
of the plan. This module answers the frequency half of the bake-off.

Usage:
    python3 analysis/policy_bakeoff.py <run_dir>/cpu_access.*.jsonl --budget-gb 30
    python3 analysis/policy_bakeoff.py <run_dir> --budget-gb 30   # auto-selects log

Outputs:
    <run_dir>/bakeoff_report_cpu<N>.json   machine-readable results
    stdout                                  human-readable table
"""
import argparse
import heapq
import json
import sys
from pathlib import Path
from typing import Any, Callable

# Reuse belady_sim's log loaders and LRU/OPT simulators without modifying it.
from belady_sim import (
    load_events,
    demand_sequence,
    simulate_lru,
    simulate_opt,
)


# ---------------------------------------------------------------------------
# Frequency-family policy simulators
# ---------------------------------------------------------------------------

def simulate_lfu(demand: list[dict], budget_bytes: int) -> dict[str, Any]:
    """LFU: evict the least-frequently-used resident chunk, oldest-first on ties.

    Frequency is counted only while a chunk is resident (no ghost history): a
    re-stored chunk restarts at count 1, matching an online policy that keeps no
    metadata for evicted chunks. This is the "pure frequency" signal.

    Lazy-deletion min-heap keyed on (freq, last_used, chunk_hash); stale entries
    (freq/last_used changed since push) are skipped on pop -- same trick as
    belady_sim.simulate_opt.
    """
    cache: dict[int, int] = {}   # chunk_hash -> size_bytes
    freq:  dict[int, int] = {}   # resident access count
    last:  dict[int, int] = {}   # last-access tick (recency tie-break)
    used = 0
    hits = misses = evictions = 0
    stored_bytes = hit_bytes = 0

    heap: list[tuple[int, int, int]] = []  # (freq, last_used, chunk_hash)

    for tick, ev in enumerate(demand):
        h = ev["chunk_hash"]
        size = ev["size_bytes"]

        if h in cache:
            hits += 1
            hit_bytes += cache[h]
            freq[h] += 1
            last[h] = tick
            heapq.heappush(heap, (freq[h], last[h], h))  # old entry now stale
        else:
            misses += 1
            stored_bytes += size
            while used + size > budget_bytes and cache:
                while heap:
                    f, l, victim = heap[0]
                    if victim in cache and freq[victim] == f and last[victim] == l:
                        break
                    heapq.heappop(heap)
                else:
                    break
                heapq.heappop(heap)
                evictions += 1
                used -= cache.pop(victim)
                del freq[victim]
                del last[victim]
            cache[h] = size
            used += size
            freq[h] = 1
            last[h] = tick
            heapq.heappush(heap, (freq[h], last[h], h))

    total = hits + misses
    return {
        "hits": hits,
        "misses": misses,
        "evictions": evictions,
        "hit_rate": hits / total if total else 0.0,
        "write_amp": stored_bytes / hit_bytes if hit_bytes else None,
    }


def simulate_lfuda(demand: list[dict], budget_bytes: int) -> dict[str, Any]:
    """LFU with Dynamic Aging: a recency x frequency blend.

    Each chunk's priority is ``count + age``; ``age`` is a global clock set to the
    priority of the most recently evicted chunk. Aging lets a once-hot chunk decay
    out over time (recency) while frequency still protects genuinely shared
    prefixes -- the question is whether blending recency into LFU beats pure LFU.

    Lazy-deletion min-heap keyed on (priority, last_used, chunk_hash).
    """
    cache: dict[int, int]   = {}   # chunk_hash -> size_bytes
    cnt:   dict[int, int]   = {}   # resident access count
    pri:   dict[int, float] = {}   # current priority = cnt + age
    last:  dict[int, int]   = {}   # last-access tick (tie-break)
    used = 0
    age = 0.0
    hits = misses = evictions = 0
    stored_bytes = hit_bytes = 0

    heap: list[tuple[float, int, int]] = []  # (priority, last_used, chunk_hash)

    def _touch(h: int, tick: int) -> None:
        cnt[h] = cnt.get(h, 0) + 1
        pri[h] = cnt[h] + age
        last[h] = tick
        heapq.heappush(heap, (pri[h], last[h], h))

    for tick, ev in enumerate(demand):
        h = ev["chunk_hash"]
        size = ev["size_bytes"]

        if h in cache:
            hits += 1
            hit_bytes += cache[h]
            _touch(h, tick)  # old heap entry now stale
        else:
            misses += 1
            stored_bytes += size
            while used + size > budget_bytes and cache:
                while heap:
                    p, l, victim = heap[0]
                    if victim in cache and pri[victim] == p and last[victim] == l:
                        break
                    heapq.heappop(heap)
                else:
                    break
                heapq.heappop(heap)
                evictions += 1
                age = pri[victim]  # dynamic aging: clock follows the evicted min
                used -= cache.pop(victim)
                del cnt[victim]
                del pri[victim]
                del last[victim]
            cache[h] = size
            used += size
            cnt[h] = 0
            _touch(h, tick)

    total = hits + misses
    return {
        "hits": hits,
        "misses": misses,
        "evictions": evictions,
        "hit_rate": hits / total if total else 0.0,
        "write_amp": stored_bytes / hit_bytes if hit_bytes else None,
    }


# Ordered so LRU is first (baseline) and OPT is last (ceiling). The middle rows
# are the frequency signals under test.
SIMULATORS: dict[str, Callable[[list[dict], int], dict[str, Any]]] = {
    "LRU": simulate_lru,
    "LFU": simulate_lfu,
    "LFUDA": simulate_lfuda,
    "OPT": simulate_opt,
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def resolve_log(path: Path) -> Path:
    """Accept a cpu_access.*.jsonl file or its parent dir (most-recent wins)."""
    if path.is_dir():
        candidates = sorted(path.glob("cpu_access.*.jsonl"), key=lambda p: p.stat().st_mtime)
        if not candidates:
            print(f"ERROR: no cpu_access.*.jsonl found in {path}", file=sys.stderr)
            sys.exit(1)
        return candidates[-1]
    return path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Offline frequency-signal bake-off for LMCache CPU access logs"
    )
    parser.add_argument(
        "log", type=Path,
        help="Path to cpu_access.*.jsonl, or its parent run directory"
    )
    parser.add_argument(
        "--budget-gb", type=float, required=True,
        help="CPU pool size in GB (LMCACHE_MAX_LOCAL_CPU_SIZE from the run)"
    )
    args = parser.parse_args()

    budget_bytes = int(args.budget_gb * 1024 ** 3)
    log_path = resolve_log(args.log)
    print(f"Loading {log_path} ...")
    events = load_events(log_path)
    demand = demand_sequence(events)
    unique_chunks = len({e["chunk_hash"] for e in demand})
    print(
        f"Demand : {len(demand):,} accesses | {unique_chunks:,} unique chunks | "
        f"Budget : {args.budget_gb:g} GB = {budget_bytes:,} bytes"
    )
    print()

    results: dict[str, dict[str, Any]] = {}
    for name, fn in SIMULATORS.items():
        print(f"Simulating {name} ...")
        results[name] = fn(demand, budget_bytes)
    print()

    lru_hr = results["LRU"]["hit_rate"]
    opt_hr = results["OPT"]["hit_rate"]
    gap = opt_hr - lru_hr  # the LRU->OPT headroom this bake-off tries to close

    # ------------------------------------------------------------------
    # Table
    # ------------------------------------------------------------------
    col = "{:<8}  {:>10}  {:>9}  {:>12}  {:>9}  {:>9}  {:>11}  {:>10}"
    header = col.format(
        "Policy", "Hit rate", "% of OPT", "Gap closed", "Hits", "Misses",
        "Evictions", "Write-amp",
    )
    print(header)
    print("-" * len(header))
    for name, r in results.items():
        pct_opt = r["hit_rate"] / opt_hr * 100 if opt_hr else 0.0
        if name == "LRU":
            closed = "  (baseline)"
        elif name == "OPT":
            closed = "  (ceiling)"
        elif gap > 0:
            closed = f"{(r['hit_rate'] - lru_hr) / gap * 100:+.1f}%"
        else:
            closed = "n/a"
        wa = f"{r['write_amp']:.3f}" if r["write_amp"] is not None else "N/A"
        print(col.format(
            name,
            f"{r['hit_rate']*100:.2f}%",
            f"{pct_opt:.1f}%",
            closed,
            f"{r['hits']:,}",
            f"{r['misses']:,}",
            f"{r['evictions']:,}",
            wa,
        ))
    print()
    print(f"LRU->OPT gap : {gap*100:+.2f} pp  (the headroom a frequency signal could close)")
    print()

    # ------------------------------------------------------------------
    # JSON report
    # ------------------------------------------------------------------
    gb_tag = f"{args.budget_gb:g}"
    report = {
        "log": str(log_path),
        "budget_gb": args.budget_gb,
        "budget_bytes": budget_bytes,
        "demand_accesses": len(demand),
        "unique_chunks": unique_chunks,
        "policies": results,
        "lru_hit_rate": lru_hr,
        "opt_hit_rate": opt_hr,
        "gap_opt_minus_lru_pp": gap * 100,
        "gap_closed_pct": {
            name: ((r["hit_rate"] - lru_hr) / gap * 100 if gap > 0 else None)
            for name, r in results.items()
            if name not in ("LRU", "OPT")
        },
    }
    report_path = log_path.parent / f"bakeoff_report_cpu{gb_tag}.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"Report written to {report_path}")


if __name__ == "__main__":
    main()
