#!/usr/bin/env python3
"""Offline Belady/OPT simulator for LMCache CPU access logs.

Reads the cpu_access.jsonl produced by Phase 1 (LMCACHE_ACCESS_LOG) and
replays the demand sequence under OPT and LRU, then compares results.

Usage:
    python analysis/belady_sim.py <run_dir>/cpu_access.jsonl --budget-gb 40

Outputs:
    <run_dir>/belady_report.json   machine-readable results
    stdout                          human-readable table + faithfulness check

Notes:
    - can_evict pinning is ignored (treated as always evictable).  Impact is
      small because pinned chunks are briefly locked during a single prefill
      and LRU already deprioritises them (they were just accessed = MRU).
    - The demand sequence (store + hit events) is approximately
      policy-independent: GPU-prefix-cache misses determine which chunks
      reach the CPU tier, which is unaffected by the CPU eviction policy.
      The LRU faithfulness check validates this assumption.
"""
import argparse
import heapq
import json
import sys
from collections import OrderedDict, defaultdict, deque
from pathlib import Path
from typing import Any

INF = sys.maxsize  # sentinel for "never reused again"


# ---------------------------------------------------------------------------
# Log loading
# ---------------------------------------------------------------------------

def load_events(path: Path) -> list[dict]:
    """Load and sort all JSONL events by seq number."""
    events = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    events.sort(key=lambda e: e["seq"])
    return events


def demand_sequence(events: list[dict]) -> list[dict]:
    """Extract the chunk-access demand trace (store + hit events only)."""
    return [e for e in events if e["op"] in ("store", "hit")]


# ---------------------------------------------------------------------------
# Next-use oracle (used by OPT)
# ---------------------------------------------------------------------------

def build_next_use_map(demand: list[dict]) -> dict[int, deque]:
    """Map chunk_hash -> deque of demand-sequence positions (ascending)."""
    pos: dict[int, list[int]] = defaultdict(list)
    for i, ev in enumerate(demand):
        pos[ev["chunk_hash"]].append(i)
    return {h: deque(ps) for h, ps in pos.items()}


def _advance_next_use(nxt_map: dict[int, deque], h: int, current: int) -> int:
    """Return the next demand position for h after `current`, or INF."""
    q = nxt_map.get(h)
    if not q:
        return INF
    while q and q[0] <= current:
        q.popleft()
    return q[0] if q else INF


# ---------------------------------------------------------------------------
# Policy simulators
# ---------------------------------------------------------------------------

def simulate_opt(demand: list[dict], budget_bytes: int) -> dict[str, Any]:
    """Belady MIN: always evict the chunk whose next use is farthest away.

    Uses a lazy-deletion min-heap keyed on (-next_use, chunk_hash).
    Negating next_use turns the min-heap into an effective max-next-use heap.
    """
    nxt = build_next_use_map(demand)
    cache: dict[int, int] = {}     # chunk_hash -> size_bytes
    used = 0
    hits = misses = evictions = 0
    stored_bytes = hit_bytes = 0

    heap: list[tuple[int, int]] = []
    # heap_nu[h] = the next_use value currently "live" in the heap for h.
    # Used to detect stale heap entries during lazy deletion.
    heap_nu: dict[int, int] = {}

    for i, ev in enumerate(demand):
        h = ev["chunk_hash"]
        size = ev["size_bytes"]
        nu = _advance_next_use(nxt, h, i)

        if h in cache:
            hits += 1
            hit_bytes += cache[h]
            # Update next-use; push new entry (old one becomes stale).
            heap_nu[h] = nu
            heapq.heappush(heap, (-nu, h))
        else:
            misses += 1
            stored_bytes += size
            # Evict the chunk(s) with farthest next use until the new chunk fits.
            while used + size > budget_bytes and cache:
                # Pop stale heap entries until we find a live one.
                while heap:
                    neg_nu, victim = heap[0]
                    if victim in cache and heap_nu.get(victim) == -neg_nu:
                        break
                    heapq.heappop(heap)
                else:
                    break  # heap exhausted (shouldn't happen in practice)
                heapq.heappop(heap)
                evictions += 1
                used -= cache.pop(victim)
                del heap_nu[victim]

            cache[h] = size
            used += size
            heap_nu[h] = nu
            heapq.heappush(heap, (-nu, h))

    total = hits + misses
    return {
        "hits": hits,
        "misses": misses,
        "evictions": evictions,
        "hit_rate": hits / total if total else 0.0,
        "write_amp": stored_bytes / hit_bytes if hit_bytes else None,
    }


def simulate_lru(demand: list[dict], budget_bytes: int) -> dict[str, Any]:
    """LRU simulation for faithfulness check against the measured log."""
    # OrderedDict: LRU at front (last=False on popitem), MRU at back.
    cache: OrderedDict[int, int] = OrderedDict()
    used = 0
    hits = misses = evictions = 0
    stored_bytes = hit_bytes = 0

    for ev in demand:
        h = ev["chunk_hash"]
        size = ev["size_bytes"]

        if h in cache:
            hits += 1
            hit_bytes += cache[h]
            cache.move_to_end(h)
        else:
            misses += 1
            stored_bytes += size
            while used + size > budget_bytes and cache:
                victim, vsize = cache.popitem(last=False)
                evictions += 1
                used -= vsize
            cache[h] = size
            used += size

    total = hits + misses
    return {
        "hits": hits,
        "misses": misses,
        "evictions": evictions,
        "hit_rate": hits / total if total else 0.0,
        "write_amp": stored_bytes / hit_bytes if hit_bytes else None,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Offline Belady/OPT simulator for LMCache CPU access logs"
    )
    parser.add_argument(
        "log", type=Path,
        help="Path to cpu_access.<ts>.<pid>.jsonl, or its parent directory "
             "(auto-selects the most recent cpu_access.*.jsonl in that dir)"
    )
    parser.add_argument(
        "--budget-gb", type=float, required=True,
        help="CPU pool size in GB (LMCACHE_MAX_LOCAL_CPU_SIZE from the run)"
    )
    parser.add_argument(
        "--faithfulness-tol", type=float, default=5.0,
        help="Max allowed pp deviation for LRU faithfulness check (default: 5.0)"
    )
    args = parser.parse_args()

    budget_bytes = int(args.budget_gb * 1024 ** 3)

    log_path = args.log
    if log_path.is_dir():
        candidates = sorted(log_path.glob("cpu_access.*.jsonl"), key=lambda p: p.stat().st_mtime)
        if not candidates:
            print(f"ERROR: no cpu_access.*.jsonl found in {log_path}", file=sys.stderr)
            sys.exit(1)
        log_path = candidates[-1]
        print(f"Auto-selected: {log_path}")

    print(f"Loading {log_path} ...")
    events = load_events(log_path)

    log_stores = sum(1 for e in events if e["op"] == "store")
    log_hits   = sum(1 for e in events if e["op"] == "hit")
    log_evicts = sum(1 for e in events if e["op"] == "evict")

    demand = demand_sequence(events)
    unique_chunks = len({e["chunk_hash"] for e in demand})

    print(
        f"Events : {len(events):,} total | "
        f"{log_stores:,} stores | {log_hits:,} hits | {log_evicts:,} evicts"
    )
    print(
        f"Demand : {len(demand):,} accesses | "
        f"{unique_chunks:,} unique chunks | "
        f"Budget : {args.budget_gb} GB = {budget_bytes:,} bytes"
    )
    print()

    # ------------------------------------------------------------------
    # LRU simulation + faithfulness check
    # ------------------------------------------------------------------
    print("Simulating LRU ...")
    lru = simulate_lru(demand, budget_bytes)

    log_hr = log_hits / (log_hits + log_stores) if (log_hits + log_stores) else 0.0
    delta_pp = abs(lru["hit_rate"] - log_hr) * 100
    faithful = delta_pp <= args.faithfulness_tol
    faith_label = (
        f"PASS  (|sim {lru['hit_rate']*100:.2f}% - log {log_hr*100:.2f}%| = {delta_pp:.2f} pp)"
        if faithful else
        f"FAIL  (|sim {lru['hit_rate']*100:.2f}% - log {log_hr*100:.2f}%| = {delta_pp:.2f} pp > {args.faithfulness_tol} pp)"
    )
    print(f"LRU faithfulness: {faith_label}")
    if not faithful:
        print(
            "  WARNING: divergence exceeds tolerance. Possible causes: "
            "can_evict pinning, concurrent access reordering in the log."
        )
    print()

    # ------------------------------------------------------------------
    # OPT simulation
    # ------------------------------------------------------------------
    print("Simulating OPT ...")
    opt = simulate_opt(demand, budget_bytes)
    print()

    # ------------------------------------------------------------------
    # Results table
    # ------------------------------------------------------------------
    policies = [("LRU", lru), ("OPT", opt)]
    opt_hr = opt["hit_rate"]

    col = "{:<8}  {:>10}  {:>10}  {:>9}  {:>9}  {:>11}  {:>10}"
    header = col.format(
        "Policy", "Hit rate", "% of OPT", "Hits", "Misses", "Evictions", "Write-amp"
    )
    print(header)
    print("-" * len(header))
    for name, r in policies:
        pct = r["hit_rate"] / opt_hr * 100 if opt_hr else 0.0
        wa  = f"{r['write_amp']:.3f}" if r["write_amp"] is not None else "N/A"
        print(col.format(
            name,
            f"{r['hit_rate']*100:.2f}%",
            f"{pct:.1f}%",
            f"{r['hits']:,}",
            f"{r['misses']:,}",
            f"{r['evictions']:,}",
            wa,
        ))

    print()
    gap_pp = (opt_hr - lru["hit_rate"]) * 100
    print(
        f"OPT − LRU gap : {gap_pp:+.2f} pp  "
        f"({'policy-closeable headroom' if gap_pp > 0 else 'LRU already optimal'} "
        f"at {args.budget_gb} GB)"
    )
    print()

    # ------------------------------------------------------------------
    # Write JSON report
    # ------------------------------------------------------------------
    report = {
        "log": str(log_path),
        "budget_gb": args.budget_gb,
        "budget_bytes": budget_bytes,
        "log_stats": {
            "stores": log_stores,
            "hits": log_hits,
            "evicts": log_evicts,
            "unique_chunks": unique_chunks,
        },
        "lru_faithfulness": {
            "pass": faithful,
            "log_hit_rate": log_hr,
            "sim_hit_rate": lru["hit_rate"],
            "delta_pp": delta_pp,
            "tolerance_pp": args.faithfulness_tol,
        },
        "policies": {name: r for name, r in policies},
        "gap_opt_minus_lru_pp": gap_pp,
    }
    report_path = log_path.parent / "belady_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"Report written to {report_path}")


if __name__ == "__main__":
    main()