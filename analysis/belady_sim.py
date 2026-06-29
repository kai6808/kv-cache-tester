#!/usr/bin/env python3
"""Offline Belady/OPT simulator for LMCache CPU access logs.

Reads the cpu_access.jsonl produced by Phase 1 (LMCACHE_ACCESS_LOG) and
replays the demand sequence under OPT (and LRU), then compares against the
measured (logged) policy.

Usage:
    # trace from an LRU run (default): OPT vs LRU, with an LRU faithfulness check
    python3 analysis/belady_sim.py <run_dir>/cpu_access.jsonl --budget-gb 40

    # trace from a non-LRU run (e.g. QSLRU): OPT vs measured, LRU shown as a
    # reference on the same demand sequence (no spurious faithfulness failure)
    python3 analysis/belady_sim.py <run_dir>/cpu_access.jsonl --budget-gb 40 --trace-policy QSLRU

Outputs:
    <run_dir>/belady_report_cpu<N>.json   machine-readable results
    stdout                                 human-readable table + (LRU) faithfulness

Notes:
    - can_evict pinning is ignored (treated as always evictable).  Impact is
      small because pinned chunks are briefly locked during a single prefill
      and LRU already deprioritises them (they were just accessed = MRU).
    - The demand sequence (store + hit events) is approximately
      policy-independent: GPU-prefix-cache misses determine which chunks
      reach the CPU tier, which is unaffected by the CPU eviction policy. So
      OPT and the LRU reference are comparable across CPU policies, and the
      LRU faithfulness check (for --trace-policy LRU) validates this assumption.
    - A non-LRU CPU policy (e.g. QSLRU) cannot be replayed offline: its
      queue-demand signal is not in the CPU access log. For --trace-policy != LRU
      the measured row is taken directly from the logged events (ground truth),
      and OPT/LRU are simulated on the (policy-independent) demand sequence.
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


def compute_opt_capacity(demand: list[dict]) -> dict[str, Any]:
    """Minimum CPU capacity for OPT to need zero evictions.

    Uses a sweep-line over chunk lifetimes: a chunk is "live" from its first
    access to its last access in the demand sequence (OPT can evict it the
    instant its last use is served).  The peak concurrent live bytes is the
    minimum budget at which OPT never needs to evict anything.

    Also computes the theoretical maximum hit rate at that capacity: every
    unique chunk incurs exactly one first-access miss; all subsequent accesses
    are hits.
    """
    first: dict[int, int] = {}
    last:  dict[int, int] = {}
    sizes: dict[int, int] = {}
    for i, ev in enumerate(demand):
        h = ev["chunk_hash"]
        if h not in first:
            first[h] = i
        last[h] = i
        sizes[h] = ev["size_bytes"]

    # Sweep line: +size at first[h], -size at last[h]+1
    delta: dict[int, int] = defaultdict(int)
    for h in first:
        delta[first[h]] += sizes[h]
        delta[last[h] + 1] -= sizes[h]

    peak_bytes = 0
    peak_pos   = 0
    cur        = 0
    for pos in range(len(demand) + 1):
        cur += delta[pos]
        if cur > peak_bytes:
            peak_bytes = cur
            peak_pos   = pos

    unique_chunks  = len(first)
    total_accesses = len(demand)
    max_hits       = total_accesses - unique_chunks  # one miss per unique chunk
    max_hit_rate   = max_hits / total_accesses if total_accesses else 0.0

    return {
        "min_capacity_bytes": peak_bytes,
        "min_capacity_gb":    peak_bytes / 1024 ** 3,
        "peak_demand_pos":    peak_pos,
        "max_hit_rate":       max_hit_rate,
        "max_hits":           max_hits,
        "max_misses":         unique_chunks,
    }


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
    """LRU simulation: faithfulness check vs the log for an LRU trace, and a
    reference baseline on the same demand sequence for a non-LRU trace."""
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


def measured_from_log(events: list[dict]) -> dict[str, Any]:
    """Ground-truth CPU-tier stats taken directly from the logged events.

    Unlike the simulators, this reflects whatever eviction policy actually ran:
    hits/misses/evictions are counted from the log rather than recomputed. Used
    as the measured baseline for a trace whose policy cannot be replayed offline
    (e.g. QSLRU, whose queue-demand signal is not in the CPU access log). A
    "store" is a CPU-tier miss (the chunk had to be (re)written); a "hit" is a
    CPU-tier hit; an "evict" is one removed chunk.
    """
    hits = misses = evictions = 0
    hit_bytes = stored_bytes = 0
    for e in events:
        op = e["op"]
        if op == "hit":
            hits += 1
            hit_bytes += e["size_bytes"]
        elif op == "store":
            misses += 1
            stored_bytes += e["size_bytes"]
        elif op == "evict":
            evictions += 1
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
    parser.add_argument(
        "--trace-policy", type=str, default="LRU",
        help="CPU eviction policy that produced the log (default: LRU). For LRU, "
             "an LRU faithfulness check runs. For any other value (e.g. QSLRU), the "
             "measured row is taken from the log and LRU is shown as a reference; "
             "OPT is policy-independent and computed either way."
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
    # Measured ground truth + LRU simulation / faithfulness
    # ------------------------------------------------------------------
    trace_policy = args.trace_policy.strip()
    is_lru_trace = trace_policy.upper() == "LRU"

    # Ground truth from the log (reflects whatever policy actually ran).
    measured = measured_from_log(events)
    log_hr = measured["hit_rate"]

    print("Simulating LRU ...")
    lru = simulate_lru(demand, budget_bytes)

    delta_pp = abs(lru["hit_rate"] - log_hr) * 100
    faithful = None
    if is_lru_trace:
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
    else:
        print(
            f"Trace policy: {trace_policy} (not LRU) — the {trace_policy} row is taken "
            f"directly from the log; LRU is a reference on the same (GPU-miss-gated, "
            f"policy-independent) demand sequence. {trace_policy} cannot be replayed "
            f"offline (its queue-demand signal is not in the CPU access log), so no "
            f"faithfulness check is run."
        )
    print()

    # ------------------------------------------------------------------
    # OPT simulation
    # ------------------------------------------------------------------
    print("Simulating OPT ...")
    opt = simulate_opt(demand, budget_bytes)
    print()

    # ------------------------------------------------------------------
    # Optimal capacity analysis
    # ------------------------------------------------------------------
    print("Computing optimal capacity ...")
    opt_cap = compute_opt_capacity(demand)
    print()

    # ------------------------------------------------------------------
    # Results table
    # ------------------------------------------------------------------
    if is_lru_trace:
        policies = [("LRU", lru), ("OPT", opt)]
    else:
        policies = [(trace_policy, measured), ("LRU(ref)", lru), ("OPT", opt)]
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
    baseline_name = "LRU" if is_lru_trace else trace_policy
    baseline_hr = lru["hit_rate"] if is_lru_trace else measured["hit_rate"]
    gap_pp = (opt_hr - baseline_hr) * 100
    gap_lru_pp = (opt_hr - lru["hit_rate"]) * 100
    print(
        f"OPT − {baseline_name} gap : {gap_pp:+.2f} pp  "
        f"({'policy-closeable headroom' if gap_pp > 0 else baseline_name + ' already optimal'} "
        f"at {args.budget_gb} GB)"
    )
    if not is_lru_trace:
        print(f"OPT − LRU(ref) gap : {gap_lru_pp:+.2f} pp")
    print()
    print(
        f"Optimal capacity  : {opt_cap['min_capacity_gb']:.3f} GB  "
        f"({opt_cap['min_capacity_bytes']:,} bytes)  — "
        f"zero evictions at or above this budget"
    )
    print(
        f"Max hit rate (OPT): {opt_cap['max_hit_rate']*100:.2f}%  "
        f"(limited only by {opt_cap['max_misses']:,} first-access misses)"
    )
    print()

    # ------------------------------------------------------------------
    # Write JSON report
    # ------------------------------------------------------------------
    # Base report: identical structure/order to the pre-QSLRU version, so an LRU
    # run's report is byte-for-byte unchanged. (gap_lru_pp == OPT-LRU gap, which
    # for an LRU trace is exactly the old gap_pp.) Non-LRU runs append extra keys.
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
        "gap_opt_minus_lru_pp": gap_lru_pp,
        "opt_capacity": opt_cap,
    }
    if not is_lru_trace:
        report["trace_policy"] = trace_policy
        report["measured"] = measured
        report["baseline_policy"] = baseline_name
        report["gap_opt_minus_baseline_pp"] = gap_pp
        report["lru_faithfulness"]["checked"] = False
    gb_tag = f"{args.budget_gb:g}"
    report_path = log_path.parent / f"belady_report_cpu{gb_tag}.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"Report written to {report_path}")


if __name__ == "__main__":
    main()