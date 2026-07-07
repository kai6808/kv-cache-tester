#!/usr/bin/env python3
"""Offline replay of queue-demand + structural CPU eviction policies.

Step 1/2/3 of docs/plans/2026-07-04-reuse-scored-cpu-eviction.md. Consumes an
*enriched* cpu_access.jsonl -- one that additionally carries the queue-demand
events ("wait"/"serve") emitted by the enriched LMCache AccessLogger
(lmcache/.../local_cpu_backend.py) -- and, unlike belady_sim/policy_bakeoff:

  1. reconstructs the prefix structure (fan-out + depth) from the ordered chunk
     lists in the "wait" events (consecutive entries = parent->child edges;
     index = depth), and
  2. replays QSLRU *offline* using the wait/serve demand timeline, which lets us
     validate the enrichment (offline QSLRU should reproduce a QSLRU run's
     measured hit rate) and, crucially, replay QSLRU on an *LRU* run's log
     (note_waiting is logged regardless of policy).

Does NOT modify belady_sim.py or policy_bakeoff.py; it imports belady_sim's
loaders and its LRU/OPT/measured helpers and adds the demand-aware simulators.

The enriched-log schema (see _AccessLogger.log_demand):
  access: {"seq", "op": store|hit|evict, "chunk_hash", "size_bytes", "can_evict", "t"}
  demand: {"seq", "op": wait|serve, "req_id", "chunks":[chunk_hash,...], "t"}
The shared "seq" interleaves both so the demand state at each access is exact.

Usage:
    python3 analysis/signal_sim.py <run_dir>/cpu_access.*.jsonl --budget-gb 30
    python3 analysis/signal_sim.py <run_dir> --budget-gb 30 --trace-policy QSLRU

Outputs:
    <run_dir>/signal_report_cpu<N>.json   machine-readable results
    stdout                                 structure summary + policy table
"""
import argparse
import json
import sys
from collections import OrderedDict, defaultdict
from pathlib import Path
from statistics import median
from typing import Any

from belady_sim import (
    load_events,
    demand_sequence,
    simulate_lru,
    simulate_opt,
    measured_from_log,
)

_DEPTH_INF = 1 << 62


# ---------------------------------------------------------------------------
# Structure reconstruction (from "wait" events)
# ---------------------------------------------------------------------------

def build_structure(events: list[dict]) -> tuple[dict[int, int], dict[int, int]]:
    """Reconstruct per-chunk fan-out and depth from the ordered wait-event chains.

    Each "wait" event lists a request's prefix chunks in order, so entry i-1 -> i
    is a parent->child edge and i is the chunk's depth. Returns (fan_out, depth):
    fan_out[c] = number of DISTINCT children of c across all requests; depth[c] =
    the shallowest position c was ever seen at (0 = sequence root / global prefix).
    """
    children: dict[int, set[int]] = defaultdict(set)
    depth: dict[int, int] = {}
    for e in events:
        if e.get("op") != "wait":
            continue
        chunks = e.get("chunks", [])
        for i, c in enumerate(chunks):
            if i < depth.get(c, _DEPTH_INF):
                depth[c] = i
            if i > 0:
                children[chunks[i - 1]].add(c)
    fan_out = {c: len(kids) for c, kids in children.items()}
    return fan_out, depth


# ---------------------------------------------------------------------------
# QSLRU replayed offline from the wait/serve demand timeline
# ---------------------------------------------------------------------------

_PRED_INF = 1 << 62


def simulate_qslru_cold(
    events: list[dict],
    budget_bytes: int,
    mode: str = "lru",
    depth: dict[int, int] | None = None,
) -> dict[str, Any]:
    """Replay QSLRU with a configurable COLD-tier victim rule (Phase B probe).

    The warm tier (ref_count>0, a queued request needs it) is always evicted last,
    in LRU order -- identical to the live QSLRU, so ``mode="lru"`` reproduces plain
    QSLRU exactly. Only the *cold* tier (ref_count==0) victim selection changes:
      "lru"    -> least-recently-used cold chunk (== plain QSLRU baseline)
      "depth"  -> deepest cold chunk first (protect the shallow shared prefix;
                  depth = shallowest prefix index seen, from build_structure)
      "prednu" -> cold chunk with the farthest predicted next use, where the
                  prediction is the last observed reuse interval (LRB-lite); a
                  chunk with no prior interval is treated as farthest.

    wait/serve maintain ref_count exactly as the live policy does (idempotent per
    req_id); store/hit are accesses (hit/miss decided by the simulated cache, as in
    belady_sim); logged evict events are ignored (we make our own decisions).
    """
    depth = depth or {}
    cache: OrderedDict[int, int] = OrderedDict()  # chunk_hash -> size_bytes (front=LRU)
    ref: dict[int, int] = {}                      # chunk_hash -> waiting count
    req_keys: dict[str, list[int]] = {}           # req_id -> chunks it registered
    last_pos: dict[int, int] = {}                 # prednu: last access (event index)
    pred: dict[int, int] = {}                     # prednu: predicted next-use position
    used = 0
    hits = misses = evictions = 0
    stored_bytes = hit_bytes = 0

    def cold_victim() -> int:
        """First cold chunk (mode lru) or the cold chunk maximising the signal;
        warm LRU tail only when no cold chunk exists (QSLRU's last resort)."""
        best: int | None = None
        best_key = -1
        for k in cache:  # front..back == LRU..MRU
            if ref.get(k, 0) != 0:
                continue
            if mode == "lru":
                return k  # first cold == LRU-cold
            score = depth.get(k, 0) if mode == "depth" else pred.get(k, _PRED_INF)
            if best is None or score > best_key:
                best, best_key = k, score
        return best if best is not None else next(iter(cache))

    for i, e in enumerate(events):
        op = e["op"]
        if op == "wait":
            rid = e["req_id"]
            if rid in req_keys:
                continue  # idempotent, matches qslru.note_waiting
            chunks = e.get("chunks", [])
            req_keys[rid] = chunks
            for c in chunks:
                ref[c] = ref.get(c, 0) + 1
        elif op == "serve":
            chunks = req_keys.pop(e["req_id"], None)
            if chunks is None:
                continue
            for c in chunks:
                remaining = ref.get(c, 0) - 1
                if remaining <= 0:
                    ref.pop(c, None)
                else:
                    ref[c] = remaining
        elif op in ("store", "hit"):
            h = e["chunk_hash"]
            size = e["size_bytes"]
            if mode == "prednu":
                pred[h] = i + (i - last_pos[h]) if h in last_pos else _PRED_INF
                last_pos[h] = i
            if h in cache:
                hits += 1
                hit_bytes += cache[h]
                cache.move_to_end(h)
            else:
                misses += 1
                stored_bytes += size
                while used + size > budget_bytes and cache:
                    victim = cold_victim()
                    used -= cache.pop(victim)
                    evictions += 1
                cache[h] = size
                used += size
        # op == "evict": ignored (offline policy makes its own decisions)

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

def resolve_log(path: Path) -> Path:
    if path.is_dir():
        cands = sorted(path.glob("cpu_access.*.jsonl"), key=lambda p: p.stat().st_mtime)
        if not cands:
            print(f"ERROR: no cpu_access.*.jsonl found in {path}", file=sys.stderr)
            sys.exit(1)
        return cands[-1]
    return path


def _pct(x: float) -> str:
    return f"{x*100:.2f}%"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Offline queue-demand + structural CPU eviction replay"
    )
    parser.add_argument("log", type=Path,
                        help="Path to an enriched cpu_access.*.jsonl, or its run dir")
    parser.add_argument("--budget-gb", type=float, required=True,
                        help="CPU pool size in GB (LMCACHE_MAX_LOCAL_CPU_SIZE)")
    parser.add_argument("--trace-policy", type=str, default="QSLRU",
                        help="Policy that produced the log (default: QSLRU). When "
                             "QSLRU, offline QSLRU is checked against the measured "
                             "hit rate as an enrichment-faithfulness gate.")
    parser.add_argument("--faithfulness-tol", type=float, default=5.0,
                        help="Max allowed pp deviation for the QSLRU replay check")
    args = parser.parse_args()

    budget_bytes = int(args.budget_gb * 1024 ** 3)
    log_path = resolve_log(args.log)
    print(f"Loading {log_path} ...")
    events = load_events(log_path)

    n_wait = sum(1 for e in events if e.get("op") == "wait")
    n_serve = sum(1 for e in events if e.get("op") == "serve")
    demand = demand_sequence(events)
    unique_chunks = len({e["chunk_hash"] for e in demand})
    print(
        f"Events : {len(events):,} total | {n_wait:,} wait | {n_serve:,} serve | "
        f"{len(demand):,} accesses | {unique_chunks:,} unique chunks"
    )
    if n_wait == 0:
        print(
            "\nWARNING: no 'wait' events found -- this log is NOT enriched. Re-run the "
            "sweep with the updated LMCache (enriched AccessLogger) so note_waiting is "
            "logged; only then can structure/QSLRU be reconstructed offline.",
            file=sys.stderr,
        )

    # ------------------------------------------------------------------
    # Structure summary
    # ------------------------------------------------------------------
    fan_out, depth = build_structure(events)
    print()
    if fan_out:
        fo = list(fan_out.values())
        branchy = sum(1 for v in fo if v > 1)
        print("Prefix structure (from wait events):")
        print(f"  chunks with >=1 child : {len(fan_out):,}")
        print(f"  fan-out > 1 (branch)  : {branchy:,}  ({branchy/len(fan_out)*100:.1f}%)")
        print(f"  fan-out  max / median : {max(fo)} / {median(fo):.0f}")
        if depth:
            dv = list(depth.values())
            print(f"  depth    max / median : {max(dv)} / {median(dv):.0f}")
        print()

    # ------------------------------------------------------------------
    # Policies: LRU / QSLRU / QSLRU+depth / QSLRU+prednu / OPT (+ faithfulness)
    # QSLRU+depth and QSLRU+prednu are Phase B probes: same queue (warm) tier as
    # QSLRU, only the cold-tier ordering swaps to a structural / reuse-interval
    # signal -- the question is whether either beats plain QSLRU on the residual gap.
    # ------------------------------------------------------------------
    print("Simulating LRU / QSLRU / QSLRU+depth / QSLRU+prednu / OPT ...")
    lru = simulate_lru(demand, budget_bytes)
    qslru = simulate_qslru_cold(events, budget_bytes, "lru")
    qs_depth = simulate_qslru_cold(events, budget_bytes, "depth", depth)
    qs_prednu = simulate_qslru_cold(events, budget_bytes, "prednu")
    opt = simulate_opt(demand, budget_bytes)
    measured = measured_from_log(events)
    print()

    lru_hr, opt_hr, qs_hr = lru["hit_rate"], opt["hit_rate"], qslru["hit_rate"]
    gap = opt_hr - lru_hr

    rows = [("LRU", lru), ("QSLRU", qslru), ("QSLRU+depth", qs_depth),
            ("QSLRU+prednu", qs_prednu), ("OPT", opt)]
    col = "{:<14}  {:>10}  {:>9}  {:>11}  {:>11}  {:>11}  {:>10}"
    header = col.format("Policy", "Hit rate", "% of OPT", "Gap closed",
                        "vs QSLRU", "Evictions", "Write-amp")
    print(header)
    print("-" * len(header))
    for name, r in rows:
        pct_opt = r["hit_rate"] / opt_hr * 100 if opt_hr else 0.0
        closed = ("  (baseline)" if name == "LRU" else "  (ceiling)" if name == "OPT"
                  else f"{(r['hit_rate'] - lru_hr) / gap * 100:+.1f}%" if gap > 0 else "n/a")
        vs_qs = ("" if name in ("LRU", "OPT", "QSLRU")
                 else f"{(r['hit_rate'] - qs_hr) * 100:+.2f} pp")
        wa = f"{r['write_amp']:.3f}" if r["write_amp"] is not None else "N/A"
        print(col.format(name, _pct(r["hit_rate"]), f"{pct_opt:.1f}%", closed,
                         vs_qs, f"{r['evictions']:,}", wa))
    print()
    print(f"LRU->OPT gap : {gap*100:+.2f} pp   |   residual QSLRU->OPT gap : "
          f"{(opt_hr - qs_hr)*100:+.2f} pp")

    # Enrichment faithfulness: for a QSLRU run, offline QSLRU must reproduce the
    # measured hit rate (validates wait/serve logging + the offline replay).
    faithful = None
    if args.trace_policy.strip().upper() == "QSLRU":
        delta_pp = abs(qslru["hit_rate"] - measured["hit_rate"]) * 100
        faithful = delta_pp <= args.faithfulness_tol
        tag = "PASS" if faithful else f"FAIL (> {args.faithfulness_tol} pp)"
        print(
            f"QSLRU replay vs measured : |{_pct(qslru['hit_rate'])} - "
            f"{_pct(measured['hit_rate'])}| = {delta_pp:.2f} pp  -> {tag}"
        )
    else:
        print(
            f"Trace policy {args.trace_policy}: QSLRU(offline) is a prediction on this "
            f"run's (policy-independent) demand + demand timeline; no faithfulness gate."
        )
    print()

    # ------------------------------------------------------------------
    # JSON report
    # ------------------------------------------------------------------
    gb_tag = f"{args.budget_gb:g}"
    report = {
        "log": str(log_path),
        "budget_gb": args.budget_gb,
        "trace_policy": args.trace_policy,
        "counts": {"wait": n_wait, "serve": n_serve,
                   "accesses": len(demand), "unique_chunks": unique_chunks},
        "structure": {
            "chunks_with_children": len(fan_out),
            "branch_nodes_fanout_gt1": sum(1 for v in fan_out.values() if v > 1),
            "fan_out_max": max(fan_out.values()) if fan_out else 0,
            "depth_max": max(depth.values()) if depth else 0,
        },
        "policies": {name: r for name, r in rows},
        "measured": measured,
        "lru_hit_rate": lru_hr,
        "opt_hit_rate": opt_hr,
        "gap_opt_minus_lru_pp": gap * 100,
        "residual_qslru_opt_gap_pp": (opt_hr - qs_hr) * 100,
        "gap_closed_pct": {
            name: ((r["hit_rate"] - lru_hr) / gap * 100 if gap > 0 else None)
            for name, r in rows if name not in ("LRU", "OPT")
        },
        "variant_vs_qslru_pp": {
            name: (r["hit_rate"] - qs_hr) * 100
            for name, r in rows if name in ("QSLRU+depth", "QSLRU+prednu")
        },
        "qslru_replay_faithful": faithful,
    }
    report_path = log_path.parent / f"signal_report_cpu{gb_tag}.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"Report written to {report_path}")


if __name__ == "__main__":
    main()