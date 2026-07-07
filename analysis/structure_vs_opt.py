#!/usr/bin/env python3
"""Does LRU's avoidable thrash concentrate on structurally "shared" chunks?

A cheap, fully-offline check before building a fan-out-weighted CPU eviction policy
(TreeLRU). Hypothesis: the Belady gap (avoidable re-store thrash) falls mostly on
high-reuse, long-lived chunks — the shared prefix / interior of the prefix tree —
which a structural signal (fan-out) would protect and plain LRU does not.

True fan-out (parent->child edges) is NOT in cpu_access.jsonl (it logs only
chunk_hash). So we use the reconstructable structural fingerprint of each chunk —
reuse count and live span — as a proxy for "shared / long-lived", and tie it to the
*measured* thrash:

    avoidable_restores(chunk) = measured_restores(chunk)  -  OPT_restores(chunk)

  - measured_restores = (store events for the chunk in the log) - 1  [ground truth of
    whatever policy actually ran: a re-store == evicted-then-needed-again].
  - OPT_restores = (misses for the chunk under simulated Belady at this budget) - 1.

We then ask: are the chunks carrying that avoidable thrash systematically higher in
reuse count / live span than the average chunk? If yes, a structural signal is
predictive and worth building.

Usage:
    python3 analysis/structure_vs_opt.py <run_dir-or-cpu_access.jsonl> --budget-gb 30

Outputs:
    stdout                                   human-readable buckets + verdict
    <dir>/structure_vs_opt_cpu<N>.json       machine-readable
"""
import argparse
import heapq
import json
import statistics
import sys
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

INF = sys.maxsize


def load_events(path: Path) -> list[dict]:
    events = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    events.sort(key=lambda e: e["seq"])
    return events


def demand_sequence(events: list[dict]) -> list[dict]:
    return [e for e in events if e["op"] in ("store", "hit")]


def simulate_opt_misses(demand: list[dict], budget_bytes: int) -> dict[int, int]:
    """Belady MIN; return misses-per-chunk (lazy-deletion max-next-use heap)."""
    pos: dict[int, list[int]] = defaultdict(list)
    for i, ev in enumerate(demand):
        pos[ev["chunk_hash"]].append(i)
    nxt = {h: deque(ps) for h, ps in pos.items()}

    def advance(h: int, cur: int) -> int:
        q = nxt.get(h)
        if not q:
            return INF
        while q and q[0] <= cur:
            q.popleft()
        return q[0] if q else INF

    cache: dict[int, int] = {}
    used = 0
    misses: dict[int, int] = defaultdict(int)
    heap: list[tuple[int, int]] = []
    heap_nu: dict[int, int] = {}

    for i, ev in enumerate(demand):
        h, size = ev["chunk_hash"], ev["size_bytes"]
        nu = advance(h, i)
        if h in cache:
            heap_nu[h] = nu
            heapq.heappush(heap, (-nu, h))
        else:
            misses[h] += 1
            while used + size > budget_bytes and cache:
                while heap:
                    neg_nu, victim = heap[0]
                    if victim in cache and heap_nu.get(victim) == -neg_nu:
                        break
                    heapq.heappop(heap)
                else:
                    break
                heapq.heappop(heap)
                used -= cache.pop(victim)
                del heap_nu[victim]
            cache[h] = size
            used += size
            heap_nu[h] = nu
            heapq.heappush(heap, (-nu, h))
    return misses


def chunk_features(demand: list[dict]) -> dict[int, dict]:
    """Per-chunk reconstructable structural fingerprint."""
    acc: dict[int, list[int]] = defaultdict(list)
    size: dict[int, int] = {}
    for i, ev in enumerate(demand):
        acc[ev["chunk_hash"]].append(i)
        size[ev["chunk_hash"]] = ev["size_bytes"]
    n = len(demand)
    feats = {}
    for h, positions in acc.items():
        span = positions[-1] - positions[0]
        feats[h] = {
            "n_access": len(positions),     # reuse count (freq)
            "span": span,                   # live distance in the demand stream
            "live_frac": span / n if n else 0.0,
            "size": size[h],
        }
    return feats


def quartile_buckets(chunks: list[int], key: dict[int, float]) -> list[list[int]]:
    """Split chunks into 4 buckets by ascending `key` value (Q1 low .. Q4 high)."""
    ordered = sorted(chunks, key=lambda h: key[h])
    m = len(ordered)
    return [ordered[i * m // 4:(i + 1) * m // 4] for i in range(4)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log", type=Path,
                    help="cpu_access.*.jsonl or its parent dir (auto-selects newest)")
    ap.add_argument("--budget-gb", type=float, required=True,
                    help="CPU pool size in GB (LMCACHE_MAX_LOCAL_CPU_SIZE of the run)")
    args = ap.parse_args()

    log_path = args.log
    if log_path.is_dir():
        cands = sorted(log_path.glob("cpu_access.*.jsonl"), key=lambda p: p.stat().st_mtime)
        if not cands:
            print(f"ERROR: no cpu_access.*.jsonl in {log_path}", file=sys.stderr)
            sys.exit(1)
        log_path = cands[-1]
        print(f"Auto-selected: {log_path}")

    budget_bytes = int(args.budget_gb * 1024 ** 3)
    events = load_events(log_path)
    demand = demand_sequence(events)
    feats = chunk_features(demand)

    # measured re-stores (ground truth of the policy that ran)
    store_count: dict[int, int] = defaultdict(int)
    for e in events:
        if e["op"] == "store":
            store_count[e["chunk_hash"]] += 1
    measured_restores = {h: max(0, c - 1) for h, c in store_count.items()}

    # OPT re-stores at this budget
    opt_misses = simulate_opt_misses(demand, budget_bytes)
    opt_restores = {h: max(0, opt_misses.get(h, 0) - 1) for h in feats}

    avoidable = {h: measured_restores.get(h, 0) - opt_restores.get(h, 0) for h in feats}
    # clamp tiny negatives (OPT shouldn't thrash more than the real policy)
    avoidable = {h: max(0, v) for h, v in avoidable.items()}

    tot_measured = sum(measured_restores.values())
    tot_opt = sum(opt_restores.values())
    tot_avoid = sum(avoidable.values())

    chunks = list(feats)
    print(f"\nLog    : {log_path}")
    print(f"Budget : {args.budget_gb} GB | demand={len(demand):,} accesses | "
          f"unique chunks={len(chunks):,}")
    print(f"Re-stores: measured(policy)={tot_measured:,}  OPT={tot_opt:,}  "
          f"avoidable={tot_avoid:,}\n")

    if tot_avoid == 0:
        print("No avoidable thrash at this budget (pool large enough). Nothing to attribute.")
        return

    # ---- concentration: how few chunks carry the avoidable thrash ----
    by_av = sorted(chunks, key=lambda h: avoidable[h], reverse=True)
    cum = 0
    k80 = 0
    for k80, h in enumerate(by_av, 1):
        cum += avoidable[h]
        if cum >= 0.8 * tot_avoid:
            break
    thrashed = [h for h in chunks if avoidable[h] > 0]
    print(f"Concentration: top {k80:,} chunks ({k80/len(chunks)*100:.1f}% of chunks) "
          f"carry 80% of avoidable thrash; {len(thrashed):,} chunks thrash at all.\n")

    # ---- attribute avoidable thrash to structural-feature quartiles ----
    col = "{:<10} {:>14} {:>14} {:>16} {:>16}"
    print(col.format("Quartile", "avoid (n_access)", "share%", "avoid (span)", "share%"))
    print("-" * 74)
    nb = quartile_buckets(chunks, {h: feats[h]["n_access"] for h in chunks})
    sb = quartile_buckets(chunks, {h: feats[h]["span"] for h in chunks})
    for i in range(4):
        an = sum(avoidable[h] for h in nb[i])
        asp = sum(avoidable[h] for h in sb[i])
        print(col.format(
            f"Q{i+1}{' (low)' if i==0 else ' (high)' if i==3 else ''}",
            f"{an:,}", f"{an/tot_avoid*100:.1f}", f"{asp:,}", f"{asp/tot_avoid*100:.1f}"))
    print("(Q4 = highest reuse-count / longest span = most 'shared / long-lived')\n")

    # ---- thrashed-vs-typical chunk profile ----
    def med(hs, f): return statistics.median([feats[h][f] for h in hs]) if hs else 0
    long_lived_share = (sum(avoidable[h] for h in thrashed
                            if feats[h]["span"] > med(chunks, "span")) / tot_avoid * 100)
    print("Median feature — thrashed chunks vs all chunks:")
    print(f"  n_access : {med(thrashed,'n_access'):.0f}  vs  {med(chunks,'n_access'):.0f}")
    print(f"  span     : {med(thrashed,'span'):.0f}  vs  {med(chunks,'span'):.0f}")
    print(f"  live_frac: {med(thrashed,'live_frac'):.3f}  vs  {med(chunks,'live_frac'):.3f}")
    # Top-half (Q3+Q4) share of each feature, and the above-median-span share, vs the
    # 50% you'd see if thrash were structure-agnostic.
    top_half_n = sum(avoidable[h] for h in nb[2] + nb[3]) / tot_avoid * 100
    top_half_s = sum(avoidable[h] for h in sb[2] + sb[3]) / tot_avoid * 100
    print(f"\nTop-half (Q3+Q4) share of avoidable thrash: "
          f"{top_half_n:.1f}% by reuse-count, {top_half_s:.1f}% by span  (null = 50%)")
    print(f"Share of avoidable thrash on above-median-span chunks: {long_lived_share:.1f}%  "
          f"(null = 50%)")

    signal = (long_lived_share + top_half_s) / 2  # both measure long-lived concentration
    if signal >= 65:
        tier = ("PREDICTIVE — avoidable thrash concentrates on long-lived / shared chunks "
                "(well above the 50% null). A structural (fan-out) signal should help. "
                "Note it is diffuse (see concentration line), so expect partial, not full, "
                "gap closure.")
    elif signal >= 55:
        tier = ("MODERATE — a structural lean exists but is mild; fan-out may give a small "
                "gain. Weigh against implementation cost.")
    else:
        tier = ("WEAK — thrash is near structure-agnostic; reconsider before building.")
    print(f"\nVerdict: {tier}")

    report = {
        "log": str(log_path),
        "budget_gb": args.budget_gb,
        "demand_accesses": len(demand),
        "unique_chunks": len(chunks),
        "restores": {"measured": tot_measured, "opt": tot_opt, "avoidable": tot_avoid},
        "concentration_top_chunks_for_80pct": k80,
        "quartile_share_pct": {
            "by_n_access": [sum(avoidable[h] for h in nb[i]) / tot_avoid * 100 for i in range(4)],
            "by_span": [sum(avoidable[h] for h in sb[i]) / tot_avoid * 100 for i in range(4)],
        },
        "above_median_span_share_pct": long_lived_share,
        "top_half_share_pct": {"by_n_access": top_half_n, "by_span": top_half_s},
        "structural_signal_pct": signal,
    }
    gb_tag = f"{args.budget_gb:g}"
    out = log_path.parent / f"structure_vs_opt_cpu{gb_tag}.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"\nReport written to {out}")


if __name__ == "__main__":
    main()
