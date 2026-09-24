"""Compare fixed-workload runs on identical requests, with the GPU / L1 split.

usage: compare_fixed_workload.py N name=dir [name=dir ...]
  N    window: each arm from its start to its N-th completed request, so a
       150-request run lines up with the start of a 400-request run
  dir  one arm's output dir (holds detailed_results.csv, server_metrics.json,
       mp_status.json)

"Identical requests" = the (trace_id, request_idx) pairs every arm served in
its window; that is the fair hit-rate comparison (same prompts, same
denominator). Needs vLLM's --enable-prompt-tokens-details for per-request
cached tokens.
Per-request `cached_tokens` = GPU + L1 hits together; the split comes from
the vLLM counters over the same window (30 s snapshots, nearest after).
vllm:prompt_tokens_cached is local + external (L1-loaded) tokens, so GPU hits
= cached - external_prefix_cache_hits and overall = cached. (external hits are
also re-recorded when a preempted request is rescheduled; preemptions are few.)
"""
import csv, json, os, re, statistics as st, sys
csv.field_size_limit(10**9)


def pct(xs, p):
    xs = sorted(xs); k = (len(xs) - 1) * p / 100; f = int(k); c = min(f + 1, len(xs) - 1)
    return xs[f] + (xs[c] - xs[f]) * (k - f)


def compare(N, arms):
    """Compare arms over their first N completed requests.

    Args:
        N: Window size in completed requests.
        arms: ``[(name, dir), ...]`` in column order.

    Returns:
        ``(n_identical, rows)``; each row is ``(key, label, fmt, {name: value})``
        with ``value`` None where an arm lacks the data. ``fmt`` is a
        ``str.format`` spec for numbers, or None for values printed as-is.
    """
    R, W = {}, {}
    for name, d in arms:
        rows = sorted((r for r in csv.DictReader(open(f"{d}/detailed_results.csv")) if r["success"] == "True"),
                      key=lambda r: float(r["request_complete_time"]))
        t0 = min(float(r["request_start_time"]) for r in rows)
        t_end = float(rows[min(N, len(rows)) - 1]["request_complete_time"])
        elapsed = float(rows[-1]["request_complete_time"]) - t0
        R[name] = rows[:N]
        snaps = json.load(open(f"{d}/server_metrics.json"))
        s1 = next((s for s in snaps if s["wall_time"] >= t_end), snaps[-1])
        C = lambda s, k: s["counters"].get(k, 0.0)
        dd = lambda k: C(s1, k) - C(snaps[0], k)
        p = dd("vllm:prompt_tokens_total"); g = dd("vllm:prompt_tokens_cached_total")
        e = dd("vllm:external_prefix_cache_hits_total")
        st_ = {}
        try:
            ms = json.load(open(f"{d}/mp_status.json"))
            ec = ms["storage_manager"]["l1_eviction_controller"]
            st_ = {"od": ec.get("eviction_on_demand"), "oom": ec.get("store_oom_keys"),
                   "od_ev": ec.get("on_demand_evicted"),
                   "l1_use": ms["storage_manager"]["l1_manager"]["memory_usage_ratio"]}
        except Exception:
            pass
        # L1 eviction volume over the whole run, the same way for every policy
        # (LRU reports no eviction count): chunks stored minus chunks resident at
        # the end, from the arm's MP-server log next to its dir. Includes the ~25
        # warm-up chunks cleared before the measured client starts.
        try:
            chunk = json.load(open(f"{d}/mp_status.json"))["chunk_size"]
            stored = trig = 0
            for line in open(os.path.join(os.path.dirname(d), os.path.basename(d) + ".mpserver.log")):
                m = re.search(r"Stored (\d+) tokens", line)
                if m:
                    stored += int(m.group(1)) // chunk
                elif "triggering eviction" in line:
                    trig += 1
            l1 = ms["storage_manager"]["l1_manager"]
            cap = l1["total_object_count"] / max(l1["memory_usage_ratio"], 1e-9)
            st_.update(ev=stored - l1["total_object_count"], trig=trig, turn=(stored - l1["total_object_count"]) / cap)
        except Exception:
            pass
        W[name] = dict(win_s=t_end - t0, gpu=100 * (g - e) / p, l1=100 * e / p, all=100 * g / p,
                       preempt=dd("vllm:num_preemptions_total"), elapsed_s=elapsed, **st_)

    key = lambda r: (r["trace_id"], r["request_idx"])
    common = set.intersection(*[set(map(key, R[n])) for n, _ in arms])
    M = {n: [r for r in R[n] if key(r) in common] for n, _ in arms}
    names = [n for n, _ in arms]
    out = []

    def row(k, label, f, fmt):
        out.append((k, label, fmt, {n: f(n) for n in names}))

    row("win_s", "time for the window (s)", lambda n: W[n]["win_s"], "{:16.0f}")
    row("gpu", "GPU hit % of prompt (window)", lambda n: W[n]["gpu"], "{:16.1f}")
    row("l1", "L1 hit % of prompt (window)", lambda n: W[n]["l1"], "{:16.1f}")
    row("all", "overall hit % (window)", lambda n: W[n]["all"], "{:16.1f}")
    ct = lambda n: [r for r in M[n] if r.get("cached_tokens")]
    row("ident_hit", "hit % on identical requests", lambda n: 100 * sum(float(r["cached_tokens"]) for r in ct(n)) /
        sum(float(r["server_prompt_tokens"]) for r in ct(n)), "{:16.1f}")
    row("prefill_m", "prefill computed, identical (M)", lambda n: sum(float(r["server_prompt_tokens"]) - float(r["cached_tokens"])
        for r in ct(n)) / 1e6, "{:16.3f}")
    tt = lambda n: [float(r["ttft"]) for r in M[n]]
    row("ttft_mean", "TTFT mean (s)", lambda n: st.mean(tt(n)), "{:16.2f}")
    row("ttft_p50", "TTFT p50 (s)", lambda n: pct(tt(n), 50), "{:16.2f}")
    row("ttft_p99", "TTFT p99 (s)", lambda n: pct(tt(n), 99), "{:16.2f}")
    tp = lambda n: [(float(r["ttlt"]) - float(r["ttft"])) / (int(r["output_tokens_actual"]) - 1) * 1000
                     for r in M[n] if int(r["output_tokens_actual"] or 0) > 1]
    row("tpot_mean", "TPOT mean (ms)", lambda n: st.mean(tp(n)), "{:16.1f}")
    row("tpot_p50", "TPOT p50 (ms)", lambda n: pct(tp(n), 50), "{:16.1f}")
    row("tpot_p99", "TPOT p99 (ms)", lambda n: pct(tp(n), 99), "{:16.1f}")
    e2e = lambda n: [float(r["ttlt"]) for r in M[n]]
    row("e2e_mean", "E2E latency mean (s)", lambda n: st.mean(e2e(n)), "{:16.2f}")
    row("e2e_p50", "E2E latency p50 (s)", lambda n: pct(e2e(n), 50), "{:16.2f}")
    row("e2e_p99", "E2E latency p99 (s)", lambda n: pct(e2e(n), 99), "{:16.2f}")
    # Throughput over each arm's own window (all N requests, not just identical).
    row("req_min", "requests/min (window)", lambda n: 60 * len(R[n]) / W[n]["win_s"], "{:16.2f}")
    row("out_tps", "output tok/s (window)", lambda n: sum(int(r["output_tokens_actual"] or 0) for r in R[n]) / W[n]["win_s"],
        "{:16.1f}")
    row("prompt_tps", "prompt tok/s (window)", lambda n: sum(float(r["server_prompt_tokens"] or 0) for r in R[n]) / W[n]["win_s"],
        "{:16.0f}")
    row("elapsed_s", "client elapsed, all requests (s)", lambda n: W[n]["elapsed_s"], "{:16.0f}")
    row("preempt", "vLLM preemptions (window)", lambda n: W[n]["preempt"], "{:16.0f}")
    row("l1_evicted", "L1 chunks evicted (run)", lambda n: W[n].get("ev"), "{:16.0f}")
    row("l1_turnovers", "L1 turnovers (evicted / capacity)", lambda n: W[n].get("turn"), "{:16.1f}")
    row("l1_rounds", "L1 watermark eviction rounds", lambda n: W[n].get("trig"), "{:16.0f}")
    row("od_evicted", "L1 on-demand evictions (chunks)", lambda n: W[n].get("od_ev"), "{:16.0f}")
    row("od", "L1 on-demand eviction", lambda n: W[n].get("od"), None)
    row("dropped", "stores dropped (L1 full, end)", lambda n: W[n].get("oom"), "{:16.0f}")
    row("l1_use", "L1 usage at end", lambda n: W[n].get("l1_use"), "{:16.2f}")
    return len(common), out


def _cell(v, fmt):
    if v is None:
        return f"{'-':>16}"
    if fmt is None:
        return f"{str(v):>16}"
    return fmt.format(v)


def main(argv):
    N = int(argv[1])
    arms = [a.split("=", 1) for a in argv[2:]]
    names = [n for n, _ in arms]
    n_common, rows = compare(N, arms)
    print(f"window = first {N} completed requests per arm; identical requests in every arm: {n_common}")
    print("latency rows use the identical requests; throughput rows use each arm's whole window")
    print(f"{'':34}" + "".join(f"{n:>16}" for n in names))
    for _, label, fmt, vals in rows:
        print(f"{label:34}" + "".join(_cell(vals[n], fmt) for n in names))


if __name__ == "__main__":
    main(sys.argv)
