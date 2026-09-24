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
"""
import csv, json, statistics as st, sys
csv.field_size_limit(10**9)
N = int(sys.argv[1])
arms = [a.split("=", 1) for a in sys.argv[2:]]

def pct(xs, p):
    xs = sorted(xs); k = (len(xs) - 1) * p / 100; f = int(k); c = min(f + 1, len(xs) - 1)
    return xs[f] + (xs[c] - xs[f]) * (k - f)

R, W = {}, {}
for name, d in arms:
    rows = sorted((r for r in csv.DictReader(open(f"{d}/detailed_results.csv")) if r["success"] == "True"),
                  key=lambda r: float(r["request_complete_time"]))
    t0 = min(float(r["request_start_time"]) for r in rows)
    t_end = float(rows[min(N, len(rows)) - 1]["request_complete_time"])
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
    W[name] = dict(win_s=t_end - t0, gpu=100 * g / p, l1=100 * e / p, all=100 * (g + e) / p,
                   preempt=dd("vllm:num_preemptions_total"), **st_)

key = lambda r: (r["trace_id"], r["request_idx"])
common = set.intersection(*[set(map(key, R[n])) for n, _ in arms])
M = {n: [r for r in R[n] if key(r) in common] for n, _ in arms}
names = [n for n, _ in arms]
print(f"window = first {N} completed requests per arm; identical requests in every arm: {len(common)}")
print(f"{'':34}" + "".join(f"{n:>16}" for n in names))
def row(label, f, fmt):
    print(f"{label:34}" + "".join(fmt.format(f(n)) for n in names))
row("time for the window (s)", lambda n: W[n]["win_s"], "{:16.0f}")
row("GPU hit % of prompt (window)", lambda n: W[n]["gpu"], "{:16.1f}")
row("L1 hit % of prompt (window)", lambda n: W[n]["l1"], "{:16.1f}")
row("overall hit % (window)", lambda n: W[n]["all"], "{:16.1f}")
ct = lambda n: [r for r in M[n] if r.get("cached_tokens")]
row("hit % on identical requests", lambda n: 100 * sum(float(r["cached_tokens"]) for r in ct(n)) /
    sum(float(r["server_prompt_tokens"]) for r in ct(n)), "{:16.1f}")
row("prefill computed, identical (M)", lambda n: sum(float(r["server_prompt_tokens"]) - float(r["cached_tokens"])
    for r in ct(n)) / 1e6, "{:16.3f}")
tt = lambda n: [float(r["ttft"]) for r in M[n]]
row("TTFT mean (s)", lambda n: st.mean(tt(n)), "{:16.2f}")
row("TTFT p50 (s)", lambda n: pct(tt(n), 50), "{:16.2f}")
row("TTFT p99 (s)", lambda n: pct(tt(n), 99), "{:16.2f}")
row("TPOT mean (ms)", lambda n: st.mean([(float(r["ttlt"]) - float(r["ttft"])) / (int(r["output_tokens_actual"]) - 1) * 1000
    for r in M[n] if int(r["output_tokens_actual"] or 0) > 1]), "{:16.1f}")
row("vLLM preemptions (window)", lambda n: W[n]["preempt"], "{:16.0f}")
row("L1 on-demand eviction", lambda n: str(W[n].get("od", "-")), "{:>16}")
row("stores dropped (L1 full, end)", lambda n: str(W[n].get("oom", "-")), "{:>16}")
row("L1 usage at end", lambda n: ("%.2f" % W[n]["l1_use"]) if "l1_use" in W[n] else "-", "{:>16}")
