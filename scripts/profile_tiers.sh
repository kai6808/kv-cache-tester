#!/usr/bin/env bash
# =============================================================================
# profile_tiers.sh — lightweight GPU/CPU/SSD tier bandwidth+latency PROFILE
# on the MP+DP=2 shared-pool path: one `lmcache server` (shared CPU pool +
# SSD/L2 fs adapter) in front of one `vllm serve --data-parallel-size 2`.
#
# This is NOT a sweep (see run_sweep_mp.sh for that): it's a single short run
# whose only job is to (1) push enough traffic to exercise all three tiers —
# GPU eviction, CPU spill to SSD, SSD reload — and (2) compute bandwidth +
# latency per tier from what LMCache already writes to disk.
#
# NOTE on where the numbers come from (verified 2026-09-09 against the first
# real run on amd-01 — the naive plan didn't work, this is the fallback):
# LMCache's `lmcache server` DOES compute proper OTel throughput histograms
# (lmcache_mp_l0_l1_*_throughput, lmcache_mp_l2_*_throughput, both GB/s) —
# but upstream commit ed3fa706 hardcodes start_prometheus_http_server=False
# for this entrypoint (lmcache/v1/multiprocess/http_server.py) and nothing
# else mounts a /metrics route for it, so those histograms are UNREACHABLE
# over HTTP on this stack right now. Fixing that needs a source patch to the
# lmcache fork, which was explicitly declined (kv-cache-tester-only changes
# preferred) — so instead this script derives the same two numbers itself
# from files LMCache already produces, no source changes:
#   - GPU<->CPU (L0<->L1): per-transfer latency + bandwidth, computed by
#     pairing offload_*/prefetch_* submit/start/end events in the MP-path
#     LMCACHE_ACCESS_LOG JSONL (same event stream the real l0_l1_throughput
#     subscriber consumes — just paired here instead of inside the process).
#   - CPU<->SSD (L1<->L2): AVERAGE bandwidth only (no per-transfer latency —
#     fs_l2_adapter.py's DEBUG log lines carry bytes per key but not timing),
#     computed as total bytes moved / wall-clock span of those DEBUG lines in
#     the server log. Weaker signal than L0<->L1 on purpose — see
#     compute_tier_metrics() below for the exact caveat text this writes out.
#
# Copied from run_sweep_mp.sh's server-launch + safety scaffolding (setsid
# process groups, EXIT/INT/TERM trap, bounded health waits — see that file's
# header for the full safety-model writeup) rather than editing it in place,
# so the real sweep flow stays untouched. The delta from run_sweep_mp.sh:
#   - single (fixed) combo, not a policy x cpu_size loop
#   - `lmcache server` gets --l2-adapter/--l2-store-policy/--l2-prefetch-policy
#     (the SSD tier), LMCACHE_LOG_LEVEL=DEBUG, and LMCACHE_ACCESS_LOG on
#   - a tier-metrics computation step after teardown (see compute_tier_metrics)
#
# Usage:
#   scripts/profile_tiers.sh [path/to/profile_tiers.conf]   # default: scripts/profile_tiers.conf
# =============================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TESTER_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
CONFIG="${1:-$SCRIPT_DIR/profile_tiers.conf}"

[[ -f "$CONFIG" ]] || { echo "ERROR: config not found: $CONFIG" >&2; exit 1; }
# shellcheck disable=SC1090
source "$CONFIG"

mkdir -p "$BASE_OUTPUT_DIR"
MASTER_LOG="$BASE_OUTPUT_DIR/profile_$(date +%Y%m%d_%H%M%S).log"

SERVER_PID=""; SERVER_PGID=""
MP_SERVER_PID=""; MP_SERVER_PGID=""

log() { echo "[$(date '+%F %T')] $*" | tee -a "$MASTER_LOG"; }

health_ok() { curl -fsS -o /dev/null "http://$HOST:$PORT/health" 2>/dev/null; }
mp_health_ok() { curl -fsS -o /dev/null "http://$MP_HTTP_HOST:$MP_HTTP_PORT/healthcheck" 2>/dev/null; }

# Kill the vLLM process group; idempotent; clears SERVER_PID when done.
stop_server() {
    [[ -z "$SERVER_PID" ]] && return 0
    if kill -0 "$SERVER_PID" 2>/dev/null; then
        log "Stopping vLLM (pid=$SERVER_PID pgid=$SERVER_PGID), SIGINT then wait..."
        kill -INT "-$SERVER_PGID" 2>/dev/null || kill -INT "$SERVER_PID" 2>/dev/null || true
        local i
        for ((i = 0; i < SERVER_STOP_GRACE; i++)); do
            kill -0 "$SERVER_PID" 2>/dev/null || break
            sleep 1
        done
        if kill -0 "$SERVER_PID" 2>/dev/null; then
            log "vLLM still alive after ${SERVER_STOP_GRACE}s, SIGKILL group."
            kill -KILL "-$SERVER_PGID" 2>/dev/null || kill -KILL "$SERVER_PID" 2>/dev/null || true
        fi
    fi
    SERVER_PID=""; SERVER_PGID=""
    local deadline=$((SECONDS + SERVER_STOP_TIMEOUT))
    while ((SECONDS < deadline)); do health_ok || return 0; sleep "$POLL_INTERVAL"; done
    log "WARN: something still answers vLLM /health after stop timeout."
}

# Kill the lmcache server process group; idempotent; clears MP_SERVER_PID when done.
stop_mp_server() {
    [[ -z "$MP_SERVER_PID" ]] && return 0
    if kill -0 "$MP_SERVER_PID" 2>/dev/null; then
        log "Stopping lmcache server (pid=$MP_SERVER_PID pgid=$MP_SERVER_PGID), SIGINT then wait..."
        kill -INT "-$MP_SERVER_PGID" 2>/dev/null || kill -INT "$MP_SERVER_PID" 2>/dev/null || true
        local i
        for ((i = 0; i < MP_SERVER_STOP_GRACE; i++)); do
            kill -0 "$MP_SERVER_PID" 2>/dev/null || break
            sleep 1
        done
        if kill -0 "$MP_SERVER_PID" 2>/dev/null; then
            log "lmcache server still alive after ${MP_SERVER_STOP_GRACE}s, SIGKILL group."
            kill -KILL "-$MP_SERVER_PGID" 2>/dev/null || kill -KILL "$MP_SERVER_PID" 2>/dev/null || true
        fi
    fi
    MP_SERVER_PID=""; MP_SERVER_PGID=""
    local deadline=$((SECONDS + MP_SERVER_STOP_TIMEOUT))
    while ((SECONDS < deadline)); do mp_health_ok || return 0; sleep "$POLL_INTERVAL"; done
    log "WARN: something still answers lmcache /healthcheck after stop timeout."
}

# Wipes the SSD/L2 scratch dir. Safe to call unconditionally: SSD_L2_PATH is
# exclusive to this profile (freshly wiped at launch too, see launch_mp_server)
# and holds nothing but disposable cache blobs — never the run's real output
# (logs / tier_metrics_summary.txt / trace_replay_tester.py's HTML+CSV live
# under $outdir / $BASE_OUTPUT_DIR instead, untouched by this).
wipe_ssd_scratch() {
    [[ -n "${SSD_L2_PATH:-}" && "$SSD_L2_PATH" != "/" ]] || return 0
    rm -rf "$SSD_L2_PATH"
}

# vLLM depends on the MP server while running: tear it down first. SSD wipe
# runs in every exit path (normal, Ctrl-C, startup failure) via this trap,
# not just the happy path at the bottom of the script.
cleanup() { stop_server; stop_mp_server; wipe_ssd_scratch; }
trap cleanup EXIT INT TERM

# 0.5 -> g0p5 ; build "64k_g0p5_dp2_l1_20gb_16u_300s"
build_run_name() {
    local gpu_tag ctx_k
    gpu_tag="g$(printf '%s' "$GPU_MEM_UTIL" | tr '.' 'p')"
    ctx_k=$((MAX_CONTEXT / 1000))
    printf '%s%sk_%s_dp%s_l1_%sgb_%su_%ss' \
        "${RUN_PREFIX:+${RUN_PREFIX}_}" "$ctx_k" "$gpu_tag" "$DATA_PARALLEL_SIZE" \
        "$L1_SIZE_GB" "$MAX_USERS" "$TEST_DURATION"
}

launch_mp_server() {
    local logf="$1" access_log="$2"
    # FSL2Adapter persists files across shutdowns by default (lookup always
    # checks disk on miss), so a stale SSD_L2_PATH from a prior run would
    # start this run's SSD tier already warm — measuring a pre-populated
    # cache instead of the spill-and-reload this profile exists to observe,
    # and quietly eating the disk budget across runs. Wipe it fresh here.
    [[ -n "$SSD_L2_PATH" && "$SSD_L2_PATH" != "/" ]] && rm -rf "$SSD_L2_PATH"
    mkdir -p "$SSD_L2_PATH"
    local l2_adapter_json
    l2_adapter_json=$(printf '{"type":"fs","base_path":"%s"}' "$SSD_L2_PATH")
    # LMCACHE_LOG_LEVEL=DEBUG: fs_l2_adapter.py only logs per-key byte counts
    # (SSD tier bandwidth) at DEBUG. LMCACHE_ACCESS_LOG: the JSONL stream
    # compute_tier_metrics() parses for GPU<->CPU per-transfer latency. Both
    # required now that the Prometheus route (see header) is unreachable.
    setsid env \
        PYTHONHASHSEED=0 \
        LMCACHE_LOG_LEVEL="$MP_LOG_LEVEL" \
        LMCACHE_ACCESS_LOG="$access_log" \
        lmcache server \
            --host "$MP_HOST" --port "$MP_PORT" \
            --http-host "$MP_HTTP_HOST" --http-port "$MP_HTTP_PORT" \
            --max-workers "$MP_MAX_WORKERS" \
            --chunk-size "$MP_CHUNK_SIZE" \
            --l1-size-gb "$L1_SIZE_GB" \
            --l1-write-ttl-seconds "$MP_L1_WRITE_TTL_SECONDS" \
            --l1-read-ttl-seconds "$MP_L1_READ_TTL_SECONDS" \
            --eviction-policy "$EVICTION_POLICY" \
            --l2-adapter "$l2_adapter_json" \
            --l2-store-policy "$L2_STORE_POLICY" \
            --l2-prefetch-policy "$L2_PREFETCH_POLICY" \
            >"$logf" 2>&1 &
    MP_SERVER_PID=$!
    MP_SERVER_PGID="$(ps -o pgid= -p "$MP_SERVER_PID" 2>/dev/null | tr -d ' ')"
    [[ -z "$MP_SERVER_PGID" ]] && MP_SERVER_PGID="$MP_SERVER_PID"
}

wait_mp_server_ready() {
    local logf="$1" deadline=$((SECONDS + MP_SERVER_START_TIMEOUT))
    while ((SECONDS < deadline)); do
        if ! kill -0 "$MP_SERVER_PID" 2>/dev/null; then
            log "ERROR: lmcache server exited during startup (tail $logf):"
            tail -n 15 "$logf" | tee -a "$MASTER_LOG"
            return 1
        fi
        mp_health_ok && return 0
        sleep "$POLL_INTERVAL"
    done
    log "ERROR: lmcache server not ready within ${MP_SERVER_START_TIMEOUT}s."
    return 1
}

launch_server() {
    local logf="$1"
    rm -rf "$PROM_DIR" && mkdir -p "$PROM_DIR"
    local kv_transfer_config
    kv_transfer_config=$(printf '{"kv_connector":"LMCacheMPConnector","kv_role":"kv_both","kv_load_failure_policy":"%s","kv_connector_extra_config":{"lmcache.mp.port":%s,"lmcache.mp.mq_timeout":%s}}' \
        "$KV_LOAD_FAILURE_POLICY" "$MP_PORT" "$MP_MQ_TIMEOUT")
    # setsid => new process group we can kill wholesale (vllm spawns DP workers).
    setsid env \
        PYTHONHASHSEED=0 \
        PROMETHEUS_MULTIPROC_DIR="$PROM_DIR" \
        HIP_VISIBLE_DEVICES="$HIP_VISIBLE_DEVICES" \
        vllm serve "$MODEL" \
            --host "$HOST" --port "$PORT" \
            --data-parallel-size "$DATA_PARALLEL_SIZE" \
            --block-size "$BLOCK_SIZE" \
            --enable-prefix-caching \
            --gpu-memory-utilization "$GPU_MEM_UTIL" \
            --max-model-len "$MAX_MODEL_LEN" \
            --max-num-seqs "$MAX_NUM_SEQS" \
            --kv-cache-metrics \
            --kv-cache-metrics-sample "$KV_CACHE_METRICS_SAMPLE" \
            --hf-overrides "$HF_OVERRIDES" \
            --kv-transfer-config "$kv_transfer_config" \
            >"$logf" 2>&1 &
    SERVER_PID=$!
    SERVER_PGID="$(ps -o pgid= -p "$SERVER_PID" 2>/dev/null | tr -d ' ')"
    [[ -z "$SERVER_PGID" ]] && SERVER_PGID="$SERVER_PID"
}

wait_server_ready() {
    local logf="$1" deadline=$((SECONDS + SERVER_START_TIMEOUT))
    while ((SECONDS < deadline)); do
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            log "ERROR: vLLM exited during startup (tail $logf):"; tail -n 15 "$logf" | tee -a "$MASTER_LOG"
            return 1
        fi
        health_ok && return 0
        sleep "$POLL_INTERVAL"
    done
    log "ERROR: vLLM not ready within ${SERVER_START_TIMEOUT}s."
    return 1
}

# Runs the client to completion; full terminal output (banner + 'Test Complete')
# is tee'd into <run>.client.log. Returns the client's exit status.
run_client() {
    local outdir="$1" logf="$2" cap=$((TEST_DURATION + CLIENT_TIMEOUT_BUFFER))
    local _max_req=()
    [[ -n "${MAX_REQUESTS:-}" ]] && _max_req=(--max-requests "$MAX_REQUESTS")
    ( cd "$TESTER_DIR" && \
        timeout --signal=INT "$cap" \
        python3 trace_replay_tester.py \
            --api-endpoint "http://$HOST:$PORT" \
            --trace-directory "$TRACE_DIR" \
            --output-dir "$outdir" \
            --tokenizer "$TOKENIZER" \
            --max-context "$MAX_CONTEXT" \
            --chunk-size "$CHUNK_SIZE" \
            --max-concurrent-requests "$MAX_CONCURRENT" \
            --start-users "$START_USERS" --max-users "$MAX_USERS" \
            --max-traces "$MAX_TRACES" \
            --test-duration "$TEST_DURATION" \
            "${_max_req[@]}" \
            --server-metrics \
            --timing-strategy "$TIMING_STRATEGY" \
            --trace-seed "$SEED" --prompt-seed "$SEED" --seed "$SEED" \
            --max-ttft "$MAX_TTFT" \
    ) 2>&1 | tee "$logf"
    return "${PIPESTATUS[0]}"
}

# Computes per-tier bandwidth+latency from files LMCache already wrote —
# no live server needed, safe to call after teardown. See header NOTE for
# why this exists instead of scraping Prometheus.
#   - GPU<->CPU (L0<->L1): pairs offload_*/prefetch_* submit/start/end
#     events in the access-log JSONL by session_id (mirrors what the real
#     l0_l1_throughput OTel subscriber does inside the process) -> real
#     per-transfer latency + bandwidth.
#   - CPU<->SSD (L1<->L2): sums FSL2Adapter's per-key DEBUG byte counts and
#     divides by the wall-clock span between the first and last such line ->
#     one AVERAGE bandwidth number for the whole run, no per-transfer
#     latency (that field isn't in fs_l2_adapter.py's log output).
compute_tier_metrics() {
    local mplog="$1" access_log="$2" summary="$3"
    python3 - "$mplog" "$access_log" "$summary" <<'PYEOF'
import glob, json, os, re, sys
from datetime import datetime

mplog_path, access_log_base, summary_path = sys.argv[1:4]

# LMCACHE_ACCESS_LOG is a *base* path — lmcache.utils.timestamped_pid_path()
# rewrites it to "<base>.<YYYYMMDD_HHMMSS>.<pid><ext>" before ever opening a
# file (single process here, so exactly one match expected; newest by mtime
# wins if launch_mp_server's pre-run glob-clean below ever misses one).
_base, _ext = os.path.splitext(access_log_base)
_matches = sorted(glob.glob(f"{_base}.*{_ext}"), key=os.path.getmtime, reverse=True)
access_log_path = _matches[0] if _matches else access_log_base

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
TS_RE = re.compile(r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),(\d{3})\]")
STORE_RE = re.compile(r"FSL2Adapter stored key \S+ \((\d+) bytes\)")
LOAD_RE = re.compile(r"FSL2Adapter loaded key \S+ \((\d+) bytes")


def parse_ts(line):
    m = TS_RE.search(line)
    if not m:
        return None
    dt = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
    return dt.timestamp() + int(m.group(2)) / 1000.0


# ---- CPU<->SSD (L1<->L2): average bandwidth from DEBUG byte-count lines ----
store_bytes = store_n = load_bytes = load_n = 0
first_ts = last_ts = None
try:
    with open(mplog_path, errors="replace") as f:
        for raw in f:
            line = ANSI_RE.sub("", raw)
            ts = parse_ts(line)
            m = STORE_RE.search(line)
            if m:
                store_bytes += int(m.group(1))
                store_n += 1
            else:
                m = LOAD_RE.search(line)
                if m:
                    load_bytes += int(m.group(1))
                    load_n += 1
                else:
                    continue
            if ts is not None:
                first_ts = ts if first_ts is None else min(first_ts, ts)
                last_ts = ts if last_ts is None else max(last_ts, ts)
except FileNotFoundError:
    pass

window = last_ts - first_ts if (first_ts is not None and last_ts is not None and last_ts > first_ts) else None


# ---- GPU<->CPU (L0<->L1): per-transfer latency+bandwidth from access log --
def paired_samples(events, start_op, end_op, bytes_key):
    # Simplifying assumption for a short profiling run: one start->end pair
    # per session_id per direction (matches "three lines per RPC, correlated
    # by session_id" in the access-log schema docstring). A session_id that
    # legitimately issues >1 RPC in one direction will only keep the latest
    # pending start — acceptable slop for a lightweight profile, not for a
    # rigorous measurement.
    pending, samples = {}, []
    for e in events:
        op = e.get("op")
        sid = e.get("session_id")
        if op == start_op:
            pending[sid] = e.get("t")
        elif op == end_op:
            t0 = pending.pop(sid, None)
            t1, b = e.get("t"), e.get(bytes_key, 0)
            if t0 is not None and t1 is not None and t1 > t0 and b:
                samples.append((t1 - t0, b))
    return samples


offload_events, prefetch_events = [], []
try:
    with open(access_log_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            op = e.get("op", "")
            if op.startswith("offload_"):
                offload_events.append(e)
            elif op.startswith("prefetch_") and op != "prefetch_lookup":
                prefetch_events.append(e)
except FileNotFoundError:
    pass

store_samples = paired_samples(offload_events, "offload_start", "offload_end", "total_bytes")
load_samples = paired_samples(prefetch_events, "prefetch_start", "prefetch_end", "total_bytes")


def fmt_samples(name, samples):
    if not samples:
        return f"{name}: no completed transfers observed in this run\n"
    n = len(samples)
    total_bytes = sum(b for _, b in samples)
    total_lat = sum(t for t, _ in samples)
    mean_gbps = sum((b / 1e9) / t for t, b in samples if t > 0) / n
    overall_gbps = (total_bytes / 1e9) / total_lat if total_lat > 0 else 0.0
    mean_lat_ms = (total_lat / n) * 1000
    return (
        f"{name}: n={n}  mean={mean_gbps:.3f} GB/s  "
        f"overall(bytes/time)={overall_gbps:.3f} GB/s  mean_latency={mean_lat_ms:.1f} ms  "
        f"total={total_bytes / 1e9:.3f} GB\n"
    )


out = []
out.append(f"=== GPU<->CPU (L0<->L1), per-transfer, from LMCACHE_ACCESS_LOG ({access_log_path}) ===\n")
out.append("NOTE: t is when the JSONL writer received each event over the MP bus, not\n")
out.append("necessarily the GPU-stream timestamp the real l0_l1_throughput OTel histogram uses\n")
out.append("internally -- read these as submit->complete RPC latency (incl. queueing), not a\n")
out.append("pure wire transfer rate.\n")
out.append(fmt_samples("L0->L1 store (GPU->CPU)", store_samples))
out.append(fmt_samples("L1->L0 load  (CPU->GPU)", load_samples))
out.append("\n")
out.append(f"=== CPU<->SSD (L1<->L2), AVERAGE over the run window, from LMCACHE_LOG_LEVEL=DEBUG lines ({mplog_path}) ===\n")
out.append("NOTE: fs_l2_adapter.py logs bytes per key but not per-transfer duration, so this is\n")
out.append("total_bytes / (first-to-last DEBUG-line wall-clock span) -- an AVERAGE rate over the\n")
out.append("whole run, NOT a per-transfer latency distribution like the L0<->L1 numbers above.\n")
out.append("Real per-transfer L2 latency needs the lmcache_mp_l2_*_throughput OTel histograms,\n")
out.append("which needs a source patch to lmcache/v1/multiprocess/http_server.py (see this\n")
out.append("script's header) -- skipped per your no-lmcache-source-changes preference.\n")
if window and store_n:
    out.append(f"L1->L2 store (CPU->SSD): n={store_n}  total={store_bytes / 1e9:.3f} GB  window={window:.1f}s  avg={(store_bytes / 1e9) / window:.3f} GB/s\n")
else:
    out.append(f"L1->L2 store (CPU->SSD): n={store_n}  total={store_bytes / 1e9:.3f} GB  (window unavailable -- rate not computed)\n")
if window and load_n:
    out.append(f"L2->L1 load  (SSD->CPU): n={load_n}  total={load_bytes / 1e9:.3f} GB  window={window:.1f}s  avg={(load_bytes / 1e9) / window:.3f} GB/s\n")
else:
    out.append(f"L2->L1 load  (SSD->CPU): n={load_n}  total={load_bytes / 1e9:.3f} GB  (window unavailable -- rate not computed)\n")
if store_n == 0 and load_n == 0:
    out.append("\nZero L2 (SSD) events: either LMCACHE_LOG_LEVEL=DEBUG wasn't applied, or L1_SIZE_GB\n")
    out.append("was large enough that nothing spilled to SSD this run -- try lowering L1_SIZE_GB.\n")

with open(summary_path, "w") as f:
    f.writelines(out)
print("".join(out), end="")
PYEOF
}

# ---- preflight --------------------------------------------------------------
command -v curl >/dev/null    || { echo "ERROR: curl required" >&2; exit 1; }
command -v setsid >/dev/null  || { echo "ERROR: setsid required" >&2; exit 1; }
command -v vllm >/dev/null    || { echo "ERROR: vllm not on PATH" >&2; exit 1; }
command -v lmcache >/dev/null || { echo "ERROR: lmcache CLI not on PATH" >&2; exit 1; }
if health_ok; then
    log "ERROR: something is already serving on $HOST:$PORT — refusing to start. Stop it first."
    exit 1
fi
if mp_health_ok; then
    log "ERROR: something is already serving on $MP_HTTP_HOST:$MP_HTTP_PORT — refusing to start. Stop it first."
    exit 1
fi

run="$(build_run_name)"
outdir="$BASE_OUTPUT_DIR/$run"
mplog="$BASE_OUTPUT_DIR/${run}.mpserver.log"
slog="$BASE_OUTPUT_DIR/${run}.server.log"
clog="$BASE_OUTPUT_DIR/${run}.client.log"
access_log="$outdir/cpu_access.jsonl"
summary_metrics="$outdir/tier_metrics_summary.txt"
mkdir -p "$outdir"
# build_run_name() is deterministic on the conf values, so re-running this
# same conf reuses the same $outdir. LMCACHE_ACCESS_LOG's actual filename
# gets a timestamp+pid inserted by lmcache (see compute_tier_metrics), so a
# stale file from a prior run here would otherwise make the glob ambiguous
# (or, worse, silently mix two runs' transfers into one summary).
rm -f "$outdir"/cpu_access.*.jsonl

log "Profile start: dp=$DATA_PARALLEL_SIZE l1=${L1_SIZE_GB}GB ssd=$SSD_L2_PATH -> $run"
log "Tester=$TESTER_DIR  Base=$BASE_OUTPUT_DIR  Master log=$MASTER_LOG"

launch_mp_server "$mplog" "$access_log"
log "lmcache server launched (pid=$MP_SERVER_PID), waiting for /healthcheck ..."
if ! wait_mp_server_ready "$mplog"; then
    log "ABORT: lmcache server failed to start."
    exit 1
fi

launch_server "$slog"
log "vLLM launched (pid=$SERVER_PID), waiting for /health ..."
if ! wait_server_ready "$slog"; then
    log "ABORT: vLLM failed to start."
    exit 1
fi
log "servers READY -> running client (log: $clog)"

if run_client "$outdir" "$clog"; then
    log "client OK -> $outdir"
else
    rc=$?
    log "client FAILED/timeout (rc=$rc) — computing tier metrics anyway for whatever traffic did happen"
fi

stop_server
stop_mp_server

# Files are flushed once both processes exit cleanly (graceful SIGINT above,
# not -9), so this runs after teardown rather than racing a live server.
compute_tier_metrics "$mplog" "$access_log" "$summary_metrics"
log "Tier metrics summary -> $summary_metrics"
log "Profile done."
if [[ -f "$summary_metrics" ]]; then
    echo
    echo "=== Tier bandwidth/latency summary ($summary_metrics) ==="
    cat "$summary_metrics"
fi
