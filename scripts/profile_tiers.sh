#!/usr/bin/env bash
# =============================================================================
# profile_tiers.sh — lightweight GPU/CPU/SSD tier bandwidth+latency PROFILE
# on the MP+DP=2 shared-pool path: one `lmcache server` (shared CPU pool +
# SSD/L2 fs adapter) in front of one `vllm serve --data-parallel-size 2`.
#
# This is NOT a sweep (see run_sweep_mp.sh for that): it's a single short run
# whose only job is to (1) push enough traffic to exercise all three tiers —
# GPU eviction, CPU spill to SSD, SSD reload — and (2) read the answer off
# LMCache's own Prometheus histograms on the `lmcache server` process:
#   lmcache_mp_l0_l1_store_throughput / l0_l1_load_throughput  (GPU<->CPU, GB/s)
#   lmcache_mp_l2_store_throughput   / l2_load_throughput      (CPU<->SSD, GB/s)
# No access-log / long-run data collection is enabled — see profile_tiers.conf
# for the SSD/L2 caveats (no CPU-off mode, no built-in size cap).
#
# Copied from run_sweep_mp.sh's server-launch + safety scaffolding (setsid
# process groups, EXIT/INT/TERM trap, bounded health waits — see that file's
# header for the full safety-model writeup) rather than editing it in place,
# so the real sweep flow stays untouched. The delta from run_sweep_mp.sh:
#   - single (fixed) combo, not a policy x cpu_size loop
#   - `lmcache server` gets --l2-adapter/--l2-store-policy/--l2-prefetch-policy
#     (the SSD tier) and --prometheus-port (so its GB/s histograms are scrapable)
#   - a metrics-scrape step after the client exits, before teardown
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

# vLLM depends on the MP server while running: tear it down first.
cleanup() { stop_server; stop_mp_server; }
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
    local logf="$1"
    # FSL2Adapter persists files across shutdowns by default (lookup always
    # checks disk on miss), so a stale SSD_L2_PATH from a prior run would
    # start this run's SSD tier already warm — measuring a pre-populated
    # cache instead of the spill-and-reload this profile exists to observe,
    # and quietly eating the disk budget across runs. Wipe it fresh here.
    [[ -n "$SSD_L2_PATH" && "$SSD_L2_PATH" != "/" ]] && rm -rf "$SSD_L2_PATH"
    mkdir -p "$SSD_L2_PATH"
    local l2_adapter_json
    l2_adapter_json=$(printf '{"type":"fs","base_path":"%s"}' "$SSD_L2_PATH")
    setsid env \
        PYTHONHASHSEED=0 \
        lmcache server \
            --host "$MP_HOST" --port "$MP_PORT" \
            --http-host "$MP_HTTP_HOST" --http-port "$MP_HTTP_PORT" \
            --prometheus-port "$MP_PROM_PORT" \
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

# Scrapes the lmcache server's own Prometheus /metrics (GPU<->CPU and
# CPU<->SSD throughput histograms live there — see header). Must run BEFORE
# stop_mp_server, or the process (and the endpoint) is already gone.
scrape_tier_metrics() {
    local raw="$1" summary="$2" i ok=0
    for ((i = 0; i < METRICS_SCRAPE_RETRIES; i++)); do
        if curl -fsS "http://$MP_HTTP_HOST:$MP_PROM_PORT/metrics" -o "$raw" 2>/dev/null; then
            ok=1; break
        fi
        sleep 1
    done
    if [[ "$ok" != "1" ]]; then
        log "WARN: could not scrape /metrics from lmcache server on port $MP_PROM_PORT"
        return 1
    fi
    {
        echo "# GPU<->CPU (L0<->L1) and CPU<->SSD (L2) tier throughput, from lmcache"
        echo "# server's own Prometheus histograms (GB/s, submit->complete latency incl.)."
        echo "# _sum/_count on a histogram = mean GB/s across completed transfers;"
        echo "# _bucket lines give the distribution. Raw scrape: $raw"
        echo
        grep -E '^lmcache_mp_(l0_l1|l2)_(store|load)_throughput' "$raw"
    } > "$summary"
    log "Tier metrics summary -> $summary"
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
raw_metrics="$outdir/tier_metrics_raw.prom"
summary_metrics="$outdir/tier_metrics_summary.txt"
mkdir -p "$outdir"

log "Profile start: dp=$DATA_PARALLEL_SIZE l1=${L1_SIZE_GB}GB ssd=$SSD_L2_PATH -> $run"
log "Tester=$TESTER_DIR  Base=$BASE_OUTPUT_DIR  Master log=$MASTER_LOG"

launch_mp_server "$mplog"
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
    log "client FAILED/timeout (rc=$rc) — scraping metrics anyway for whatever traffic did happen"
fi

scrape_tier_metrics "$raw_metrics" "$summary_metrics"

stop_server
stop_mp_server
log "Profile done."
if [[ -f "$summary_metrics" ]]; then
    echo
    echo "=== Tier bandwidth/latency summary ($summary_metrics) ==="
    cat "$summary_metrics"
fi
