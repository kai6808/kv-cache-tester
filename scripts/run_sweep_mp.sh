#!/usr/bin/env bash
# =============================================================================
# run_sweep_mp.sh — Rung 0 MP+DP variant of run_sweep.sh. Sweeps
# `lmcache server` eviction policy x L1 pool size, with vLLM running
# `--data-parallel-size N` against a standalone shared-pool MP server instead
# of the in-process LMCacheConnectorV1. See
# .claude/plans/2026-07-14-rung0-multiworker-headroom-gate.md for the recipe
# and background; this script is milestone 1.2.
#
# Each run now has TWO server processes to manage instead of one:
#   1. `lmcache server` (ZMQ + HTTP) — the shared L1 pool, started first.
#   2. `vllm serve --data-parallel-size N` — connects to it via LMCacheMPConnector.
# Same safety discipline as run_sweep.sh: each server runs in its own process
# group (setsid); an EXIT/INT/TERM trap always tears both down, in dependency
# order (vLLM before the MP server it depends on); every wait has a timeout;
# one failing combo does not abort the sweep.
#
# Usage:
#   scripts/run_sweep_mp.sh [path/to/sweep_mp.conf]   # default: scripts/sweep_mp.conf
# =============================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TESTER_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
CONFIG="${1:-$SCRIPT_DIR/sweep_mp.conf}"

[[ -f "$CONFIG" ]] || { echo "ERROR: config not found: $CONFIG" >&2; exit 1; }
# shellcheck disable=SC1090
source "$CONFIG"

mkdir -p "$BASE_OUTPUT_DIR"
MASTER_LOG="$BASE_OUTPUT_DIR/sweep_mp_$(date +%Y%m%d_%H%M%S).log"

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

# 0.5 -> g0p5 ; build "64k_g0p5_mp_dp2_cpu20_lru_50u_3600s"
build_run_name() {
    local policy="$1" cpu="$2" gpu_tag ctx_k pol
    gpu_tag="g$(printf '%s' "$GPU_MEM_UTIL" | tr '.' 'p')"
    ctx_k=$((MAX_CONTEXT / 1000))
    pol="$(printf '%s' "$policy" | tr '[:upper:]' '[:lower:]')"
    printf '%s%sk_%s_mp_dp%s_cpu%s_%s_%su_%ss' \
        "${RUN_PREFIX:+${RUN_PREFIX}_}" "$ctx_k" "$gpu_tag" "$DATA_PARALLEL_SIZE" \
        "$cpu" "$pol" "$MAX_USERS" "$TEST_DURATION"
}

launch_mp_server() {
    local policy="$1" cpu="$2" logf="$3" outdir="$4"
    local _access_log_env=()
    if [[ "${ENABLE_ACCESS_LOG:-0}" == "1" ]]; then
        mkdir -p "$outdir"
        _access_log_env=("LMCACHE_ACCESS_LOG=$outdir/cpu_access.jsonl")
    fi
    setsid env \
        PYTHONHASHSEED=0 \
        "${_access_log_env[@]}" \
        lmcache server \
            --host "$MP_HOST" --port "$MP_PORT" \
            --http-host "$MP_HTTP_HOST" --http-port "$MP_HTTP_PORT" \
            --max-workers "$MP_MAX_WORKERS" \
            --chunk-size "$MP_CHUNK_SIZE" \
            --l1-size-gb "$cpu" \
            --l1-write-ttl-seconds "$MP_L1_WRITE_TTL_SECONDS" \
            --l1-read-ttl-seconds "$MP_L1_READ_TTL_SECONDS" \
            --eviction-policy "$policy" \
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
    local logf="$1" outdir="$2"
    rm -rf "$PROM_DIR" && mkdir -p "$PROM_DIR"
    local kv_transfer_config
    kv_transfer_config=$(printf '{"kv_connector":"LMCacheMPConnector","kv_role":"kv_both","kv_load_failure_policy":"%s","kv_connector_extra_config":{"lmcache.mp.port":%s,"lmcache.mp.mq_timeout":%s}}' \
        "$KV_LOAD_FAILURE_POLICY" "$MP_PORT" "$MP_MQ_TIMEOUT")
    # Routing/scheduling/GPU-eviction logging (Milestone 2b) lives in the vLLM
    # process, not the MP server — needs its own LMCACHE_ACCESS_LOG, same
    # convention as launch_mp_server() above. Same base path is fine: each
    # process's AccessLogSubscriber/lmcache_access_log disambiguates by its
    # own timestamp+PID (timestamped_pid_path()), so the API-server and each
    # DP worker each get their own file even though they're handed the same
    # env var value.
    local _access_log_env=()
    if [[ "${ENABLE_ACCESS_LOG:-0}" == "1" ]]; then
        mkdir -p "$outdir"
        _access_log_env=("LMCACHE_ACCESS_LOG=$outdir/cpu_access.jsonl")
    fi
    # setsid => new process group we can kill wholesale (vllm spawns DP workers).
    setsid env \
        PYTHONHASHSEED=0 \
        PROMETHEUS_MULTIPROC_DIR="$PROM_DIR" \
        HIP_VISIBLE_DEVICES="$HIP_VISIBLE_DEVICES" \
        "${_access_log_env[@]}" \
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

# Returns 0 when /health is up; 1 if the process dies or the timeout elapses.
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

# ---- preflight --------------------------------------------------------------
command -v curl >/dev/null   || { echo "ERROR: curl required" >&2; exit 1; }
command -v setsid >/dev/null || { echo "ERROR: setsid required" >&2; exit 1; }
command -v vllm >/dev/null   || { echo "ERROR: vllm not on PATH" >&2; exit 1; }
command -v lmcache >/dev/null || { echo "ERROR: lmcache CLI not on PATH" >&2; exit 1; }
if health_ok; then
    log "ERROR: something is already serving on $HOST:$PORT — refusing to start. Stop it first."
    exit 1
fi
if mp_health_ok; then
    log "ERROR: something is already serving on $MP_HTTP_HOST:$MP_HTTP_PORT — refusing to start. Stop it first."
    exit 1
fi
((${#EVICTION_POLICIES[@]} && ${#CPU_SIZES_GB[@]})) || { echo "ERROR: EVICTION_POLICIES/CPU_SIZES_GB empty" >&2; exit 1; }

total=$(( ${#EVICTION_POLICIES[@]} * ${#CPU_SIZES_GB[@]} ))
log "MP sweep start: $total run(s) = policies(${EVICTION_POLICIES[*]}) x cpu(${CPU_SIZES_GB[*]}GB), dp=$DATA_PARALLEL_SIZE"
log "Tester=$TESTER_DIR  Base=$BASE_OUTPUT_DIR  Master log=$MASTER_LOG"

ok=0; fail=0; n=0
for policy in "${EVICTION_POLICIES[@]}"; do
    for cpu in "${CPU_SIZES_GB[@]}"; do
        n=$((n + 1))
        run="$(build_run_name "$policy" "$cpu")"
        outdir="$BASE_OUTPUT_DIR/$run"
        mplog="$BASE_OUTPUT_DIR/${run}.mpserver.log"
        slog="$BASE_OUTPUT_DIR/${run}.server.log"
        clog="$BASE_OUTPUT_DIR/${run}.client.log"
        log "======== [$n/$total] RUN $run  (policy=$policy cpu=${cpu}GB dp=$DATA_PARALLEL_SIZE) ========"

        launch_mp_server "$policy" "$cpu" "$mplog" "$outdir"
        log "lmcache server launched (pid=$MP_SERVER_PID), waiting for /healthcheck ..."
        if ! wait_mp_server_ready "$mplog"; then
            log "[$n/$total] SKIP $run — lmcache server failed to start."
            stop_mp_server; fail=$((fail + 1)); sleep "$SERVER_SETTLE"; continue
        fi

        launch_server "$slog" "$outdir"
        log "vLLM launched (pid=$SERVER_PID), waiting for /health ..."
        if ! wait_server_ready "$slog"; then
            log "[$n/$total] SKIP $run — vLLM failed to start."
            stop_server; stop_mp_server; fail=$((fail + 1)); sleep "$SERVER_SETTLE"; continue
        fi
        log "servers READY -> running client (log: $clog)"

        if run_client "$outdir" "$clog"; then
            log "[$n/$total] client OK -> $outdir"; ok=$((ok + 1))
        else
            rc=$?
            log "[$n/$total] client FAILED/timeout (rc=$rc) — partial results may exist in $outdir"
            fail=$((fail + 1))
        fi

        stop_server
        stop_mp_server
        log "settling ${SERVER_SETTLE}s for GPU/host-mem release ..."
        sleep "$SERVER_SETTLE"
    done
done

log "MP sweep done: $ok ok, $fail failed/skipped, $total total. Logs under $BASE_OUTPUT_DIR"
