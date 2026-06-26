#!/usr/bin/env bash
# =============================================================================
# run_sweep.sh — sweep LMCache eviction policy x CPU-pool size for the agentic
# trace-replay eviction study. Replaces the manual two-tmux flow:
#   - starts vllm serve, waits until it is FULLY ready (HTTP /health),
#   - only then runs trace_replay_tester.py to completion,
#   - only after the client fully exits, shuts the server down cleanly,
#   - then moves to the next (policy, cpu) combo.
#
# Safety (the #1 requirement — never leave a half-stuck / zombie server):
#   - server runs in its own process group (setsid); cleanup kills the GROUP.
#   - a trap on EXIT/INT/TERM ALWAYS tears the server down, even on Ctrl-C/error.
#   - every wait has a timeout: a server that never becomes ready is skipped,
#     a client that overruns is SIGINT'd (it saves partial results), nothing
#     blocks forever.
#   - one failing combo does not abort the sweep; it is logged and skipped.
#   (A hard `kill -9` of THIS script is the only case cleanup can't cover.)
#
# Usage:
#   scripts/run_sweep.sh [path/to/sweep.conf]   # default: scripts/sweep.conf
# =============================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TESTER_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
CONFIG="${1:-$SCRIPT_DIR/sweep.conf}"

[[ -f "$CONFIG" ]] || { echo "ERROR: config not found: $CONFIG" >&2; exit 1; }
# shellcheck disable=SC1090
source "$CONFIG"

mkdir -p "$BASE_OUTPUT_DIR"
MASTER_LOG="$BASE_OUTPUT_DIR/sweep_$(date +%Y%m%d_%H%M%S).log"

SERVER_PID=""
SERVER_PGID=""

log() { echo "[$(date '+%F %T')] $*" | tee -a "$MASTER_LOG"; }

health_ok() { curl -fsS -o /dev/null "http://$HOST:$PORT/health" 2>/dev/null; }

# Kill the whole server process group; idempotent; clears SERVER_PID when done.
stop_server() {
    [[ -z "$SERVER_PID" ]] && return 0
    if kill -0 "$SERVER_PID" 2>/dev/null; then
        log "Stopping server (pid=$SERVER_PID pgid=$SERVER_PGID), SIGINT then wait..."
        kill -INT "-$SERVER_PGID" 2>/dev/null || kill -INT "$SERVER_PID" 2>/dev/null || true
        local i
        for ((i = 0; i < SERVER_STOP_GRACE; i++)); do
            kill -0 "$SERVER_PID" 2>/dev/null || break
            sleep 1
        done
        if kill -0 "$SERVER_PID" 2>/dev/null; then
            log "Server still alive after ${SERVER_STOP_GRACE}s, SIGKILL group."
            kill -KILL "-$SERVER_PGID" 2>/dev/null || kill -KILL "$SERVER_PID" 2>/dev/null || true
        fi
    fi
    SERVER_PID=""; SERVER_PGID=""
    # Wait until /health stops answering (port released) before returning.
    local deadline=$((SECONDS + SERVER_STOP_TIMEOUT))
    while ((SECONDS < deadline)); do health_ok || return 0; sleep "$POLL_INTERVAL"; done
    log "WARN: something still answers /health after stop timeout."
}

cleanup() { stop_server; }
trap cleanup EXIT INT TERM

# 0.5 -> g0p5 ; build "64k_g0p5_cpu40_lru_50u_3600s"
build_run_name() {
    local policy="$1" cpu="$2" gpu_tag ctx_k pol
    gpu_tag="g$(printf '%s' "$GPU_MEM_UTIL" | tr '.' 'p')"
    ctx_k=$((MAX_CONTEXT / 1000))
    pol="$(printf '%s' "$policy" | tr '[:upper:]' '[:lower:]')"
    printf '%s%sk_%s_cpu%s_%s_%su_%ss' \
        "${RUN_PREFIX:+${RUN_PREFIX}_}" "$ctx_k" "$gpu_tag" "$cpu" "$pol" "$MAX_USERS" "$TEST_DURATION"
}

launch_server() {
    local policy="$1" cpu="$2" logf="$3"
    rm -rf "$PROM_DIR" && mkdir -p "$PROM_DIR"
    local _access_log_env=()
    if [[ "${ENABLE_ACCESS_LOG:-0}" == "1" ]]; then
        mkdir -p "$outdir"
        _access_log_env=("LMCACHE_ACCESS_LOG=$outdir/cpu_access.jsonl")
    fi
    # setsid => new process group we can kill wholesale (vllm spawns workers).
    setsid env \
        PYTHONHASHSEED=0 \
        PROMETHEUS_MULTIPROC_DIR="$PROM_DIR" \
        LMCACHE_LOCAL_CPU=True \
        LMCACHE_MAX_LOCAL_CPU_SIZE="$cpu" \
        LMCACHE_CHUNK_SIZE="$LMCACHE_CHUNK_SIZE" \
        LMCACHE_CACHE_POLICY="$policy" \
        "${_access_log_env[@]}" \
        HIP_VISIBLE_DEVICES="$HIP_VISIBLE_DEVICES" \
        vllm serve "$MODEL" \
            --port "$PORT" \
            --block-size "$BLOCK_SIZE" \
            --enable-prefix-caching \
            --gpu-memory-utilization "$GPU_MEM_UTIL" \
            --max-model-len "$MAX_MODEL_LEN" \
            --max-num-seqs "$MAX_NUM_SEQS" \
            --kv-cache-metrics \
            --kv-cache-metrics-sample "$KV_CACHE_METRICS_SAMPLE" \
            --hf-overrides "$HF_OVERRIDES" \
            --kv-transfer-config "$KV_TRANSFER_CONFIG" \
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
            log "ERROR: server exited during startup (tail $logf):"; tail -n 15 "$logf" | tee -a "$MASTER_LOG"
            return 1
        fi
        health_ok && return 0
        sleep "$POLL_INTERVAL"
    done
    log "ERROR: server not ready within ${SERVER_START_TIMEOUT}s."
    return 1
}

# Runs the client to completion; full terminal output (banner + 'Test Complete')
# is tee'd into <run>.client.log. Returns the client's exit status.
run_client() {
    local outdir="$1" logf="$2" cap=$((TEST_DURATION + CLIENT_TIMEOUT_BUFFER))
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
            --server-metrics \
            --timing-strategy "$TIMING_STRATEGY" \
            --trace-seed "$SEED" --prompt-seed "$SEED" --seed "$SEED" \
            --max-ttft "$MAX_TTFT" \
    ) 2>&1 | tee "$logf"
    return "${PIPESTATUS[0]}"
}

# ---- preflight --------------------------------------------------------------
command -v curl >/dev/null  || { echo "ERROR: curl required" >&2; exit 1; }
command -v setsid >/dev/null || { echo "ERROR: setsid required" >&2; exit 1; }
command -v vllm >/dev/null   || { echo "ERROR: vllm not on PATH" >&2; exit 1; }
if health_ok; then
    log "ERROR: something is already serving on $HOST:$PORT — refusing to start. Stop it first."
    exit 1
fi
((${#POLICIES[@]} && ${#CPU_SIZES_GB[@]})) || { echo "ERROR: POLICIES/CPU_SIZES_GB empty" >&2; exit 1; }

total=$(( ${#POLICIES[@]} * ${#CPU_SIZES_GB[@]} ))
log "Sweep start: $total run(s) = policies(${POLICIES[*]}) x cpu(${CPU_SIZES_GB[*]}GB)"
log "Tester=$TESTER_DIR  Base=$BASE_OUTPUT_DIR  Master log=$MASTER_LOG"

ok=0; fail=0; n=0
for policy in "${POLICIES[@]}"; do
    for cpu in "${CPU_SIZES_GB[@]}"; do
        n=$((n + 1))
        run="$(build_run_name "$policy" "$cpu")"
        outdir="$BASE_OUTPUT_DIR/$run"
        slog="$BASE_OUTPUT_DIR/${run}.server.log"
        clog="$BASE_OUTPUT_DIR/${run}.client.log"
        log "======== [$n/$total] RUN $run  (policy=$policy cpu=${cpu}GB) ========"

        launch_server "$policy" "$cpu" "$slog"
        log "server launched (pid=$SERVER_PID), waiting for /health ..."
        if ! wait_server_ready "$slog"; then
            log "[$n/$total] SKIP $run — server failed to start."
            stop_server; fail=$((fail + 1)); sleep "$SERVER_SETTLE"; continue
        fi
        log "server READY -> running client (log: $clog)"

        if run_client "$outdir" "$clog"; then
            log "[$n/$total] client OK -> $outdir"; ok=$((ok + 1))
        else
            rc=$?
            log "[$n/$total] client FAILED/timeout (rc=$rc) — partial results may exist in $outdir"
            fail=$((fail + 1))
        fi

        stop_server
        log "settling ${SERVER_SETTLE}s for GPU/host-mem release ..."
        sleep "$SERVER_SETTLE"
    done
done

log "Sweep done: $ok ok, $fail failed/skipped, $total total. Logs under $BASE_OUTPUT_DIR"
