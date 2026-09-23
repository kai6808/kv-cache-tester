#!/usr/bin/env bash
# =============================================================================
# run_coupled.sh — one file to run the coupled GPU + L1 eviction comparison.
#
#   ./scripts/run_coupled.sh [config]          # default: scripts/coupled.conf
#   ./scripts/run_coupled.sh --smoke [config]  # every arm, short, real shape
#
# Runs each ARM in the config through scripts/run_sweep_mp.sh -- the same
# orchestration the DP2 collection uses -- then writes a comparison summary.
#
# Each arm differs ONLY in configuration: the eviction policy's term weights
# and two switches. No code path differs between arms, which is what makes the
# ladder a fair comparison.
# =============================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TESTER_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

SMOKE=0
if [[ "${1:-}" == "--smoke" ]]; then SMOKE=1; shift; fi
CONFIG="${1:-$SCRIPT_DIR/coupled.conf}"
[[ -f "$CONFIG" ]] || { echo "ERROR: config not found: $CONFIG" >&2; exit 1; }
# shellcheck source=/dev/null
source "$CONFIG"

if ((SMOKE)); then
    # Every configured arm, short, at the REAL shape: the conf's users,
    # traces, concurrency and pool size are kept; only the duration shrinks.
    #
    # Keep the real load: a 2-user smoke is not what the real round runs, and
    # it is where the (since fixed) MP event-destroy hang went unnoticed as
    # "the coupling blocks the engine" (cachelab design doc section 11).
    TEST_DURATION="${SMOKE_DURATION:-300}"
    MAX_REQUESTS=""
    BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR}_smoke"
    # The smoke fails (non-zero exit) unless every arm's warm-up flowed, no
    # stall or engine death followed it, and the notice arms were live.
    SMOKE_SUMMARY_ARGS=(--require-clean)
fi

mkdir -p "$BASE_OUTPUT_DIR"
MASTER_LOG="$BASE_OUTPUT_DIR/coupled.log"
SUMMARY="$BASE_OUTPUT_DIR/summary.md"

log() { printf '[%s] %s\n' "$(date '+%F %T')" "$*" | tee -a "$MASTER_LOG"; }

# ---- preflight --------------------------------------------------------------
((${#ARMS[@]})) || { echo "ERROR: ARMS is empty" >&2; exit 1; }
((${#CPU_SIZES_GB[@]} == 1)) || {
    echo "ERROR: coupled runs take exactly one pool size; got ${CPU_SIZES_GB[*]}" >&2
    exit 1
}
[[ -x "$SCRIPT_DIR/run_sweep_mp.sh" ]] || {
    echo "ERROR: $SCRIPT_DIR/run_sweep_mp.sh missing or not executable" >&2; exit 1; }

CPU_GB="${CPU_SIZES_GB[0]}"
log "coupled round: ${#ARMS[@]} arm(s), dp=$DATA_PARALLEL_SIZE, L1=${CPU_GB}GB, ${TEST_DURATION}s each"
log "output: $BASE_OUTPUT_DIR"
((SMOKE)) && log "SMOKE MODE: short run, results are for wiring validation only"

# ---- run one arm ------------------------------------------------------------
# Generates a conf that sources the shared one and then overrides the per-arm
# bits, so the arm definition stays in coupled.conf and this script stays
# mechanical.
run_arm() {
    local name="$1" policy="$2" flags="$3" gpu_side="$4" demand="$5"
    local armconf="$BASE_OUTPUT_DIR/.arm_${name}.conf"

    {
        printf 'source %q\n' "$CONFIG"
        printf 'EVICTION_POLICIES=(%q)\n' "$policy"
        printf 'CPU_SIZES_GB=(%q)\n' "$CPU_GB"
        printf 'BASE_OUTPUT_DIR=%q\n' "$BASE_OUTPUT_DIR"
        printf 'RUN_NAME_OVERRIDE=%q\n' "$name"
        printf 'TEST_DURATION=%q\n' "$TEST_DURATION"
        printf 'MAX_REQUESTS=%q\n' "$MAX_REQUESTS"
        printf 'MAX_USERS=%q\nSTART_USERS=%q\nMAX_TRACES=%q\nMAX_CONCURRENT=%q\n' \
            "$MAX_USERS" "$START_USERS" "$MAX_TRACES" "$MAX_CONCURRENT"
        printf 'ENABLE_ACCESS_LOG=%q\n' "$ENABLE_ACCESS_LOG"

        # Extra `lmcache server` flags for this arm.
        printf 'MP_EXTRA_ARGS=('
        # shellcheck disable=SC2086 -- flags is a deliberate word-split list
        for f in $flags; do printf '%q ' "$f"; done
        printf '"--coupled-survival-horizon-s" %q ' "$COUPLED_SURVIVAL_HORIZON_S"
        printf '"--coupled-tie-break" %q ' "$COUPLED_TIE_BREAK"
        printf ')\n'

        # vLLM-side GPU mode:
        #   0      nothing on the GPU side
        #   track  chunk residency tracked and reported (the exclusive term's
        #          input) but allocation order untouched -- the overhead gate
        #   1      tracking + "GPU evicts L1-backed first"
        case "$gpu_side" in
            1)
                printf 'VLLM_EXTRA_ENV=(LMCACHE_GPU_EVICT_L1_BACKED=1 '
                printf 'LMCACHE_GPU_EVICT_L1_BACKED_WINDOW=%q ' "$GPU_EVICT_WINDOW"
                printf 'LMCACHE_CHUNK_SIZE=%q)\n' "$MP_CHUNK_SIZE"
                ;;
            track)
                printf 'VLLM_EXTRA_ENV=(LMCACHE_GPU_RESIDENCY_TRACK=1 '
                printf 'LMCACHE_CHUNK_SIZE=%q)\n' "$MP_CHUNK_SIZE"
                ;;
            0)
                printf 'declare -a VLLM_EXTRA_ENV\n'
                ;;
            *)
                echo "ERROR: arm '$name': gpu_side must be 0, track or 1, got '$gpu_side'" >&2
                return 1
                ;;
        esac

        # Whether the connector sends demand/residency notices at all.
        if [[ "$demand" == "1" ]]; then
            printf 'KV_EXTRA_CONFIG=%q\n' ',"lmcache.mp.coupled_demand_notice":true'
        else
            printf 'KV_EXTRA_CONFIG=""\n'
        fi
    } > "$armconf"

    # The COUPLED flags are only meaningful to the COUPLED policy; passing them
    # to LRU would make `lmcache server` reject unknown-for-this-policy args.
    if [[ "$policy" != "COUPLED" ]]; then
        sed -i 's/^MP_EXTRA_ARGS=(.*/declare -a MP_EXTRA_ARGS/' "$armconf"
    fi

    log "---- arm '$name' (policy=$policy gpu_side=$gpu_side demand=$demand) ----"
    bash "$SCRIPT_DIR/run_sweep_mp.sh" "$armconf" 2>&1 | tee -a "$MASTER_LOG"
    return "${PIPESTATUS[0]}"
}

# ---- main -------------------------------------------------------------------
declare -a RAN=()
ok=0; fail=0
for arm in "${ARMS[@]}"; do
    IFS='|' read -r a_name a_policy a_flags a_gpu a_demand <<<"$arm"
    [[ -n "$a_name" ]] || { log "skipping malformed ARM entry: $arm"; continue; }
    if run_arm "$a_name" "$a_policy" "$a_flags" "$a_gpu" "$a_demand"; then
        ok=$((ok + 1)); RAN+=("$a_name")
    else
        log "arm '$a_name' FAILED -- continuing so the other arms still produce data"
        fail=$((fail + 1))
    fi
done

log "arms complete: $ok ok, $fail failed"

# ---- summary ----------------------------------------------------------------
# SUMMARY_ARGS (optional, from the conf) is passed through, e.g. (--gate) for
# notice_gate.conf, which then fails the run unless every gate criterion holds.
declare -a SUMMARY_ARGS SMOKE_SUMMARY_ARGS
SUMMARY_ARGS+=(${SMOKE_SUMMARY_ARGS[@]+"${SMOKE_SUMMARY_ARGS[@]}"})
log "building comparison summary -> $SUMMARY"
python3 "$SCRIPT_DIR/coupled_summary.py" \
    --base "$BASE_OUTPUT_DIR" \
    --arms "${RAN[@]:-}" \
    --out "$SUMMARY" \
    ${SUMMARY_ARGS[@]+"${SUMMARY_ARGS[@]}"} 2>&1 | tee -a "$MASTER_LOG"
summary_rc="${PIPESTATUS[0]}"

log "DONE. Summary: $SUMMARY"
[[ -f "$SUMMARY" ]] && cat "$SUMMARY"
exit $(( fail > 0 || summary_rc != 0 ))
