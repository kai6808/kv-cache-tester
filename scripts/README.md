# scripts/ — eviction sweep orchestrator

Automates the server+client eviction runs (replaces the manual two-tmux flow) and
sweeps **LMCache eviction policy × CPU-pool size** in one safe pass.

## Files
- `run_sweep.sh` — the orchestrator (bash, Linux; needs `vllm`, `curl`, `setsid`, `timeout`).
- `sweep.conf` — the config, **sourced** by bash (no YAML/parsing deps → can't half-parse).

## Run
```bash
cd kv-cache-tester           # or /workspace/kv-cache-tester on the server
scripts/run_sweep.sh         # uses scripts/sweep.conf
scripts/run_sweep.sh my.conf # or a custom config
```

## What one combo does
For every `(policy, cpu_gb)` in `POLICIES × CPU_SIZES_GB`:
1. start `vllm serve` with `LMCACHE_CACHE_POLICY=<policy>` and `LMCACHE_MAX_LOCAL_CPU_SIZE=<cpu>`;
2. wait until the server is **fully ready** (HTTP `/health`), not just launched;
3. run `trace_replay_tester.py --server-metrics` to completion;
4. only after the client fully exits, **shut the server down cleanly**;
5. settle, then next combo.

Policy and cpu are derived from the same loop vars used for the run name, so the
server env and the `--output-dir` can never disagree (e.g. `64k_g0p5_cpu40_lru_50u_3600s`).

## Outputs (under `BASE_OUTPUT_DIR`)
- `<run>/` — the tester's per-run results (HTML, CSV, `server_metrics.json`).
- `<run>.client.log` — full client terminal output (startup banner … `Test Complete`).
- `<run>.server.log` — full vllm server output (for OOM/error postmortems).
- `sweep_<timestamp>.log` — master timeline of the whole sweep.

## Safety model (never leaves a stuck/zombie server)
- The server runs in its **own process group** (`setsid`); teardown kills the group
  (vllm + workers), not just the parent.
- A `trap` on `EXIT`/`INT`/`TERM` **always** tears the server down — on Ctrl-C, on
  error, on normal finish. (Only a `kill -9` of the script itself can bypass it; then
  kill the leftover `vllm` group by hand.)
- Every wait is bounded: a server that never reaches `/health` within
  `SERVER_START_TIMEOUT` is **skipped**; a client that overruns
  `TEST_DURATION + CLIENT_TIMEOUT_BUFFER` is `SIGINT`'d (it saves partial results).
- One failing combo is logged and skipped; the sweep continues.
- Refuses to start if something already answers on `HOST:PORT` (won't kill a stranger).

## Common edits
- **Sweep policies**: `POLICIES=(LRU LFU ARC S3FIFO WTINYLFU FIFO MRU)`.
- **Sweep CPU sizes**: `CPU_SIZES_GB=(20 40 80)`.
- **Cache-everything (no-eviction) baseline**: add a CPU size large enough to hold the
  whole working set, e.g. `CPU_SIZES_GB=(40 320)` — but mind host RAM (the CPU pool is
  host memory). This is the simple "optimal capacity" reference; see
  `docs/plans/2026-06-24-belady-optimal-cpu-eviction.md` for what it does vs. doesn't
  measure relative to Belady.
- **GPU eviction counts**: keep `KV_CACHE_METRICS_SAMPLE=1.0` for true counts; lower it
  to cut overhead (counts then become a sample).
- **GPU/CPU pressure tuning**: `GPU_MEM_UTIL`, `MAX_NUM_SEQS`, `MAX_USERS` — watch the
  "GPU KV usage (%)" panel in `server_lmcache_eviction.html` (aim ~70–90%, not pinned).
