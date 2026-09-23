#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Build the arm-by-arm comparison for a coupled GPU + L1 eviction round.

Reads what the harness writes -- ``server_metrics.json`` (vLLM and LMCache
counters), ``summary_trace_replay.csv`` (client-side TTFT and throughput),
``mp_status.json`` (the MP server's demand-plane counters) and the
``coupled stats`` lines in each arm's vLLM log -- and emits Markdown tables:
the results, the change relative to the first arm, and the mechanism (proof
the couplings were live, and what they cost). No per-event tracing: a real
round writes kilobytes per arm.

Rates are taken over the MEASURED window only: counters are the last sample
minus the first (the first is taken when the measured client starts, i.e.
after the warm-up), and stalls are counted only after the warm-up finished.

``--gate`` checks the notice-path overhead gate (notice_gate.conf): the second
arm (notices on) against the first (off). Exit status 1 unless it passes.

Every rate is defined explicitly in the output, because "hit rate" is
ambiguous across tiers and the whole point of the round is comparing them.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import re
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

# Counter names as vLLM/LMCache emit them into server_metrics.json.
PROMPT_TOKENS = "vllm:prompt_tokens_total"
PROMPT_CACHED = "vllm:prompt_tokens_cached_total"
EXT_HITS = "vllm:external_prefix_cache_hits_total"
EXT_QUERIES = "vllm:external_prefix_cache_queries_total"
TTFT_SUM = "vllm:time_to_first_token_seconds_sum"
TTFT_COUNT = "vllm:time_to_first_token_seconds_count"
PREFILL_SUM = "vllm:request_prefill_time_seconds_sum"
PREEMPTIONS = "vllm:num_preemptions_total"


def _final_counters(run_dir: Path) -> dict[str, float]:
    """Return the counters accumulated over the measured window.

    vLLM's Prometheus counters run from engine start, so they include the
    warm-up. The client takes its first sample when the measured run starts,
    so last-minus-first is exactly the measured window.

    Returns:
        Counter deltas plus ``_elapsed``, or {} when the file is absent.
    """
    path = run_dir / "server_metrics.json"
    if not path.is_file():
        return {}
    try:
        samples = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    if not samples:
        return {}
    first = samples[0].get("counters", {})
    final = samples[-1]
    counters = {
        k: float(v) - float(first.get(k, 0.0) or 0.0)
        for k, v in final.get("counters", {}).items()
        if isinstance(v, (int, float))
    }
    counters["_elapsed"] = float(final.get("elapsed", 0.0) or 0.0) - float(
        samples[0].get("elapsed", 0.0) or 0.0
    )
    return counters


STALL = "No available shared memory broadcast block"
_LOG_TS = re.compile(r"(\d\d)-(\d\d) (\d\d):(\d\d):(\d\d)")
_STATS_LINE = re.compile(
    r"coupled stats dp=(\d+) shim=(\{.*?\}) client=(\{.*?\}) awaiting_store=(\d+)"
)


def _warmup_epoch(run_dir: Path) -> float | None:
    """When the warm-up finished (the measured window's start), if recorded."""
    try:
        return float((run_dir / "warmup_done_epoch").read_text().strip())
    except (OSError, ValueError):
        return None


def _log_epoch(line: str, year: int) -> float | None:
    """Epoch of a vLLM log line's ``MM-DD HH:MM:SS`` stamp (container is UTC)."""
    m = _LOG_TS.search(line)
    if not m:
        return None
    mo, d, h, mi, se = (int(x) for x in m.groups())
    return datetime(year, mo, d, h, mi, se, tzinfo=timezone.utc).timestamp()


def _health(base: Path, arm: str) -> dict[str, float | None]:
    """Stalls and engine deaths in the measured window of one arm.

    A stall is vLLM's 60-second "No available shared memory broadcast block"
    warning. The one-time startup stall is expected during the warm-up and is
    excluded; any stall after it is a real finding.
    """
    try:
        status = (base / arm / "warmup_status").read_text().strip()
        warmup_ok: float | None = 1.0 if status == "ok" else 0.0
    except OSError:
        warmup_ok = None  # no warm-up configured
    log = base / f"{arm}.server.log"
    if not log.is_file():
        return {"stalls": None, "engine_deaths": None, "warmup_ok": warmup_ok}
    start = _warmup_epoch(base / arm)
    year = datetime.fromtimestamp(start or 0, tz=timezone.utc).year if start else 1970
    stalls = deaths = 0
    for line in log.read_text(errors="replace").splitlines():
        if STALL not in line and "EngineDeadError" not in line:
            continue
        if start is not None:
            t = _log_epoch(line, year)
            if t is not None and t < start:
                continue
        if STALL in line:
            stalls += 1
        else:
            deaths += 1
    return {"stalls": stalls, "engine_deaths": deaths, "warmup_ok": warmup_ok}


def _mechanism(base: Path, arm: str) -> dict[str, float | None]:
    """The couplings' own counters: were they live, and what did they cost.

    Server side from ``mp_status.json`` (whole run, including the warm-up);
    GPU/client side from the last ``coupled stats`` line of each DP rank,
    summed over ranks.
    """
    out: dict[str, float | None] = {}
    try:
        status = json.loads((base / arm / "mp_status.json").read_text())
    except (OSError, json.JSONDecodeError):
        status = {}
    for key in (
        "notices_waiting",
        "notices_residency",
        "notice_handler_p50_us",
        "notice_handler_p99_us",
        "keys_held_by_gpu",
        "race_losses",
        "evicted_backed_checked",
        "invalidations_sent",
        "invalidation_overflows",
        "demand_notices_dropped",
    ):
        out[key] = status.get(key)
    waiting = status.get("waiting_hashes")
    out["waiting_hash_in_l1_pct"] = _ratio(status.get("waiting_hashes_in_l1"), waiting)

    last: dict[str, tuple[dict, dict]] = {}
    log = base / f"{arm}.server.log"
    if log.is_file():
        for line in log.read_text(errors="replace").splitlines():
            m = _STATS_LINE.search(line)
            if m:
                try:
                    last[m.group(1)] = (
                        ast.literal_eval(m.group(2)),
                        ast.literal_eval(m.group(3)),
                    )
                except (ValueError, SyntaxError):
                    continue
    if last:
        shims = [v[0] for v in last.values()]
        clients = [v[1] for v in last.values()]
        for key in (
            "chunks_completed",
            "chunks_released",
            "backed_first_picks",
            "evicted_backed",
            "invalidations_applied",
            "backed_chunks",
        ):
            out[key] = float(sum(sh.get(key, 0) for sh in shims))
        out["notices_sent"] = float(sum(c.get("notices_sent", 0) for c in clients))
        out["notice_hash_bytes"] = float(
            sum(c.get("notice_hash_bytes", 0) for c in clients)
        )
        out["reply_backlog_max"] = float(
            max(c.get("reply_backlog_max", 0) for c in clients)
        )
    return out


def _client_rows(run_dir: Path) -> list[dict[str, str]]:
    """Return the per-period client rows, or [] when the file is absent."""
    path = run_dir / "summary_trace_replay.csv"
    if not path.is_file():
        return []
    try:
        with path.open(newline="") as fh:
            return list(csv.DictReader(fh))
    except OSError:
        return []


def _f(row: dict[str, str], key: str) -> float | None:
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return None


def _mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _ratio(num: float | None, den: float | None) -> float | None:
    if num is None or not den:
        return None
    return 100.0 * num / den


def collect(base: Path, arm: str) -> dict[str, float | None]:
    """Gather one arm's metrics.

    Args:
        base: The round's output directory.
        arm: The arm name, which is also its run directory name.

    Returns:
        Metric name -> value, with None wherever the source was missing.
    """
    run_dir = base / arm
    c = _final_counters(run_dir)
    rows = _client_rows(run_dir)

    ttft_count = c.get(TTFT_COUNT)
    prompt = c.get(PROMPT_TOKENS)
    elapsed = c.get("_elapsed") or None

    p50 = _mean([v for r in rows if (v := _f(r, "ttft_p50")) is not None])
    p95 = _mean([v for r in rows if (v := _f(r, "ttft_p95")) is not None])
    out_tps = _mean([v for r in rows if (v := _f(r, "output_tokens_per_second")) is not None])
    rps = _mean([v for r in rows if (v := _f(r, "requests_per_second")) is not None])
    completed = max(
        (v for r in rows if (v := _f(r, "requests_completed")) is not None),
        default=None,
    )

    return {
        **_health(base, arm),
        **_mechanism(base, arm),
        # Client-observed latency: what a user feels.
        "ttft_mean_s": (c[TTFT_SUM] / ttft_count) if ttft_count else None,
        "ttft_p50_s": p50,
        "ttft_p95_s": p95,
        # Work done in the fixed wall-clock window.
        "requests": completed,
        "prompt_tokens": prompt,
        "out_tok_per_s": out_tps,
        "req_per_s": rps,
        # Per-tier reuse. GPU = vLLM's own prefix cache over prompt tokens;
        # L1 = the shared pool answering what the GPU missed.
        "gpu_hit_pct": _ratio(c.get(PROMPT_CACHED), prompt),
        "l1_hit_pct": _ratio(c.get(EXT_HITS), c.get(EXT_QUERIES)),
        # Overall = any prompt token that did not need a fresh prefill.
        "overall_hit_pct": _ratio(
            (c.get(PROMPT_CACHED) or 0) + (c.get(EXT_HITS) or 0), prompt
        ),
        # Cost side.
        "prefill_s": c.get(PREFILL_SUM),
        "preemptions": c.get(PREEMPTIONS),
        "elapsed_s": elapsed,
    }


ROWS: list[tuple[str, str, str]] = [
    ("ttft_mean_s", "TTFT mean (s)", "{:.2f}"),
    ("ttft_p50_s", "TTFT p50 (s)", "{:.2f}"),
    ("ttft_p95_s", "TTFT p95 (s)", "{:.2f}"),
    ("requests", "requests completed", "{:.0f}"),
    ("prompt_tokens", "prompt tokens", "{:.0f}"),
    ("out_tok_per_s", "output tok/s", "{:.1f}"),
    ("req_per_s", "requests/s", "{:.3f}"),
    ("gpu_hit_pct", "GPU prefix hit %", "{:.1f}"),
    ("l1_hit_pct", "L1 pool hit %", "{:.1f}"),
    ("overall_hit_pct", "overall hit %", "{:.1f}"),
    ("prefill_s", "prefill time (s)", "{:.1f}"),
    ("preemptions", "preemptions", "{:.0f}"),
    ("elapsed_s", "elapsed (s)", "{:.0f}"),
]


MECH_ROWS: list[tuple[str, str, str]] = [
    ("warmup_ok", "warm-up completed (1 = yes, must be 1)", "{:.0f}"),
    ("stalls", "stalls after warm-up (must be 0)", "{:.0f}"),
    ("engine_deaths", "EngineDeadError after warm-up", "{:.0f}"),
    ("notices_sent", "notices sent (all ranks)", "{:.0f}"),
    ("notice_hash_bytes", "notice payload bytes (hashes)", "{:.0f}"),
    ("notice_handler_p50_us", "server handler p50 (us)", "{:.0f}"),
    ("notice_handler_p99_us", "server handler p99 (us)", "{:.0f}"),
    ("reply_backlog_max", "max replies in flight", "{:.0f}"),
    ("waiting_hash_in_l1_pct", "queued-prefix hashes found in L1 %", "{:.1f}"),
    ("chunks_completed", "GPU chunks completed", "{:.0f}"),
    ("chunks_released", "GPU chunks released", "{:.0f}"),
    ("keys_held_by_gpu", "L1 keys held by some GPU (end)", "{:.0f}"),
    ("backed_chunks", "chunks believed L1-backed (end)", "{:.0f}"),
    ("backed_first_picks", "backed-first evictions", "{:.0f}"),
    ("invalidations_sent", "L1->GPU invalidations sent", "{:.0f}"),
    ("invalidation_overflows", "invalidation overflows", "{:.0f}"),
    ("evicted_backed_checked", "GPU evictions of backed chunks", "{:.0f}"),
    ("race_losses", "...of which already gone from L1 (stale)", "{:.0f}"),
    ("demand_notices_dropped", "notices dropped (non-COUPLED policy)", "{:.0f}"),
]


def _table(arms: list[str], data: dict, rows: list[tuple[str, str, str]]) -> list[str]:
    out = [
        "| metric | " + " | ".join(arms) + " |",
        "|---|" + "---|" * len(arms),
    ]
    for key, label, fmt in rows:
        cells = []
        for arm in arms:
            v = data[arm].get(key)
            cells.append(fmt.format(v) if isinstance(v, (int, float)) else "n/a")
        out.append(f"| {label} | " + " | ".join(cells) + " |")
    return out


def render(base: Path, arms: list[str]) -> str:
    data = {arm: collect(base, arm) for arm in arms}
    out: list[str] = [
        f"# Coupled GPU + L1 eviction — {base.name}",
        "",
        "Each arm differs only in configuration; no code path differs between "
        "them. Arms ran back to back on the same box, one at a time.",
        "",
    ]
    out += _table(arms, data, ROWS)

    baseline = arms[0] if arms else None
    if baseline and len(arms) > 1:
        out += [
            "",
            f"## Relative to `{baseline}`",
            "",
            "| metric | " + " | ".join(arms[1:]) + " |",
            "|---|" + "---|" * (len(arms) - 1),
        ]
        for key, label, _ in ROWS:
            base_v = data[baseline].get(key)
            cells = []
            for arm in arms[1:]:
                v = data[arm].get(key)
                if isinstance(v, (int, float)) and isinstance(base_v, (int, float)) and base_v:
                    cells.append(f"{100.0 * v / base_v:.1f}%")
                else:
                    cells.append("n/a")
            out.append(f"| {label} | " + " | ".join(cells) + " |")

    out += [
        "",
        "## Mechanism",
        "",
        "Proof the couplings were live, and their cost. `n/a` = that arm does "
        "not run the mechanism. Server counters cover the whole run including "
        "the warm-up; stalls and deaths cover only the measured window.",
        "",
    ]
    out += _table(arms, data, MECH_ROWS)

    out += [
        "",
        "## What the rates mean",
        "",
        "- **GPU prefix hit %** = `prompt_tokens_cached_total / prompt_tokens_total`",
        "  — prompt tokens vLLM served from its own HBM prefix cache.",
        "- **L1 pool hit %** = `external_prefix_cache_hits_total / "
        "external_prefix_cache_queries_total` — of what the GPU missed and "
        "asked the shared pool for, the share the pool could answer.",
        "- **overall hit %** = `(prompt_tokens_cached + external hits) / "
        "prompt_tokens` — the share of prompt tokens that needed no fresh prefill.",
        "",
        "Arms ran for a fixed wall clock, so a better policy serves MORE "
        "requests in the same window; read `requests completed` and "
        "`output tok/s` as results, not as confounds.",
        "",
        "**A caveat the offline replay could not show:** the sim had no TTFT "
        "model, so extra L1 loads never queued there. TTFT here is the first "
        "real evidence on that axis, and it is the number most likely to "
        "disagree with the simulated ranking.",
    ]
    return "\n".join(out) + "\n"


GATE_TOLERANCE_PCT = 5.0
GATE_HANDLER_P99_US = 1000.0


def gate(base: Path, arms: list[str]) -> tuple[bool, str]:
    """The notice-path overhead gate: notices on (arms[1]) vs off (arms[0]).

    Criteria (cachelab decision 2026-09-22): TTFT p50 and output tok/s within
    +-5 %, zero stalls and engine deaths after warm-up in both arms, and the
    server's notice handler p99 under 1 ms.

    Returns:
        (passed, Markdown section).
    """
    if len(arms) != 2:
        return False, "## Gate\n\nFAIL: the gate needs exactly two arms (off, on).\n"
    off, on = (collect(base, a) for a in arms)
    checks: list[tuple[str, bool, str]] = []

    def within(key: str, label: str) -> None:
        a, b = off.get(key), on.get(key)
        if not isinstance(a, (int, float)) or not isinstance(b, (int, float)) or not a:
            checks.append((label, False, "missing"))
            return
        delta = 100.0 * (b - a) / a
        checks.append((label, abs(delta) <= GATE_TOLERANCE_PCT, f"{delta:+.1f}%"))

    within("ttft_p50_s", f"TTFT p50 within +-{GATE_TOLERANCE_PCT:.0f}%")
    within("out_tok_per_s", f"output tok/s within +-{GATE_TOLERANCE_PCT:.0f}%")
    for arm, d in zip(arms, (off, on)):
        bad = (d.get("stalls") or 0) + (d.get("engine_deaths") or 0)
        ok = d.get("stalls") is not None and bad == 0
        checks.append((f"no stalls/deaths after warm-up ({arm})", ok, f"{bad:.0f}"))
        checks.append((
            f"warm-up completed ({arm})",
            d.get("warmup_ok") == 1.0,
            {1.0: "yes", 0.0: "no", None: "n/a"}[d.get("warmup_ok")],
        ))
    p99 = on.get("notice_handler_p99_us")
    checks.append((
        f"handler p99 < {GATE_HANDLER_P99_US:.0f} us",
        isinstance(p99, (int, float)) and p99 < GATE_HANDLER_P99_US,
        f"{p99}" if p99 is not None else "missing",
    ))
    sent = on.get("notices_sent")
    checks.append(("notices actually flowed (on)", bool(sent), f"{sent}"))

    passed = all(ok for _, ok, _ in checks)
    lines = [
        f"## Gate: `{arms[1]}` vs `{arms[0]}` -- {'PASS' if passed else 'FAIL'}",
        "",
        "| criterion | result | value |",
        "|---|---|---|",
    ]
    lines += [f"| {c} | {'PASS' if ok else 'FAIL'} | {v} |" for c, ok, v in checks]
    return passed, "\n".join(lines) + "\n"


def require_clean(base: Path, arms: list[str]) -> tuple[bool, str]:
    """The smoke verdict: every arm healthy, and every coupling actually live.

    Per arm: the warm-up completed, requests completed, and no stalls or
    engine deaths followed the warm-up. Arms that send notices must have sent some and completed GPU
    chunks. The queued-prefix hash hit ratio is reported as a warning only: a
    short run may queue no follow-up turn whose prefix is already in L1.

    Returns:
        (passed, Markdown section).
    """
    rows: list[tuple[str, str, bool, str]] = []
    for arm in arms:
        d = collect(base, arm)
        bad = (d.get("stalls") or 0) + (d.get("engine_deaths") or 0)
        # Missing data is a failure, never a pass: an arm whose servers did
        # not start leaves no warm-up status and no completed requests.
        rows.append((arm, "warm-up completed", d.get("warmup_ok") == 1.0,
                     {1.0: "yes", 0.0: "no", None: "missing"}[d.get("warmup_ok")]))
        rows.append((arm, "requests completed", bool(d.get("requests")),
                     f"{d.get('requests') or 0:.0f}"))
        rows.append((arm, "no stalls/deaths after warm-up",
                     d.get("stalls") is not None and bad == 0, f"{bad:.0f}"))
        if d.get("notices_sent") is not None:
            rows.append((arm, "notices sent", bool(d.get("notices_sent")),
                         f"{d.get('notices_sent'):.0f}"))
            rows.append((arm, "GPU chunks completed", bool(d.get("chunks_completed")),
                         f"{d.get('chunks_completed') or 0:.0f}"))
            hit = d.get("waiting_hash_in_l1_pct")
            rows.append((arm, "queued-prefix hashes found in L1 (warning only)", True,
                         "n/a" if hit is None else f"{hit:.1f}%"))
    passed = all(ok for _, _, ok, _ in rows)
    lines = [
        f"## Smoke verdict -- {'PASS' if passed else 'FAIL'}",
        "",
        "| arm | check | result | value |",
        "|---|---|---|---|",
    ]
    lines += [f"| {a} | {c} | {'PASS' if ok else 'FAIL'} | {v} |" for a, c, ok, v in rows]
    return passed, "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", required=True, type=Path)
    ap.add_argument("--arms", nargs="*", default=[])
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument(
        "--require-clean",
        action="store_true",
        help="smoke verdict: exit 1 unless every arm is healthy and live",
    )
    ap.add_argument(
        "--gate",
        action="store_true",
        help="check the notice overhead gate (arm 1 = off, arm 2 = on); "
        "exit 1 unless it passes",
    )
    args = ap.parse_args()

    arms = [a for a in args.arms if a]
    if not arms:
        args.out.write_text("# Coupled round\n\nNo arms completed.\n")
        print("no arms to summarise")
        sys.exit(1 if args.gate else 0)

    text = render(args.base, arms)
    passed = True
    if args.require_clean:
        ok, section = require_clean(args.base, arms)
        passed = passed and ok
        text += "\n" + section
        print(section)
    if args.gate:
        ok, section = gate(args.base, arms)
        passed = passed and ok
        text += "\n" + section
        print(section)
    args.out.write_text(text)
    print(f"wrote {args.out}")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
