#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Build the arm-by-arm comparison for a coupled GPU + L1 eviction round.

Reads what the harness already writes -- ``server_metrics.json`` (vLLM and
LMCache counters) and ``summary_trace_replay.csv`` (client-side TTFT and
throughput) -- and emits one Markdown table. No new instrumentation, and no
per-event tracing: a real round writes kilobytes per arm.

Every rate is defined explicitly in the output, because "hit rate" is
ambiguous across tiers and the whole point of the round is comparing them.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
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
    """Return the last sample's counters, or {} when the file is absent."""
    path = run_dir / "server_metrics.json"
    if not path.is_file():
        return {}
    try:
        samples = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    if not samples:
        return {}
    final = samples[-1]
    counters = dict(final.get("counters", {}))
    counters["_elapsed"] = float(final.get("elapsed", 0.0) or 0.0)
    return counters


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


def render(base: Path, arms: list[str]) -> str:
    data = {arm: collect(base, arm) for arm in arms}
    out: list[str] = [
        f"# Coupled GPU + L1 eviction — {base.name}",
        "",
        "Each arm differs only in configuration; no code path differs between "
        "them. Arms ran back to back on the same box, one at a time.",
        "",
        "| metric | " + " | ".join(arms) + " |",
        "|---|" + "---|" * len(arms),
    ]
    for key, label, fmt in ROWS:
        cells = []
        for arm in arms:
            v = data[arm].get(key)
            cells.append(fmt.format(v) if isinstance(v, (int, float)) else "n/a")
        out.append(f"| {label} | " + " | ".join(cells) + " |")

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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", required=True, type=Path)
    ap.add_argument("--arms", nargs="*", default=[])
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    arms = [a for a in args.arms if a]
    if not arms:
        args.out.write_text("# Coupled round\n\nNo arms completed.\n")
        print("no arms to summarise")
        return

    args.out.write_text(render(args.base, arms))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
