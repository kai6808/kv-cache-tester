#!/usr/bin/env python3
"""Replay a Qwen Bailian usage trace (qwen_traceA / qwen_traceB, 16-token
hash_id blocks) against an OpenAI-compatible vLLM server, alongside
trace_replay_tester.py's agentic sessions (cachelab mixed workload, Sep 29 IV).

Prompts are token ids, not text: every hash_id maps to 16 deterministic token
ids (seeded by the id and the trace class), sent through /v1/completions. Two
requests that share a hash_id prefix in the trace therefore share exactly that
token prefix, so vLLM / LMCache see the trace's cross-request prefix sharing
(e.g. trace B's system prompts) with no tokenizer boundary effects. The model's
own outputs are not fed back: a follow-up turn's prompt is the trace's blocks
(so the previous response's blocks are new tokens to the server; a small
under-count of real chat reuse).

Pacing:
  * root requests (turn 1) and every trace B request: open loop, sent at their
    trace timestamp (relative to the window start) x --time-scale;
  * trace A follow-up turns: closed loop + trace gap -- sent when the parent
    turn completed plus (ts_child - ts_parent) x --time-scale.
Sampling: conversations whose root's trace timestamp falls in [--window-start,
--window-start + --window-s) are kept with probability --sample-frac; of a
sampled conversation, only turns whose trace timestamp is inside the window are
replayed (--keep-tails replays the later turns too; they can run ~1 h past a
short window). The sampled set is fixed by --seed, so every arm
replays exactly the same requests (fixed workload); the run ends when all of
them completed (or --deadline-s).
Routing: --dp-affinity N pins a request with X-data-parallel-rank = crc32(key)
% N; key = root chat_id (--sticky-key conversation) or, for --sticky-key
prefix, the first --prefix-blocks hash_ids when the prompt has that many blocks
(short prompts fall back to the chat_id).

Output: <output-dir>/qwen_<class>_results.csv, one row per request.
"""
import argparse
import asyncio
import csv
import json
import os
import random
import time
import zlib

import aiohttp

BLOCK = 16
# Qwen2.5 vocab: 151,643 regular ids; stay clear of the low ids (bytes /
# whitespace merges) and of the special tokens at the top.
TOK_LO, TOK_HI = 1000, 150000


def block_tokens(hash_id: int, salt: int) -> list:
    rng = random.Random((hash_id * 1000003) ^ salt)
    return [rng.randrange(TOK_LO, TOK_HI) for _ in range(BLOCK)]


def build_prompt(rec: dict, salt: int, cache: dict) -> list:
    toks = []
    for h in rec["hash_ids"]:
        b = cache.get(h)
        if b is None:
            b = cache[h] = block_tokens(h, salt)
        toks.extend(b)
    n = max(1, min(rec["input_length"], len(toks)))
    return toks[:n]


def load_sample(path, window_start, window_s, frac, seed, max_requests, truncate=True):
    recs = [json.loads(line) for line in open(path)]
    by_id = {r["chat_id"]: r for r in recs}
    children = {}
    for r in recs:
        if r["parent_chat_id"] != -1:
            children.setdefault(r["parent_chat_id"], []).append(r)

    def root_of(r):
        while r["parent_chat_id"] in by_id:
            r = by_id[r["parent_chat_id"]]
        return r

    rng = random.Random(seed)
    lo, hi = window_start, window_start + window_s
    roots = [r for r in recs if r["parent_chat_id"] == -1 or r["parent_chat_id"] not in by_id]
    roots = [r for r in roots if lo <= r["timestamp"] < hi]
    picked = [r for r in roots if rng.random() < frac]
    sample = []
    for root in picked:
        stack = [root]
        while stack:
            r = stack.pop()
            if truncate and r["timestamp"] >= hi:
                continue  # turn arrives after the window: dropped (with its descendants)
            sample.append(r)
            stack.extend(children.get(r["chat_id"], []))
        if max_requests and len(sample) >= max_requests:
            break
    return sample, children, lo, root_of


class Replayer:
    def __init__(self, args):
        self.a = args
        self.salt = zlib.crc32(args.trace_class.encode())
        self.block_cache = {}
        self.rows_by_id = {}
        self.t0 = None

    def headers(self, rec, root_id):
        if not self.a.dp_affinity:
            return {}
        if self.a.sticky_key == "prefix" and len(rec["hash_ids"]) >= self.a.prefix_blocks:
            key = ",".join(map(str, rec["hash_ids"][: self.a.prefix_blocks]))
        else:
            key = str(root_id)
        return {"X-data-parallel-rank": str(zlib.crc32(key.encode()) % self.a.dp_affinity)}

    def _row(self, rec, root_id):
        """Every sampled request gets a row up front (status not_sent), so a
        deadline, SIGINT or crash can never make a request vanish from the CSV."""
        row = self.rows_by_id.get(rec["chat_id"])
        if row is None:
            row = {"class": self.a.trace_class, "chat_id": rec["chat_id"], "root_id": root_id,
                   "turn": rec["turn"], "trace_ts": rec["timestamp"], "t_sched": "", "t_send": "",
                   "t_done": "", "input_tokens": min(rec["input_length"], len(rec["hash_ids"]) * BLOCK),
                   "output_tokens_expected": max(1, min(rec["output_length"], self.a.max_output_tokens)),
                   "rank_header": self.headers(rec, root_id).get("X-data-parallel-rank", ""),
                   "status": "not_sent", "success": False, "ttft": "", "e2e": "",
                   "cached_tokens": "", "output_tokens": "", "error": ""}
            self.rows_by_id[rec["chat_id"]] = row
        return row

    async def one(self, session, rec, root_id, t_sched):
        prompt = build_prompt(rec, self.salt, self.block_cache)
        out_len = max(1, min(rec["output_length"], self.a.max_output_tokens))
        body = {"model": self.a.model, "prompt": prompt, "max_tokens": out_len,
                "min_tokens": out_len, "ignore_eos": True, "temperature": 0.0,
                "stream": True, "stream_options": {"include_usage": True}}
        hdr = self.headers(rec, root_id)
        row = self._row(rec, root_id)
        row.update({"t_sched": t_sched, "t_send": time.time(), "input_tokens": len(prompt),
                    "status": "in_flight"})
        try:
            async with session.post(self.a.endpoint + "/v1/completions", json=body, headers=hdr,
                                    timeout=aiohttp.ClientTimeout(total=self.a.request_timeout_s)) as resp:
                if resp.status != 200:
                    row["error"] = f"HTTP {resp.status}: {(await resp.text())[:200]}"
                else:
                    first = None
                    done = False
                    usage_seen = None
                    async for raw in resp.content:
                        line = raw.decode().strip()
                        if not line.startswith("data:"):
                            continue
                        data = line[5:].strip()
                        if data == "[DONE]":
                            done = True
                            break
                        ev = json.loads(data)
                        if ev.get("error"):
                            # vLLM reports a mid-stream failure as an error event, then [DONE]
                            row["error"] = ("stream error: " + json.dumps(ev["error"]))[:200]
                            continue
                        if first is None and ev.get("choices") and ev["choices"][0].get("text") is not None:
                            first = time.time()
                        usage = ev.get("usage")
                        if usage:
                            usage_seen = usage
                            row["output_tokens"] = usage.get("completion_tokens", "")
                            det = usage.get("prompt_tokens_details") or {}
                            row["cached_tokens"] = det.get("cached_tokens", "")
                    t_end = time.time()
                    row["ttft"] = (first - row["t_send"]) if first else ""
                    row["e2e"] = t_end - row["t_send"]
                    if not row["error"]:
                        if not done:
                            row["error"] = "stream ended without [DONE]"
                        elif usage_seen is None:
                            row["error"] = "no usage record"
                        elif usage_seen.get("completion_tokens") != out_len:
                            row["error"] = f"completion_tokens {usage_seen.get('completion_tokens')} != {out_len}"
                        elif first is None:
                            row["error"] = "no token received"
                    row["success"] = not row["error"]
        except asyncio.CancelledError:
            row["status"] = "cancelled"
            row["t_done"] = time.time()
            raise
        except Exception as e:  # recorded, never raised: one failure must not stop the run
            row["error"] = repr(e)[:200]
        row["status"] = "ok" if row["success"] else "error"
        row["t_done"] = time.time()
        return row

    async def conversation(self, session, root, children, lo):
        # root at its (scaled) trace time; follow-ups closed loop + trace gap
        t_sched = self.t0 + (root["timestamp"] - lo) * self.a.time_scale
        await asyncio.sleep(max(0.0, t_sched - time.time()))
        stack = [(root, t_sched)]
        tasks = []
        while stack:
            rec, ts = stack.pop()
            row = await self.one(session, rec, root["chat_id"], ts)
            for child in sorted(children.get(rec["chat_id"], []), key=lambda r: r["timestamp"]):
                gap = (child["timestamp"] - rec["timestamp"]) * self.a.time_scale
                t_next = row["t_done"] + max(0.0, gap)
                tasks.append(asyncio.create_task(self.follow(session, child, root["chat_id"], t_next, children)))
        if tasks:
            await asyncio.gather(*tasks)

    async def follow(self, session, rec, root_id, t_next, children):
        await asyncio.sleep(max(0.0, t_next - time.time()))
        row = await self.one(session, rec, root_id, t_next)
        subs = []
        for child in sorted(children.get(rec["chat_id"], []), key=lambda r: r["timestamp"]):
            gap = (child["timestamp"] - rec["timestamp"]) * self.a.time_scale
            subs.append(asyncio.create_task(self.follow(session, child, root_id, row["t_done"] + max(0.0, gap), children)))
        if subs:
            await asyncio.gather(*subs)

    def write_csv(self, n_sample):
        a = self.a
        os.makedirs(a.output_dir, exist_ok=True)
        path = os.path.join(a.output_dir, f"qwen_{a.trace_class}_results.csv")
        cols = ["class", "chat_id", "root_id", "turn", "trace_ts", "t_sched", "t_send", "t_done", "ttft", "e2e",
                "input_tokens", "cached_tokens", "output_tokens_expected", "output_tokens", "rank_header",
                "status", "success", "error"]
        rows = sorted(self.rows_by_id.values(), key=lambda r: (r["trace_ts"], r["chat_id"]))
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in rows:
                w.writerow({c: r.get(c, "") for c in cols})
        st = {}
        for r in rows:
            st[r["status"]] = st.get(r["status"], 0) + 1
        el = time.time() - self.t0 if self.t0 else 0
        print(f"[qwen_replay {a.trace_class}] {n_sample} sampled, status {st}, {el:.0f} s -> {path}", flush=True)

    async def run(self):
        a = self.a
        sample, children, lo, _ = load_sample(a.trace, a.window_start, a.window_s, a.sample_frac,
                                              a.seed, a.max_requests, truncate=not a.keep_tails)
        ids = {r["chat_id"] for r in sample}
        children = {k: [c for c in v if c["chat_id"] in ids] for k, v in children.items()}
        roots = [r for r in sample if r["parent_chat_id"] == -1 or r["parent_chat_id"] not in ids]
        root_of = {}
        for r in roots:
            stack = [r]
            while stack:
                x = stack.pop()
                root_of[x["chat_id"]] = r["chat_id"]
                stack.extend(children.get(x["chat_id"], []))
        for r in sample:
            self._row(r, root_of[r["chat_id"]])
        tok = sum(r["input_length"] for r in sample)
        span = (max(r["timestamp"] for r in sample) - lo) * a.time_scale if sample else 0.0
        print(f"[qwen_replay {a.trace_class}] {len(sample)} requests ({len(roots)} conversations), "
              f"{tok/1e6:.2f} M prompt tokens, window {a.window_s:.0f} s x scale {a.time_scale} "
              f"-> offered {tok / (a.window_s * a.time_scale):.0f} prompt tok/s; "
              f"last trace arrival at +{span:.0f} s", flush=True)
        if span > 0.8 * a.deadline_s:
            print(f"[qwen_replay {a.trace_class}] WARNING: scaled trace span {span:.0f} s is close to or past "
                  f"--deadline-s {a.deadline_s:.0f}; closed-loop follow-ups would be cut", flush=True)
        if a.dry_run:
            return
        if a.start_at:
            await asyncio.sleep(max(0.0, a.start_at - time.time()))
        self.t0 = time.time()
        # A capped connection pool on the loopback address: with thousands of
        # open-loop requests in flight, an unlimited pool to "localhost"
        # exhausted the resolver / sockets (4% ClientConnector errors on Sep 30).
        # A request waiting for a connection is still timed from t_send, so
        # the queueing shows up in its TTFT.
        a.endpoint = a.endpoint.replace("//localhost", "//127.0.0.1")
        conn = aiohttp.TCPConnector(limit=a.max_connections)
        try:
            async with aiohttp.ClientSession(connector=conn) as session:
                work = asyncio.gather(*[self.conversation(session, r, children, lo) for r in roots])
                try:
                    await asyncio.wait_for(work, timeout=a.deadline_s)
                except asyncio.TimeoutError:
                    print(f"[qwen_replay {a.trace_class}] DEADLINE reached", flush=True)
        finally:
            self.write_csv(len(sample))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--trace", required=True)
    p.add_argument("--trace-class", required=True, help="label, e.g. A or B (also salts the token ids)")
    p.add_argument("--endpoint", default="http://localhost:8100")
    p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--window-start", type=float, default=0.0)
    p.add_argument("--window-s", type=float, default=1800.0)
    p.add_argument("--sample-frac", type=float, default=1.0)
    p.add_argument("--time-scale", type=float, default=1.0, help=">1 stretches the timeline (lower rate)")
    p.add_argument("--max-requests", type=int, default=0)
    p.add_argument("--keep-tails", action="store_true",
                   help="also replay turns whose trace time is after the window (default: drop them, "
                        "so every class's load ends with the window)")
    p.add_argument("--max-output-tokens", type=int, default=2048)
    p.add_argument("--max-connections", type=int, default=256,
                   help="open HTTP connections at most (queued requests wait; their TTFT includes the wait)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dp-affinity", type=int, default=0)
    p.add_argument("--sticky-key", choices=["conversation", "prefix"], default="conversation")
    p.add_argument("--prefix-blocks", type=int, default=16, help="16 x 16 tokens = one 256-token chunk")
    p.add_argument("--start-at", type=float, default=0.0, help="epoch seconds to start (sync with other clients)")
    p.add_argument("--deadline-s", type=float, default=5400.0)
    p.add_argument("--request-timeout-s", type=float, default=1800.0)
    p.add_argument("--output-dir", default=".")
    p.add_argument("--dry-run", action="store_true", help="print the sample size and offered load, send nothing")
    asyncio.run(Replayer(p.parse_args()).run())


if __name__ == "__main__":
    main()
