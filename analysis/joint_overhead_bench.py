#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Microbenchmark: LMCache's LRU cache policy vs. the JOINT adapter.

Task A6 of the Rung 1 Stage A plan: before trusting JOINT as a
drop-in CPU eviction policy, measure its per-op
overhead relative to plain LRU on the three calls the storage backend
actually makes -- update_on_put, update_on_hit, and the one-candidate-
at-a-time get_evict_candidates(1) scan used by local_cpu_backend.py.

Each measured round constructs a fresh policy (and, for JOINT, resets
the process-wide JointController singleton first) so no round's timing
is contaminated by a previous round's resident keys -- update_on_put in
particular is only meaningful starting from an empty cache_dict.

Usage:
    python3 analysis/joint_overhead_bench.py

Exit code is 1 if any budget criterion fails, 0 if all pass.
"""

# Standard
from dataclasses import dataclass
import statistics
import sys
import time

# Third Party
import torch

# First Party
from lmcache.utils import CacheEngineKey
from lmcache.v1.joint import reset_joint_controller_for_testing
from lmcache.v1.storage_backend.cache_policy import get_cache_policy
from lmcache.v1.storage_backend.cache_policy.base_policy import BaseCachePolicy

CHUNK_SIZE_BYTES = 14 * 2**20  # 14MB, a realistic KV chunk size
N_VALUES = (2_000, 20_000)
ROUNDS = 5
EVICT_CALLS = 200
POLICY_NAMES = ("LRU", "JOINT")

# Budget criteria (see check_budgets()):
PUT_HIT_MAX_RATIO = 3.0  # JOINT put/hit must stay within this x LRU, at every n
EVICT_BUDGET_US = 1000.0  # JOINT get_evict_candidates(1) must stay below this
EVICT_BUDGET_N = 2_000  # ...measured at this n


class DummyMemoryObj:
    """Minimal MemoryObj stand-in: an evictable flag and a byte size, the
    only fields demand_model.CacheEntry / LRUCachePolicy need."""

    can_evict: bool = True

    def get_size(self) -> int:
        """Return the entry's size in bytes (fixed 14MB chunk)."""
        return CHUNK_SIZE_BYTES


@dataclass(frozen=True)
class RoundResult:
    """Per-op timings (microseconds) from a single fresh-policy round."""

    put_us_per_op: float
    hit_us_per_op: float
    evict_us_per_call: float


def build_keys(n: int) -> list[CacheEngineKey]:
    """Build n distinct real CacheEngineKey objects.

    Mirrors tests/v1/utils.py's dumb_cache_engine_key, imported directly
    from lmcache.utils rather than the test helper.
    """
    return [
        CacheEngineKey(
            model_name="joint_overhead_bench",
            world_size=1,
            worker_id=0,
            chunk_hash=i,
            dtype=torch.bfloat16,
        )
        for i in range(n)
    ]


def measure_put(
    policy: BaseCachePolicy, keys: list[CacheEngineKey], cache_dict: dict
) -> int:
    """Time populating cache_dict with all keys via the backend's real
    call order: insert into the mapping, then notify the policy.

    Returns:
        Elapsed wall time in nanoseconds for the whole loop.
    """
    start = time.perf_counter_ns()
    for key in keys:
        cache_dict[key] = DummyMemoryObj()
        policy.update_on_put(key)
    return time.perf_counter_ns() - start


def measure_hit(
    policy: BaseCachePolicy, keys: list[CacheEngineKey], cache_dict: dict
) -> int:
    """Time one update_on_hit call per key, cycling through all keys once.

    Returns:
        Elapsed wall time in nanoseconds for the whole loop.
    """
    start = time.perf_counter_ns()
    for key in keys:
        policy.update_on_hit(key, cache_dict)
    return time.perf_counter_ns() - start


def measure_evict(policy: BaseCachePolicy, cache_dict: dict, calls: int) -> int:
    """Time repeated get_evict_candidates(1) calls, removing each returned
    key from the mapping exactly as the backend's eviction loop does
    (batched_remove pops hot_cache with no policy callback,
    local_cpu_backend.py:770-800). This exercises JOINT's memoized ranking
    at its real amortized cost: serving from the memo between removals,
    re-scanning when it drains. `calls` must stay below the resident count.

    Returns:
        Elapsed wall time in nanoseconds for the whole loop.
    """
    start = time.perf_counter_ns()
    for _ in range(calls):
        evict_keys = policy.get_evict_candidates(cache_dict, num_candidates=1)
        del cache_dict[evict_keys[0]]
    return time.perf_counter_ns() - start


def run_round(policy_name: str, keys: list[CacheEngineKey]) -> RoundResult:
    """Construct a fresh policy and time one full put/hit/evict round.

    Args:
        policy_name: "LRU" or "JOINT".
        keys: Keys to populate the cache with (len(keys) == n).

    Returns:
        Per-op timings in microseconds for this round.
    """
    if policy_name == "JOINT":
        # The adapter binds a process-wide singleton (get_joint_controller);
        # reset it so this round's controller starts cold, not carrying
        # state from the previous round.
        reset_joint_controller_for_testing()
    policy = get_cache_policy(policy_name)
    cache_dict = policy.init_mutable_mapping()

    n = len(keys)
    put_ns = measure_put(policy, keys, cache_dict)
    hit_ns = measure_hit(policy, keys, cache_dict)
    evict_ns = measure_evict(policy, cache_dict, EVICT_CALLS)

    return RoundResult(
        put_us_per_op=put_ns / n / 1000.0,
        hit_us_per_op=hit_ns / n / 1000.0,
        evict_us_per_call=evict_ns / EVICT_CALLS / 1000.0,
    )


def median_result(rounds: list[RoundResult]) -> RoundResult:
    """Take the per-field median across rounds."""
    return RoundResult(
        put_us_per_op=statistics.median(r.put_us_per_op for r in rounds),
        hit_us_per_op=statistics.median(r.hit_us_per_op for r in rounds),
        evict_us_per_call=statistics.median(r.evict_us_per_call for r in rounds),
    )


def print_table(results: dict[int, dict[str, RoundResult]]) -> None:
    """Print an aligned (n, op) x (LRU, JOINT, ratio) table."""
    header = f"{'n':>8}  {'op':<22}{'LRU (us)':>12}{'JOINT (us)':>12}{'ratio':>10}"
    print(header)
    print("-" * len(header))
    for n in N_VALUES:
        lru = results[n]["LRU"]
        joint = results[n]["JOINT"]
        for op, lru_val, joint_val in (
            ("update_on_put", lru.put_us_per_op, joint.put_us_per_op),
            ("update_on_hit", lru.hit_us_per_op, joint.hit_us_per_op),
            ("get_evict_candidates(1)", lru.evict_us_per_call, joint.evict_us_per_call),
        ):
            ratio = joint_val / lru_val if lru_val > 0 else float("inf")
            print(f"{n:>8}  {op:<22}{lru_val:>12.3f}{joint_val:>12.3f}{ratio:>10.2f}")


def check_budgets(results: dict[int, dict[str, RoundResult]]) -> bool:
    """Check and print PASS/FAIL for each budget criterion.

    Returns:
        True if every criterion passed, False otherwise.
    """
    all_passed = True
    print()
    print("Budget checks:")
    for n in N_VALUES:
        lru = results[n]["LRU"]
        joint = results[n]["JOINT"]
        for op, lru_val, joint_val in (
            ("update_on_put", lru.put_us_per_op, joint.put_us_per_op),
            ("update_on_hit", lru.hit_us_per_op, joint.hit_us_per_op),
        ):
            ratio = joint_val / lru_val if lru_val > 0 else float("inf")
            passed = ratio <= PUT_HIT_MAX_RATIO
            all_passed &= passed
            status = "PASS" if passed else "FAIL"
            print(
                f"  [{status}] n={n:<7} JOINT {op} within "
                f"{PUT_HIT_MAX_RATIO}x of LRU: ratio={ratio:.2f} "
                f"(LRU={lru_val:.3f}us, JOINT={joint_val:.3f}us)"
            )

    evict_us = results[EVICT_BUDGET_N]["JOINT"].evict_us_per_call
    evict_passed = evict_us < EVICT_BUDGET_US
    all_passed &= evict_passed
    status = "PASS" if evict_passed else "FAIL"
    print(
        f"  [{status}] n={EVICT_BUDGET_N:<7} JOINT get_evict_candidates(1) < "
        f"{EVICT_BUDGET_US:.0f}us: measured={evict_us:.3f}us"
    )
    return all_passed


def main() -> int:
    """Run the LRU-vs-JOINT overhead microbenchmark.

    Returns:
        0 if all budget criteria pass, 1 otherwise.
    """
    results: dict[int, dict[str, RoundResult]] = {}
    for n in N_VALUES:
        keys = build_keys(n)
        results[n] = {}
        for policy_name in POLICY_NAMES:
            rounds = [run_round(policy_name, keys) for _ in range(ROUNDS)]
            results[n][policy_name] = median_result(rounds)

    print_table(results)
    passed = check_budgets(results)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
