"""Generate synthetic runs/ matching schema.py, for testing analyze.py - no GPU, no
tau2. Adapted from simswap's fake_runs.py: that script planted a rank inversion
across (agent, user_model); this one plants a per-condition accuracy delta across the
6 CONDITIONS, one agent/user pair fixed. Three scenarios:
  --scenario null          no effect anywhere - analyze.py must not report one
  --scenario uniform       all 5 non-control conditions get the same --delta
  --scenario policy_inert  matches this study's hypothesis: no_policy hurts a lot,
                            policy_truncated/schema_names_only barely hurt (the
                            "most of the policy document is inert" case),
                            history_truncated/system_minimized hurt moderately
Prints ground_truth.json so analyze.py's recovery of it can be checked.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from schema import CONDITIONS, ActionTally, RunRecord

AGENT_NAME = "Qwen2.5-7B-Instruct-AWQ"
USER_NAME = "Qwen2.5-7B-Instruct-AWQ"

SCENARIOS = ("null", "uniform", "policy_inert")

# policy_inert scenario: fraction of --delta actually applied per condition.
# 1.0 = full effect (as damaging as the theoretical worst case), 0.0 = inert.
POLICY_INERT_FRACTIONS = {
    "no_policy": 1.0,
    "policy_truncated": 0.05,
    "schema_names_only": 0.6,
    "history_truncated": 0.5,
    "system_minimized": 0.3,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--n-tasks", type=int, default=20)
    p.add_argument("--n-seeds", type=int, default=2)
    p.add_argument("--out", type=Path, default=Path("runs"))
    p.add_argument("--scenario", choices=SCENARIOS, default="uniform")
    p.add_argument("--delta", type=float, default=0.15,
                    help="accuracy-scale effect size for the planted condition(s) (p-scale)")
    p.add_argument("--history-k", default="4", help="condition_param value for history_truncated")
    p.add_argument("--rng-seed", type=int, default=42, help="seed for the fake data generator itself")
    p.add_argument("--domain", default="retail")
    return p.parse_args()


def condition_deltas(scenario: str, delta: float) -> dict[str, float]:
    """Map condition -> planted accuracy delta (positive = condition hurts accuracy
    relative to full_context). full_context itself is always 0.0 by construction."""
    if scenario == "null":
        return {c: 0.0 for c in CONDITIONS}
    if scenario == "uniform":
        return {c: (delta if c != "full_context" else 0.0) for c in CONDITIONS}
    if scenario == "policy_inert":
        return {
            c: (delta * POLICY_INERT_FRACTIONS[c] if c != "full_context" else 0.0)
            for c in CONDITIONS
        }
    raise ValueError(scenario)


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.rng_seed)

    tasks = [f"task{i:02d}" for i in range(args.n_tasks)]
    seeds = list(range(args.n_seeds))
    deltas = condition_deltas(args.scenario, args.delta)

    base_skill = 0.55  # full_context baseline action-correctness, arbitrary/synthetic
    # task difficulty offsets - some tasks near-floor for everyone, mirrors simswap's
    # fake_runs.py convention (real per-task variance, not flat).
    task_difficulty = rng.normal(0, 0.05, args.n_tasks)
    n_floor = max(1, round(args.n_tasks * 0.13)) if args.n_tasks > 0 else 0
    floor_idx = rng.choice(args.n_tasks, size=n_floor, replace=False) if n_floor else []
    for i in floor_idx:
        task_difficulty[i] = -0.35

    ground_truth = {
        "scenario": args.scenario,
        "delta": args.delta,
        "condition_deltas": deltas,
        "note": "condition_deltas is the planted accuracy DROP (p-scale) vs full_context, "
                "0.0 for full_context by construction. analyze.py's recovered per-condition "
                "contrast should track this, up to noise.",
    }

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "transcripts").mkdir(parents=True, exist_ok=True)

    status_counts = {"completed": 0, "error": 0, "timeout": 0}
    reason_counts: dict[str, int] = {}
    n_written = 0

    for condition in CONDITIONS:
        condition_param = args.history_k if condition == "history_truncated" else None
        cond_slug = f"cond-{condition}" + (f"-{condition_param}" if condition_param else "")
        for ti, task_id in enumerate(tasks):
            for seed in seeds:
                run_id = f"{task_id}__{AGENT_NAME}__{USER_NAME}__{cond_slug}__seed{seed}"

                p = base_skill - deltas[condition] + task_difficulty[ti]
                p += rng.normal(0, 0.03)
                p = float(np.clip(p, 0.02, 0.95))

                n_turns = max(2, int(round(9 + rng.integers(-2, 3))))

                dice = rng.random()
                if dice < 0.05:
                    status, reason = "error", "harness_crash"
                    error = "Traceback (most recent call last):\n  RuntimeError: vLLM server connection reset"
                    reward = None
                    task_success = None
                    read = ActionTally(0, 0)
                    write = ActionTally(0, 0)
                    n_turns = min(n_turns, int(rng.integers(0, 3)))
                    wall_clock = round(float(rng.uniform(1, 15)), 2)
                elif dice < 0.08:
                    status, reason = "completed", "max_turns"
                    error = None
                    reward = 0.0
                    task_success = False
                    n_read = int(rng.integers(1, 5))
                    n_write = int(rng.integers(1, 4))
                    read = ActionTally(n_read, int(rng.binomial(n_read, p)))
                    write = ActionTally(n_write, int(rng.binomial(n_write, p)))
                    n_turns = max(n_turns, 20 + int(rng.integers(0, 5)))
                    wall_clock = round(float(rng.uniform(30, 90)), 2)
                elif dice < 0.10:
                    status, reason = "error", "agent_error"
                    error = "Traceback (most recent call last):\n  ValueError: malformed tool call arguments"
                    reward = None
                    task_success = None
                    n_read = int(rng.integers(0, 3))
                    n_write = int(rng.integers(0, 2))
                    read = ActionTally(n_read, int(rng.binomial(n_read, p)))
                    write = ActionTally(n_write, int(rng.binomial(n_write, p)))
                    wall_clock = round(float(rng.uniform(2, 20)), 2)
                else:
                    status, reason = "completed", "normal"
                    error = None
                    p_success = p**1.5
                    reward = 1.0 if rng.random() < p_success else 0.0
                    task_success = reward >= 1.0 - 1e-6
                    n_read = int(rng.integers(4, 14))
                    n_write = int(rng.integers(3, 10))
                    read = ActionTally(n_read, int(rng.binomial(n_read, p)))
                    write = ActionTally(n_write, int(rng.binomial(n_write, p)))
                    wall_clock = round(float(rng.uniform(15, 60)), 2)

                status_counts[status] += 1
                reason_counts[reason] = reason_counts.get(reason, 0) + 1

                transcript_rel = f"transcripts/{run_id}.json"
                transcript = [
                    {"turn": t, "role": "user" if t % 2 == 0 else "assistant", "text": "..."}
                    for t in range(n_turns)
                ]
                (args.out / transcript_rel).write_text(json.dumps(transcript))

                record = RunRecord(
                    schema_version=1,
                    run_id=run_id,
                    tau2_version="1.0.1",
                    tau2_commit="fake0000tau2",
                    attrib_commit="fake0000attrib",
                    task_id=task_id,
                    domain=args.domain,
                    agent_model=AGENT_NAME,
                    agent_model_revision="fake-rev",
                    user_model=USER_NAME,
                    user_model_revision="fake-rev",
                    agent_quantization="fake-quant",
                    user_quantization="fake-quant",
                    seed=seed,
                    condition=condition,
                    condition_param=condition_param,
                    temperature=0.0,
                    gpu_name="fake0000gpu",
                    timestamp="2026-08-17T00:00:00Z",
                    max_steps=200,
                    status=status,
                    termination_reason=reason,
                    error=error,
                    task_success=task_success,
                    reward=reward,
                    reward_info_note=None,
                    n_turns=n_turns,
                    read=read,
                    write=write,
                    wall_clock_seconds=wall_clock,
                    agent_cost=0.0,
                    user_cost=0.0,
                    transcript_path=transcript_rel,
                )
                record.save(args.out / f"{run_id}.json")
                n_written += 1

    (args.out / "ground_truth.json").write_text(json.dumps(ground_truth, indent=2))

    print(f"wrote {n_written} runs to {args.out}")
    print(f"status counts: {status_counts}")
    print(f"termination reason counts: {reason_counts}")
    print("GROUND TRUTH:")
    print(json.dumps(ground_truth, indent=2))


if __name__ == "__main__":
    main()
