"""Minimum detectable per-condition GROUNDING-RATE delta at the actual grid size
(n_tasks=20, n_seeds=2) - rerun of detectability.py's design for the proposed new
primary metric (see ANALYSIS_PLAN.md's ARCHIVED section and PROGRESS.md's
2026-08-17 metric-swap entry). `detectability.py` itself is untouched and stays as
the historical record for the old action_accuracy-based primary - this is a
separate script, not an edit, mirroring the "archive, don't delete" treatment
ANALYSIS_PLAN.md's frozen primary section got.

Same statistical machinery as detectability.py (same `contrast`-shaped pipeline via
analyze.py's `paired_permutation_test`/`holm_correct`, same strict "all 5
non-control conditions Holm-significant" detection criterion, same 80% power over
60 replicates) - only the per-cell data-generating process changes, to match
grounding rate's real shape instead of action_accuracy's.

Two real, not-invented numbers anchor this simulation:
- BASE_SKILL (grounded rate = 1 - ungrounded_fraction_by_argument_conditional):
  0.611, from this project's OWN real pilot data - `full_context`'s single real
  episode (task 93) had conditional ungrounded rate 14/36 = 38.9%, i.e. grounded
  61.1%. Computed via `grounding.py::analyze_conversation()` against
  `data/pilot_run_20260817/transcripts/pilot_93__full_context__seed0.json` - see
  PROGRESS.md for the full per-condition table. n=1, real but noisy - flagged, not
  hidden.
- DELTAS below is a plain curve (5/10/15/20/30 percentage points), not anchored to
  a single "reference effect." simswap's real observed Qwen-vs-Llama CONDITIONAL
  grounding-rate gap (34.9pp, `F:/simswap/docs/FAILURE_ANALYSIS.md:274-277`, "gap
  vs Qwen (conditional)" column, matched-user-simulator numbers) is NOT used as a
  planted delta here and is not reported as a single verdict - it is a gap between
  two different AGENT MODELS (Qwen vs Llama) under simswap's design, not between
  two context-ablation CONDITIONS of the same model under this study's design, so
  treating it as a credible prior for THIS grid's ablation effect size would be
  invented precision. The curve is the actual answer to "what can this grid
  detect"; 34.9pp is mentioned only as context, not as the number being tested
  against.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from analyze import CONTROL, holm_correct, paired_permutation_test
from schema import CONDITIONS

REPLICATES = 60
POWER_TARGET = 0.80
N_PERM = 2000

N_TASKS = 20  # the actual grid, per the operator's explicit ask - not swept here
N_SEEDS = 2
DELTAS = [0.05, 0.10, 0.15, 0.20, 0.30]  # operator's explicit curve, 2026-08-17 -
# NOT anchored to simswap's 34.9pp Qwen-vs-Llama gap as "the" verdict: that gap is
# a cross-model comparison, not a credible prior for a context-ablation delta on
# the same model - see this file's own module docstring. Report the curve, not a
# single number.

BASE_SKILL_GROUNDED = 0.611  # 1 - 0.389, this project's own real full_context pilot episode

# Real range of total_id_arguments observed across this pilot's 6 episodes (n=1
# each): 7 (schema_names_only) to 44 (full_context). Used as the per-episode
# "how many ID-bearing tool-call arguments were attempted" sampling range - the
# same role n_read/n_write played in detectability.py's accuracy simulation.
MIN_ID_ARGS, MAX_ID_ARGS = 7, 44


def simulate_condition_grounding(n_tasks: int, n_seeds: int, condition_delta: float,
                                  rng: np.random.Generator) -> dict[tuple[str, int], float]:
    """One condition's (task, seed) -> grounded_rate cells. Same task-difficulty/
    noise/exclusion shape as detectability.py's simulate_condition_accuracies, but
    the per-cell binomial draw is over ID-bearing tool-call arguments (grounded vs
    ungrounded), not read/write actions."""
    task_difficulty = rng.normal(0, 0.05, n_tasks)
    n_floor = max(1, round(n_tasks * 0.13)) if n_tasks > 0 else 0
    floor_idx = rng.choice(n_tasks, size=n_floor, replace=False) if n_floor else []
    for i in floor_idx:
        task_difficulty[i] = -0.35

    cells: dict[tuple[str, int], float] = {}
    for ti in range(n_tasks):
        task_id = f"task{ti:02d}"
        for seed in range(n_seeds):
            p_grounded = BASE_SKILL_GROUNDED - condition_delta + task_difficulty[ti]
            p_grounded += rng.normal(0, 0.03)
            p_grounded = float(np.clip(p_grounded, 0.02, 0.98))

            dice = rng.random()
            if dice < 0.05:
                continue  # harness_crash-equivalent -> no episode, excluded
            n_id_args = int(rng.integers(MIN_ID_ARGS, MAX_ID_ARGS + 1))
            if n_id_args == 0:
                continue
            n_grounded = int(rng.binomial(n_id_args, p_grounded))
            cells[(task_id, seed)] = n_grounded / n_id_args
    return cells


def trial_detects(n_tasks: int, n_seeds: int, delta: float, rng: np.random.Generator) -> bool:
    """Same strict criterion as detectability.py: plant `delta` on all 5 non-control
    conditions, require all 5 Holm-significant to count as detected."""
    control_cells = simulate_condition_grounding(n_tasks, n_seeds, 0.0, rng)
    other_conditions = [c for c in CONDITIONS if c != CONTROL]

    raw_p = []
    for _ in other_conditions:
        cond_cells = simulate_condition_grounding(n_tasks, n_seeds, delta, rng)
        shared = sorted(set(control_cells) & set(cond_cells))
        if len(shared) < 2:
            raw_p.append(1.0)
            continue
        diffs = np.array([control_cells[k] - cond_cells[k] for k in shared])
        raw_p.append(paired_permutation_test(diffs, N_PERM, rng))

    p_holm = holm_correct(raw_p)
    return all(p < 0.05 for p in p_holm)


def main() -> None:
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--n-tasks", type=int, default=N_TASKS,
                   help="override the module default (20) - used 2026-08-17 to cost "
                        "scope-cut options against session 1's real crash/attrition "
                        "data, see PROGRESS.md. Default (20) reproduces the frozen "
                        "tables/detectability_grounding.csv result unchanged.")
    p.add_argument("--n-seeds", type=int, default=N_SEEDS)
    p.add_argument("--out-suffix", default="",
                   help="appended to the output CSV filename so scope-option reruns "
                        "don't overwrite the frozen n_tasks=20/n_seeds=2 result.")
    args = p.parse_args()

    n_tasks, n_seeds = args.n_tasks, args.n_seeds
    rng = np.random.default_rng(20260902)
    tables_dir = Path("tables")
    tables_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    min_detected_delta = None
    for delta in DELTAS:
        hits = sum(trial_detects(n_tasks, n_seeds, delta, rng) for _ in range(REPLICATES))
        power = hits / REPLICATES
        rows.append({"n_tasks": n_tasks, "n_seeds": n_seeds, "delta": delta,
                     "replicates": REPLICATES, "power": power})
        print(f"n_tasks={n_tasks} n_seeds={n_seeds} delta={delta:.3f}  power={power:.2f}")
        if power >= POWER_TARGET and min_detected_delta is None:
            min_detected_delta = delta

    print(f"\n-> at n_tasks={n_tasks}, n_seeds={n_seeds}: smallest tested delta reaching "
          f"{POWER_TARGET:.0%} power = "
          f"{min_detected_delta if min_detected_delta is not None else 'NOT REACHED in DELTAS grid'}")
    print("NOTE: simswap's 34.9pp Qwen-vs-Llama gap is NOT reported as a single "
          "verdict here - it is a cross-model comparison, not a credible prior for "
          "a context-ablation delta on the same model. The curve above is the "
          "actual answer to 'what can this grid detect.'")

    out_csv = tables_dir / f"detectability_grounding{args.out_suffix}.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["n_tasks", "n_seeds", "delta", "replicates", "power"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {out_csv}")


if __name__ == "__main__":
    main()
