"""Minimum detectable per-condition accuracy delta at candidate grid sizes - the
design justification for this study's task/seed count, before any GPU time.
Mirrors simswap's detectability.py (plant a known effect on synthetic data, run the
real analyze.py statistics, sweep until reliably detected) but in-memory rather than
through disk - this sweep runs (grid x delta x replicate) trials and each trial's
data-generating process must stay fast, so it reimplements fake_runs.py's per-cell
accuracy draw directly rather than writing/reading real RunRecord files per trial.
fake_runs.py itself (disk-based, real RunRecord/schema.py objects) is the one that
validates the actual file format end to end; this script only needs the same
STATISTICAL process, not the file I/O.

"Detected" = the condition carrying the planted delta is Holm-significant at
alpha=0.05 in analyze.py's own paired-permutation-plus-Holm test. "Reliably
detected" = detected in >= POWER_TARGET of REPLICATES independent draws at that
(grid, delta) - a standard power-analysis convention, not a single lucky draw.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from analyze import CONTROL, holm_correct, paired_permutation_test
from schema import CONDITIONS

REPLICATES = 60
POWER_TARGET = 0.80
N_PERM = 2000  # lower than analyze.py's real-run default (10000) - this sweep runs
                # (grid x delta x REPLICATES x 5 conditions) permutation tests, needs
                # to stay fast; 2000 is still enough resolution for p vs alpha=0.05
                # decisions at these n_pairs (worst case min p ~ 1/2001 = 0.0005).

N_TASKS_GRID = [15, 20, 25, 30]
N_SEEDS = 2
DELTAS = [0.05, 0.08, 0.10, 0.12, 0.15, 0.20]

BASE_SKILL = 0.55


def simulate_condition_accuracies(n_tasks: int, n_seeds: int, condition_delta: float,
                                   rng: np.random.Generator) -> dict[tuple[str, int], float]:
    """One condition's (task, seed) -> accuracy cells, same per-cell process as
    fake_runs.py: task-difficulty offset, p-level noise, binomial action sampling,
    and the same ~5%/3%/92% error/max_turns/normal status mix (errors contribute
    read=write=ActionTally(0,0), i.e. accuracy=None, same exclusion as analyze.py
    applies to real runs)."""
    task_difficulty = rng.normal(0, 0.05, n_tasks)
    n_floor = max(1, round(n_tasks * 0.13)) if n_tasks > 0 else 0
    floor_idx = rng.choice(n_tasks, size=n_floor, replace=False) if n_floor else []
    for i in floor_idx:
        task_difficulty[i] = -0.35

    cells: dict[tuple[str, int], float] = {}
    for ti in range(n_tasks):
        task_id = f"task{ti:02d}"
        for seed in range(n_seeds):
            p = BASE_SKILL - condition_delta + task_difficulty[ti]
            p += rng.normal(0, 0.03)
            p = float(np.clip(p, 0.02, 0.95))

            dice = rng.random()
            if dice < 0.05:
                continue  # harness_crash -> accuracy=None, excluded, matches analyze.py
            elif dice < 0.10:
                n_read, n_write = int(rng.integers(1, 5)), int(rng.integers(1, 4))
            else:
                n_read, n_write = int(rng.integers(4, 14)), int(rng.integers(3, 10))
            n_read_correct = int(rng.binomial(n_read, p))
            n_write_correct = int(rng.binomial(n_write, p))
            total = n_read + n_write
            if total == 0:
                continue
            cells[(task_id, seed)] = (n_read_correct + n_write_correct) / total
    return cells


def trial_detects(n_tasks: int, n_seeds: int, delta: float, rng: np.random.Generator) -> bool:
    """One replicate: plant `delta` on ALL 5 non-control conditions (uniform
    scenario), run the real contrast+Holm pipeline, return whether every one of the
    5 planted conditions came out Holm-significant. This is the strict criterion -
    "reliably detects the effect wherever it's real", not just "detects it
    somewhere" - matching this study's actual use of the test (attributing failure
    to a SPECIFIC condition, not just noticing that something moved)."""
    control_cells = simulate_condition_accuracies(n_tasks, n_seeds, 0.0, rng)
    other_conditions = [c for c in CONDITIONS if c != CONTROL]

    raw_p = []
    for _ in other_conditions:
        cond_cells = simulate_condition_accuracies(n_tasks, n_seeds, delta, rng)
        shared = sorted(set(control_cells) & set(cond_cells))
        if len(shared) < 2:
            raw_p.append(1.0)
            continue
        diffs = np.array([control_cells[k] - cond_cells[k] for k in shared])
        raw_p.append(paired_permutation_test(diffs, N_PERM, rng))

    p_holm = holm_correct(raw_p)
    return all(p < 0.05 for p in p_holm)


def main() -> None:
    rng = np.random.default_rng(20260902)  # ATTRIB deadline date, arbitrary but fixed/documented
    tables_dir = Path("tables")
    tables_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for n_tasks in N_TASKS_GRID:
        min_detected_delta = None
        for delta in DELTAS:
            hits = sum(trial_detects(n_tasks, N_SEEDS, delta, rng) for _ in range(REPLICATES))
            power = hits / REPLICATES
            rows.append({"n_tasks": n_tasks, "n_seeds": N_SEEDS, "delta": delta,
                         "replicates": REPLICATES, "power": power})
            print(f"n_tasks={n_tasks:3d} n_seeds={N_SEEDS} delta={delta:.2f}  power={power:.2f}")
            if power >= POWER_TARGET and min_detected_delta is None:
                min_detected_delta = delta
        print(f"  -> n_tasks={n_tasks}: smallest delta reaching {POWER_TARGET:.0%} power "
              f"= {min_detected_delta if min_detected_delta is not None else 'NOT REACHED in DELTAS grid'}")

    out_csv = tables_dir / "detectability.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["n_tasks", "n_seeds", "delta", "replicates", "power"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {out_csv}")


if __name__ == "__main__":
    main()
