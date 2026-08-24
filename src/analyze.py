"""Read runs/, compute per-condition action-accuracy and attribute drops to
conditions vs the full_context control. No hand-computed numbers - every reported
value comes out of this script. Primary metric: action-level correctness
(read+write tally), paired by (task_id, seed) against the full_context control run
for the same cell (same agent/user model throughout this study, only context varies).

Statistic per condition: paired mean accuracy difference (control - condition,
positive = condition hurt). Significance: Monte Carlo paired sign-flip permutation
test (exact enumeration is infeasible once n_pairs exceeds ~20, unlike simswap's
3-task case), Holm-corrected across the non-control conditions present in the data.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from schema import CONDITIONS, RunRecord

CONTROL = "full_context"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--runs-dir", type=Path, default=Path("runs"))
    p.add_argument("--tables-dir", type=Path, default=Path("tables"))
    p.add_argument("--n-perm", type=int, default=10000)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--rng-seed", type=int, default=0)
    return p.parse_args()


def load_runs(runs_dir: Path) -> list[RunRecord]:
    records = []
    for p in sorted(runs_dir.glob("*.json")):
        if p.name == "ground_truth.json":
            continue
        records.append(RunRecord.load(p))
    return records


def accuracy_by_cell(
    records: list[RunRecord],
    exclude_truncated: bool = False,
    exclude_agent_error: bool = False,
) -> dict[str, dict[tuple[str, int], float]]:
    """condition -> {(task_id, seed): action_accuracy}, completed runs with >=1 action only.

    PRIMARY (both flags False, the frozen default): does NOT filter or weight on
    `termination_reason`. An episode that hits max_steps (`termination_reason ==
    "max_turns"`) or too many tool-call errors (`termination_reason == "agent_error"`)
    is a harness-level `status == "completed"` outcome, not a failure - it still
    scores whatever actions it took, same as a normally-ended episode. `status` (did
    the harness get a scoreable result at all) and `termination_reason` (why the
    episode itself ended) are kept separate everywhere in this module; only `status`
    gates inclusion here by default, and `action_accuracy`/`task_success` never look
    at `termination_reason`.

    SECONDARY robustness check (exclude_truncated=True): additionally drops runs with
    `termination_reason == "max_turns"`. Pre-registered in ANALYSIS_PLAN.md's
    "PRE-REGISTERED SECONDARY" section, written blind before any pilot output existed.

    THIRD view (exclude_agent_error=True, added 2026-08-17 alongside exclude_truncated
    - "normal-terminations-only" when both are True): additionally drops runs with
    `termination_reason == "agent_error"`. Pre-registered in ANALYSIS_PLAN.md's
    "PRE-REGISTERED: agent_error handling" section, added after the pilot showed
    `agent_error` is not rare - written from the same principle as the max_turns
    check above, not tuned to any specific accuracy number.

    All views reported alongside each other in `main()` regardless of whether they
    agree - none is ever used to silently replace another."""
    out: dict[str, dict[tuple[str, int], float]] = {c: {} for c in CONDITIONS}
    for r in records:
        if r.status != "completed":
            continue
        if exclude_truncated and r.termination_reason == "max_turns":
            continue
        if exclude_agent_error and r.termination_reason == "agent_error":
            continue
        acc = r.action_accuracy
        if acc is None:
            continue
        out[r.condition][(r.task_id, r.seed)] = acc
    return out


def truncation_rate_by_condition(records: list[RunRecord]) -> dict[str, dict]:
    """condition -> {n_completed, n_truncated, truncation_rate}.

    Truncated = `status == "completed"` AND `termination_reason == "max_turns"` - a
    completed (scoreable) episode that ran out of steps, not a harness failure (see
    `accuracy_by_cell`'s docstring). Reported as a first-class result in
    ANALYSIS_PLAN.md's "PRE-REGISTERED SECONDARY" section, not a footnote -
    `truncation_rate = None` (not 0.0) when a condition has zero completed runs, so a
    missing rate is never silently plotted/printed as "no truncation"."""
    out: dict[str, dict] = {c: {"n_completed": 0, "n_truncated": 0} for c in CONDITIONS}
    for r in records:
        if r.status != "completed":
            continue
        out[r.condition]["n_completed"] += 1
        if r.termination_reason == "max_turns":
            out[r.condition]["n_truncated"] += 1
    for c in out:
        n = out[c]["n_completed"]
        out[c]["truncation_rate"] = out[c]["n_truncated"] / n if n > 0 else None
    return out


TRUNCATION_CONFOUND_THRESHOLD = 0.10  # absolute points - ANALYSIS_PLAN.md, pre-registered blind


def check_truncation_confound(rates: dict[str, dict]) -> list[str]:
    """Pre-registered interpretation rule (ANALYSIS_PLAN.md's "PRE-REGISTERED
    SECONDARY" section, written blind before any pilot output existed): if any two
    conditions' truncation rates differ by more than 10 absolute points, the primary
    accuracy effect is confounded with step exhaustion and must be reported as such,
    not as a pure grounding/context effect. Returns one warning string per offending
    pair (empty list = no confound detected)."""
    warnings = []
    conds = [c for c in rates if rates[c]["truncation_rate"] is not None]
    for i, c1 in enumerate(conds):
        for c2 in conds[i + 1:]:
            diff = abs(rates[c1]["truncation_rate"] - rates[c2]["truncation_rate"])
            if diff > TRUNCATION_CONFOUND_THRESHOLD:
                warnings.append(
                    f"CONFOUND: {c1} truncation_rate={rates[c1]['truncation_rate']:.1%} vs "
                    f"{c2} truncation_rate={rates[c2]['truncation_rate']:.1%} "
                    f"(diff={diff:.1%} > {TRUNCATION_CONFOUND_THRESHOLD:.0%} threshold) - "
                    f"the accuracy contrast between these conditions may be confounded "
                    f"with step exhaustion, not a pure context effect"
                )
    return warnings


def agent_error_rate_by_condition(records: list[RunRecord]) -> dict[str, dict]:
    """condition -> {n_completed, n_agent_error, agent_error_rate}. Same shape and
    same rules as `truncation_rate_by_condition` (see its docstring for why
    `agent_error_rate = None`, not 0.0, when a condition has zero completed runs),
    for `termination_reason == "agent_error"` instead of `"max_turns"`. Added
    2026-08-17 per ANALYSIS_PLAN.md's "PRE-REGISTERED: agent_error handling" -
    the pilot showed `agent_error` (tau2's `too_many_errors`, `orchestrator.py:747-
    749`, threshold `max_errors=10`, `orchestrator.py:326-327` counts each executed
    tool call that returns `ToolMessage(error=True)`) is not rare, so it gets the
    same first-class treatment `max_turns` already has, not folded into "normal"."""
    out: dict[str, dict] = {c: {"n_completed": 0, "n_agent_error": 0} for c in CONDITIONS}
    for r in records:
        if r.status != "completed":
            continue
        out[r.condition]["n_completed"] += 1
        if r.termination_reason == "agent_error":
            out[r.condition]["n_agent_error"] += 1
    for c in out:
        n = out[c]["n_completed"]
        out[c]["agent_error_rate"] = out[c]["n_agent_error"] / n if n > 0 else None
    return out


AGENT_ERROR_CONFOUND_THRESHOLD = 0.10  # absolute points - same threshold as max_turns, symmetric rule


def check_agent_error_confound(rates: dict[str, dict]) -> list[str]:
    """Pre-registered interpretation rule (ANALYSIS_PLAN.md's "PRE-REGISTERED:
    agent_error handling" section, written 2026-08-17 by direct symmetry with
    `check_truncation_confound` - not tuned to any specific accuracy number): if any
    two conditions' agent_error rates differ by more than 10 absolute points, the
    primary accuracy effect between those conditions is confounded with tool-call
    error propensity and must be reported as such, not as a pure context effect."""
    warnings = []
    conds = [c for c in rates if rates[c]["agent_error_rate"] is not None]
    for i, c1 in enumerate(conds):
        for c2 in conds[i + 1:]:
            diff = abs(rates[c1]["agent_error_rate"] - rates[c2]["agent_error_rate"])
            if diff > AGENT_ERROR_CONFOUND_THRESHOLD:
                warnings.append(
                    f"CONFOUND: {c1} agent_error_rate={rates[c1]['agent_error_rate']:.1%} vs "
                    f"{c2} agent_error_rate={rates[c2]['agent_error_rate']:.1%} "
                    f"(diff={diff:.1%} > {AGENT_ERROR_CONFOUND_THRESHOLD:.0%} threshold) - "
                    f"the accuracy contrast between these conditions may be confounded "
                    f"with tool-call error propensity, not a pure context effect"
                )
    return warnings


def holm_correct(p_values: list[float]) -> list[float]:
    """Standard Holm step-down: sort ascending, p_holm_i = max(p_holm_{i-1}, (n-i)*p_i), clipped to 1."""
    n = len(p_values)
    order = sorted(range(n), key=lambda i: p_values[i])
    adjusted = [0.0] * n
    running_max = 0.0
    for rank, idx in enumerate(order):
        val = (n - rank) * p_values[idx]
        running_max = max(running_max, val)
        adjusted[idx] = min(1.0, running_max)
    return adjusted


def paired_permutation_test(diffs: np.ndarray, n_perm: int, rng: np.random.Generator) -> float:
    """Two-sided Monte Carlo sign-flip test. p = (b+1)/(n_perm+1), b = permutations
    with |stat| >= |observed|. Matches the exact-enumeration convention used in
    simswap's permutation_test.py, generalized to Monte Carlo once n_pairs is too
    large to enumerate 2**n_pairs sign assignments exactly."""
    observed = float(np.mean(diffs))
    abs_diffs = np.abs(diffs)
    signs = rng.choice([1.0, -1.0], size=(n_perm, len(diffs)))
    perm_stats = (signs * abs_diffs).mean(axis=1)
    b = int(np.sum(np.abs(perm_stats) >= abs(observed) - 1e-12))
    return (b + 1) / (n_perm + 1)


def contrast(
    by_cell: dict[str, dict[tuple[str, int], float]],
    n_perm: int, alpha: float, rng: np.random.Generator,
) -> list[dict]:
    """Shared contrast pipeline (paired diff, permutation test, Holm correction)
    used for both the primary and the excl-truncated secondary check, so the two
    can never silently diverge in method - only in which records feed `by_cell`."""
    control_cells = by_cell[CONTROL]
    if not control_cells:
        return []
    other_conditions = [c for c in CONDITIONS if c != CONTROL and by_cell[c]]

    rows = []
    raw_p = []
    for condition in other_conditions:
        cond_cells = by_cell[condition]
        shared_keys = sorted(set(control_cells) & set(cond_cells))
        if not shared_keys:
            continue
        diffs = np.array([control_cells[k] - cond_cells[k] for k in shared_keys])
        p_raw = paired_permutation_test(diffs, n_perm, rng)
        rows.append({
            "condition": condition,
            "n_pairs": len(shared_keys),
            "mean_accuracy_control": float(np.mean([control_cells[k] for k in shared_keys])),
            "mean_accuracy_condition": float(np.mean([cond_cells[k] for k in shared_keys])),
            "mean_diff_control_minus_condition": float(np.mean(diffs)),
            "p_raw": p_raw,
        })
        raw_p.append(p_raw)

    if not rows:
        return []

    p_holm = holm_correct(raw_p)
    for row, ph in zip(rows, p_holm):
        row["p_holm"] = ph
        row["holm_significant"] = ph < alpha
    return rows


def print_contrast(rows: list[dict], label: str, n_perm: int, alpha: float) -> None:
    print(f"\n[{label}] {len(rows)} condition(s) contrasted against {CONTROL}, "
          f"n_perm={n_perm}, alpha={alpha}")
    for row in sorted(rows, key=lambda r: r["p_holm"]):
        flag = "SIGNIFICANT" if row["holm_significant"] else "not significant"
        print(f"  {row['condition']:20s} n={row['n_pairs']:3d}  "
              f"control={row['mean_accuracy_control']:.3f}  condition={row['mean_accuracy_condition']:.3f}  "
              f"diff={row['mean_diff_control_minus_condition']:+.3f}  "
              f"p_raw={row['p_raw']:.4f}  p_holm={row['p_holm']:.4f}  {flag}")


def write_csv(rows: list[dict], path: Path) -> None:
    import csv
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path}")


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.rng_seed)
    args.tables_dir.mkdir(parents=True, exist_ok=True)

    records = load_runs(args.runs_dir)
    if not records:
        print(f"no run records found in {args.runs_dir}")
        return

    # --- truncation rate: first-class result, computed and reported before any
    # accuracy contrast (ANALYSIS_PLAN.md's "PRE-REGISTERED SECONDARY" section).
    rates = truncation_rate_by_condition(records)
    print("truncation rate by condition (status=='completed', termination_reason=='max_turns'):")
    for c in CONDITIONS:
        r = rates[c]
        rate_str = f"{r['truncation_rate']:.1%}" if r["truncation_rate"] is not None else "n/a (0 completed)"
        print(f"  {c:20s} n_completed={r['n_completed']:3d}  n_truncated={r['n_truncated']:3d}  rate={rate_str}")
    import csv
    trunc_csv = args.tables_dir / "truncation_rate_by_condition.csv"
    with open(trunc_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["condition", "n_completed", "n_truncated", "truncation_rate"])
        w.writeheader()
        for c in CONDITIONS:
            w.writerow({"condition": c, **rates[c]})
    print(f"wrote {trunc_csv}")

    confounds = check_truncation_confound(rates)
    if confounds:
        print("\n*** TRUNCATION CONFOUND DETECTED (>10 absolute points between conditions) ***")
        for w in confounds:
            print(f"  {w}")
    else:
        print("\nno truncation-rate confound detected (all pairwise diffs <= "
              f"{TRUNCATION_CONFOUND_THRESHOLD:.0%})")

    # --- agent_error rate: same first-class treatment as truncation rate, added
    # 2026-08-17 (ANALYSIS_PLAN.md's "PRE-REGISTERED: agent_error handling").
    ae_rates = agent_error_rate_by_condition(records)
    print("\nagent_error rate by condition (status=='completed', termination_reason=='agent_error'):")
    for c in CONDITIONS:
        r = ae_rates[c]
        rate_str = f"{r['agent_error_rate']:.1%}" if r["agent_error_rate"] is not None else "n/a (0 completed)"
        print(f"  {c:20s} n_completed={r['n_completed']:3d}  n_agent_error={r['n_agent_error']:3d}  rate={rate_str}")
    ae_csv = args.tables_dir / "agent_error_rate_by_condition.csv"
    with open(ae_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["condition", "n_completed", "n_agent_error", "agent_error_rate"])
        w.writeheader()
        for c in CONDITIONS:
            w.writerow({"condition": c, **ae_rates[c]})
    print(f"wrote {ae_csv}")

    ae_confounds = check_agent_error_confound(ae_rates)
    if ae_confounds:
        print("\n*** AGENT_ERROR CONFOUND DETECTED (>10 absolute points between conditions) ***")
        for w in ae_confounds:
            print(f"  {w}")
    else:
        print("\nno agent_error-rate confound detected (all pairwise diffs <= "
              f"{AGENT_ERROR_CONFOUND_THRESHOLD:.0%})")

    # --- PRIMARY (frozen): all completed episodes, truncated ones included and
    # scored as-is. Do not change this call's exclude_truncated default.
    primary_rows = contrast(accuracy_by_cell(records), args.n_perm, args.alpha, rng)
    if not primary_rows:
        print(f"\nno completed {CONTROL} runs (or no shared cells) - cannot test the primary metric")
    else:
        print_contrast(primary_rows, "PRIMARY (all completed episodes)", args.n_perm, args.alpha)
        write_csv(primary_rows, args.tables_dir / "condition_contrasts.csv")

    # --- SECONDARY robustness check (pre-registered, ANALYSIS_PLAN.md): same
    # pipeline, truncated episodes excluded. Reported alongside the primary
    # regardless of whether it agrees - never substituted for the primary.
    secondary_rows = contrast(accuracy_by_cell(records, exclude_truncated=True), args.n_perm, args.alpha, rng)
    if not secondary_rows:
        print(f"\n[SECONDARY excl-truncated] no contrastable data "
              "(e.g. every completed run in some condition hit max_turns)")
    else:
        print_contrast(secondary_rows, "SECONDARY (excl. truncated episodes)", args.n_perm, args.alpha)
        write_csv(secondary_rows, args.tables_dir / "condition_contrasts_excl_truncated.csv")

    # --- THIRD view (pre-registered 2026-08-17, ANALYSIS_PLAN.md's "PRE-REGISTERED:
    # agent_error handling"): normal-terminations-only - excludes BOTH max_turns AND
    # agent_error. Reported alongside the other two, never substituted for either.
    normal_only_rows = contrast(
        accuracy_by_cell(records, exclude_truncated=True, exclude_agent_error=True),
        args.n_perm, args.alpha, rng,
    )
    if not normal_only_rows:
        print(f"\n[THIRD normal-terminations-only] no contrastable data "
              "(e.g. every completed run in some condition ended max_turns or agent_error)")
    else:
        print_contrast(normal_only_rows, "THIRD (normal terminations only)", args.n_perm, args.alpha)
        write_csv(normal_only_rows, args.tables_dir / "condition_contrasts_normal_only.csv")

    if confounds:
        print("\nREMINDER: truncation-rate confound flagged above - read the primary/secondary "
              "diffs together with that in mind, do not report either as a pure context effect alone.")
    if ae_confounds:
        print("\nREMINDER: agent_error-rate confound flagged above - read the primary/third-view "
              "diffs together with that in mind, do not report either as a pure context effect alone.")


if __name__ == "__main__":
    main()
