# Results for paper - ATTRIB context-ablation study

All numbers below come from files that exist on disk. Source file noted per entry.
One decimal place max. No interpretation - see PROGRESS.md for narrative.

## Design

- Conditions: 6 (full_context, no_policy, policy_truncated, schema_names_only,
  history_truncated, system_minimized)
  [src/schema.py:CONDITIONS]
- Grid: 12 tasks x 2 seeds x 6 conditions = 144 triples
  [PROGRESS.md: "Grid size: REVISED 2026-08-18"]
- Episodes completed: 138 of 144 (6 errors/timeouts)
  [PROGRESS.md: 2026-08-23 attrition table]
- Tasks: 93, 43, 98, 77, 66, 94, 4, 46, 107, 64, 102, 110
  [PROGRESS.md: 2026-08-18, item 1]
- Agent/user model: Qwen2.5-7B-Instruct-AWQ (self-play, vLLM 0.27.1)
  [ANALYSIS_PLAN.md: Fixed factors]
- tau2-bench pinned commit: fc0055dc4e0a316c3f83133267fbd6faaa770992
  [ANALYSIS_PLAN.md: Fixed factors]
- MDE at n_tasks=12, n_seeds=2: 20pp at 87% power (0.867)
  [src/tables/detectability_grounding_12t2s.csv, delta=0.2 row]

## Per-condition completion (n=24 each)

| condition | n_completed | n_normal | n_max_turns | n_agent_error | n_cwe | n_timeout | n_error |
|---|---|---|---|---|---|---|---|
| full_context | 23 | 11 | 3 | 9 | 0 | 1 | 0 |
| no_policy | 22 | 11 | 4 | 7 | 0 | 0 | 2 |
| policy_truncated | 23 | 11 | 8 | 4 | 0 | 0 | 1 |
| schema_names_only | 24 | 10 | 3 | 11 | 0 | 0 | 0 |
| history_truncated | 24 | 5 | 2 | 17 | 0 | 0 | 0 |
| system_minimized | 22 | 4 | 6 | 9 | 3 | 2 | 0 |

Source: [PROGRESS.md, 2026-08-23 item 3 attrition table]. Values verified against
all 144 RunRecord files in data/grid_session1_20260817/runs/ and
data/grid_session2_20260818/runs/.

## Truncation (max_turns) rate

| condition | n_completed | n_truncated | truncation_rate |
|---|---|---|---|
| full_context | 23 | 3 | 13.0% |
| no_policy | 22 | 4 | 18.2% |
| policy_truncated | 23 | 8 | 34.8% |
| schema_names_only | 24 | 3 | 12.5% |
| history_truncated | 24 | 2 | 8.3% |
| system_minimized | 22 | 6 | 27.3% |

Source: [tables/truncation_rate_by_condition.csv]

## Agent error rate

| condition | n_completed | n_agent_error | agent_error_rate |
|---|---|---|---|
| full_context | 23 | 9 | 39.1% |
| no_policy | 22 | 7 | 31.8% |
| policy_truncated | 23 | 4 | 17.4% |
| schema_names_only | 24 | 11 | 45.8% |
| history_truncated | 24 | 17 | 70.8% |
| system_minimized | 22 | 9 | 40.9% |

Source: [tables/agent_error_rate_by_condition.csv]

## Primary metric: grounding rate (per-episode-averaged, CONDITIONAL)

Ungrounded fraction by argument (conditional), macro-averaged over episodes with
a defined rate. Positive "increase" = worse grounding than full_context.

| condition | ungrounded_rate | n_in_avg (of 24) | increase vs full_context |
|---|---|---|---|
| full_context | 71.1% | 22 | - (control) |
| no_policy | 71.7% | 12 | +0.6pp |
| policy_truncated | 72.6% | 22 | +1.5pp |
| schema_names_only | 73.2% | 20 | +2.0pp |
| history_truncated | 80.2% | 21 | +9.1pp |
| system_minimized | 76.6% | 13 | +5.5pp |

Source: computed by src/grounding.py:average_by_condition() over all 144 RunRecord
files in data/grid_session1_20260817/runs/ and data/grid_session2_20260818/runs/.
Not stored in a table CSV - numbers are from PROGRESS.md 2026-08-23 item 4a.

## Grounded-rate contrast vs full_context (paired permutation test, Holm-corrected)

Cells with a defined grounded rate in BOTH the condition and full_context are paired
by (task_id, seed). Grounded rate = 1 - ungrounded_fraction_by_argument_conditional.

| condition | n_pairs | control (grounded) | condition (grounded) | diff (control-cond) | p_raw | p_holm | significant |
|---|---|---|---|---|---|---|---|
| no_policy | 11 | 0.096 | 0.309 | -0.212 | 0.047 | 0.235 | No |
| policy_truncated | 20 | 0.265 | 0.240 | +0.025 | 0.816 | 1.000 | No |
| schema_names_only | 18 | 0.198 | 0.261 | -0.063 | 0.485 | 1.000 | No |
| history_truncated | 19 | 0.304 | 0.180 | +0.123 | 0.135 | 0.542 | No |
| system_minimized | 11 | 0.299 | 0.208 | +0.090 | 0.517 | 1.000 | No |

Source: [tables/grounding_condition_contrasts.csv]
Note: control values are grounded rates on the PAIRED subset only, not the full
marginal average for full_context.

## Crash-exclusion bound (increase vs full_context under worst/best imputation)

| condition | (a) excluded | (b) worst | (c) best | straddles 20pp? |
|---|---|---|---|---|
| no_policy | +0.6pp | +3.4pp | -6.6pp | No |
| policy_truncated | +1.5pp | +1.4pp | +1.4pp | No |
| schema_names_only | +2.0pp | +0.8pp | +5.1pp | No |
| history_truncated | +9.1pp | +7.8pp | +12.2pp | No |
| system_minimized | +5.5pp | +10.7pp | -12.7pp | No |

Source: PROGRESS.md 2026-08-23 item 4b. Crashed episodes: full_context=1,
no_policy=2, policy_truncated=1, schema_names_only=0, history_truncated=0,
system_minimized=5. (a) excludes crashed; (b) imputes crashed as 100% ungrounded;
(c) imputes crashed as 0% ungrounded.

## action_accuracy: defined-cell counts

| condition | scored | not reached | evaluation failed |
|---|---|---|---|
| no_policy | 8 | 13 | 3 |
| policy_truncated | 8 | 13 | 3 |
| full_context | 7 | 13 | 4 |
| schema_names_only | 4 | 14 | 6 |
| system_minimized | 3 | 20 | 1 |
| history_truncated | 1 | 19 | 4 |

Source: Recomputed from `reward_info_note` across all 144 RunRecord files in
data/grid_session1_20260817/runs/ and data/grid_session2_20260818/runs/.
Categories: scored = note contains "No nl_assertions to evaluate"; evaluation
failed = note contains "AuthenticationError"; not reached = everything else
(e.g. too_many_errors, max_steps, or empty/missing note).

Regeneration snippet:
```python
import json, os
from collections import defaultdict
counts = defaultdict(lambda: {"scored": 0, "auth_error": 0, "other": 0})
for d in ["data/grid_session1_20260817/runs", "data/grid_session2_20260818/runs"]:
    for f in sorted(os.listdir(d)):
        if not f.endswith(".json"): continue
        rec = json.load(open(os.path.join(d, f)))
        cond = rec.get("condition", "unknown")
        note = rec.get("reward_info_note", "") or ""
        if not isinstance(note, str): note = str(note)
        if "No nl_assertions to evaluate" in note: counts[cond]["scored"] += 1
        elif "AuthenticationError" in note: counts[cond]["auth_error"] += 1
        else: counts[cond]["other"] += 1
for c in ["full_context","no_policy","policy_truncated","schema_names_only",
           "history_truncated","system_minimized"]:
    d = counts[c]
    print("{}: scored={} not_reached={} eval_failed={}".format(
        c, d["scored"], d["other"], d["auth_error"]))
```

## action_accuracy contrast (paired permutation test)

| condition | n_pairs | control | condition | diff | p_raw | p_holm | significant |
|---|---|---|---|---|---|---|---|
| no_policy | 3 | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 | No |
| policy_truncated | 3 | 0.000 | 0.333 | -0.333 | 1.000 | 1.000 | No |
| schema_names_only | 4 | 0.031 | 0.031 | 0.000 | 1.000 | 1.000 | No |
| system_minimized | 3 | 0.067 | 0.067 | 0.000 | 1.000 | 1.000 | No |

Source: [tables/condition_contrasts.csv]
Note: history_truncated absent - only 1 defined cell, no shared (task_id, seed) key
with full_context survives pairing.

## action_accuracy: normal-terminations-only contrast

| condition | n_pairs | control | condition | diff | p_raw | p_holm | significant |
|---|---|---|---|---|---|---|---|
| no_policy | 3 | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 | No |
| policy_truncated | 3 | 0.000 | 0.333 | -0.333 | 1.000 | 1.000 | No |
| schema_names_only | 4 | 0.031 | 0.031 | 0.000 | 1.000 | 1.000 | No |
| system_minimized | 3 | 0.067 | 0.067 | 0.000 | 1.000 | 1.000 | No |

Source: [tables/condition_contrasts_normal_only.csv]

## action_accuracy: excl-truncated contrast

| condition | n_pairs | control | condition | diff | p_raw | p_holm | significant |
|---|---|---|---|---|---|---|---|
| no_policy | 3 | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 | No |
| policy_truncated | 3 | 0.000 | 0.333 | -0.333 | 1.000 | 1.000 | No |
| schema_names_only | 4 | 0.031 | 0.031 | 0.000 | 1.000 | 1.000 | No |
| system_minimized | 3 | 0.067 | 0.067 | 0.000 | 1.000 | 1.000 | No |

Source: [tables/condition_contrasts_excl_truncated.csv]

## Selection-artifact check: correlations

| view | vs n_turns | vs total_tool_calls |
|---|---|---|
| FLAT (n=135) | r=+0.034 | r=+0.363 |
| CONDITIONAL (n=110) | r=+0.021 | r=+0.153 |

Source: PROGRESS.md 2026-08-23 item 1a, computed from all 144 RunRecords.
Positive = episodes making more tool calls trend toward better grounding.

## Normal-only grounding rates

| condition | n_normal | mean_n_turns | grounded_rate | n_in_avg |
|---|---|---|---|---|
| full_context | 11 | 41.9 | 0.330 | 11 |
| no_policy | 11 | 26.3 | 0.290 | 5 |
| policy_truncated | 11 | 42.6 | 0.264 | 10 |
| schema_names_only | 10 | 34.6 | 0.380 | 8 |
| history_truncated | 5 | 33.6 | 0.500 | 2 |
| system_minimized | 4 | 30.5 | 0.376 | 3 |

Source: PROGRESS.md 2026-08-23 item 1c. Restricts to
termination_reason="normal" only. Note: history_truncated (n=2) and
system_minimized (n=3) averages are not reliable at this sample size.

## Payload verification (CPU-verified, all 20 tasks x 6 conditions)

| condition | system_prompt_chars | n_tools | n_tools_with_descriptions | n_history_msgs_sent |
|---|---|---|---|---|
| full_context | 7056 | 16 | 16 | 11 |
| no_policy | 357 | 16 | 16 | 11 |
| policy_truncated | 2311-4621 (task-varying) | 16 | 16 | 11 |
| schema_names_only | 7056 | 16 | 0 | 11 |
| history_truncated | 7056 | 16 | 16 | 3 |
| system_minimized | 6913 | 16 | 16 | 11 |

Source: [tables/condition_payload_dump_20tasks.csv]. policy_truncated range is
2311-4621 chars across 20 tasks (task-sensitive mapping via ACTION_TO_SECTION dict
in src/conditions.py). history_truncated sent only 3 non-system messages (k=3)
on the first turn (nothing to truncate yet).

## evaluator.py outcome-selection finding

tau2/evaluator/evaluator.py:119-129: if termination_reason not in
{AGENT_STOP, USER_STOP}, returns RewardInfo(reward=0.0, reward_basis=None)
immediately - ActionEvaluator.calculate_reward() is never reached. This causes
action_accuracy to be undefined (None) for any episode ending via agent_error,
max_turns, or context_window_exceeded. Defined-cell counts above confirm
differential attrition: history_truncated has 1/24 defined, full_context has 7/24.

Source: [ANALYSIS_PLAN.md, "METRIC CHANGE, 2026-08-17" section; confirmed by reading
tau2/evaluator/evaluator.py lines 119-129 in the pinned clone]

## Detectability: grounding-rate power curve (n_tasks=20, n_seeds=2)

| delta (pp) | power (60 replicates) |
|---|---|
| 5 | 0% |
| 10 | 30% |
| 15 | 92% |
| 20 | 100% |
| 30 | 100% |

Source: [tables/detectability_grounding.csv]

## Detectability: grounding-rate power at this grid (n_tasks=12, n_seeds=2)

| delta (pp) | power (60 replicates) |
|---|---|
| 5 | 0% |
| 10 | 8% |
| 15 | 43% |
| 20 | 87% |
| 30 | 100% |

Source: [src/tables/detectability_grounding_12t2s.csv]

## Detectability: accuracy-based power curve (n_tasks=20, n_seeds=2)

| delta (pp) | power (60 replicates) |
|---|---|
| 5 | 2% |
| 10 | 10% |
| 15 | 70% |
| 20 | 97% |

Source: [tables/detectability.csv, n_tasks=20 rows]

## Turn-count statistics

| condition | mean_n_turns (all 24) |
|---|---|
| full_context | 66.8 |
| no_policy | 57.6 |
| policy_truncated | 94.8 |
| schema_names_only | 60.5 |
| history_truncated | 72.5 |
| system_minimized | 73.9 |

Source: PROGRESS.md 2026-08-23 item 4c, derived from all 144 RunRecords.
