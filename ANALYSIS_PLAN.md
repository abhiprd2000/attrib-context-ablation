# Analysis plan - ATTRIB workshop, context-ablation study

Pre-registered 2026-08-17, before any real tau2-bench data exists (before any GPU time
at all - see PROGRESS.md). Not edited after real data lands; deviations get a new dated
section, not a silent edit to the rules here. Reuses simswap's harness (`src/schema.py`,
`src/sweep.py`, forked 2026-08-17) and its grounding-rate algorithm (`src/grounding.py`,
ported); does not touch the simswap repository.

**PRIMARY METRIC CHANGED 2026-08-17 (still before any 240-episode grid data - the
6-episode pilot only, run for wall-clock/plumbing, is what triggered this).**
Grounding rate replaced `action_accuracy` as the sole primary metric after a real
tau2-source finding (`tau2/evaluator/evaluator.py:119-129`) showed `action_accuracy`
is undefined for any episode that doesn't terminate via `agent_stop`/`user_stop`,
regardless of the task - differential attrition, not a floor. See "METRIC CHANGE,
2026-08-17" below for the finding and "ARCHIVED, 2026-08-17" at the end of this file
for the complete original accuracy-based plan, kept verbatim, not deleted.

## Question

Which part of the agent's context causes a tool-call failure? Isolate the contribution
of the policy document, tool schemas, conversation history, and system instructions by
removing/degrading each in turn and measuring the resulting change in behavior.

## Hypothesis under test (REWRITTEN 2026-08-17 in grounding-rate vocabulary - see
## "Metric change" section below for why; the original accuracy-vocabulary version
## is preserved verbatim in the ARCHIVED block at the end of this file)

Most of the policy document is inert: agents fail on tasks whose governing policy
section was in context the whole time. If true, that reframes policy-adherence failure
as retrieval/attention rather than reasoning - the model had the right text available
and still didn't act on it, similar in spirit to simswap's finding that Qwen mostly
fails by trusting an unverified user claim rather than fabricating one (a "didn't check
what was available" failure, not a "had nothing to check" one). Concretely, in this
study's terms: the agent has a real order/item/user ID available somewhere in context
(the policy, the conversation, a prior successful tool result) and still calls a tool
with a fabricated one instead - a grounding failure, not a missing-information one.

Falsifiable prediction: if the hypothesis holds, `policy_truncated` (task-relevant
policy sections only) should show a MUCH smaller INCREASE in ungrounded-ID rate than
`no_policy` (whole document removed) - most of the document was never load-bearing for
the tasks it covers, so trimming it shouldn't make the agent fabricate more IDs than it
already does. If the hypothesis is false, `policy_truncated`'s ungrounded-rate increase
is nearly as large as `no_policy`'s - the model was actually using material outside the
task-relevant section (e.g. cross-referencing policy for edge cases), and truncation
destroys that, driving up fabrication the same way removing the whole document does.

## Fixed factors (not swept)

- Agent: Qwen2.5-7B-Instruct-AWQ (`Qwen/Qwen2.5-7B-Instruct-AWQ`, vLLM `hermes` parser).
- User simulator: the SAME model, self-play - matches simswap's matched-user-simulator
  design (see that project's `docs/FAILURE_ANALYSIS.md` "CONFOUND CHECK" section for why
  a mismatched user simulator is a real confound, not a labeling detail).
- Domain: tau2-bench v1.0.1, pinned commit `fc0055dc4e0a316c3f83133267fbd6faaa770992`.
- Seed count, temperature (0.0), max_steps (200): inherited from simswap's convention.

**Efficiency note vs. simswap's original design**: because the agent and user model
never change across conditions, vLLM servers start ONCE per session, not once per
(agent, user) pair - no restart-overhead multiplier across the grid, unlike simswap's
16-pair sweep.

## The 6 conditions

| condition | intervention | `condition_param` |
|---|---|---|
| `full_context` | none - control | - |
| `no_policy` | policy document removed entirely | - |
| `policy_truncated` | policy reduced to task-relevant section(s) only | - |
| `schema_names_only` | tool schemas stripped of descriptions, names/signatures only | - |
| `history_truncated` | conversation history truncated to the last k turns | k (string) |
| `system_minimized` | system instructions reduced to a bare minimum - **functions as a negative control, not a content ablation; see below** | - |

`src/schema.py`'s `RunRecord.condition`/`condition_param` fields and `CONDITIONS` tuple
enforce this set; `src/sweep.py`'s CLI validates `--condition`/`--condition-param`
against it.

### `system_minimized`: labeled a negative control, 2026-08-17

`AGENT_INSTRUCTION` (`tau2/agent/llm_agent.py:24-32` in the pinned tau2 source) is
already the ENTIRE non-policy instructions block in the system prompt - there is
nothing else labeled "system instructions" left to strip once the policy document is
excluded (that's `no_policy`'s job) and the tool-call/JSON-format mechanics are kept
intact (kept deliberately - stripping those would confound this condition with
`schema_names_only`-style output-format breakage, a different failure mode). The
actual diff (`conditions.py:52-58` vs. the real `AGENT_INSTRUCTION` constant, dumped
verbatim by `verify_all_tasks.py`, see `PROGRESS.md`): 306 -> 163 chars, 143 chars
removed (2.0% of the full 7056-char `full_context` system prompt), exactly two
sentences - "You are a customer service agent that helps the user according to the
<policy> provided below." and "Try to be helpful and always follow the policy." -
with the tool-call-format instructions and "Always make sure you generate valid JSON
only." kept verbatim.

At 2% of the full prompt, with the entire policy document still present and
unchanged, this condition is not expected to move behavior much on its own merits -
it isolates one narrow question ("does the model need to be told, in so many words,
to follow the policy, given the policy is fully in context regardless?") that is
orthogonal to this paper's actual hypothesis (which is about the POLICY document's
content, not this framing sentence). **Labeled here as a negative control**: a large
effect from `system_minimized` would be surprising and worth its own investigation: a
near-zero effect is the expected result and validates that the harness isn't
mistaking normal LLM sampling noise for a real ablation effect. It is not read as
evidence about the "policy is inert" hypothesis either way - only `no_policy` and
`policy_truncated` bear on that (see Detectability and the frozen final-analysis
section below).

## Condition -> intervention mapping: IMPLEMENTED and payload-verified, 2026-08-17

`src/conditions.py::apply_condition()` implements all 6 conditions. Verified against
a real clone of tau2-bench at the pinned commit (not guessed): `TextRunConfig` turned
out to have no field for policy/tools/system-instructions/history at all (an earlier
version of this section assumed it did - see `SOURCE_NOTES.md`'s "Correction to
ANALYSIS_PLAN.md's premise"). The real interception is monkeypatching
`tau2.runner.build.build_agent`, `tau2.agent.llm_agent.AGENT_INSTRUCTION`, and
`tau2.agent.llm_agent.LLMAgent._generate_next_message`, installed for the duration of
one `run_single_task()` call and reverted in `finally`. Every condition's exact
object/field, file:line, and the `policy_truncated` task->section mapping's
provenance are in `SOURCE_NOTES.md`.

**Payload-verified CPU-only, no GPU, against all 20 real tasks in the actual sweep
grid** (`sweep.select_tasks("retail", 20)`, seed 20260814) x all 6 conditions - see
`PROGRESS.md`'s 2026-08-17 entries and `tables/condition_payload_dump_20tasks.csv`
for the full dump. No task's `policy_truncated` landed near-full (>=6500 chars, i.e.
nothing dropped) or near-empty (<=800 chars, i.e. everything dropped) - real range
across the 20 tasks was 2311-4621 chars, strictly between `no_policy`'s 357 and
`full_context`'s 7056 on every one.

**`policy_truncated`'s task->section mapping provenance**: a hand-built, static
Python dict (`conditions.py:31-38`, `ACTION_TO_SECTION`), written after manually
reading `data/tau2/domains/retail/policy.md`'s 6 `##` section headers and
cross-referencing every distinct `action.name` that actually appears in
`data/tau2/domains/retail/tasks.json` (grepped in full, 250+ matches, all 6 mutating
action names covered, none unmapped). At runtime, `truncated_policy_for_task()`
(`conditions.py:82-96`) does one dict lookup per action name off the task's own
ground-truth `evaluation_criteria.actions` - `ACTION_TO_SECTION.get(name)`
(`conditions.py:92`). This is neither model-derived (no LLM call decides section
relevance) nor keyword-matched against free text at runtime (no fuzzy/substring
search) - it is a fixed lookup table built once by hand from source facts, applied
mechanically. Not circular.

**`policy_truncated` is an ORACLE retrieval condition, not a retrieval simulation.**
The kept section(s) are chosen from the task's own ground-truth `evaluation_criteria
.actions` (`conditions.py:87-89` reads `task.evaluation_criteria.actions[*].name`
directly) - information the agent under test never sees and no real retriever would
have available either. A real retrieval-based context-trimming system would have to
infer which policy sections are relevant from the conversation so far, and could get
that wrong (miss a needed section, include an irrelevant one, or need multiple
retrieval rounds as the conversation develops). This condition instead always keeps
exactly the right section(s), every time, by construction. It therefore measures
"if policy retrieval were perfect, how much would trimming to only the relevant
section change behavior" - an upper bound on what a real retriever could achieve, not
a prediction of what one would achieve. A small `policy_truncated` drop supports
"most of the document is inert for the tasks it covers," but does NOT by itself imply
that trimming the policy via a real (imperfect) retriever would be safe in practice -
that is a separate, unanswered question this condition does not address.

## METRIC CHANGE, 2026-08-17: why action_accuracy was demoted - the evaluator finding

**This is the headline reason for the primary-metric change below, recorded here
before the new primary/secondary sections so the "why" is not buried after the
"what."**

`tau2/evaluator/evaluator.py:119-129` (function `evaluate_simulation`, verified
against the real pinned-commit source):

```python
if simulation.termination_reason not in {
    TerminationReason.AGENT_STOP,
    TerminationReason.USER_STOP,
}:
    return RewardInfo(
        reward=0.0,
        reward_basis=None,
        info={"note": f"Simulation terminated prematurely. Termination reason: {simulation.termination_reason.value}"},
    )
```

This is a hard gate at the top of tau2's OWN evaluator, unconditional on the task.
**Any simulation that does not end via `agent_stop`/`user_stop` never reaches
`ActionEvaluator.calculate_reward()`** (`evaluator_action.py:68-109`, which is the
only code path that ever populates `RewardInfo.action_checks` from a task's golden
actions) - `action_checks` stays unset (`None`), so `RunRecord.read`/`write` are
`ActionTally(0,0)` and `action_accuracy` is `None`, **regardless of the task's
golden-action set and regardless of what the agent actually did.**

Verified this is NOT a task-selection artifact: read all 20 tasks in this study's
actual sweep grid (`sweep.select_tasks("retail", 20)`) - **20/20 have a non-empty
`evaluation_criteria.actions` set** (range 1-13 golden actions per task, e.g. task 93
= 1 action, task 4 and task 55 = 13 each; see PROGRESS.md for the full per-task
breakdown). Task selection is not the bug.

**This is differential attrition, not a floor.** `termination_reason` correlates with
`condition` (a condition that degrades the agent's context can plausibly make
`agent_error`/`max_turns` more likely - exactly the mechanism the agent_error
confound rule below already exists to catch), and whether `action_accuracy` is even
DEFINED for an episode depends entirely on `termination_reason` via the evaluator gate
above - not on task difficulty, not on how well the agent actually performed. A metric
that is selectively undefined based on an outcome correlated with the treatment is
attrition bias, the same failure mode a clinical trial guards against when dropout
rate differs by arm. This is why `action_accuracy` is demoted below, not merely
"underpowered."

**Cross-tab, this pilot's 6 episodes (n=1 per condition, not a finding - the
mechanism, not the numbers, is what generalizes):**

| termination_reason | action_accuracy defined | action_accuracy undefined (None) |
|---|---|---|
| `normal` (`no_policy`, `policy_truncated`) | 2 | 0 |
| `agent_error` (`full_context`, `history_truncated`, `system_minimized`) | 0 | 3 |
| `max_turns` (`schema_names_only`) | 0 | 1 |

A perfect correlation in this pilot: defined iff `normal`. Not claimed to be exactly
100% at n=240 - claimed to be structurally guaranteed to correlate, by the evaluator
gate above, which is unconditional.

## Primary metric: per-call grounding rate (CHANGED 2026-08-17, see above)

**`src/grounding.py`, SOLE primary as of 2026-08-17** (was secondary; the original
action_accuracy-based primary is ARCHIVED verbatim at the end of this file, not
deleted). Ported from simswap's `failure_analysis.py` (algorithm unchanged, already
validated there - 11 regression tests inherited + this project's own 6). An ID-shaped
tool-call argument is grounded if its exact value appears as a substring of an earlier
successful tool result in the same conversation; CONDITIONAL rate (restricted to calls
made after the grounding pool already held >=1 success) is the headline sub-metric,
FLAT (all ID-shaped arguments) reported alongside it - see simswap's
`docs/FAILURE_ANALYSIS.md` for why conditional is the honest headline.

**DECLARED 2026-08-17, before any grid data exists: per-episode-averaged
(macro-average) is THE primary rate. Pooled (micro-average) is a reported
secondary, never used to override the primary.** Reason: equal weight per episode
means one long episode cannot dominate a condition's denominator - pooling lets
exactly that happen (see the real, verified 14.0pp conditional-rate disagreement on
this pilot's data below, driven by `full_context` being both the best-grounded and
the longest episode). Both are still computed and reported every run - this
declaration closes the degree of freedom of "which one is THE number," it does not
stop reporting the other:
- **PRIMARY - per-episode-averaged (macro-average)**: compute each episode's own
  rate, then average those rates, each episode weighted equally regardless of how
  many tool calls it made. `src/grounding.py::average_by_condition()`.
- **SECONDARY - pooled (micro-average)**: sum raw ungrounded/total counts across all
  of a condition's episodes, then divide. `src/grounding.py::pool_by_condition()`.

**Flagged gap, not yet built**: `pool_by_condition()`/`average_by_condition()` both
produce ONE aggregate rate per condition (for descriptive reporting), not a
`{condition: {(task_id, seed): rate}}` per-cell dict shaped like `analyze.py::
accuracy_by_cell()`'s output - which is what `analyze.py::contrast()` actually needs
to run the paired significance test the Attribution rule below depends on. This
function does not exist yet for grounding rate. Needed before the real grid's
significance test can run; not built speculatively this session since it wasn't
part of what was asked for.

**These disagree, verified on this pilot's real 6 episodes** (not invented, computed
via the real functions against `data/pilot_run_20260817/transcripts/`): pooled across
all 6 as one group, FLAT = 84.1% ungrounded, CONDITIONAL = 65.6% ungrounded.
Per-episode-averaged: FLAT = 91.7% (mean of 6 episode rates), CONDITIONAL = 79.6%
(mean of the 3 episodes with a defined conditional rate; 3 excluded, pool never
filled, not counted as 0% or 100%). **Disagreement: 7.6pp (flat), 14.0pp
(conditional)** - because `full_context` (this pilot's best-grounded episode, 20 tool
calls, the longest) dominates the pooled denominator and pulls the pooled rate down
relative to its equal 1/6 weight in the average. Reason pooling was flagged in the
first place: pilot episodes ran 14-201 turns, so a long episode's tool calls can
swamp a short episode's in a pooled statistic - confirmed real on this data, not
hypothetical. Both numbers are reported for the real grid; neither is silently
preferred.

## ATTRITION IS A REPORTED OUTCOME, not a silently-dropped denominator (added 2026-08-17)

Grid session 1 (real GPU data, 7 tasks x 2 seeds x 6 conditions = 84 episodes, see
PROGRESS.md) surfaced a THIRD attrition mechanism beyond `max_turns`/`agent_error`:
`litellm.exceptions.ContextWindowExceededError` when a conversation's prompt exceeds
`--max-model-len` (32768 tokens). Verified against the real trace: this is an
UNCAUGHT exception - tau2's own `TerminationReason.CONTEXT_WINDOW_EXCEEDED` enum
value exists but is never assigned anywhere in the half-duplex orchestrator, so the
exception propagated out of `run_single_task()` entirely, before any `SimulationRun`
existed. 3 of 84 session-1 episodes crashed this way (`status="error"`,
`termination_reason="harness_crash"` in the pre-fix schema).

**Per-condition crash rate and termination_reason distribution, first-class, not
folded into a denominator silently** (session 1, real data - n=1 grid so far, not
yet a statistical finding, but the mechanism and its condition-correlation are real):

| condition | n | crash_rate | termination_reason distribution |
|---|---|---|---|
| `full_context` | 14 | 0% | agent_error:6, max_turns:2, normal:6 |
| `no_policy` | 14 | **14.3%** | normal:7, max_turns:4, agent_error:1, harness_crash:2 |
| `policy_truncated` | 14 | **7.1%** | normal:9, max_turns:4, harness_crash:1 |
| `schema_names_only` | 14 | 0% | max_turns:2, agent_error:7, normal:5 |
| `history_truncated` | 14 | 0% | agent_error:8, normal:5, max_turns:1 |
| `system_minimized` | 14 | 0% | agent_error:7, max_turns:4, normal:3 |

**The crashes cluster on `no_policy` and `policy_truncated` specifically because
those conditions run long, not because of anything about policy content per se** -
both conditions also carry the two highest `max_turns` rates observed in session 1
(33.3% and 30.8% respectively, vs 7.1-28.6% for the other four conditions; see
PROGRESS.md for the full per-condition turns table) and the two highest mean
wall-clock times (882.7s and 436.8s vs 135.5-594.2s elsewhere). A condition that
tends to produce longer conversations is mechanically more likely to hit BOTH
`max_turns` (200-step cap) AND the context-window cap (32768 tokens) - the same
underlying cause (long conversations), two different caps. **This is
condition-dependent attrition and is disclosed here, not silently excluded from
whatever denominator a summary table uses.**

**Fixed 2026-08-17** (`sweep.py::run_cell()`, CPU-verified against the real litellm
exception class, not guessed): `ContextWindowExceededError` is now caught
specifically, never retried (deterministic at temperature=0.0 - one session-1
episode burned 73 minutes retrying this exact, unwinnable failure 3 times before
giving up), and recorded via `convert_context_window_exceeded()` as
`status="completed"`, `termination_reason="context_window_exceeded"` (new category,
`schema.py::TERMINATION_REASONS`), `reward=0.0` - a real, recordable outcome, not a
harness failure, matching how `max_turns`/`agent_error` are already treated.

**How crashed/context-window-exceeded episodes are handled in the PRIMARY grounding
rate**: a crashed episode (old, pre-fix `harness_crash`) or a context-window-exceeded
episode (new, post-fix) has NO transcript - the exception fires before any messages
exist to analyze. `grounding.py::analyze_conversation()` on an empty message list
produces `total_id_arguments=0`, so every rate field (`ungrounded_fraction_by_
argument`, `..._by_argument_conditional`) is `None` via the existing "zero
denominator -> None, never a silent 0%/100%" discipline (same guard
`test_grounding.py`'s empty-pool regression test already covers). Per
`average_by_condition()` (the declared PRIMARY aggregate, see above), a `None` rate
is EXCLUDED from that condition's average, not counted as 0% or 100% - so a crashed/
context-window-exceeded episode contributes NOTHING to the primary grounding rate,
neither inflating nor deflating it, but it DOES still count in the condition's
`agent_error`/crash-rate reporting above. A condition with a high crash rate is
therefore scored on FEWER real episodes for its grounding rate than a condition with
a low crash rate - smaller effective n, not a biased rate - and this must be read
alongside the crash-rate table above, not in isolation.

**Crash-exclusion bound, pre-registered 2026-08-18, BEFORE any grounding rate from
the expanded (N=12) grid has been read.** The PRIMARY handling above (crashed
episodes excluded, contribute `None`, not 0% or 100%) is one defensible choice among
several, and crashes are condition-dependent (14.3% `no_policy`, 7.1%
`policy_truncated`, 0% the other four - session 1 real numbers, restated from the
table above). To bound how much that choice could be driving any result, three views
are computed and reported together, not just the primary one:

- **(a) PRIMARY - excluded**: crashed/context-window-exceeded episodes contribute
  `None`, excluded from `average_by_condition()`. Already the current behavior,
  unchanged.
- **(b) WORST CASE - imputed fully ungrounded**: every crashed episode's missing
  rate is set to `1.0` (100% ungrounded) before averaging, for every condition that
  had any crashes.
- **(c) BEST CASE - imputed fully grounded**: every crashed episode's missing rate is
  set to `0.0` (0% ungrounded) before averaging, same conditions.

**Robustness rule, written now, before the numbers exist**: for any condition whose
primary-view (a) verdict matters to the falsification criterion above, compute (b)
and (c) for that condition too. If the ungrounded-rate increase (condition minus
control) under (b) and (c) fall on OPPOSITE sides of the 20pp threshold - i.e. one
view clears 20pp and the other does not - the result for that condition is NOT
ROBUST to the crash-handling choice and must be reported as such explicitly, not
rounded to whichever view happens to match the primary-view verdict. This rule is
symmetric: it applies whether the primary verdict is "load-bearing" or "inert." Only
conditions with a nonzero crash rate in the relevant comparison need (b)/(c)
computed - `full_context`, `schema_names_only`, `history_truncated`, and
`system_minimized` had 0% crashes in session 1, so their (a)/(b)/(c) views coincide
by construction unless session 2's 5 new tasks introduce new crashes there.

## Secondary metric: action_accuracy (DEMOTED 2026-08-17)

`RunRecord.action_accuracy` (`src/schema.py:158-160`), reported as a secondary/
diagnostic metric, not run through the primary significance test. **Carries an
explicit differential-attrition note wherever it is reported**: undefined (`None`) in
all 6 of this pilot's real episodes' non-`normal`-terminated cases and all 4 of
simswap's real Qwen2.5-7B-Instruct-AWQ episodes (10/10 real episodes across both
projects show `action_accuracy` either `None` or `0.0`, zero exceptions) - per the
evaluator finding above, this is not a floor effect, it is selective non-definition
correlated with `termination_reason`, which itself plausibly correlates with
`condition`. Any reported `action_accuracy` contrast MUST be read alongside its
`n_pairs` and the corresponding `agent_error`/`max_turns` rates for the conditions
being compared (see "PRE-REGISTERED: agent_error handling" and "PRE-REGISTERED
SECONDARY" sections below, both still in force, written for this metric originally -
the same exclude-by-termination-reason mechanism applies conceptually to grounding
rate too, but `grounding.py`'s aggregation functions do not yet have an equivalent
termination-reason filter; not implemented speculatively ahead of the real grid
showing whether it's needed).

## Attribution rule (REWRITTEN 2026-08-17 in grounding vocabulary)

A condition is read as a load-bearing context component if its Holm-corrected p-value
is significant AND its ungrounded-rate INCREASE (condition minus control - positive
means the condition made fabrication worse) exceeds the minimum reliably-detectable
delta at the chosen grid size (see Detectability, grounding rate, below) -
significance alone is not enough at small grids, same reasoning as the original
accuracy-based rule (preserved in the ARCHIVED block). A condition is read as inert if
its increase is small AND not significant, or significant but below the
practical-effect threshold.

## Detectability, grounding rate (NEW 2026-08-17) - the actual grid, curve not one number

`src/detectability_grounding.py`: same statistical machinery as
`detectability.py` (same `paired_permutation_test`/`holm_correct` via `analyze.py`,
same strict "all 5 non-control conditions Holm-significant" criterion, same 80%
power over 60 replicates), rerun for the actual grid (`n_tasks=20, n_seeds=2`) with
a per-cell data-generating process shaped like grounding rate instead of
action_accuracy. Anchored on two real numbers, both cited precisely in the script's
own docstring: baseline grounded rate 0.611 (1 - 38.9% ungrounded-conditional, this
project's own real `full_context` pilot episode, task 93) and a realistic per-episode
ID-argument-count range (7-44, this pilot's real observed range).

**Run for real, `tables/detectability_grounding.csv`:**

| delta (percentage points) | power (60 replicates) |
|---|---|
| 5pp | 0% |
| 10pp | 30% |
| 15pp | 92% |
| 20pp | 100% |
| 30pp | 100% |

**Minimum delta reaching 80% power at n_tasks=20, n_seeds=2: 15 percentage points.**
This is the smallest ungrounded-rate increase this grid can reliably attribute to a
specific condition - stated as a curve, not a single verdict, per the operator's
explicit instruction. **simswap's real 34.9pp Qwen-vs-Llama conditional gap
(`F:/simswap/docs/FAILURE_ANALYSIS.md:274-277`) is NOT used as the tested reference
effect** - it is a cross-model comparison (two different agent models), not a
context-ablation delta on the same model, and treating it as a credible prior for
this study's effect size would be invented precision. It is cited only as evidence
that grounding-rate gaps of this general magnitude are not implausible in this
codebase's lineage, nothing more. This grid's real, tested MDE (15pp) - not 34.9pp -
is what sets the falsification threshold below.

## Detectability, action_accuracy (ARCHIVED reference - see the ARCHIVED block at
## the end of this file for why this metric was demoted; kept here as the historical
## basis for the OLD grid-size decision, not re-derived or re-endorsed)

`src/detectability.py`: plants a uniform delta on all 5 non-control conditions
simultaneously, runs the real `analyze.py` contrast+Holm pipeline, requires ALL 5
planted conditions to come out Holm-significant (the strict per-condition-attribution
criterion, not just "something moved somewhere"). "Reliably detected" = >=80% power
over 60 independent replicate draws per (grid, delta) cell. Full sweep:
`tables/detectability.csv`.

| n_tasks | n_seeds | smallest delta reaching 80% power (60 replicates) |
|---|---|---|
| 15 | 2 | 0.20 |
| 20 | 2 | 0.20 (0.15 reaches only ~70% power) |
| 25 | 2 | 0.15 |
| 30 | 2 | 0.15 |

Power estimates carry real Monte Carlo noise at 60 replicates (binomial SE ~0.05-0.06
near p=0.5-0.7) - the non-monotonic jump between some adjacent delta values in the raw
sweep output is that noise, not a real non-monotonicity in the underlying design; the
table above reports the smallest delta that cleared 80% in this run, not a smoothed
curve.

**Grid size: DECIDED, 2026-08-17 - `n_tasks=20, n_seeds=2, 6 conditions` (240
episodes, ~14.3-18.9h estimated GPU-hours - see PROGRESS.md's real pilot-derived
range, not re-derived here). Operator's call, made under the OLD accuracy-based
detectability numbers** (`n_tasks=25` would have reliably resolved down to 0.15
accuracy-scale; this grid only reliably resolved ~0.20 on that metric) - the grid
size itself was NOT revisited when the primary metric changed to grounding rate,
since the new metric's own detectability was rerun AT this grid (`n_tasks=20,
n_seeds=2`, see "Detectability, grounding rate" above) rather than the grid being
re-picked to fit a metric. The effect-size floor used in the falsification criterion
below is grounding rate's own 15pp MDE at this grid, not the old accuracy-based
0.20.

## Grid size: REVISED 2026-08-18 - cut to N=12 tasks x 2 seeds (GPU budget, not a
## data-dependent decision - see LIMITATIONS)

Operator's call, 2026-08-18, made from PROGRESS.md's real scope-options table
(session 1's real per-condition wall-clock means, 10 remaining cells at 7.46h
projected, real `detectability_grounding.py` reruns per candidate grid size) -
session 1's real grounding-rate numbers themselves were NOT read before this
decision. **10 cells remain.** Final grid: 12 task_ids x 2 seeds x 6 conditions =
144 episodes (84 from session 1 + 60 new).

**Task selection is a PREFIX of the original 20-task list, not a re-pick after
seeing data.** `sweep.select_tasks("retail", 20)` at `TASK_SAMPLE_SEED=20260814`
returns, in order:
`['93', '43', '98', '77', '66', '94', '4', '46', '107', '64', '102', '110', '82',
'73', '7', '74', '55', '89', '15', '13']`. The first 12 of that list, in the same
order, are the final grid:
`['93', '43', '98', '77', '66', '94', '4', '46', '107', '64', '102', '110']`.
Verified 2026-08-18 by re-running `select_tasks` against the pinned tau2 commit
(CPU-only, no GPU/network): all 7 already-run session-1 task_ids (`4, 43, 66, 77,
93, 94, 98`) fall inside this prefix, no exceptions - checked programmatically, not
by inspection. The 5 new task_ids needed for session 2 are `46, 107, 64, 102, 110`.
A prefix was used, rather than any other 12-element subset, because it is the only
selection rule that requires no choice conditioned on anything the operator has
seen: the ordering itself was fixed by the seed on 2026-08-14, before any GPU
episode ever ran.

**Threshold revised to 20pp for this grid** (was 15pp at `n_tasks=20`). Real MDE at
`n_tasks=12, n_seeds=2` from the same `detectability_grounding.py` machinery (see
PROGRESS.md's scope-options table, 2026-08-17/18 session): 20pp reaches 87% power
(above the 80% target); the 15pp threshold used for the `n_tasks=20` design does not
transfer to a smaller grid and is not re-tested here - see the ARCHIVED note
immediately below the falsification section for the superseded 15pp version.

## Falsification criterion for "policy is inert" (REWRITTEN 2026-08-17, grounding
## vocabulary, threshold set from this grid's real 15pp MDE - not a threshold the
## grid cannot detect, per the operator's explicit instruction)
## *** ARCHIVED 2026-08-18: superseded by the 20pp/N=12 version below. Kept
## verbatim, not deleted - this was written for the n_tasks=20 grid, which this
## project no longer plans to run in full. ***

- **Hypothesis SUPPORTED** if `no_policy` is load-bearing (`p_holm < 0.05` AND
  ungrounded-rate increase `>= 0.15`) AND `policy_truncated` is NOT load-bearing
  (either `p_holm >= 0.05`, or `p_holm < 0.05` but increase `< 0.15`) - i.e. removing
  the whole document drives up fabrication a lot, keeping only the task-relevant
  section does not, at this grid's resolution.
- **Hypothesis FALSIFIED** if `policy_truncated` is ALSO load-bearing (`p_holm <
  0.05` AND increase `>= 0.15`) - task-irrelevant policy text was actually
  load-bearing for staying grounded, contradicting the "most of the document is
  inert" claim.
- **Inconclusive at this grid** if `no_policy` itself is not load-bearing, or if
  `policy_truncated`'s increase falls in a range directionally consistent with either
  reading but not clearly resolved by the 15pp floor - reported as inconclusive, not
  rounded to a conclusion. Since this grid's MDE (15pp) is smaller than the old
  accuracy-based grid's (20pp on this same `n_tasks=20` grid), grounding rate is, if
  anything, LESS likely to land in this inconclusive band than the original metric
  would have been at the same grid size - not a reason to expect more ambiguity from
  the metric change, if anything less.

`schema_names_only` and `history_truncated` are read against the same attribution
rule but do not bear on the "policy is inert" hypothesis specifically. `system_
minimized` remains a negative control (see its section above), read the same way
under grounding rate as it was under accuracy: a large ungrounded-rate increase there
would be a surprising finding worth its own investigation, not evidence for or
against this hypothesis.

## Falsification criterion for "policy is inert" (REVISED 2026-08-18, threshold
## 20pp for the N=12 grid - see "Grid size: REVISED 2026-08-18" above for why 15pp
## no longer applies)

Same structure as the archived 15pp version above, threshold only changed:

- **Hypothesis SUPPORTED** if `no_policy` is load-bearing (`p_holm < 0.05` AND
  ungrounded-rate increase `>= 0.20`) AND `policy_truncated` is NOT load-bearing
  (either `p_holm >= 0.05`, or `p_holm < 0.05` but increase `< 0.20`).
- **Hypothesis FALSIFIED** if `policy_truncated` is ALSO load-bearing (`p_holm <
  0.05` AND increase `>= 0.20`).
- **Inconclusive at this grid** if `no_policy` itself is not load-bearing, or if
  `policy_truncated`'s increase falls in a range directionally consistent with either
  reading but not clearly resolved by the 20pp floor - reported as inconclusive, not
  rounded to a conclusion.

`schema_names_only`, `history_truncated`, and `system_minimized` are read the same
way as in the archived version (attribution rule and negative-control reading
unchanged, threshold only).

## What is NOT decided here

- Whether/how to capture policy text per condition for origin classification (see
  Primary metric section above - still genuinely open, unrelated to grid size or the
  metric change).
- GPU-hours budget and session count - see PROGRESS.md's real pilot-derived estimate
  (18.9h at the observed per-condition means, which did not fit the original 2-session
  18h budget) - unrelated to the metric change, not re-derived here.
- Whether `grounding.py`'s aggregation functions need their own termination-reason
  exclusion (mirroring `accuracy_by_cell`'s `exclude_truncated`/`exclude_agent_error`)
  - flagged in the Secondary metric section above, not implemented speculatively.

## ARCHIVED, 2026-08-17 - the ORIGINAL action_accuracy-based analysis plan, preserved
## verbatim below, NOT deleted, NOT re-endorsed. Superseded by the sections above
## (Hypothesis, Primary/Secondary metric, Attribution rule, Detectability, and
## Falsification criterion) once the evaluator finding showed action_accuracy is
## selectively undefined by a mechanism correlated with condition (differential
## attrition), not merely underpowered. Everything from this heading to the end of
## the file is the pre-metric-change text, unedited except for this header and the
## explicit archival note before the old "FROZEN" section - read it as a historical
## record of what was decided before the pilot ran, not as current guidance.

### ARCHIVED - original Hypothesis under test (accuracy/grounding-hybrid vocabulary,
### as originally written 2026-08-17 before any GPU time)

Most of the policy document is inert: agents fail on tasks whose governing policy
section was in context the whole time. If true, that reframes policy-adherence failure
as retrieval/attention rather than reasoning - the model had the right text available
and still didn't act on it, similar in spirit to simswap's finding that Qwen mostly
fails by trusting an unverified user claim rather than fabricating one (a "didn't check
what was available" failure, not a "had nothing to check" one).

Falsifiable prediction: if the hypothesis holds, `policy_truncated` (task-relevant
policy sections only) should show a MUCH smaller accuracy/grounding drop than
`no_policy` (whole document removed) - most of the document was never load-bearing for
the tasks it covers. If the hypothesis is false, `policy_truncated` drops nearly as
much as `no_policy` - the model was actually using material outside the task-relevant
section (e.g. cross-referencing policy for edge cases), and truncation destroys that.

### ARCHIVED - original Primary metric: action-level correctness

Same `read`/`write` `ActionTally` -> `action_accuracy` as simswap. For each non-control
condition, paired by `(task_id, seed)` against the `full_context` control run for the
same cell: `diff = accuracy(control) - accuracy(condition)`, positive means the
condition hurt. Two-sided Monte Carlo paired sign-flip permutation test (exact
enumeration is infeasible once n_pairs exceeds ~20, unlike simswap's 3-task case -
`src/analyze.py::paired_permutation_test`, `n_perm=10000` default), Holm-corrected
across the (up to 5) non-control conditions.

### ARCHIVED - original Secondary metric: grounding rate, per condition

`src/grounding.py`, ported from simswap's `failure_analysis.py` (algorithm unchanged,
already validated there). Reports FLAT and CONDITIONAL ungrounded-ID rate per
condition, pooled via `pool_by_condition` (sums raw counts, does not average
per-conversation fractions - see `tests/test_grounding.py`).

**Known gap, disclosed not hidden**: origin classification (user-stated vs. invented),
which was central to simswap's headline finding, is NOT implemented here yet - it
needs the conversation's policy text and user utterances, and this project's
transcript format (`sweep.py`'s bare message-list write) does not currently capture
policy text at all. Needs a `sweep.py` change to save it per condition (trivial once
`apply_condition()` exists, since that's exactly where the effective policy text will
be constructed) - not done speculatively ahead of that.

### ARCHIVED - original Attribution rule

A condition is read as a load-bearing context component if its Holm-corrected p-value
is significant AND its accuracy/grounding drop exceeds the minimum reliably-detectable
delta at the chosen grid size (see Detectability below) - significance alone is not
enough at small grids (the `policy_inert` fake-data validation run below hit p_holm <
0.05 on a condition with a genuinely tiny planted effect at n=20 tasks in one draw;
see PROGRESS.md). A condition is read as inert if its drop is small AND not
significant, or significant but below the practical-effect threshold.

### ARCHIVED - original "FROZEN" final analysis, written 2026-08-17 before any real (GPU) data exists

Pre-registered per the operator's explicit instruction: written now, not edited after
seeing results. A result that contradicts this section gets a new dated section
elsewhere in this file recording the deviation - this section itself does not change.

**Primary metric**: `RunRecord.action_accuracy` (`src/schema.py:158-160`) -
`action_correct / action_total`, read and write actions combined, per (task_id,
seed, condition) cell. Cells with zero actions are excluded (`None`), not treated as
zero (`src/analyze.py:45-87`, `accuracy_by_cell` - line range corrected 2026-08-17
after the agent_error extension below, body logic unchanged).

**Secondary metric**: grounding rate (`src/grounding.py`), FLAT and CONDITIONAL
ungrounded-ID rate per condition, pooled by summing raw counts across a condition's
runs (`pool_by_condition`, not averaged per-conversation fractions - see
`tests/test_grounding.py`). Descriptive/diagnostic, not run through the same
significance test as the primary metric - no p-value is computed for it here.

**Test**: two-sided Monte Carlo paired sign-flip permutation test
(`src/analyze.py:200-210`, `paired_permutation_test` - line number corrected again
2026-08-17 after the agent_error extension moved it further down; body
byte-identical both times, substance unchanged, see PROGRESS.md), paired by `(task_id, seed)`
against the `full_context` control run for the same cell, `diff = accuracy(control) -
accuracy(condition)`. `n_perm=10000` (`analyze.py:30`, CLI default), `rng_seed=0`
(`analyze.py:32`, CLI default - fixed for reproducibility, not re-rolled per run).

**Alpha**: 0.05 (`analyze.py:31`, CLI default; matches the hardcoded threshold
already used in `detectability.py:100`'s power sweep, so the frozen test and the
grid-size power calculation that motivated it use the same alpha).

**Multiple-comparison correction**: Holm step-down (`analyze.py:187-197`,
`holm_correct` - line number corrected again 2026-08-17, same note as above) across
the up-to-5 non-control conditions' raw p-values
(`no_policy`, `policy_truncated`, `schema_names_only`, `history_truncated`,
`system_minimized`). A condition counts as significant iff `p_holm < 0.05`.

**Attribution rule** (restated from above, now frozen alongside the rest): a
condition is load-bearing iff `p_holm < 0.05` AND its accuracy drop >= 0.20 (this
grid's minimum reliably-detectable delta at 80% power, `tables/detectability.csv`).
Significance alone is not sufficient at `n_tasks=20` - a p_holm < 0.05 result on a
drop below 0.20 is reported as "statistically detected but below this grid's
practical-effect floor," not as "load-bearing."

**Falsification criterion for "policy is inert"** (operationalized from the
Hypothesis section above, using this grid's 0.20 floor):

- **Hypothesis SUPPORTED** if `no_policy` is load-bearing (`p_holm < 0.05` AND drop
  `>= 0.20`) AND `policy_truncated` is NOT load-bearing (either `p_holm >= 0.05`, or
  `p_holm < 0.05` but drop `< 0.20`) - i.e. removing the whole document hurts a lot,
  keeping only the task-relevant section does not, at this grid's resolution.
- **Hypothesis FALSIFIED** if `policy_truncated` is ALSO load-bearing (`p_holm <
  0.05` AND drop `>= 0.20`) - task-irrelevant policy text was actually load-bearing
  (e.g. cross-referencing for edge cases), contradicting the "most of the document is
  inert" claim.
- **Inconclusive at this grid** if `no_policy` itself is not load-bearing (the
  precondition for testing the hypothesis at all did not hold - the whole document
  didn't clearly matter, so nothing can be said about which part of it does) or if
  `policy_truncated`'s drop falls in a range that is directionally consistent with
  either reading but not clearly resolved by the 0.20 floor (e.g. detected but small,
  or a large point-estimate drop that fails significance) - reported as inconclusive,
  not rounded to a conclusion in either direction. A larger grid (`n_tasks=25`,
  0.15 floor, not run here) would be the next step if this happens.

`schema_names_only` and `history_truncated` are read against the same attribution
rule but do not bear on the "policy is inert" hypothesis specifically - they answer
separate questions (tool-schema descriptions; conversation-history length) about
which part of context is load-bearing. `system_minimized` is a negative control (see
its section above) and is not read as evidence for or against any hypothesis; a
load-bearing result there would be reported as a surprising finding worth its own
investigation, not folded into the primary attribution table.

## PRE-REGISTERED SECONDARY - written 2026-08-17, before any pilot output was read

The pilot kernel (`abhiprd2000/attrib-pilot-run`) was RUNNING on Kaggle when this
section was written. Its output had not been read, downloaded, or looked at in any
form. This section is not a reaction to pilot numbers - it exists because `max_steps`
truncation is a structural possibility in every condition (see `sweep.py`'s `MAX_STEPS
= 200` provenance note) and needed a pre-committed handling rule before any data,
pilot or real, could tempt a post-hoc one.

**The FROZEN primary section above is unchanged and unedited by this section** -
truncated-at-cap episodes (`termination_reason == "max_turns"`) remain included in
the primary `action_accuracy` contrast, scored as-is on whatever actions they took,
exactly as the frozen text already specified. This section adds reporting
obligations and one interpretation rule around that primary result; it does not
replace or gate it.

**Truncation rate, first-class, not a footnote.** For every condition: `n_completed`,
`n_truncated` (`termination_reason == "max_turns"` among completed runs), and
`truncation_rate = n_truncated / n_completed`. Computed by
`analyze.py:truncation_rate_by_condition()`, printed and written to
`tables/truncation_rate_by_condition.csv` by `analyze.py::main()` on every run,
unconditionally - not something that only appears if it looks interesting.

**Secondary robustness check**: `action_accuracy` recomputed with truncated episodes
excluded (`accuracy_by_cell(records, exclude_truncated=True)`,
`analyze.py:45-87`), run through the identical contrast pipeline as the primary
(`analyze.py:contrast()` at `analyze.py:213`, factored out specifically so primary and secondary can
never silently diverge in method - only in which records feed them). Written to
`tables/condition_contrasts_excl_truncated.csv`. **Reported alongside the primary
every time, regardless of whether the two agree** - agreement is not a precondition
for publishing either number, and the secondary is never substituted for the primary
even if the secondary looks "cleaner."

**Interpretation rule, stated blind, before any pilot or real data existed**: if any
two conditions' truncation rates differ by more than 10 absolute points
(`analyze.py::check_truncation_confound()`, `TRUNCATION_CONFOUND_THRESHOLD = 0.10`),
the primary accuracy effect between those conditions is confounded with step
exhaustion and MUST be reported as such - not as a pure grounding/context effect, no
matter how clean the accuracy contrast itself looks. Concretely: a condition that
strips context an agent needs could plausibly cause longer, more confused
conversations that run out of steps more often than the control - in that case a
large accuracy drop is at least partly "the harness cut it off," not purely "the
agent acted on what it had." The secondary (excl-truncated) contrast is the first
thing to check when this rule fires; if the secondary drop is much smaller than the
primary drop for the same condition, that is itself evidence the confound is real,
not just flagged.

This rule is pre-committed now specifically so it cannot be relaxed later if a real
truncation-rate gap turns out to be inconvenient for the "policy is inert" narrative
- the threshold (10 points) and the action (report as confounded) were fixed before
any number existed to motivate picking a more favorable threshold.

## PRE-REGISTERED: agent_error handling - written 2026-08-17, before reading or
## reporting this pilot's accuracy numbers

The pilot's 6 records were read for plumbing diagnosis (transcripts, tool-call error
content, termination reasons, token counts - see PROGRESS.md) - `action_accuracy`/
`reward`/`read`/`write` values were incidentally visible in that same pass, since
they live in the same `RunRecord` JSON. This section was written by direct symmetry
with the already-frozen `max_turns` treatment above, not by looking at or tuning to
those numbers - disclosed plainly rather than claiming a stricter blindness than
actually held. What this section commits to is the RULE, not a number, and the rule
is identical in form to `max_turns`'s, just applied to a second termination reason.

**Why this needed writing now, not later**: `ANALYSIS_PLAN.md`'s existing
pre-registration only ever named `max_turns` vs `normal` explicitly.
`termination_reason == "agent_error"` (tau2's `TerminationReason.TOO_MANY_ERRORS`) is
a third real category, structurally identical to `max_turns` in the one way that
matters here - it is a `status == "completed"` outcome, not a harness failure - and
the pilot showed it is not a rare edge case. Extending the rule blind, before the
240-episode grid, closes the same gap `max_turns`'s rule closed for step exhaustion.

**tau2 source, verified against the pinned commit** (see `SOURCE_NOTES.md`'s
citation conventions - same clone, same discipline): `orchestrator.py:326-327`
(`Orchestrator._execute_tool_calls`) - `if tool_result.error: self.num_errors += 1`,
incremented once per executed tool call whose `ToolMessage.error` is `True` (set in
`Environment.get_response()`, `environment.py:465-490`, whenever the underlying tool
method raises - e.g. a business-logic/lookup failure, not necessarily a malformed
call). `orchestrator.py:747-749` (half-duplex path, the one this project uses) -
`if self.num_errors >= self.max_errors: self.done = True; self.termination_reason =
TerminationReason.TOO_MANY_ERRORS`. Threshold `max_errors`: default `10`
(`tau2/config.py:5`, `DEFAULT_MAX_ERRORS = 10`; `TextRunConfig`/`BaseRunConfig` never
overrides it, confirmed by reading `data_model/simulation.py`'s field default and
`sweep.py::build_config()`, which does not set `max_errors` at all) - so an episode
must accumulate exactly 10 failed tool-call executions to end this way, not a soft
heuristic.

**1. Primary scoring**: identical treatment to `max_turns`, extended, not
special-cased. `agent_error` episodes are `status == "completed"` and are INCLUDED in
the primary `action_accuracy` contrast, scored on whatever actions they took before
the 10th tool-call error ended the episode - exactly the same rule already frozen for
`max_turns`, now stated to cover both. `accuracy_by_cell(records)` (both exclusion
flags default `False`) is unchanged and still the frozen primary.

**2. Secondary exclusion - yes, and it gets a THIRD accuracy view, not a replacement
of the existing one.** `accuracy_by_cell()` gained a second, independent exclusion
flag: `exclude_agent_error` (`analyze.py:45-87`, defaults `False`, `max_turns`'
`exclude_truncated` flag is untouched and still defaults the same way it always has).
Three views are now computed and reported every run, side by side, none silently
preferred:
- **PRIMARY**: all completed episodes (`condition_contrasts.csv`, unchanged).
- **SECONDARY**: excludes `max_turns` only (`condition_contrasts_excl_truncated.csv`,
  unchanged, already frozen).
- **THIRD** (new): excludes BOTH `max_turns` AND `agent_error` - i.e. normal
  terminations only (`condition_contrasts_normal_only.csv`,
  `accuracy_by_cell(records, exclude_truncated=True, exclude_agent_error=True)`).

**3. Interpretation rule, stated blind**: `agent_error_rate_by_condition()` and
`check_agent_error_confound()` (`analyze.py`, added alongside their `max_turns`
counterparts, same shape, same logic) - if any two conditions' `agent_error` rates
differ by more than 10 absolute points (`AGENT_ERROR_CONFOUND_THRESHOLD = 0.10`,
same threshold as the `max_turns` rule, chosen for consistency not re-derived), the
primary accuracy effect between those conditions is confounded with tool-call error
propensity and MUST be reported as such, not as a pure context effect - same
reasoning as the `max_turns` rule: a condition that degrades the agent's ability to
call tools correctly (e.g. `schema_names_only`, which removes the descriptions that
tell it how to format arguments) could plausibly cause more failed tool calls
independent of whether the underlying task-relevant information was present, and
that is a different mechanism than "the model didn't act on content it had." Printed
and written to `tables/agent_error_rate_by_condition.csv` unconditionally, every run
- first-class, not a footnote, same standard as `max_turns`.

**Reported first-class result from the pilot itself (n=1, not interpreted as an
accuracy finding)**: 3 of 6 conditions in the pilot ended `agent_error`
(`full_context`, `history_truncated`, `system_minimized`), 1 ended `max_turns`
(`schema_names_only`), 2 ended `normal` (`no_policy`, `policy_truncated`) - see
PROGRESS.md for the diagnosis. This is exactly the situation this section exists to
handle, and is why it was written now rather than left for the real grid to surface.

## LIMITATIONS - written 2026-08-17

**The `agent_error` handling rule (the "PRE-REGISTERED: agent_error handling"
section above) is a disclosed deviation, not a clean pre-registration.** Unlike the
FROZEN primary section and the "PRE-REGISTERED SECONDARY" (`max_turns`) section -
both written and this repo snapshotted before any GPU episode had ever run - the
`agent_error` rule was written AFTER this pilot's 6 episodes had been read in full,
including their `action_accuracy`/`reward`/`read`/`write` fields (read for the
plumbing/termination diagnosis this session, not hidden from view before the rule was
drafted). The rule's actual content (threshold, structure, what gets excluded) was
derived by direct symmetry with the already-frozen `max_turns` rule, not by tuning to
what this pilot's specific numbers showed - but the ordering (numbers seen, then rule
written) is real and is recorded here plainly rather than described as blind. Anyone
auditing this project's pre-registration discipline later should treat this one
section differently from the two written before any data existed.

**RESOLVED 2026-08-17 (update to the paragraph below, which described this as
undecided as of the previous session): the primary metric was NOT a floor problem -
it was differential attrition** (`tau2/evaluator/evaluator.py:119-129` gates action
scoring on `termination_reason`, unconditional on the task; verified 20/20 selected
tasks have non-empty golden action sets, so task selection was not the cause - see
the "METRIC CHANGE" section near the top of this file for the full finding and the
real termination_reason-vs-defined cross-tab). Grounding rate is now the SOLE
primary; `action_accuracy` is demoted to a reported secondary carrying this
attrition note wherever it appears. The original text below is kept for the
record of what this section said before that finding:

*(original, superseded)* "The primary metric's floor problem, surfaced by this
pilot, is not yet resolved. `action_accuracy` was `None` (zero scoreable actions) or
`0.0` in every one of this pilot's 6 episodes, and in all 4 of simswap's own real
Qwen2.5-7B-Instruct-AWQ episodes (`F:/simswap/data/runs/qwen_task{0,1,2,3}.json`,
read-only) - 10/10 real episodes across both projects, no exceptions. Whether the
frozen primary metric should change is a live, undecided question - see
PROGRESS.md's 2026-08-17 'metric proposal' entry for the argument both ways. This
section records that the question exists; it does not answer it. `ANALYSIS_PLAN.md`'s
FROZEN primary section is unchanged and remains in force until the operator decides
otherwise."

**The 15pp MDE rests on a single-episode baseline - if that baseline is wrong, the
real MDE is wrong too.** `detectability_grounding.py`'s `BASE_SKILL_GROUNDED = 0.611`
is `full_context`'s conditional grounded rate from exactly ONE pilot episode (task
93, n=1). The 15pp minimum-detectable-delta figure in the "Detectability, grounding
rate" section above is only as good as that one number. If the true baseline
grounded rate across the real 20-task grid differs materially from 61.1% (in either
direction - a higher true baseline would generally make small deltas easier to
detect on a rate metric bounded by an already-high pool, a lower one harder), the
real MDE at `n_tasks=20, n_seeds=2` differs from 15pp too. **Not re-derived here -
there is no better baseline estimate available until the grid itself runs.** Stated
plainly and left as a known limitation of this pre-registration, not resolved by
guessing a "safer" number.

**The grid was cut from N=20 to N=12 for GPU budget, not for any data-dependent
reason.** The decision (see "Grid size: REVISED 2026-08-18" above) was made from
PROGRESS.md's scope-options table - real session-1 wall-clock means and real
`detectability_grounding.py` reruns per candidate grid size - not from session 1's
actual grounding-rate results, which were not consulted when N=12 was chosen. The
consequence is a real one, disclosed here rather than smoothed over: the falsification
threshold is 20pp instead of 15pp, a coarser resolution, so an effect between 15pp
and 20pp that the original N=20 design could have detected will now read as
inconclusive. This is a genuine loss of power traded for GPU-hours, not a choice that
biases the result toward any particular verdict.

**`grounding_rate_by_cell()` was written after this project's grounding-rate point
estimates had already been seen - a disclosed deviation, not a clean pre-registration,
same category as the `agent_error` handling rule above.** The PRIMARY significance
test (paired permutation test + Holm correction on grounding rate) was frozen in this
file on 2026-08-17, but the per-cell `{condition: {(task_id, seed): rate}}` shape
`analyze.py::contrast()` needs to actually run it did not exist until 2026-08-23 - by
which point the macro-averaged/pooled point estimates (`pool_by_condition`/
`average_by_condition`, no significance attached) had already been computed and
reported for the full 144-episode grid, including the real per-condition ungrounded-
rate increases over `full_context`. `grounding_rate_by_cell()` (`src/grounding.py`)
was built by direct structural symmetry with `analyze.py::accuracy_by_cell()` - same
shape, same one-value-per-cell design, same "undefined rate excluded, never
0%/100%" discipline already used throughout `grounding.py` - not tuned to make any
specific condition's p-value come out a particular way, but the ordering (point
estimates seen, then the significance-test plumbing built) is real and stated here
plainly rather than described as blind. The resulting p-values (see PROGRESS.md's
2026-08-23 RESULTS entry) are reported as-is, including the one raw p<0.05 result
(`no_policy`, p_raw=0.047) that does NOT survive Holm correction (p_holm=0.235) and
points in the opposite direction from this hypothesis's prediction on that specific
paired subset - not suppressed or reframed because of how it came out.
