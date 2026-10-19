# Progress log

Append-only, newest entries at the bottom. No git commits this session (repo is
`git init`-ed, zero commits) - this file plus directory snapshots
(`../attrib-snap-YYYYMMDD-HHMM/`) are the crash-recovery mechanism, same convention
as the simswap project.

## 2026-08-17 - new project opened, pre-GPU validation pass, nothing touched a GPU

Operator opened a new project for the ATTRIB workshop (deadline 2026-09-02): which
part of the agent's context causes a tool-call failure, via a context-ablation study
reusing the simswap harness. Explicit instruction: new directory, separate repo, do
not contaminate simswap (did not edit `simswap/paper/` or `simswap/src/` - only read
`simswap/src/schema.py`/`sweep.py` to copy them as a starting point).

**Created `F:\attrib`, its own `git init`-ed repo (no commits).**

**Step 1-2: schema.py + sweep.py forked.** Copied from simswap, then: added
`CONDITIONS` (6: `full_context` control + `no_policy`/`policy_truncated`/
`schema_names_only`/`history_truncated`/`system_minimized`) and `condition`/
`condition_param` fields to `RunRecord`, with validation (`condition_param` required
iff `condition=="history_truncated"`, forbidden otherwise). Renamed
`simswap_commit`->`attrib_commit` throughout (this is a different repo now).
`sweep.py`'s CLI gained `--condition`/`--condition-param`; `run_id` now encodes the
condition. **`apply_condition()` (the actual context-stripping logic) deliberately
raises `NotImplementedError` for every condition except `full_context`** - the real
tau2 interception point (which object actually carries the policy document/tool
schemas/system instructions into the served prompt) is not yet verified against real
tau2 source, and guessing at it before checking would risk silently running the wrong
experiment. This is flagged in `ANALYSIS_PLAN.md`, not hidden or worked around.
Validated: `python sweep.py --dry-run` runs the real write path end to end (mocked
tau2 result, `attrib_commit` correctly falls back to `"no-commit-yet"` since this repo
has zero commits by design) - confirms schema.py/sweep.py's plumbing is sound before
any tau2/GPU dependency is even needed.

**Step 3: analysis script, validated on fake data.** Built `src/fake_runs.py`
(synthetic `RunRecord` generator, 3 scenarios: `null`, `uniform`, `policy_inert` - the
last one plants exactly the hypothesis pattern, no_policy hurts a lot,
policy_truncated barely, for validating the pipeline can recover that shape) and
`src/analyze.py` (per-condition action-accuracy, paired by (task_id, seed) against the
`full_context` control, Monte Carlo paired sign-flip permutation test, Holm-corrected
across conditions - generalizes simswap's exact-enumeration `permutation_test.py` to
the larger n_pairs a single-factor 6-condition design allows).

Ran both together: `--scenario policy_inert --delta 0.20` (one draw, `--n-tasks 20`) -
recovered the correct RANK ORDER (no_policy biggest drop 0.238, policy_truncated
smallest 0.086) but **policy_truncated's true planted delta was only 0.01
("near-inert") and still came out Holm-significant (p_holm=0.043) in this single
draw** - not a bug (traced: per-run binomial sampling noise on small action counts is
large enough that a single draw can push a near-zero true effect over the
significance line at n=20 pairs), but exactly the false-positive risk a real power
analysis needs to characterize rather than a single lucky/unlucky draw. Also ran
`--scenario null` (no planted effect anywhere): 0/5 conditions Holm-significant,
closest raw p was 0.03 on `policy_truncated` (would have looked "significant" without
Holm correction) - confirms Holm correction is doing real, necessary work here.

**Step 3b: grounding-rate module.** Ported simswap's `failure_analysis.py` algorithm
(unchanged - it was already validated there via 11 regression tests and 2 independent
hand-traces) into `src/grounding.py`, adapted to this project's transcript format
(sweep.py writes a bare message list, not simswap's `sim_dump` wrapper) and to group
by `condition` instead of by agent model. Validated the same way simswap validated the
original - hand-built fixture regression tests, not synthetic statistics, since
grounding is a deterministic per-conversation property: `tests/test_grounding.py`,
6/6 passing (4 cases ported directly from simswap's proven suite, 1 new for the
condition-pooling aggregation - built with a fixture where pooling-then-computing
vs. averaging-fractions would visibly disagree, same guard pattern as simswap's
`test_matched_pool_aggregates_correctly`). **Known, disclosed gap**: origin
classification (user-stated vs. invented - simswap's headline distinction) is not
ported, because it needs policy text and user utterances that this project's
transcript format does not currently capture. Not implemented speculatively - flagged
in `ANALYSIS_PLAN.md` as a `sweep.py` change to make once `apply_condition()` exists
(the natural point to also capture the effective policy text per condition).

**Step 4: grid size and GPU-hours - built `src/detectability.py`, ran it for real
numbers, twice (20 then 60 replicates/cell, second run kept - the first run's power
estimates were too noisy at 20 replicates to trust, ~40s vs ~128s wall-clock, both
cheap, CPU-only, reran without hesitation).** Plants a uniform delta on all 5
non-control conditions simultaneously, requires ALL 5 to be Holm-significant
("reliably attributes to the SPECIFIC condition", not just "notices something moved"
- the stricter, correct criterion for an attribution study), 80% power target over 60
independent replicate draws per (grid, delta) cell. Real result
(`tables/detectability.csv`):

| n_tasks | n_seeds | smallest delta @ 80% power |
|---|---|---|
| 15 | 2 | 0.20 |
| 20 | 2 | 0.20 (0.15 only reaches ~70%) |
| 25 | 2 | 0.15 |
| 30 | 2 | 0.15 |

**GPU-hours estimate - bounded by exactly ONE real number, everything else flagged as
unmeasured.** simswap's real Qwen2.5-7B-AWQ self-play wall-clock (4 real episodes,
same agent+user models this project fixes throughout): 74.10s, 210.28s, 346.37s,
228.94s (mean 214.92s) - this is the ONLY empirical basis available, since
`apply_condition()` isn't implemented and no episode under any of the other 5
conditions has ever run. Applying that mean uniformly across all 6 conditions
(explicitly an assumption, not a measurement - stripped-context conditions could run
faster due to shorter prompts, or differently if the agent errors out sooner/later
without full context; no real number exists for that yet):

| grid (n_tasks x 2 seeds x 6 conditions) | episodes | estimated GPU-hours | fits one 12h Kaggle session? |
|---|---|---|---|
| 15 | 180 | ~10.7h | yes, barely, no margin |
| 20 | 240 | ~14.3h | no, needs 2 sessions |
| 25 | 300 | ~17.9h | no, needs 2 sessions |
| 30 | 360 | ~21.5h | no, needs 2 sessions, near the 24h (2-session) ceiling |

No vLLM restart-overhead term needed (unlike simswap's 16-pair sweep) - agent/user
models never change across conditions, so servers start once per session, not once
per condition.

**Reported to the operator, per the "before any GPU time" brief: grid size and cost
are NOT decided here.** n_tasks=25 (300 runs, ~17.9h estimated, 2 sessions) is the
smallest grid in the sweep that reliably attributes a 0.15-scale effect to a specific
condition; n_tasks=20 (240 runs, ~14.3h) is cheaper but only reliably resolves ~0.20
effects. Given the 2026-09-02 deadline this is an operator tradeoff, not resolved
unilaterally - matches the standing project norm (see simswap's `DECISIONS_NEEDED.md`
pattern) of surfacing options rather than picking silently.

**Nothing touched a GPU this session.** No kernel pushed, no Kaggle job checked. No
git commits (repo has zero commits, by design). Directory snapshot to follow.

## 2026-08-17 - apply_condition() implemented and verified against real tau2 source, still no GPU

Operator's brief: verify against real tau2-bench source first, write it down, only
then implement, then verify by dumping the actual request payload per condition.
Followed in that order.

**Step 1: cloned tau2-bench at the pinned commit for real** (`fc0055dc4e0a316c3f83
133267fbd6faaa770992`, confirmed via `git log -1`: "chore: prepare release v1.0.1").
Full clone (shallow fetch of just that commit - `git fetch --depth 1 origin
<sha>`) into a scratch dir outside this repo, read-only, not committed here. This
machine hit repeated process-spawn/fork resource exhaustion during the session
(`fork: retry: Resource temporarily unavailable`, `EPERM uv_spawn` from both Git
Bash and PowerShell, intermittently, several separate times) - not a tau2 or network
problem, a local machine problem; retried through it each time rather than working
around it with guesses.

**Step 2: `SOURCE_NOTES.md` written**, and it overturned the plan's premise.
`ANALYSIS_PLAN.md` assumed `apply_condition()` would mutate/rebuild the tau2
`TextRunConfig` before it reaches `run_single_task`. **Verified false**:
`TextRunConfig` (`tau2/data_model/simulation.py:514-596`) has no field for policy
text, tool schemas, system instructions, or history - `sweep.py` calls
`tau2.runner.run_single_task(config, task, seed=seed)` as one black-box call with no
hook. The real interception point, verified by reading the full call chain
(`run_single_task` -> `build_orchestrator` -> `build_text_orchestrator` ->
`build_environment` + `build_agent`, `tau2/runner/build.py` and
`tau2/runner/batch.py`), is monkeypatching: `tau2.runner.build.build_agent` (for
`no_policy`/`policy_truncated`/`schema_names_only`), `tau2.agent.llm_agent
.AGENT_INSTRUCTION` (for `system_minimized`), and `tau2.agent.llm_agent.LLMAgent
._generate_next_message` (for `history_truncated`), installed for the duration of one
`run_single_task()` call and reverted in a `finally` block. Every condition's exact
object/field and file:line is in `SOURCE_NOTES.md`, including the `policy_truncated`
task-relevance mapping (verified against every distinct `action.name` across the real
`data/tau2/domains/retail/tasks.json` - all 6 mutating action names map cleanly to
one of policy.md's 6 `##` sections, no unmapped names) and a documented design call
on what "minimized" system instructions means (kept the tool-call/JSON-format
mechanics, dropped the persona/policy-following framing - avoids confounding this
condition with schema_names_only-style output-format breakage).

**Step 3: implemented.** New `src/conditions.py` (`apply_condition()`, now a context
manager over `(condition, condition_param, task)`, not a `TextRunConfig` mutator).
`sweep.py::build_config()` now returns a plain `TextRunConfig`; `run_cell()` wraps
the `run_single_task()` call in `with apply_condition(...)`. `full_context` installs
no patches at all - a true no-op, not a patch that happens to do nothing.

**Step 4: verified by dumping and diffing the actual payload, CPU-only, no GPU.**
Built a throwaway venv (`verify-venv`, gitignored, outside the repo) with tau2's
real runtime dependencies (Python 3.11 - tau2's own `pyproject.toml` declares
`requires-python>=3.12`, but that only gates `pip install`ing the `tau2` distribution
itself; running its modules via `sys.path` against real Python 3.11 hit no
3.12-only syntax). Script (`verify_conditions.py`, scratch, not committed) calls
`build_text_orchestrator()` - the exact function `run_single_task` calls internally -
under each condition's patches, against a real retail task (`id=74`, actions
`cancel_pending_order` + `modify_pending_order_items`), and diffs the resulting
`agent.system_prompt` and `[tool.openai_schema for tool in agent.tools]` against
`full_context`. One real bug caught and fixed by this process (in the verify script,
not `conditions.py`): `system_prompt` is a lazily-evaluated property that re-reads
`AGENT_INSTRUCTION` live on every access, unlike `domain_policy` which freezes into a
plain attribute at construction time - reading it after the patch context had already
exited silently returned the unpatched value. Fixed by reading it while still inside
`with apply_condition(...)`, which is also what proves `system_minimized` is safe in
production: `get_init_state()` (which bakes `system_prompt` into the first
`SystemMessage`) always runs inside `run_single_task()`, which `sweep.py` wraps
entirely in `with apply_condition(...)`.

Real results, all 6 conditions, real tau2 source, one shared baseline task:

| condition | system_prompt | tools | check |
|---|---|---|---|
| `full_context` | 7056 chars | 16 tools | baseline |
| `no_policy` | 357 chars | 16/16 byte-identical to baseline | PASS: `<policy>` block empty, tools untouched |
| `policy_truncated` | 4303 chars | 16/16 byte-identical to baseline | PASS: kept "Cancel pending order", dropped "Return delivered order" and "Exchange delivered order" |
| `schema_names_only` | 7056 chars (identical to baseline) | 0/16 byte-identical - all descriptions gone | PASS: policy untouched, every tool's function-level and per-parameter descriptions stripped (confirmed via tau2's own `logger.warning(f"Tool {name} has no description.")` firing for all 16) |
| `history_truncated` (k=3) | 7056 chars (identical - doesn't touch system_prompt) | 16/16 byte-identical | PASS on turn probe: 6 simulated user turns -> `state.messages` holds 12 (untruncated, stored history intact) but `generate()` was sent only 3 non-system messages on the final turn |
| `system_minimized` | 6913 chars | 16/16 byte-identical | PASS: "Try to be helpful and always follow the policy" and the persona line gone, "valid JSON only" kept |

No cross-condition leakage observed (each condition's diff touched only what it
claimed to - e.g. `schema_names_only` left `system_prompt` byte-identical to
baseline, `no_policy`/`policy_truncated` left all 16 tool schemas byte-identical).

**Still not done, still no GPU spent**: the operator's pilot step (one real episode
per condition, 6 total, same task, against actual vLLM-served Qwen2.5-7B-Instruct-AWQ,
reporting real per-condition wall-clock) needs the Kaggle GPU session this machine
doesn't have - per the standing rule, that is pushed and logged, not run or polled
here. `src/conditions.py` and `SOURCE_NOTES.md` are ready for that session whenever
the operator wants to start it.

## 2026-08-17 - four more pre-GPU checks: mapping provenance, all-20-tasks dump, system_minimized diff, history confound. Still no GPU.

Operator asked for four checks before any GPU time, plus freezing the final analysis.
All four checks ran against the real cloned tau2 source (same scratch clone/venv as
the earlier session today), CPU-only. Answered in order:

**1. Policy mapping provenance.** `ACTION_TO_SECTION` (`src/conditions.py:31-38`) is
a static, hand-built Python dict - written once by reading `data/tau2/domains/retail/
policy.md`'s 6 `##` headers and cross-referencing every distinct `action.name` in
`data/tau2/domains/retail/tasks.json` (grepped in full this session, 250+ matches).
Not model-derived (no LLM call picks the section at runtime), not keyword-matched
(no fuzzy/substring search against policy text) - `truncated_policy_for_task()`
(`conditions.py:82-96`) does exactly one dict lookup per task action name
(`ACTION_TO_SECTION.get(name)`, `conditions.py:92`) against the task's own
ground-truth `evaluation_criteria.actions`. Not circular. `ANALYSIS_PLAN.md` now
carries this provenance note in the mapping section.

**2. All 20 tasks x 6 conditions, not one.** Re-ran the payload dump
(`verify_all_tasks.py`, scratch) across the exact 20 task IDs `sweep.select_tasks
("retail", 20)` actually selects (seed 20260814): `['93','43','98','77','66','94',
'4','46','107','64','102','110','82','73','7','74','55','89','15','13']`. One real
bug caught and fixed along the way (test-script-only, not `conditions.py`):
`llm_agent.py` does `from tau2.utils.llm_utils import generate` at import time, so
patching `tau2.utils.llm_utils.generate` alone (as the single-task verify script did
earlier) doesn't reach the REAL, unpatched `_generate_next_message` used by 5 of 6
conditions - only `history_truncated`'s own patched copy (which does a fresh local
import each call) saw the fake. Fixed by also patching `tau2.agent.llm_agent
.generate`. Full 120-row dump saved to `tables/condition_payload_dump_20tasks.csv`.
**Flags: none** - every task's `policy_truncated` landed strictly between
`no_policy` (357 chars) and `full_context` (7056 chars); real range across the 20
tasks was 2311-4621 chars, confirming the mapping is actually task-sensitive, not a
constant.

**3. system_minimized is 2%.** Exact diff dumped via `difflib.unified_diff` against
the real `AGENT_INSTRUCTION` constant: 306 -> 163 chars, 143 chars removed (2.0% of
`full_context`'s 7056-char prompt) - exactly two sentences, "You are a customer
service agent that helps the user according to the <policy> provided below." and
"Try to be helpful and always follow the policy.", with the tool-call-format
instructions and "Always make sure you generate valid JSON only." kept verbatim (kept
deliberately, see the earlier session's note on why). Determination: **labeled a
negative control in `ANALYSIS_PLAN.md`, not a content ablation** -
`AGENT_INSTRUCTION` already was the entire non-policy instructions block, so there
was nothing more to strip without confounding tool-call comprehension; at 2% of the
prompt with the full policy document still present, a near-zero effect is the
expected, hypothesis-neutral result, not a broken ablation.

**4. history_truncated confound check.** Confirmed by reading `LLMAgentState`
(`tau2/agent/llm_agent.py:44-48` in pinned source): `system_messages` and `messages`
are separate fields; the patched `_generate_next_message` only slices `state.messages
[-k:]` and always prepends the full `state.system_messages` unsliced
(`conditions.py`'s `patched_generate_next_message`). Empirically confirmed too - real
role sequence sent to the model on the final turn of a 6-turn probe, k=3:
`['system', 'user', 'assistant', 'user']` - system message always present, always at
index 0, count exactly 1, and exactly 3 non-system messages regardless of how much
history had accumulated (12 messages were stored in `state.messages` by that point).
**No confound**: the system message (which carries the policy, per `SYSTEM_PROMPT`'s
`<policy>` block) is never inside the truncation window at any k. `history_truncated`
ablates conversation history only, not policy.

**5. Froze the final analysis.** `ANALYSIS_PLAN.md` now has a `FROZEN` section
(metric, test, alpha=0.05, Holm correction, and an operationalized falsification
criterion for "policy is inert" using this grid's real 0.20 minimum-detectable-delta
floor from `tables/detectability.csv`) - written before any GPU data exists, marked
not to be edited after seeing results; deviations get a new dated section instead.
Grid size marked DECIDED in `ANALYSIS_PLAN.md` (`n_tasks=20, n_seeds=2`, 240
episodes) - the operator's call, recorded as such, not re-litigated here.

**Still no GPU spent, no git commits.** Snapshot follows.

## 2026-08-17 - pilot pushed to Kaggle. First real GPU time this project, not waited on.

Operator's 3 pre-push checks, then build+push the pilot, per their explicit brief.

**1. Max-step cap.** Already identical across all 6 conditions with no code change
needed - confirmed by reading `conditions.py` in full: no condition, patched function,
or module-level patch ever touches `max_steps`/`TextRunConfig.max_steps`, only
policy/tools/instructions/history. `sweep.py`'s `MAX_STEPS = 200` comment upgraded
with real provenance (two independent sources, not invented): (1) `tau2.config
.DEFAULT_MAX_STEPS` verified directly against the real cloned source this session
(`tau2/config.py:4` = `200`, not just trusted from an inherited comment), and (2)
simswap's own real observed episodes (`F:/simswap/data/runs/*.json`, 7 real runs,
read-only reference - never touched that repo) - `n_turns` ranged 30-201 across
those, and `qwen_task2.json` genuinely hit `termination_reason="max_turns"` at
`n_turns=201`, so 200 is a bound that was actually exercised, not theoretical.
`analyze.py::accuracy_by_cell()` got an explicit doc-comment stating the invariant
already true in the code: it filters only on `status`, never on `termination_reason`
- a `max_turns`-terminated episode is `status="completed"` (harness got a scoreable
result) and scores its partial actions like any other completed run. Confirmed by
reading the function in full, not just asserted.

**2. Holm family confirmed.** `ANALYSIS_PLAN.md`'s frozen section already lists all
5 non-control conditions in the correction family, `system_minimized` included:
"Holm-corrected across the up-to-5 non-control conditions' raw p-values (`no_policy`,
`policy_truncated`, `schema_names_only`, `history_truncated`, `system_minimized`)."
Left as originally frozen, stated back per the operator's request, not re-edited.

**3. `policy_truncated` labeled an ORACLE retrieval condition** in `ANALYSIS_PLAN.md`
- the kept section comes from the task's own ground-truth `evaluation_criteria
.actions`, which neither the agent under test nor any real retriever would have
access to. Documented as an upper bound on what perfect retrieval could achieve, not
a retrieval simulation - a small drop here supports "most of the document is inert"
but says nothing about whether a real (imperfect) retriever would be safe in
practice.

**Pilot built.** `src/pilot_core.py` (moved from an initial `kernels/` draft to match
simswap's own established layout - the "core" runner lives in `src/` alongside
`schema.py`/`sweep.py`/`conditions.py` so a flat dataset upload of `src/`'s contents
carries everything the kernel needs) + `kernels/pilot_run.py` (thin Kaggle entry
point) + `kernels/kernel-metadata.json` / `src/dataset-metadata.json` (real Kaggle
username filled in, `abhiprd2000`, not templates) + `kernels/README.md`. Runs all 6
conditions once each, same task (`sweep.select_tasks("retail", 20)[0]`, i.e. an
actual task from the real 20-task grid, not arbitrary), same seed (0), against one
pair of vLLM 0.27.1 servers (Qwen2.5-7B-Instruct-AWQ self-play, `--quantization awq`
- not `awq_marlin`, T4/SM75 has no Marlin kernel, same constraint simswap's own real
T4 runs already verified) started once for the whole session, through
`sweep.run_cell()` - the exact function the real 240-episode sweep calls, just
invoked 6 times here instead of 240. Ported `patch_evaluator_crash_guard` from
simswap's `gate1_core.py` (read-only reference) - re-verified independently this
session against this project's own tau2 clone (`tau2/runner/simulation.py`, read in
full while writing `SOURCE_NOTES.md`) that `run_simulation` has no try/except around
`evaluate_simulation`, same finding, same fix, reused not re-guessed. Pilot-only,
flagged-not-decided choice: `history_truncated`'s `condition_param="3"` for this
pilot - `ANALYSIS_PLAN.md` does not yet pin the real sweep's k value.

**Pushed - first real GPU time this project has spent.**
- Dataset: `kaggle datasets create -p src` -> `abhiprd2000/attrib-code`, confirmed
  `status: ready` before the kernel push (checked once, not polled in a loop).
- Kernel: `kaggle kernels push -p kernels` -> **`abhiprd2000/attrib-pilot-run`,
  version 1**, https://www.kaggle.com/code/abhiprd2000/attrib-pilot-run
- One status check immediately after push (not a wait/poll loop): `kaggle kernels
  status abhiprd2000/attrib-pilot-run` -> `KernelWorkerStatus.RUNNING`, confirmed at
  push time. Not checked again after that - per the standing rule, next check happens
  when the operator brings back the output, not before.

No git commits. `F:/simswap` was read from (kernel patterns, real observed run
records for the max_steps and quantization provenance above) but never written to.
Snapshot follows, then stop per the operator's instruction.

## 2026-08-17 - blind pre-registration while the pilot runs. Kernel not touched.

Operator's brief, explicit: pilot kernel (`abhiprd2000/attrib-pilot-run`) was
RUNNING on Kaggle for this entire session - not polled, not re-pushed, confirmed by
never calling `kaggle kernels status`/`push` this session (grep the session's own
tool calls if in doubt - only local file edits and one local Python smoke test ran).

**1. `termination_reason`/`n_turns` first-class fields - confirmed, no fix needed.**
Read `schema.py`'s `RunRecord` dataclass and both `sweep.py::convert_success()` /
`convert_failure()` in full: `termination_reason: str` (`schema.py:92`) and
`n_turns: int` (`schema.py:97`) are plain top-level dataclass fields, both actually
populated by every code path that builds a `RunRecord` (`convert_success` at
`sweep.py:236,241`, `convert_failure` at `sweep.py:276,281`).
`RunRecord.to_dict()` is a bare `dataclasses.asdict(self)` - top-level fields
serialize as top-level JSON keys, nothing nested inside `reward_info_note` or any
other blob. Nothing to fix; the pilot already running writes records this schema
already guarantees are safe to re-read.

**2. `ANALYSIS_PLAN.md`'s "PRE-REGISTERED SECONDARY" section written, timestamped,
blind** - the pilot kernel was confirmed RUNNING (not yet finished, output not read
in any form) at the time this was written. States plainly: primary stays frozen and
unedited (truncated episodes included, scored as-is); truncation rate is a
first-class per-condition result, not a footnote; a secondary robustness check
(accuracy excluding truncated episodes) is reported alongside the primary
unconditionally, agreement or not; and the >10-absolute-point truncation-rate-gap
confound rule is fixed now, before any number exists that could motivate picking a
more convenient threshold later.

**Not just written - implemented and smoke-tested**, matching this project's
standing practice of never citing an analysis method that isn't real, working code
(same discipline `detectability.py`/`analyze.py`'s existing frozen pipeline were
built under). `analyze.py` changes:
- `accuracy_by_cell()` gained an `exclude_truncated` parameter (default `False`,
  preserving the frozen primary's exact original behavior) - `analyze.py:45-75`.
- New `truncation_rate_by_condition()` (`analyze.py:77-99`) and
  `check_truncation_confound()` (`analyze.py:102-122`,
  `TRUNCATION_CONFOUND_THRESHOLD = 0.10`).
- `main()`'s contrast logic factored into a shared `contrast()` helper
  (`analyze.py:151-189`) so the primary and secondary runs literally cannot diverge
  in method, only in which records feed them - called twice, writes both
  `tables/condition_contrasts.csv` (primary, unchanged filename/shape) and the new
  `tables/condition_contrasts_excl_truncated.csv`, plus
  `tables/truncation_rate_by_condition.csv`.
- **Smoke-tested for real** against `fake_runs.py --scenario policy_inert --delta
  0.20` (240 fake records, CPU-only, no GPU, in a scratch dir - not written into this
  repo's `runs/`, which stays empty until real data exists): truncation rates printed
  correctly (0.0-2.7% across conditions in that run, correctly flagged as no
  confound), primary and secondary contrasts both computed and written, closely
  agreed as expected since truncation was rare and condition-independent in the
  synthetic data. Additionally unit-checked `check_truncation_confound()` directly on
  hand-built inputs: a 7-point gap correctly produced no warning, a 15-point gap
  correctly fired, and a `None` rate (zero completed runs) was correctly skipped
  without crashing.

**Corrected two stale file:line citations in the FROZEN primary section** -
`paired_permutation_test` and `holm_correct` physically moved (function bodies
byte-identical, confirmed by re-reading both in full before and after) when the new
functions above were inserted ahead of them in the file. This is a citation-location
fix only, explicitly not a substance edit: the frozen test, alpha, correction method,
and falsification criterion are all unchanged text. Flagging this distinction
explicitly since the operator's instruction was "do not edit the FROZEN primary" -
this was the one edit made to that section, and it changes zero science.

**3. `MAX_STEPS` not touched.** No edit to it this session (previous session's `200`
plus its provenance comment stand as written).

No git commits. Kaggle kernel not polled or re-pushed at any point this session.
Snapshot follows, then stop and wait for the operator to bring back pilot output, per
their explicit instruction.

## 2026-08-17 - pilot output pulled and read. Real GPU-hours estimate revised upward. Grid NOT launched.

Kernel `abhiprd2000/attrib-pilot-run` confirmed `KernelWorkerStatus.COMPLETE` (one
status check, not a poll), output pulled via `kaggle kernels output`. All 6 condition
records + transcripts + logs downloaded to a scratch dir, read in full. Real numbers
below, nothing invented.

**1. Per-condition table, task 93, seed 0, all 6 conditions - every condition wrote a
valid, schema-passing record (none reached `convert_failure` - schema `__post_init__`
would have rejected a malformed one and no `status` other than `"completed"`
appears):**

| condition | wall_clock_s | n_turns | termination_reason | record written |
|---|---|---|---|---|
| full_context | 262.41 | 102 | agent_error | yes |
| no_policy | 27.13 | 14 | normal | yes |
| policy_truncated | 39.73 | 24 | normal | yes |
| schema_names_only | 1093.18 | 201 | **max_turns** | yes |
| history_truncated | 174.35 | 78 | agent_error | yes |
| system_minimized | 101.43 | 58 | agent_error | yes |

**Flags**: `schema_names_only` hit `max_turns` (201 turns = the full 200-step cap,
`reward_info_note`: "Simulation terminated prematurely. Termination reason:
max_steps") - the ONE condition that ran out of room. 4 of 6 conditions
(`full_context`, `history_truncated`, `system_minimized`, and implicitly counted
above) ended `termination_reason=agent_error` (tau2's own `too_many_errors` -
`reward_info_note`: "Termination reason: too_many_errors") - a real tau2-level
behavioral outcome, distinct from a harness failure (`status` stayed `"completed"`
for all 6, `status="error"`/`"timeout"` never occurred - zero harness crashes,
zero missing records). Worth noting plainly: at n=1 per condition this is not yet
interpretable as "the agent errors more under condition X" - it is one data point
per condition, exactly the caveat item 2 exists to enforce.

**2. Plumbing sanity check - served payload vs the CPU-verified dump, PASSED.**
`accuracy`/`task_success` are NOT reported as findings (n=1 per condition, per the
operator's explicit instruction) - what was actually checked is whether the real
served request matched what `SOURCE_NOTES.md`'s CPU-only verification predicted.
Real litellm-reported `usage.prompt_tokens` on each condition's first substantive
agent-role LLM call (task 93):

| condition | CPU dump: system_prompt chars | CPU dump: tools w/ descriptions | real first-call prompt_tokens |
|---|---|---|---|
| full_context | 7056 | 16/16 | 4923 |
| no_policy | 357 | 16/16 | 3516 |
| policy_truncated | 3136 | 16/16 | 4085 |
| schema_names_only | 7056 (unchanged) | 0/16 | 3255 |
| history_truncated | 7056 (unchanged, 1st turn) | 16/16 | **4923 (bit-identical to full_context)** |
| system_minimized | 6913 | 16/16 | 4894 |

Every relative ordering matches: `no_policy < policy_truncated < full_context`
(policy-stripping direction and monotonicity correct); `history_truncated`'s first
call is bit-for-bit identical to `full_context`'s (correct - nothing to truncate yet
on turn 1, exactly as `SOURCE_NOTES.md` predicted); `system_minimized` sits just
below `full_context` (29 fewer tokens, consistent with the real 143-char/two-sentence
diff already dumped). One real, plumbing-relevant (not accuracy-relevant) surprise:
`schema_names_only`'s real token count (3255) was the LOWEST of all six, lower even
than `no_policy` - stripping all 16 tools' descriptions removed more prompt content
than stripping the entire policy document did, for this task. Plausibly connected to
why that condition alone ran to `max_turns` (201 turns) - flagged as a real
observation about prompt composition, explicitly NOT claimed as an explanation of
behavior or an accuracy finding.

**3-4. GPU-hours arithmetic - real numbers, computed via script, not by hand.**
Sum of the 6 real `wall_clock_seconds` above (one full pass through all 6 conditions,
n=1 each) = **1698.24s**. The real grid is 40 episodes per condition
(`n_tasks=20 x n_seeds=2`), so: `40 x 1698.24s = 67,929.4s = **18.87 GPU-hours**`.

Compared to the prior estimate: PROGRESS.md's earlier `~14.3h` was `240 x 214.92s`
(simswap's `full_context`-only mean, applied uniformly to all 6 conditions - always
flagged as an assumption, never a measurement, until now). The real, condition-
specific pilot numbers come in **31.7% higher** (+4.54h) than that assumption -
almost entirely because `schema_names_only`'s single real observation (1093.18s,
because it hit `max_turns`) is ~5x the assumed-uniform 214.92s and is 64% of the
6-condition sum by itself.

**Does it fit 2 Kaggle sessions at 9h each (18h budget)? NO.** `18.87h > 18.0h` -
deficit of **-0.87h (52 minutes over)**, using this exact per-condition-mean model.
Not rescoped silently - reporting this and stopping per the operator's explicit
instruction not to launch the real grid.

**The single biggest source of uncertainty in this whole estimate**: `schema_names_
only`'s 1093.18s is ONE observation, on ONE task, and it happened to be the one
condition that hit the step cap. Whether that's representative of `schema_names_
only`'s true behavior across the other 19 tasks, or an artifact of task 93
specifically being unusually hard without tool descriptions, is unknown at n=1 -
this dominates the 18.87h estimate more than any of the rescoping choices below do.

**Options, real cost of each, none applied:**

| option | total estimated GPU-hours | fits? | slack |
|---|---|---|---|
| as-is (`n_tasks=20, n_seeds=2`), 2 sessions (18h) | 18.87h | **NO** | **-0.87h** |
| as-is, 3rd session added (27h budget) | 18.87h | yes | **+8.13h** (30% margin, room for a substantial partial rerun) |
| `n_tasks=19, n_seeds=2` (drop 1 task), 2 sessions | 17.93h | yes, barely | **+0.07h** (~4 minutes - no real margin, not safe given the n=1 uncertainty above) |
| `n_tasks=20, n_seeds=1` (drop to 1 seed), 2 sessions | 9.43h | yes | **+8.57h** (huge margin, room for a full rerun) |

Costs not captured in the hours column: a 3rd session means more Kaggle GPU quota
consumed and another manual push/monitor cycle - remaining weekly quota not checked
this session, worth confirming before committing. Dropping to `n_tasks=19` buys
almost nothing (1 fewer task) for essentially zero real safety margin - not
recommended given how noisy this estimate already is. Dropping to `n_seeds=1` roughly
halves the number of paired `(task_id, seed)` cells feeding the frozen permutation
test per condition (up to 40 -> up to 20) - `tables/detectability.csv` was computed
at `n_seeds=2` and does NOT currently have a number for `n_seeds=1`'s minimum
reliably-detectable delta; `detectability.py` would need to be rerun before this
option's real statistical cost is known, not assumed comparable to the `n_seeds=2`
numbers already on file.

Real pilot output (6 `RunRecord`s, transcripts, `pilot_summary.json`, `pilot_log.txt`)
copied into the repo at `data/pilot_run_20260817/` so it survives in the snapshot,
not just the session's scratch dir - matches simswap's own `data/runs/` convention
for keeping real observed episodes (read-only reference).

**Grid NOT launched.** No `sweep.py` run started, no new Kaggle kernel pushed, no
commits. Snapshot follows, then stop, per the operator's explicit instruction.

## 2026-08-17 - diagnosed agent_error before touching the grid. Correction to my own prior count.

**Correction first**: the prior session's report said "4 of 6 conditions... ended
`termination_reason=agent_error`" and only named 3 (`full_context`,
`history_truncated`, `system_minimized`). That was wrong - re-verified against
`pilot_summary.json` directly: **3 of 6 are `agent_error`**, 1 (`schema_names_only`)
is `max_turns`, 2 (`no_policy`, `policy_truncated`) are `normal`. 3+1+2=6. The
operator's message this turn repeated my mistaken "4" - flagging so the correction is
visible, not silently absorbed.

**1. What actually errored - read every agent_error transcript in full, not
sampled.** All 3 (`full_context`: 20 tool calls/10 errors; `history_truncated`: 13/10;
`system_minimized`: 10/10 - every single tool call it made failed) hit exactly 10
`ToolMessage(error=True)` results, matching `max_errors=10` precisely. **Classified
all 30 errors: zero are malformed JSON, zero are hallucinated tool NAMES, zero are
bad argument schemas.** Every tool name used is real (`exchange_delivered_order_
items`, `get_order_details`, `find_user_id_by_email`, `get_product_details`,
`get_item_details`) and every call's arguments are well-typed and well-formed. **All
30 errors are business-logic/lookup failures** - `Error: Order not found` (7x),
`Error: Non-delivered order cannot be exchanged` (3x), `Error: User not found` (1x),
`Error: Product not found` (2x), `Error: Item not found` (2x), plus repeats - the
agent supplies plausible-looking but fabricated order/item/product/user IDs (e.g.
`123456789`, `#L123456789`, `ABC123`/`DEF456`, `jane.doe@example.com`) instead of
obtaining real ones from the user or a correct lookup path, then keeps retrying with
different fabricated IDs until it hits the cap. This is "something else" on the
operator's classification list - hallucinated ENTITY VALUES, not a parsing/schema/
tool-name problem. Full verbatim transcript excerpts (last 3 agent messages + every
tool error, per condition) captured in this session's scratch output, not repeated
here in full - see `data/pilot_run_20260817/transcripts/` directly for the source.

tau2 source, verified against the pinned clone (also now cited in `ANALYSIS_PLAN.md`):
`orchestrator.py:326-327` (`if tool_result.error: self.num_errors += 1`),
`orchestrator.py:747-749` (half-duplex path - `if self.num_errors >= self.max_errors:
... TerminationReason.TOO_MANY_ERRORS`), threshold `max_errors=10`
(`tau2/config.py:5`, `DEFAULT_MAX_ERRORS = 10` - `TextRunConfig` never overrides it).

**2. Compared to simswap (read-only) - simswap's rate is HIGHER, not lower.**
`F:/simswap/data/runs/qwen_task{0,1,2,3}.json` (the only 4 real Qwen2.5-7B-Instruct-
AWQ self-play records that repo has): `qwen_task0`=agent_error, `qwen_task1`=
agent_error, `qwen_task2`=max_turns, `qwen_task3`=agent_error - **3/4 = 75%
agent_error**, 1/4 max_turns, **0/4 normal**. ATTRIB's pilot (3/6=50% agent_error) is
actually LOWER than simswap's own published-model precedent. Since simswap's rate is
not "materially lower," the operator's conditional "list every difference between
serving configs" branch doesn't strictly trigger - checked anyway for due diligence:
`--dtype float16`, `--gpu-memory-utilization 0.90`, `--tool-call-parser hermes`,
`--enable-auto-tool-choice`, `--max-model-len 32768`, no `--chat-template` (none
needed for Qwen), tau2 commit `fc0055dc4e0a316c3f83133267fbd6faaa770992` - all
identical between the two. One real difference found: ATTRIB's `pilot_core.py`
passes `--quantization awq` explicitly; simswap's `gate1_run.py` hardcodes
`agent_quantization=None` (leaves it to vLLM's auto-detect) for every run through
that entry point, including the real qwen self-play runs. Neither crashed, meaning
auto-detect evidently resolved to the same working kernel on T4 either way - noted,
not treated as an explanation, since it doesn't crash-differ and the whole point of
the explicit flag (per `gate1_core.py`'s own comment) was avoiding an unsafe
`awq_marlin` auto-pick, which didn't happen in either project's real runs.

**Additional real observation, not asked for but relevant, flagged not fixed**:
`agent_vllm_gpu0_port8001.log:55` - vLLM logged "Default vLLM sampling parameters
have been overridden by the model's `generation_config.json`:
`{'repetition_penalty': 1.05, 'temperature': 0.7, 'top_k': 20, 'top_p': 0.8}`."
ATTRIB's harness explicitly sets `temperature=0.0` on every call
(`llm_args={"temperature": 0.0, ...}`, `sweep.py`), which should override this
default per-request - but `top_p`/`top_k`/`repetition_penalty` are never explicitly
set by the harness, so those fall back to the model's tuned values rather than
vLLM's neutral ones. At `temperature=0.0` (greedy/argmax decoding), `top_p`/`top_k`
should be moot, but `repetition_penalty=1.05` is NOT moot - it still perturbs the
logits before the deterministic argmax choice. **Confirmed simswap's `sweep.py` has
the exact same `llm_args` construction (`{"temperature": 0.0, "timeout": ...}`,
`F:/simswap/src/sweep.py:145`)** - this is a shared, pre-existing characteristic of
both projects' serving config, not something ATTRIB introduced. Not verified
causally against the hallucinated-ID pattern above (would need an actual A/B test
with `--generation-config vllm` to know) - flagged as a real, previously-undocumented
fact about the serving config, not claimed as an explanation.

**3. Tool-call parser - confirmed correctly set, from the real log, not just
source.** `pilot_log.txt:182-183` (the actual command that ran, both GPUs):
```
python3 -m vllm.entrypoints.openai.api_server --model Qwen/Qwen2.5-7B-Instruct-AWQ
  --dtype float16 --port 8001 --gpu-memory-utilization 0.90 --max-model-len 32768
  --quantization awq --enable-auto-tool-choice --tool-call-parser hermes
```
Both `--enable-auto-tool-choice` and `--tool-call-parser hermes` ARE present. Grepped
both vLLM server logs for parse-failure warnings - none found. Combined with item 1's
finding (every failed tool call was a clean, correctly-parsed, correctly-named,
correctly-typed call that failed on data lookup, not parsing) - **the tool-call
parser is not the cause of these terminations.** Not patched, because nothing here
indicates it is broken.

**4. `ANALYSIS_PLAN.md`'s "PRE-REGISTERED: agent_error handling" section written**,
timestamped, extending the already-frozen `max_turns` treatment by direct symmetry
(same threshold, same structure) rather than by looking at what would make this
pilot's numbers read better - disclosed that `action_accuracy`/`reward`/`read`/
`write` values were incidentally visible while reading these same JSON files for the
plumbing diagnosis, so strict blindness wasn't literally maintained, but the rule
itself was derived from principle, not reverse-engineered from a number. Implemented
in `analyze.py`, not just described: `accuracy_by_cell()` gained an independent
`exclude_agent_error` flag (defaults `False`, `max_turns`' existing flag and the
frozen primary's behavior both fully unchanged); new `agent_error_rate_by_condition()`
and `check_agent_error_confound()` (`AGENT_ERROR_CONFOUND_THRESHOLD = 0.10`, same
value as the `max_turns` rule) mirror their `max_turns` counterparts exactly;
`main()` now reports THREE accuracy views every run - PRIMARY (all), SECONDARY (excl.
`max_turns`, already frozen, unchanged), and a new THIRD (excl. both `max_turns` AND
`agent_error` - normal terminations only), writing `tables/agent_error_rate_by_
condition.csv` and `tables/condition_contrasts_normal_only.csv`. Smoke-tested against
the same `fake_runs.py` synthetic data as before - all three views compute and write
cleanly (fake data has 0% `agent_error` throughout, since `fake_runs.py` doesn't
simulate it, so the new confound check correctly reports "none detected" there,
consistent with a straightforward no-op on data that doesn't exercise it). Two stale
file:line citations in the FROZEN primary section were corrected AGAIN (`holm_correct`,
`paired_permutation_test` moved a second time when the new functions were inserted
ahead of them) - bodies re-verified byte-identical, citation-only fix, logged the
same way as last time.

**5. Recommendation: accept `agent_error` as real, do NOT rerun the pilot chasing a
config fix.** Basis: (a) the parser is confirmed correctly configured from the actual
log; (b) every single tool error is a clean, correctly-formed call failing on
fabricated entity IDs, not a parsing/schema/tool-name defect - there is no config bug
to fix here; (c) simswap's own real, already-relied-upon Qwen2.5-7B-Instruct-AWQ
self-play data shows an EVEN HIGHER agent_error rate (75% vs ATTRIB's 50%) under a
near-identical config - this is consistent, expected behavior for this model on this
benchmark, not an ATTRIB-specific misconfiguration. The correct response is what item
4 above already did: pre-register how `agent_error` is scored and when it signals a
confound, so the real grid's analysis handles it correctly - not chase a fix for
behavior that isn't a bug. Residual honesty: n=1 per condition, the sampling-param
observation above is flagged but not causally verified, and I cannot rule out some
other config difference I did not find - stated plainly, not papered over.

**Grid still NOT launched.** No `sweep.py` run, no new kernel push, no commits.
Snapshot follows, then stop.

## 2026-08-17 - floor check: the primary metric may have no headroom. Grid still not launched.

Operator's framing, now confirmed with real data: the 30 fabricated-ID tool errors
diagnosed above are simswap's own grounding failure surfacing as termination in this
harness. That reframes the floor question as urgent - answered first, before
anything else.

**1. FLOOR CHECK - answered directly: yes, baseline accuracy is at or below the
floor.** `full_context`'s pilot record (n=1, floor probe only, not a finding):
`task_success=false`, `reward=0.0`, `read={count:0,correct:0}`,
`write={count:0,correct:0}` - `action_total=0`, so `RunRecord.action_accuracy` is
**`None`** (undefined), not merely "near zero." Read all 6 pilot records: 4/6
(`full_context`, `schema_names_only`, `history_truncated`, `system_minimized`) have
`action_total=0` -> `action_accuracy=None`; the other 2 (`no_policy`,
`policy_truncated`) have exactly 1 write action attempted, 0 correct ->
`action_accuracy=0.0`. Zero of 6 pilot episodes have a positive `action_accuracy`.

**Read `F:/simswap/data/runs/qwen_task{0,1,2,3}.json` in full (read-only) - all 4 of
simswap's own real Qwen2.5-7B-Instruct-AWQ episodes**: `task_success=false`,
`reward=0.0`, `read={0,0}`, `write={0,0}` in every single one -
`action_accuracy=None` for all 4. **10 of 10 real episodes across both projects
(this pilot's 6 + simswap's 4) have `action_accuracy` that is either undefined or
exactly zero. Zero exceptions.**

**Stated plainly, as instructed**: if this pattern holds across the wider grid,
`accuracy_by_cell()` (`analyze.py:45-87`) will exclude most cells from the primary
contrast entirely (it only pairs cells where `action_accuracy is not None`), and
whatever cells DO pair will mostly be pairing zero against zero. **No ablation
condition can show a "drop" below a floor the control has already hit.** As designed,
with this primary metric, the frozen contrast may not be measuring degradation at
all - it may just be measuring "how often does at least one action get attempted,"
which is a different, narrower question than the one `ANALYSIS_PLAN.md`'s hypothesis
is asking. This is real evidence from every real episode either project has ever run,
not n=1 noise - flagging it as exactly the failure mode the operator asked about,
not softening it.

**2. CENSORING - confirmed, and now labeled where it appears.** Read `grounding.py`
in full: `analyze_conversation()` and `pool_by_condition()` both compute proper RATES
(`ungrounded_fraction_by_argument`, `..._by_call`, `..._by_argument_conditional` -
always `ungrounded / total`, at both per-conversation and per-condition-pooled
level, the pooled version summing raw counts THEN dividing, not averaging fractions).
Raw counts (`total_id_arguments`, `ungrounded_id_arguments`, `total_tool_calls`,
etc.) DO appear in the output dicts alongside the rates, for transparency/debugging -
but they were not previously labeled as bounded by episode length. **Added that
label now** (`grounding.py`, both `analyze_conversation()`'s return dict and
`pool_by_condition()`'s): a condition whose episodes hit `max_errors=10` sooner gets
fewer total tool calls and therefore a smaller RAW ungrounded count independent of
its true error rate - only the `ungrounded_fraction_*` fields are valid for
cross-conversation/cross-condition comparison, the raw counts are not. Comment-only
change (dict-literal documentation, no logic touched) - re-ran `tests/
test_grounding.py`: still 6/6 passing.

**3. METRIC PROPOSAL - argued both ways, NOT implemented, `ANALYSIS_PLAN.md`'s
FROZEN primary section NOT touched, per instruction.**

*For* moving primary to per-call grounding rate: (a) it has real headroom where
`action_accuracy` does not - even a fully agent_error-terminated episode makes 10-20
tool calls before hitting the cap (per this pilot's transcripts), so grounding rate
is computable in cases where `action_accuracy` is `None`; (b) it already has a
real, large observed effect size in this codebase's own lineage -
`ANALYSIS_PLAN.md`'s existing Detectability section cites simswap's "own conditional
grounding gap was 34.9 points" as a real prior (not invented this session, already on
record); (c) it directly targets the mechanism just diagnosed - the 30 pilot tool
errors are ALL fabricated-ID failures, which is exactly what grounding rate measures,
so it would turn a currently-invisible-to-`action_accuracy` failure mode into the
headline signal instead of a downstream truncation event; (d) it is already fully
implemented and validated (11 regression tests inherited from simswap + this
project's own 6, deterministic per-conversation property, not a statistical estimate
needing its own detectability sweep the way `action_accuracy` did).

*Against*: (a) it is a genuinely different construct, not a drop-in replacement -
`action_accuracy` measures task-completion correctness against tau2's ground truth
(what the paper currently claims to explain), grounding rate measures whether ID
arguments trace to prior content; a model could ground perfectly and still fail the
task for unrelated reasons (wrong policy decision, wrong tool, wrong order), so
switching changes what the paper is actually about, not just how it's measured; (b)
grounding rate's own denominator (`total_id_arguments[_conditional]`) is STILL
shaped by episode length per item 2 above - conditions that error out fast get fewer
observations and a noisier per-condition rate estimate, so switching does not remove
every censoring-adjacent concern, only the "raw count" version of it; (c) no
detectability analysis has ever been run for grounding rate as the primary test
statistic - `detectability.py` and its 0.20-delta grid-size floor are specific to
`action_accuracy`; adopting grounding rate as primary without redoing that analysis
would reintroduce exactly the "we don't know if this grid can detect what we're
claiming to test" gap the whole exercise existed to close; (d) the Hypothesis and
Falsification-criterion sections of `ANALYSIS_PLAN.md` are written in
`action_accuracy`'s vocabulary (accuracy drop, 0.20 floor) - switching is a real
rewrite of those sections, not a substitution of one number for another.

**My recommendation, offered as requested, not acted on**: given 10/10 real episodes
across two independent projects show `action_accuracy` at or below the floor with
zero exceptions - not a marginal or n=1 concern - I lean toward promoting grounding
rate to at least CO-PRIMARY status (report both, every run, not one replacing the
other) rather than a silent full swap, since `action_accuracy` still answers a real
question (task completion) that grounding rate cannot answer on its own. A full
primary swap is defensible too, but carries the real, non-trivial follow-up cost of
re-deriving Detectability, Hypothesis, and the falsification criterion for the new
statistic - not a costless edit. This is the operator's call, not resolved here.

**4. `ANALYSIS_PLAN.md`'s "LIMITATIONS" section added**, stating plainly that the
`agent_error` handling rule (previous session) was written after this pilot's
accuracy numbers had already been read - a disclosed deviation from this project's
otherwise-blind pre-registration discipline, not a clean one - and that the floor
problem surfaced here is a live, undecided question the FROZEN primary section does
not yet address. The FROZEN section itself remains unedited and in force.

**Grid still NOT launched.** No `sweep.py` run, no new kernel push, no commits.
Snapshot follows, then stop.

## 2026-08-17 - the metric rewrite, done for real. Root cause of the floor: evaluator, not tasks.

Resumed after a usage-limit checkpoint mid-session (several background shell tasks
were killed mid-flight, including a `pilot_core.py` move). Operator required item 0
(repo-state / data-loss check) to close before anything else.

**Item 0 - data was NOT lost.** `pilot_core.py` lives only at `src/pilot_core.py`
(the move completed before the kill; no duplicate in `kernels/`).
`F:\attrib\data\pilot_run_20260817\` exists (`dir` confirmed directly) with all 6
records + 6 transcripts + `pilot_log.txt` + `pilot_summary.json`, byte-sizes matching
the original download exactly, every file parses as valid JSON, no missing required
fields (`condition`/`wall_clock_seconds`/`n_turns`/`termination_reason`/`status` all
present in all 6). An earlier `Glob` call on a specific nested double-star pattern
returned a false "no files found" - a tool quirk on that path form, not real data
loss; a plain listing and a targeted `Glob` both found everything immediately after.

**Item 1 (root cause) - `action_total=0` is neither "agent never called a write
tool" nor "task has zero expected actions." It's a third thing: tau2's evaluator
refuses to score ANY action for any non-`agent_stop`/`user_stop` termination,
regardless of the task.** `tau2/evaluator/evaluator.py:119-129`
(`evaluate_simulation`): `if simulation.termination_reason not in {AGENT_STOP,
USER_STOP}: return RewardInfo(reward=0.0, reward_basis=None, info={"note": f"...
{termination_reason}"})` - a hard gate BEFORE `ActionEvaluator.calculate_reward()`
(`evaluator_action.py:68-109`, the only code that ever builds `action_checks` from a
task's golden actions) is ever reached. Verified NOT a task-selection bug: wrote and
ran a script reading all 20 real selected tasks (`sweep.select_tasks("retail", 20)`)
- **20/20 have non-empty `evaluation_criteria.actions`**, ranging 1 (task 93) to 13
(tasks 4, 55) golden actions, zero empty, zero missing criteria. Cross-tab, this
pilot's 6 episodes: `normal` -> 2/2 defined, `agent_error` -> 0/3 defined, `max_turns`
-> 0/1 defined - perfect correlation with `termination_reason` in this pilot.
**This is differential attrition** (`termination_reason` plausibly correlates with
`condition`, and whether `action_accuracy` exists at all depends entirely on
`termination_reason` via this evaluator gate) - written into `ANALYSIS_PLAN.md`'s new
"METRIC CHANGE" section as the stated reason, not "a floor."

**Item 2 (detectability, real curve, not one number) - run for real, `n_tasks=20,
n_seeds=2`:**

| delta | power (60 replicates) |
|---|---|
| 5pp | 0% |
| 10pp | 30% |
| 15pp | 92% |
| 20pp | 100% |
| 30pp | 100% |

Minimum detectable delta at 80% power: **15 percentage points**. `src/
detectability_grounding.py` (new, `detectability.py` untouched - separate script,
same "archive don't delete" discipline applied to code) - same `analyze.py` machinery
(`paired_permutation_test`/`holm_correct`, strict all-5-Holm-significant criterion),
anchored on two real numbers: baseline grounded rate 0.611 (this project's own real
`full_context` pilot episode, 1 - 38.9% conditional-ungrounded) and the real observed
per-episode ID-argument-count range (7-44, this pilot's 6 episodes). **simswap's
34.9pp Qwen-vs-Llama gap was explicitly NOT tested as a single verdict** - removed
from the deltas list per the operator's correction that it's a cross-model
comparison, not a credible context-ablation prior; kept only as descriptive context
in the script's docstring, clearly labeled as such.

**Item 3 (unequal denominators) - pooled and per-episode-averaged rates DISAGREE on
the real pilot data, confirmed via the actual functions, not hand math.** New
`grounding.py::average_by_condition()` (mirrors `pool_by_condition`'s interface,
macro-averages instead of pooling; conversations with an undefined rate excluded
from that average, not counted as 0%/100% - same discipline as the pooled version).
Run against the pilot's 6 real transcripts, treated as one group: pooled FLAT=84.1%
ungrounded, averaged FLAT=91.7% (**7.6pp disagreement**); pooled CONDITIONAL=65.6%,
averaged CONDITIONAL=79.6% over the 3 episodes with a defined conditional rate
(**14.0pp disagreement**). Direction: pooling makes the group look better (lower
ungrounded%) because `full_context` (this pilot's best-grounded episode, also its
longest at 20 tool calls) dominates the pooled denominator, pulling it down relative
to its equal 1/6 weight in an average - confirms the operator's concern was real, not
hypothetical. `tests/test_grounding.py` still 6/6 passing after the addition.

**Item 4 (the rewrite) - done.** `ANALYSIS_PLAN.md`: grounding rate is now the SOLE
primary (Primary metric section rewritten, reports pooled AND averaged, every run);
`action_accuracy` demoted to a reported secondary carrying an explicit attrition note
citing the 10/10-real-episodes finding; Hypothesis and the Falsification criterion
rewritten in grounding vocabulary, threshold set to **15pp** (this grid's real MDE
from item 2 above - not the old accuracy-based 0.20, and not a threshold the grid
cannot detect, per the operator's instruction); a new "METRIC CHANGE" section states
the evaluator finding as the reason, with the cross-tab. **The complete original
accuracy-based plan (Hypothesis, Primary metric, Secondary metric, Attribution rule,
FROZEN final analysis) is preserved verbatim in a new `## ARCHIVED` block at the end
of the file** - relabeled with clear "ARCHIVED - original ..." headers, not deleted,
not rewritten in place. Timestamped 2026-08-17 throughout, explicitly stated as made
before any 240-episode grid data exists (only the 6-episode pilot, run for wall-clock
and plumbing, informs this).

**Backup note, stated per the operator's explicit instruction, not fixed**: this
repo has zero git commits by design (see the top of this file). Directory snapshots
(`F:\attrib-snap-YYYYMMDD-HHMM\`) are the ONLY backup mechanism. Between snapshots,
anything created or changed is unrecoverable if lost - there is no commit history to
fall back to. This is a real, standing fragility of this project's workflow, not
resolved here.

**Grid still NOT launched.** No `sweep.py` run, no new kernel push, no commits.
Snapshot follows, then stop.

## 2026-08-17 - two ANALYSIS_PLAN.md edits, then SESSION 1 launched. First real grid GPU time.

**1. Declared per-episode-averaged the primary grounding rate**, pooled a reported
secondary - `ANALYSIS_PLAN.md`'s Primary metric section, timestamped, before any
grid data exists. Reason stated: equal weight per episode means one long episode
cannot dominate a condition's denominator, unlike pooling (real, already-verified
14.0pp conditional-rate disagreement on the pilot data, from the previous session).
Also flagged, honestly, while there: `grounding.py` has no `{condition: {(task_id,
seed): rate}}` per-cell function yet (`accuracy_by_cell()`'s shape) - needed before
the real grid's significance test can actually run on grounding rate. Not built this
session; noted as a gap, not silently left implicit.

**2. MDE limitation added to `ANALYSIS_PLAN.md`'s LIMITATIONS section**: the 15pp
MDE rests on `BASE_SKILL_GROUNDED=0.611`, itself one pilot episode's (task 93,
`full_context`) conditional grounded rate, n=1. Stated plainly that the real MDE
differs if the true grid-wide baseline differs, and that there is no better estimate
until the grid runs - not re-derived, per the operator's explicit instruction.

**3. Session 1 built and launched - real numbers, computed before pushing:**

Sum of the pilot's 6 real per-condition wall-clocks (one full pass through all 6
conditions for one (task, seed) cell) = 1698.24s (`262.41 full_context + 27.13
no_policy + 39.73 policy_truncated + 1093.18 schema_names_only + 174.35
history_truncated + 101.43 system_minimized`, from `data/pilot_run_20260817/
pilot_summary.json`). Options computed via script before deciding:

| n_tasks (x2 seeds) | episodes | projected hours | margin under 9h |
|---|---|---|---|
| 6 | 72 | 5.66h | 3.34h |
| **7 (chosen)** | **84** | **6.60h (+~0.05h startup ~= 6.65h)** | **~2.35h (26%)** |
| 8 | 96 | 7.55h | 1.45h |

**Chose 7 tasks x 2 seeds = 14 cells x 6 conditions = 84 episodes** - first 7 of the
real 20-task selection (`sweep.select_tasks("retail", 20)`: `93, 43, 98, 77, 66, 94,
4`), both seeds, all 6 conditions per cell. Cell-based split (not condition-based) -
splitting by condition instead would have let `schema_names_only` alone (1093s/cell
in the pilot, by far the largest single contributor) blow the budget of whichever
session got it. Remaining 13 tasks (26 cells, 156 episodes) deferred to session 2+,
NOT decided or launched here - the full grid's real ~18.87h total needs at least 2
more sessions after this one.

**Built**: `src/grid_core.py` (new - reuses `pilot_core.py`'s `install_deps`/
`start_vllm`/`wait_for_server`/logging/VRAM-monitoring/evaluator-crash-guard
unchanged via direct import, same serving config the pilot already validated clean:
vLLM 0.27.1, `--quantization awq`, `--enable-auto-tool-choice --tool-call-parser
hermes`, temperature 0.0, `MAX_STEPS=200`, tau2 pinned `fc0055dc4...`) +
`kernels/grid_session1/grid_run.py` (thin entry point, mirrors `pilot_run.py`'s
dataset-polling pattern) + `kernels/grid_session1/kernel-metadata.json` (separate
subdirectory from the pilot kernel's own `kernels/kernel-metadata.json` - Kaggle's
CLI requires that exact filename in the push directory, so a second kernel needs its
own directory, not a same-directory rename). **Each `RunRecord` is written to disk
immediately after `sweep.run_cell()` returns for that episode** (`record.save(...)`
inside the per-condition loop, not batched at the end), and the running summary JSON
is rewritten after every episode too - a session that dies at hour 8 keeps everything
completed so far. Resumable: skips any `run_id` already on disk, same pattern
`sweep.py`'s own `main()` uses. Retries transient failures up to 3x per episode
(`max_retries=3`, matching `sweep.py`'s own default - unlike the pilot's
deliberate `max_retries=1`, which existed specifically to see raw failures
undisguised for diagnosis; this is now a real data-collection run, not a diagnostic
one).

Housekeeping caught before pushing: an earlier session's `detectability_grounding.py`
run (executed with CWD=`src/`) had written its real output to a stray `src/tables/`
instead of the repo's top-level `tables/` - moved the real CSV
(`detectability_grounding.csv`, the actual 5/10/15/20/30pp curve data) to
`tables/detectability_grounding.csv` and deleted the stray directory (plus
`src/__pycache__`) before uploading `src/` as the Kaggle dataset, so nothing
extraneous shipped.

**Pushed - first real GPU time on the actual grid, not the pilot.**
- Dataset: `kaggle datasets version -p src -m "add grid_core.py, detectability_
  grounding.py for session 1"` -> `abhiprd2000/attrib-code`, confirmed `status:
  ready` before the kernel push (checked once).
- Kernel: `kaggle kernels push -p kernels/grid_session1` -> **`abhiprd2000/
  attrib-grid-session1`, version 1**,
  https://www.kaggle.com/code/abhiprd2000/attrib-grid-session1
- One status check immediately after push (not a poll loop): `KernelWorkerStatus.
  RUNNING`, confirmed at push time. Per the standing rule, not checked again until
  the operator brings back output.

No git commits. `F:/simswap` not touched this session. Snapshot follows, then stop.

## 2026-08-17/18 - grid session 1 results: 81/84, real crash root cause, metric fix, scope options costed. Session 2 NOT launched.

Kernel `abhiprd2000/attrib-grid-session1` finished (`KernelWorkerStatus.COMPLETE`,
confirmed by a direct status check, not a poll - the operator explicitly asked
"check if it is done" first). Output pulled, preserved in-repo at
`data/grid_session1_20260817/` (174 files: 84 records, 84 transcripts, summary,
logs).

**Result: 81/84 completed, 3 errored. Real wall-clock 37784.1s = 10.50h, vs the
6.65h projected from the pilot's n=1 sample - a 58% overrun.** Real per-condition
means (n=14 each) vs the pilot's n=1 guess diverged wildly in both directions:
`no_policy` 27.1s->882.7s (32.5x), `policy_truncated` 39.7s->436.8s (11.0x),
`schema_names_only` 1093.2s->594.2s (0.5x, the pilot's own outlier didn't
generalize). This is the single clearest demonstration all session of why n=1
pilots exist and why grid math gets revised, not defended, when real data arrives.

**0. Repo-state check (before touching anything) - no data lost.** `pilot_core.py`
confirmed at `src/pilot_core.py` only (no duplicate from the earlier killed
background move). `data/pilot_run_20260817/` confirmed present and intact via
direct listing after an earlier `Glob` call returned a false "no files found" on one
nested pattern (tool quirk, not data loss - a plain `dir` and a simpler `Glob`
pattern both found everything).

**1. Which cells ran**: **7 tasks x 2 seeds = 14 cells**, not 14 tasks x 1 seed -
tasks `4, 43, 66, 77, 93, 94, 98` (the first 7 of `sweep.select_tasks`'s real
20-task selection), both seeds each, confirmed by listing every `(task_id, seed)`
pair in the summary. This means a task-count cut can reuse all 14 of these cells
directly; a seed cut can reuse only the 7 seed0 cells (seed1 data exists but a
1-seed design doesn't need it).

**2. Crash path fixed, CPU-verified, real root cause identified - not the pilot's
tool-call-error pattern.** Read all 3 crash records' full tracebacks: all 3 are
`litellm.exceptions.ContextWindowExceededError` - the prompt exceeded
`--max-model-len` (32768 tokens). Verified against the real tau2 clone: tau2's own
`TerminationReason.CONTEXT_WINDOW_EXCEEDED` enum value exists but is never assigned
anywhere in the half-duplex orchestrator - the exception propagates out of
`run_single_task()` UNCAUGHT, before any `SimulationRun` exists, so there was never
a graceful termination reason to read. Fixed in `sweep.py::run_cell()`: catches
`ContextWindowExceededError` specifically, never retries it (deterministic at
temperature=0.0 - the same conversation hits the same token count every attempt;
one session-1 episode burned 73 minutes retrying this exact unwinnable failure 3
times), and records it via new `convert_context_window_exceeded()` -
`status="completed"`, `termination_reason="context_window_exceeded"` (new category
added to `schema.py::TERMINATION_REASONS`), `reward=0.0`, matching how
`max_turns`/`agent_error` are already treated as real outcomes, not harness
failures. `grid_core.py`'s `max_retries` changed 3 -> 1 (matching `pilot_core.py`'s
original deliberate choice), for the same reason.

**CPU-verified before anything touched a GPU again** (`cpu_check_cwe_fix.py`,
mocks `tau2.runner.run_single_task` to raise a real
`litellm.exceptions.ContextWindowExceededError`, no network/GPU needed): confirmed
the new exception path is caught exactly once (not retried), produces a record that
passes `schema.py`'s `__post_init__` validation and round-trips through
`to_dict()`/`from_dict()`; confirmed a DIFFERENT exception type still retries
normally (3x) and still produces `status="error"`/`harness_crash` as before - the
new branch doesn't interfere with the existing generic path. `tests/
test_grounding.py` still 6/6 after the change. One real bug caught and fixed along
the way in the check script itself (missing `sys.path` entry for the tau2 clone -
`ModuleNotFoundError: No module named 'tau2'` on first attempt, fixed, reran clean).

**NOT pushed - the dataset update and the targeted 3-cell rerun kernel are both
still BLOCKED, not completed.** Built and CPU-verified `src/rerun_failed_core.py` +
`kernels/rerun_failed/` (own subdirectory, same Kaggle-CLI-filename-collision reason
as `grid_session1/`) for a rerun of exactly the 3 named failed cells
(`4/seed0/no_policy`, `4/seed1/no_policy`, `94/seed1/policy_truncated`) - NOT session
2, only these 3. `kaggle datasets version` was attempted twice: first attempt was
interrupted mid-upload (network hiccup, partial file list uploaded); second attempt
uploaded all 9 files successfully but the final `CreateDatasetVersion` API call
itself returned `403 Client Error: Forbidden` - **the version was never actually
created, so the Kaggle dataset still has the OLD, unfixed code as of this writing.**
Follow-up checks (`kaggle datasets status`, `kaggle kernels status` on the
already-completed `attrib-grid-session1` kernel, which worked fine earlier this same
session) then started failing CONSISTENTLY and repeatably - `403 Client Error:
Forbidden` on datasets, `Permission 'kernels.get' was denied` on kernels - not the
random empty-output/EPERM shell flakiness seen elsewhere this session (which comes
and goes across single retries), a real repeated error with real message content
across multiple separate attempts. This looks like a genuine Kaggle API access
problem (rate limiting from the rapid retry sequence, or a token/session issue on
the account side), not something retrying further from here is likely to fix -
stopped attempting rather than keep hammering an API that may already be
rate-limiting these calls. **The operator should check Kaggle access directly
(kaggle.com, or re-verify the API token) before the next push is attempted.**

**3. Attrition is now a reported outcome - `ANALYSIS_PLAN.md` updated.** New
section: per-condition crash rate (`no_policy` 14.3%, `policy_truncated` 7.1%, all
others 0% - real session-1 numbers, n=1 grid so far) and full termination_reason
distribution, both first-class, not folded into a denominator silently. Stated
plainly why the crashes cluster where they do: `no_policy`/`policy_truncated` also
carry the two highest `max_turns` rates (33.3%, 30.8%) and the two highest mean
wall-clocks (882.7s, 436.8s) - conditions that run long are mechanically more
likely to hit BOTH the 200-step cap and the 32768-token cap, not because of
anything specific to policy content. Documented exactly how crashed/context-window-
exceeded episodes are handled in the PRIMARY (per-episode-averaged) grounding rate:
no transcript exists, so every rate field is `None` via the same "zero denominator
-> None" discipline already covering empty-pool conversations, and `average_by_
condition()` EXCLUDES `None` rates from the average rather than counting them as
0%/100% - a crashed episode contributes nothing to the rate either way, but DOES
still count toward that condition's crash-rate/effective-n reporting.

**4. Turns as a result - real numbers, not interpreted as grounding findings**,
from session 1's 81 completed episodes:

| condition | n | mean n_turns | max_turns count | max_turns rate |
|---|---|---|---|---|
| full_context | 14 | 74.1 | 2 | 14.3% |
| no_policy | 12 | 83.2 | 4 | 33.3% |
| policy_truncated | 13 | 91.6 | 4 | 30.8% |
| schema_names_only | 14 | 67.9 | 2 | 14.3% |
| history_truncated | 14 | 67.9 | 1 | 7.1% |
| system_minimized | 14 | 86.3 | 4 | 28.6% |

**5. Scope options - real MDEs from actual `detectability_grounding.py` reruns
(script parameterized with `--n-tasks`/`--n-seeds`/`--out-suffix`, frozen `n_tasks=
20,n_seeds=2` result at `tables/detectability_grounding.csv` untouched), real
remaining-hours from session 1's actual per-condition means (sum 2685.3s/cell).
Proposed only, nothing launched:**

| option | remaining cells | remaining GPU-h | total GPU-h | real MDE | verdict |
|---|---|---|---|---|---|
| (a) finish all 40 cells | 26 | 19.39h | 29.89h | **15pp** | needs 3 more 9h sessions (4 total) |
| (b) 20 tasks x 1 seed | 13 (reuses 7 seed0 cells) | 9.70h | 20.19h | **20pp** | needs 2 more sessions |
| (c) N=7 (session 1 as final) | 0 | 0h | 10.50h | **30pp** | **FLAGGED - above 20pp, not worth it** |
| (c) N=10 | 6 | 4.48h | 14.97h | 20pp reaches only 75% power (below 80% target) | **borderline - does not cleanly clear the threshold either** |
| (c) N=12 | 10 | 7.46h | 17.95h | **20pp** (87% power) | cheapest option that cleanly clears 20pp; fits in ONE more 9h session (7.46h, ~17% margin) |
| (c) N=15 | 16 | 11.93h | 22.43h | **20pp** (98% power) | comfortable margin, needs 2 more sessions (11.93h > 9h) |

Standout candidate for the operator's consideration, not decided here: **(c) N=12**
is the cheapest option that cleanly reaches the 20pp threshold and is the only one
(besides stopping at N=7, which is disqualified) that needs just ONE more 9h
session rather than two or three.

No git commits. Snapshot follows, then stop, per the operator's explicit
instruction not to push session 2.

## 2026-08-18 - N=12 decided, ANALYSIS_PLAN.md pre-registration updated, session 2
## kernel built, NOT pushed

Operator's decision, made from the scope-options table above (real session-1
per-condition means, real detectability reruns) without reading session 1's real
grounding-rate numbers first: **N=12 tasks x 2 seeds**. 10 cells remain.

**1. Task selection confirmed as a prefix, not a re-pick.** Re-ran
`sweep.select_tasks("retail", 20)` (`TASK_SAMPLE_SEED=20260814`) against the pinned
tau2 commit, CPU-only, twice (once during the verification pass, once again just
before writing this entry - both runs agree): full 20-task order is `['93', '43',
'98', '77', '66', '94', '4', '46', '107', '64', '102', '110', '82', '73', '7', '74',
'55', '89', '15', '13']`. First 12 (the new grid): `['93', '43', '98', '77', '66',
'94', '4', '46', '107', '64', '102', '110']`. All 7 already-run session-1 task_ids
(`4, 43, 66, 77, 93, 94, 98`) confirmed inside this prefix, no exceptions. The 5 new
task_ids for session 2: `46, 107, 64, 102, 110`. Written into `ANALYSIS_PLAN.md`'s
new "Grid size: REVISED 2026-08-18" section with this same derivation and the
explicit reason ("prefix of the original selection, not chosen after seeing data").

**2. Falsification threshold updated 15pp -> 20pp, done before reading any
grounding rate.** `ANALYSIS_PLAN.md`'s original 15pp threshold (set for the
`n_tasks=20` design) was written for a grid this project no longer plans to run in
full; the real MDE at `n_tasks=12, n_seeds=2` is 20pp (87% power, from the scope-
options table above). The 15pp/`n_tasks=20` falsification section is marked
ARCHIVED in place (kept verbatim, not deleted); a new 20pp section follows it with
the same SUPPORTED/FALSIFIED/inconclusive structure, threshold only changed. A new
LIMITATIONS paragraph states plainly that the grid was cut for GPU budget, not for
any data-dependent reason, and that this trades away the ability to resolve any true
effect between 15pp and 20pp - a real loss of power, not a result-biasing choice.

**3. Crash-exclusion bound pre-registered, before the expanded grid's grounding
rate exists.** `ANALYSIS_PLAN.md`'s ATTRITION section now specifies three views to
compute together once the real N=12 data lands: (a) PRIMARY - crashed episodes
excluded (current behavior, unchanged), (b) WORST CASE - crashed imputed as 100%
ungrounded, (c) BEST CASE - crashed imputed as 0% ungrounded. Rule written down now:
if (b) and (c) fall on opposite sides of the 20pp threshold for a condition that
matters to the falsification criterion, that condition's result must be reported as
NOT ROBUST to the crash-handling choice, not rounded to whichever view matches the
primary verdict. Only conditions with nonzero session-1 crash rates
(`no_policy` 14.3%, `policy_truncated` 7.1%) need this in practice unless session 2's
5 new tasks introduce crashes elsewhere.

**4. Session 2 kernel built, NOT pushed.** `src/session2_core.py` (new, mirrors
`src/grid_core.py`'s structure exactly): iterates the 5 new task_ids (`46, 107, 64,
102, 110`, taken as `all_task_ids[7:12]`) x 2 seeds x 6 conditions = 60 episodes (10
cells), same serving config as session 1 (vLLM 0.27.1, Qwen2.5-7B-Instruct-AWQ,
`--quantization awq`, `--enable-auto-tool-choice --tool-call-parser hermes`,
temperature 0, MAX_STEPS=200, tau2 pinned `fc0055dc4e0a316c3f83133267fbd6faaa770992`),
includes the session-1 `ContextWindowExceededError` crash fix, `max_retries=1`,
records written to disk immediately per episode (not batched).
`kernels/grid_session2/grid_run.py` and `kernel-metadata.json` (kernel id
`abhiprd2000/attrib-grid-session2`) built alongside it, mirroring the session-1
kernel wrapper pattern. Both new Python files syntax-checked clean
(`ast.parse`, CPU-only). The `session_task_ids = all_task_ids[7:12]` slice was
independently re-verified against the real pinned tau2 source (CPU-only, no GPU/
network) immediately before this entry was written: `['46', '107', '64', '102',
'110']`, matching item 1 exactly.

**Projected hours - real numbers, from session 1's actual per-condition means (NOT
the earlier 6-episode pilot's figure)**: session 1's 6 real per-condition wall-clock
means sum to 2685.3s/cell (see the scope-options table above, same source as the
N=12 row). 10 cells x 2685.3s = 26853s = 7.459h, plus ~0.05h server startup ~=
**7.51h projected**, leaving ~1.49h (17%) margin under the 9h Kaggle GPU session
cap - the tightest margin of any option costed, but still real margin. This is
projected to be the LAST session needed for the N=12 grid: after it, all 12 task_ids
x 2 seeds x 6 conditions (144 episodes total, 84 from session 1 + 60 new) are on
disk.

**NOT pushed.** Kaggle API access was blocked by persistent 403 errors as of the end
of session 1 (`kernels.get` permission denied, dataset version creation forbidden).
The operator has not yet confirmed access is restored. Per explicit instruction:
built and verified only, no `kaggle kernels push`, no `kaggle datasets version`, no
polling. No git commits made this entry either.

## 2026-08-18 - session 2 hardened and PUSHED

Operator confirmed new Kaggle credentials (account `abhiprd20`, ~19h GPU quota
remaining). Session 1 overran its own projection by 58% (6.65h projected vs 10.50h
real); session 2's naive projection (7.51h against a 9h cap, ~17% margin) was judged
too fragile to push as-is. Hardened before pushing, not after it died:

**1. Cells ordered by real cost, most expensive first.** Recomputed per-condition
wall-clock means directly from `data/grid_session1_20260817/grid_session1_summary.json`
(the outer per-episode timer, captured for every episode regardless of status - NOT
each `RunRecord`'s own `wall_clock_seconds` field, which is `null` for the 3 crashed
pre-fix episodes and would have silently zeroed them out; caught this by testing both
and getting a 3.35x-different, wrong sum from the RunRecord field before switching to
the summary file). Real means, n=14 each:

| condition | mean wall-clock |
|---|---|
| no_policy | 882.68s |
| schema_names_only | 594.17s |
| policy_truncated | 436.76s |
| full_context | 385.74s |
| system_minimized | 250.43s |
| history_truncated | 135.51s |

Sum = 2685.27s/cell, confirming the 2685.3s figure already in the scope-options
table above. `src/session2_core.py`'s condition loop now runs in this order
(`CONDITION_ORDER_BY_COST`) instead of `schema.CONDITIONS`' declaration order, guarded
by an assert that it's a reordering, not a different set. If the Kaggle time cap
kills the box mid-cell, the two conditions left unattempted are the two cheapest
(`system_minimized`, `history_truncated`), not the two most expensive AND most
load-bearing for the falsification criterion (`no_policy`, `schema_names_only`).

**2. Checkpoint discipline confirmed by reading the code, not assumed.**
`record.save(run_path)` (`session2_core.py`) fires synchronously immediately after
each `sweep.run_cell()` returns, before the loop advances - not batched.
`RunRecord.save()` (`schema.py:179-181`) calls `Path.write_text()`, which opens,
writes, and closes the file within that single call - on disk (OS page cache,
survives a process kill) by the time `save()` returns.

**3. Resumability hardened and CPU-tested.** Bare `run_path.exists()` was replaced
with `_existing_valid_record()` - skips only if the file parses as JSON with a
non-null `status` field; a truncated/corrupt write is re-run, not silently treated
as done. CPU-tested (`scratchpad/test_resumability.py`, no GPU/network) against
session 1's real 84 records copied into a scratch dir: all 84 correctly recognized
as valid/skippable; all 60 of session 2's new (task_id, seed, condition)
combinations correctly selected to run (none collide - disjoint task_ids); a
synthetic truncated-JSON file correctly NOT skipped. All assertions passed.

**4. Account mismatch found and fixed before pushing.** The read-only
verification call (`kaggle kernels list -m`, per instruction: check auth cheaply
before pushing) succeeded under the new `abhiprd20` credentials - real account,
real existing kernels, no 403. But a second read-only check
(`kaggle datasets files abhiprd2000/attrib-code`) came back `403 Forbidden`: the
`attrib-code` dataset and session-1/rerun-failed kernels belong to `abhiprd2000`, a
different account than the one just authenticated. Stopped and asked the operator
rather than guessing; operator chose to re-upload the dataset under the new
account. Real actions taken: `src/dataset-metadata.json`'s id changed to
`abhiprd20/attrib-code`; `kaggle datasets create -p src` re-uploaded all 13 real
`.py` files (confirmed file-by-file in the CLI output) to
`abhiprd20/attrib-code`; `kernels/grid_session2/kernel-metadata.json`'s `id` and
`dataset_sources` and `grid_run.py`'s dataset-discovery `CANDIDATES` path updated
to `abhiprd20` to match.

**5. PUSHED.** `kaggle kernels push -p kernels/grid_session2` - after several
attempts lost to local shell/network flakiness (timeouts and spawn errors that
never reached Kaggle, confirmed by `kaggle kernels list -m` still not showing the
kernel after each), a clean attempt returned `Kernel version 1 successfully
pushed`. One single read-only status check afterward (not polling) confirmed
`abhiprd20/attrib-grid-session2` status `KernelWorkerStatus.QUEUED` - it started.
Slug: `abhiprd20/attrib-grid-session2`. Pushed 2026-08-18 (exact minute not logged -
not chased further after several flaky timestamp attempts, date precision is
consistent with the rest of this log). Per explicit instruction: no further polling,
no waiting on this session's outcome - stopping here.

No git commits made this entry.

## 2026-08-23 - session 2 output downloaded and renamed, full coverage check, combined
## attrition table, analysis run once on all 144 records. Grid complete: 138/144.

Resumed after several days away (prior session, same date-stamped 2026-08-23, had
already pulled session 2's Kaggle output to a scratch-named
`data/grid_session2_20260823_pull/` and reported 57/60 completed there - this session
picks up from that point per the operator's brief, which opened by requiring a
snapshot before anything else).

**1. Snapshot first, before any other change.** `robocopy F:\attrib
F:\attrib-snap-20260823-1307 /E` - real byte-for-byte check (`Get-ChildItem -Recurse
-File | Measure-Object` on both sides, not trusted from robocopy's own summary line
alone, which reported a different, inconsistent total on a re-check): source and
destination both **365 files**, 0 failed. This snapshot captures the state with
`data/grid_session2_20260823_pull/` still under its pull-date name, i.e. before the
rename below - the newest snapshot before this one was `2026-08-18-1425`, so this
closes a 5-day gap with nothing else touched in between.

**Renamed** `data/grid_session2_20260823_pull/` -> `data/grid_session2_20260818/`
(`mv`) - named for the RUN date (kernel pushed and executed 2026-08-18, per the
2026-08-18 "session 2 hardened and PUSHED" entry above), not the pull date, matching
`data/grid_session1_20260817/`'s existing convention (named for when session 1 ran,
not when its output was downloaded). Confirmed via directory listing:
`data/grid_session1_20260817/`, `data/grid_session2_20260818/`, `data/
pilot_run_20260817/` are the only three dirs under `data/`.

**2. Coverage check - clean, 144/144, zero gaps, zero duplicates.** Built the
expected set of 12 (frozen prefix) x 2 seeds x 6 conditions = 144 `(task_id, seed,
condition)` triples and cross-checked every `RunRecord` in both
`data/grid_session1_20260817/runs/` (84 files) and `data/grid_session2_20260818/
runs/` (60 files) against it (`coverage_check.py`, scratchpad, not committed - reads
`task_id`/`seed`/`condition` directly off each record, not off filenames). Result:
- **0 gaps** - every one of the 144 expected triples has exactly 1 record.
- **0 duplicates** - no triple has more than 1 record.
- **0 records** whose `task_id` falls outside the frozen 12
  (`93,43,98,77,66,94,4,46,107,64,102,110`).
- **0 unparseable records.**
- 144 records scanned total (84 + 60), matching 144 expected exactly - no more, no
  fewer. Per the operator's stop-condition: no gaps or duplicates found, so analysis
  proceeded to item 3.

**3. Combined attrition table - all 144, per condition, n=24 each.** Real field
values only (`status`, `termination_reason`), no invented categories
(`attrition_check.py`, scratchpad):

| condition | n_completed | n_normal | n_max_turns | n_agent_error | n_context_window_exceeded | n_timeout (status) | n_error (status, non-timeout harness_crash) | total_attrition_rate |
|---|---|---|---|---|---|---|---|---|
| full_context | 23 | 11 | 3 | 9 | 0 | 1 | 0 | 54.2% |
| no_policy | 22 | 11 | 4 | 7 | 0 | 0 | 2 | 54.2% |
| policy_truncated | 23 | 11 | 8 | 4 | 0 | 0 | 1 | 54.2% |
| schema_names_only | 24 | 10 | 3 | 11 | 0 | 0 | 0 | 58.3% |
| history_truncated | 24 | 5 | 2 | 17 | 0 | 0 | 0 | 79.2% |
| system_minimized | 22 | 4 | 6 | 9 | 3 | 2 | 0 | 83.3% |

Every row reconciles exactly: `n_completed = n_normal + n_max_turns + n_agent_error +
n_context_window_exceeded`; `24 = n_completed + n_timeout + n_error`. All 3
`status=="error"` records (session 1's pre-fix `ContextWindowExceededError` crashes,
`termination_reason="harness_crash"`) and all 3 `status=="timeout"` records (session
2, also `termination_reason="harness_crash"` per `sweep.py:70`'s status->reason
mapping, distinguished from `status="error"` only by the exception's type name/message
containing "timeout", `sweep.py:258`) are accounted for in the `n_timeout`/`n_error`
columns - none silently dropped.

**Confound rule, decided and stated plainly**: `total_attrition_rate` defined here as
`1 - n_normal/24` (fraction of episodes that did NOT end via tau2's own "normal"
termination - the only outcome with no attached confound flag anywhere in this
project). This specific combined metric is not itself pre-registered in
`ANALYSIS_PLAN.md` (the two individually pre-registered confound checks are
`check_truncation_confound`/`check_agent_error_confound`, each with their own 10pp
threshold, already in `analyze.py`) - stated as my own operationalization for this
one combined view, not presented as frozen. **Result: attrition ranges 54.2% (three
tied conditions) to 83.3% (`system_minimized`) - a 29.1 absolute-point spread, more
than double the 10-point threshold. Attrition differs materially across conditions.**
This is not a borderline call - the same conclusion holds independently under both of
the actually pre-registered mechanism-specific checks too (real `analyze.py` output,
item 4 below): the `max_turns`-rate confound check fired on 7 of 15 condition pairs
(largest gap 26.4pp, `policy_truncated` vs `history_truncated`) and the
`agent_error`-rate confound check fired on 10 of 15 pairs (largest gap 53.4pp, same
pair). Per the operator's instruction, this is reported, not argued with: any
grounding-rate or accuracy contrast in this dataset must be read alongside these
confound flags, not as a pure context effect.

**4. Analysis run once, `analyze.py` exactly as frozen (`--n-perm=10000
--alpha=0.05 --rng-seed=0`, all CLI defaults unchanged), against all 144 records
staged into one combined `runs/` dir (copy, session dirs on disk untouched).**
Real output CSVs written to the project's actual `tables/` dir (not scratch):
`condition_contrasts.csv`, `condition_contrasts_excl_truncated.csv`,
`condition_contrasts_normal_only.csv`, `truncation_rate_by_condition.csv`,
`agent_error_rate_by_condition.csv`.

**4a. PRIMARY per-episode-averaged grounding rate, with contrast/p-values -
NOT COMPUTABLE as specified. This is a real, standing gap, not a result.**
`src/grounding.py` has no `{condition: {(task_id, seed): rate}}` per-cell function -
only two whole-condition aggregate functions (`pool_by_condition`,
`average_by_condition`). `analyze.py::contrast()` (the only code that runs the
permutation test + Holm correction) requires the per-cell shape and only exists for
`action_accuracy`. This exact gap was already flagged in `ANALYSIS_PLAN.md` on
2026-08-17 ("Flagged gap, not yet built... Needed before the real grid's significance
test can run") and **is still open** - not built this session either, per the
operator's explicit instruction not to add metrics. Listed under EXPLORATORY below,
not implemented.

What IS real and reportable: the point estimates from the two aggregate functions that
do exist, run for real against all 144 transcripts (144/144 resolved - every record's
`transcript_path` matched a real file on disk in its session's `transcripts/` dir; the
"crashed" ones are real, valid, empty (`[]`, 2 bytes) transcript files, not missing
files - confirmed by direct inspection of all 6 non-`normal`-status records, not
assumed). **PRIMARY per ANALYSIS_PLAN.md (macro-averaged, CONDITIONAL, undefined
rates excluded)** - `ungrounded_fraction_by_argument_conditional_avg`:

| condition | rate | n in average (of 24) | increase vs full_context |
|---|---|---|---|
| full_context | 71.13% | 22 | - (control) |
| no_policy | 71.71% | 12 | +0.6pp |
| policy_truncated | 72.61% | 22 | +1.5pp |
| schema_names_only | 73.16% | 20 | +2.0pp |
| history_truncated | 80.21% | 21 | +9.1pp |
| system_minimized | 76.59% | 13 | +5.5pp |

No p-values attached to any of these - point estimates only, per the gap above.

**4b. Crash-exclusion bound (worst/best imputation) - real, computed over the
correctly-identified crashed subset only** (`status in {error, timeout}` OR
`termination_reason == "context_window_exceeded"` - literally-empty-transcript
episodes, per `ANALYSIS_PLAN.md`'s own definition; conversations with a real,
non-empty transcript whose grounding pool simply never filled before the episode
ended are a different, unrelated reason for an undefined rate and were kept excluded
in all three views, not folded in). Crashed-episode counts: `full_context`=1,
`no_policy`=2, `policy_truncated`=1, `schema_names_only`=0, `history_truncated`=0,
`system_minimized`=5. Because `full_context` (the control) itself has 1 crashed
episode, its own rate also shifts across views, which is folded correctly into every
condition's "increase vs control" number below - both sides of each comparison use
the matching view:

| condition | inc. (a) excluded | inc. (b) worst | inc. (c) best | straddles 20pp? |
|---|---|---|---|---|
| no_policy | +0.6pp | +3.4pp | -6.6pp | **No** |
| policy_truncated | +1.5pp | +1.4pp | +1.4pp | **No** |
| schema_names_only | +2.0pp | +0.8pp | +5.1pp | **No** |
| history_truncated | +9.1pp | +7.8pp | +12.2pp | **No** |
| system_minimized | +5.5pp | +10.7pp | -12.7pp | **No** |

No condition straddles 20pp under worst-case vs best-case crash handling - all three
views stay well clear of the threshold for every condition. `system_minimized` has
the widest worst/best swing (23.4pp, driven by its 5 crashed episodes, the most of any
condition) but even its worst case (+10.7pp) is roughly half the 20pp floor.

**4c. SECONDARY views, all real, all computed:**

*Pooled (micro-average) grounding rate, CONDITIONAL*: full_context 62.80%, no_policy
61.96%, policy_truncated 52.25%, schema_names_only 75.74%, history_truncated 85.93%,
system_minimized 65.98%. (FLAT pooled, for reference: 72.18% / 82.63% / 67.98% /
87.21% / 89.04% / 88.85% respectively.)

*action_accuracy, with undefined-count*: `full_context` 7/23 defined (16 undefined),
`no_policy` 8/22 (14 undefined), `policy_truncated` 8/23 (15 undefined),
`schema_names_only` 4/24 (20 undefined), `history_truncated` 1/24 (23 undefined),
`system_minimized` 3/22 (19 undefined). Consistent with the differential-attrition
finding already on record in `ANALYSIS_PLAN.md` - the vast majority of cells have no
defined `action_accuracy` at all, and `history_truncated` has exactly 1 defined cell
in the entire 144-record grid, which is why it drops out of every `analyze.py`
contrast table entirely (no shared `(task_id, seed)` key survives the pairing against
`full_context`'s defined cells) - real behavior of the frozen pipeline, not a bug,
not worked around.

*mean n_turns (all 24 records/condition, every status)*: full_context 66.8, no_policy
57.6, policy_truncated 94.8, schema_names_only 60.5, history_truncated 72.5,
system_minimized 73.9.

*max_turns rate (from `truncation_rate_by_condition.csv`, completed episodes only)*:
full_context 13.0%, no_policy 18.2%, policy_truncated 34.8%, schema_names_only 12.5%,
history_truncated 8.3%, system_minimized 27.3%.

**4d. Was the 20pp falsification threshold met? INCONCLUSIVE, for two independent
reasons, stated separately, not conflated.** (1) The frozen falsification criterion
requires `p_holm < 0.05` - not computable at all for grounding rate, per 4a's gap.
(2) Independent of significance, by point estimate alone under every one of the three
crash-handling views in 4b: `no_policy`'s own ungrounded-rate increase over
`full_context` - the hypothesis's own precondition ("if `no_policy` itself is not
load-bearing... inconclusive") - is +0.6pp to +3.4pp, nowhere near the 20pp floor. No
condition in the entire grid reaches 20pp under any view; the largest point estimate
anywhere is `history_truncated`'s +12.2pp (best-case crash handling). Per
`ANALYSIS_PLAN.md`'s own pre-registered structure, this lands squarely in
"Inconclusive at this grid" - not support, not falsification - and would be
inconclusive on point estimates alone even if the missing significance test existed.

**EXPLORATORY (not run, flagged only, per instruction):**
- Build `grounding.py`'s missing per-cell `{condition: {(task_id, seed): rate}}`
  function (mirrors `accuracy_by_cell`'s shape) so `analyze.py::contrast()` (or an
  equivalent) can compute a real permutation p-value for grounding rate. This is the
  single blocker preventing the falsification criterion from ever being evaluated as
  literally pre-registered - flagged repeatedly since 2026-08-17, still open.
- Formalize a single pre-registered "combined attrition rate" confound function in
  `analyze.py` (mirroring `check_truncation_confound`/`check_agent_error_confound`'s
  shape and 10pp threshold) rather than the ad hoc `1 - n_normal/n_total`
  operationalization used for item 3's table this session - would need the operator
  to bless a specific definition first, since more than one reasonable one exists
  (e.g. weighted by which mechanisms matter to the falsification criterion vs. a flat
  "any non-normal outcome" count as used here).

No `ANALYSIS_PLAN.md` edits this session (per instruction). No metrics added. No git
commits. Snapshot follows, then stop.

## 2026-08-23 (continued) - selection-artifact check, the frozen primary test run
## for real, RESULTS section written. Grid done, 144/144, result is a null with a
## confound.

**1. THE SELECTION ARTIFACT - checked before any interpretation, per instruction.**
Per-episode data (all 144: `total_tool_calls` and both grounded-rate views from
`grounding.analyze_conversation()`, `n_turns` from the real `RunRecord`) -
`selection_artifact.py`, scratchpad.

**1a. Correlation, grounded rate (1 - ungrounded fraction) vs episode length -
real, computed, and the OPPOSITE sign from the concern raised.**

| view | vs n_turns | vs total_tool_calls |
|---|---|---|
| FLAT (n=135) | r=+0.034 | r=+0.363 |
| CONDITIONAL (n=110) | r=+0.021 | r=+0.153 |

Both correlations are **positive, not negative**. If short episodes were
systematically scoring higher (better) due to fewer chances to fabricate, the
correlation with `total_tool_calls` would be negative. It is not - weakly-to-
moderately positive instead (r=+0.36 on the FLAT view, the larger-n one).
**Stated plainly, as instructed: this data does NOT show that short episodes
systematically score higher.** If anything, episodes that make MORE tool-call
attempts trend toward a slightly BETTER (more grounded) flat rate, plausibly
because a single early successful lookup gets reused many times over a longer
conversation, diluting the ungrounded fraction - a real, disclosed possible
artifact, but running in the opposite direction from "short-episode inflation."
With `n_turns` specifically the correlation is close to zero either way (r=+0.02
to +0.03) - turn count alone is not predictive of grounded rate at the
per-episode level.

**1b. Mean tool calls attempted per episode, per condition (all 24, every status):**

| condition | mean total_tool_calls |
|---|---|
| full_context | 11.08 |
| no_policy | 7.71 |
| policy_truncated | 8.54 |
| schema_names_only | 8.75 |
| history_truncated | **12.33** (longest) |
| system_minimized | **7.17** (shortest) |

No clean monotonic relationship between this and the earlier ungrounded-fraction
ranking: `history_truncated` is simultaneously the LONGEST condition by tool-call
count AND the worst-grounded (80.21% ungrounded, full-set macro average) - directly
contradicting a "short episode -> looks worse" story for that specific condition.
`system_minimized` IS the shortest condition (7.17 calls) and is also high-attrition
(83.3%) and poorly-grounded (76.59%) - consistent with a length/censoring effect for
THIS one condition, but not a general pattern across all six. `policy_truncated` has
the highest `n_turns` (94.75, from the prior session's mean-n_turns table) yet one of
the smallest ungrounded-rate increases (+1.5pp) - also inconsistent with a simple
length story. **Conclusion: episode length (turns or tool-call count) does not
cleanly predict the ungrounded-rate pattern across conditions - the earlier finding
that high-attrition conditions score worst is better explained by attrition itself
(fewer episodes ever reach a defined conditional rate at all - see 1c/4a-equivalent
cell counts below) than by a generic short-episode artifact.**

**1c. Restricted to `termination_reason == "normal"` only - real recompute,
n collapses to the point of being uninterpretable for 2 of 6 conditions.**

| condition | n_normal | mean_n_turns (normal only) | grounded rate, normal-only (macro avg) | n in avg |
|---|---|---|---|---|
| full_context | 11 | 41.9 | 0.3296 | 11 |
| no_policy | 11 | 26.3 | 0.2897 | 5 |
| policy_truncated | 11 | 42.6 | 0.2641 | 10 |
| schema_names_only | 10 | 34.6 | 0.3802 | 8 |
| history_truncated | 5 | 33.6 | 0.5000 | **2** |
| system_minimized | 4 | 30.5 | 0.3755 | **3** |

Side-by-side with the full-set macro-averaged grounded rate (all statuses, from the
prior session's report, restated here for direct comparison):

| condition | full-set grounded avg | normal-only grounded avg |
|---|---|---|
| full_context | 0.2887 | 0.3296 |
| no_policy | 0.2829 | 0.2897 |
| policy_truncated | 0.2739 | 0.2641 |
| schema_names_only | 0.2684 | 0.3802 |
| history_truncated | 0.1979 | 0.5000 |
| system_minimized | 0.2341 | 0.3755 |

**Flagged plainly: the two highest-attrition conditions (`history_truncated`,
`system_minimized`) have only 2 and 3 episodes respectively contributing to their
normal-only average - not a reliable per-condition estimate at that n, regardless of
which direction it points. This table is reported for transparency, not read as a
finding.** The normal-only numbers move substantially for those two conditions (both
toward LOOKING BETTER-grounded, not worse, once attrition-heavy episodes are
excluded) - consistent with attrition being where the apparent effect concentrates,
not with a length artifact in the surviving normal episodes themselves.

**2. Built `grounding.py::grounding_rate_by_cell()` - plumbing for the test frozen
2026-08-17, no metric/test/alpha/Holm changes.** `src/grounding.py` (new function,
~35 lines with docstring): `condition -> {(task_id, seed): grounded_rate}`, one real
value per episode (this grid has exactly 1 record per cell, confirmed by the
2026-08-23 coverage check above - no averaging needed), `grounded_rate = 1 -
ungrounded_fraction_by_argument_conditional` so `analyze.py::contrast()`'s existing
"positive diff = condition hurt" convention (written for `action_accuracy`) applies
unchanged and agrees in sign with `ANALYSIS_PLAN.md`'s "ungrounded-rate increase"
framing by construction. Cells with an undefined conditional rate are omitted, same
discipline as `average_by_condition()`. `analyze.py::contrast()` itself was **not
touched** - called exactly as written, same `paired_permutation_test`/
`holm_correct`, same `n_perm=10000, alpha=0.05, rng_seed=0` as the `action_accuracy`
runs. Regression test added (`tests/test_grounding.py::
test_grounding_rate_by_cell_shape_and_exclusion`, fixture-based, matching this
project's existing convention) - **7/7 tests pass**, all prior 6 unaffected.

**Disclosure added to `ANALYSIS_PLAN.md`'s LIMITATIONS section** (not a frozen
section - edited per the operator's instruction): `grounding_rate_by_cell()` was
built AFTER this project's grounding-rate point estimates (the macro-averaged/pooled
aggregates from the prior 2026-08-23 entry) had already been seen. Stated plainly,
not hidden - same disclosure category as the earlier `agent_error` handling rule.

**The frozen primary test, run for real, first time ever:**

| condition | n_pairs | control (grounded) | condition (grounded) | diff (control-condition) | p_raw | p_holm | significant? |
|---|---|---|---|---|---|---|---|
| no_policy | 11 | 0.096 | 0.309 | -0.212 | 0.0470 | 0.2350 | No |
| history_truncated | 19 | 0.304 | 0.180 | +0.123 | 0.1354 | 0.5415 | No |
| policy_truncated | 20 | 0.265 | 0.240 | +0.025 | 0.8159 | 1.0000 | No |
| schema_names_only | 18 | 0.198 | 0.261 | -0.063 | 0.4854 | 1.0000 | No |
| system_minimized | 11 | 0.299 | 0.208 | +0.090 | 0.5167 | 1.0000 | No |

Written to `tables/grounding_condition_contrasts.csv`. **Zero conditions reach Holm
significance.** `no_policy`'s raw p (0.047) clears the uncorrected 0.05 line but not
Holm's - and its point estimate on this specific PAIRED subset (n=11 cells shared
with `full_context`) points the OPPOSITE direction from the hypothesis: `no_policy`
looks MORE grounded than the paired control subset, not less. This paired-subset
control mean (0.096 grounded = 90.4% ungrounded) is far worse than `full_context`'s
overall marginal average (28.87% ungrounded across all 22 of its own valid cells,
from the prior entry) - because pairing restricts to the 11 cells where BOTH
`full_context` AND `no_policy` happened to have a defined rate, a different and much
smaller subset than either condition's own marginal set. This composition
sensitivity is itself further evidence of how unstable this primary test is at this
grid's real effective n (11-20 pairs per condition, not the nominal 24) - not
evidence for or against the hypothesis in either direction.

**3. RESULTS - written plainly, no spin.**

- `no_policy` changes the macro-averaged CONDITIONAL grounding rate by **+0.6
  percentage points** (ungrounded fraction) vs `full_context`, on the full 144-episode
  set. Under the frozen paired significance test (this entry, item 2): `p_holm =
  0.235`, not significant, and the point estimate on the paired subset actually points
  the other way.
- **No condition clears the 20pp practical-effect threshold under any of the three
  crash-handling views** (excluded / worst-case-imputed / best-case-imputed) computed
  in the prior 2026-08-23 entry. Largest point estimate anywhere: `history_truncated`
  at +12.2pp (best-case view) - still well short of 20pp.
- **Attrition spread is 29.1 percentage points** (54.2% to 83.3% across conditions),
  more than double the pre-registered 10pp confound threshold. Both
  mechanism-specific pre-registered checks independently confirm this: the
  `max_turns`-rate confound fired on 7/15 condition pairs, the `agent_error`-rate
  confound fired on 10/15 pairs (real `analyze.py` output, prior entry). **The primary
  result is confounded with step exhaustion and tool-call error propensity, exactly as
  the pre-registered rule requires reporting.**
- **`action_accuracy` is undefined for most cells and cannot carry any finding.**
  Undefined-count per condition (of 24): `full_context` 16/24, `no_policy` 14/24,
  `policy_truncated` 15/24, `schema_names_only` 20/24, `history_truncated` 23/24,
  `system_minimized` 19/24. `history_truncated` has exactly 1 defined cell in the
  entire grid.
- **Selection-artifact check (item 1): does not explain the pattern via short
  episodes scoring higher** - the per-episode correlation between grounded rate and
  activity (tool calls / turns) is weakly positive or flat, not negative, so a
  "shorter episodes look artificially better" mechanism is not supported by this
  data. The apparent condition-level pattern (high-attrition conditions score worse)
  is better attributed to attrition itself censoring which episodes ever reach a
  defined rate (item 1c's collapsed normal-only sample sizes) than to episode length
  as such.

**What CAN be claimed from this data**: the harness, conditions, grounding-rate
algorithm, and (as of this entry) the significance-test plumbing all work end to end
and produced a real, reproducible null result under the frozen test. The attrition
mechanism (agent_error/max_turns/context_window_exceeded rates varying sharply by
condition) is real, large, and already-disclosed - not itself in doubt.

**What CANNOT be claimed**: that any context-ablation condition is load-bearing or
inert for grounding, per the frozen falsification criterion - the test ran, and
returned no significant result at any condition, but it ran on a primary metric this
project's own pre-registered rule already flags as confounded with attrition at more
than 2.5x the allowed threshold. The correct reading is **inconclusive at this
grid**, not "hypothesis supported" and not "hypothesis falsified" - the same verdict
the point-estimate-only read reached in the prior entry, now confirmed rather than
merely suggested by the actual significance test.

**No GPU run. No grid re-cut. No frozen sections edited** (only `LIMITATIONS`, which
is explicitly a disclosure section, not frozen - see item 2). No git commits.
Snapshot follows, then stop.

## CAN CLAIM / CANNOT CLAIM (added 2026-08-24, structured lists)

### CAN CLAIM

- The harness, conditions, grounding-rate algorithm, and significance-test plumbing
  work end to end. [tables/grounding_condition_contrasts.csv - the frozen primary
  test produced real p-values for all 5 conditions, zero errors]
- The grid is complete: 12 tasks x 2 seeds x 6 conditions = 144 triples, 138
  completed. [PROGRESS.md 2026-08-23 coverage check; data/grid_session1_20260817/runs/
  and data/grid_session2_20260818/runs/]
- The attrition mechanism (agent_error/max_turns/context_window_exceeded rates
  varying by condition) is real, large, and already disclosed. [tables/
  agent_error_rate_by_condition.csv, tables/truncation_rate_by_condition.csv -
  attrition ranges 54.2%-83.3%, 29.1pp spread, >2x the 10pp confound threshold]
- The evaluator.py:119-129 outcome-selection gate causes action_accuracy to be
  undefined for any non-agent_stop/user_stop termination - this is differential
  attrition, not a floor. [tables/condition_contrasts.csv - history_truncated has
  0 pairs (1 defined cell, none shared with full_context); defined-cell counts:
  full_context 7/24, no_policy 8/24, policy_truncated 8/24, schema_names_only
  4/24, history_truncated 1/24, system_minimized 3/24]
- The selection-artifact check (correlation between episode length and grounded
  rate) does NOT support a short-episode-inflation explanation. [PROGRESS.md
  2026-08-23 item 1a - FLAT r=+0.034 vs n_turns, r=+0.363 vs total_tool_calls;
  CONDITIONAL r=+0.021 vs n_turns, r=+0.153 vs total_tool_calls; all positive or
  flat, none negative]
- The CPU-verified payload dump confirms each condition strips exactly what it
  claims (no cross-condition leakage). [tables/condition_payload_dump_20tasks.csv]
- No condition clears the 20pp practical-effect threshold under any of the three
  crash-handling views. [PROGRESS.md 2026-08-23 item 4b - largest point estimate:
  history_truncated +12.2pp best-case, well short of 20pp]

### CANNOT CLAIM

- That any context-ablation condition is load-bearing or inert for grounding, per
  the frozen falsification criterion - zero conditions reach Holm significance.
  [tables/grounding_condition_contrasts.csv - smallest p_holm=0.235 (no_policy)]
- The primary grounding-rate result in isolation, because attrition is confounded
  at >2.5x the pre-registered threshold. [tables/agent_error_rate_by_condition.csv
  (max pairwise gap 53.4pp) and tables/truncation_rate_by_condition.csv (max
  pairwise gap 26.4pp) - both exceed the 10pp confound threshold from
  analyze.py:check_agent_error_confound() and analyze.py:check_truncation_-
  confound()]
- action_accuracy as a finding metric - undefined for the vast majority of cells
  (137 of 144 lack a defined value in at least one condition of a pair).
  [tables/condition_contrasts.csv - max 4 pairs; tables/condition_contrasts_-
  normal_only.csv - identical; tables/condition_contrasts_excl_truncated.csv -
  identical]
- That the 20pp MDE is tight enough to detect a real effect if one exists at a
  smaller scale - it is not. The MDE rests on BASE_SKILL_GROUNDED=0.611 from one
  pilot episode (task 93, full_context), n=1. [ANALYSIS_PLAN.md LIMITATIONS section;
  src/tables/detectability_grounding_12t2s.csv shows 87% power only at delta=0.2]
- The normal-only grounding rates as reliable per-condition estimates - two
  conditions have n_in_avg < 5. [PROGRESS.md 2026-08-23 item 1c - history_truncated
  n=2, system_minimized n=3]

## CORRECTION (added 2026-09-03, repo finalization)

The ATTRIB workshop deadline referenced in this log (lines 10 and 116) was
2026-09-02. The actual deadline was **2026-09-05 AOE**, and the paper was
submitted to the **idea track**. Historical entries above are left as-is.
README.md also carried the stale deadline (line 3) and is being replaced
entirely in this finalization pass.
