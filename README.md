# attrib

ATTRIB workshop project (deadline 2026-09-02). Question: which part of the agent's
context causes a tool-call failure? Context-ablation study on tau2-bench retail, one
fixed agent/user pair (Qwen2.5-7B-Instruct-AWQ both sides), 6 conditions (see
`ANALYSIS_PLAN.md`). Forked from the simswap project's harness (`src/schema.py`,
`src/sweep.py`) and grounding-rate algorithm (`src/grounding.py`, ported); does not
touch or depend on the simswap repository.

**Status: pre-GPU validation only. No real runs exist yet.** `src/sweep.py`'s
`apply_condition()` (the actual context-stripping logic for 5 of the 6 conditions) is
deliberately unimplemented - see `ANALYSIS_PLAN.md`'s "OPEN, not yet implemented"
section. Nothing here has touched a GPU.

## Setup

```
pip install -r requirements.txt
```

## What's validated so far (no GPU, no tau2)

```
python tests/test_grounding.py             # 6/6 regression tests on hand-built fixtures
cd src && python sweep.py --dry-run --out ../data/dryrun   # exercises schema.py + sweep.py's
                                             # real write path with a mocked tau2 result
cd src && python fake_runs.py --scenario policy_inert --out ../data/fake   # synthetic
                                             # RunRecords, 6 conditions x N tasks x 2 seeds
cd src && python analyze.py --runs-dir ../data/fake --tables-dir ../data/fake_tables
                                             # per-condition contrast vs full_context,
                                             # Holm-corrected paired permutation test
cd src && python detectability.py           # the grid-size sweep behind ANALYSIS_PLAN.md's
                                             # detectability table (tables/detectability.csv)
```

## Layout

```
ANALYSIS_PLAN.md    pre-registered design, conditions, detectability numbers, open items
PROGRESS.md          append-only session log
src/
  schema.py           RunRecord + condition/condition_param fields (forked from simswap)
  sweep.py             harness CLI (forked from simswap), apply_condition() NOT implemented
  fake_runs.py          synthetic RunRecord generator, 3 scenarios (null/uniform/policy_inert)
  analyze.py             per-condition action-accuracy contrast vs control, Holm-corrected
  detectability.py        in-memory synthetic power sweep -> tables/detectability.csv
  grounding.py             grounding-rate algorithm, ported from simswap, condition-aware
tests/
  test_grounding.py   6 hand-built-fixture regression tests for grounding.py
tables/
  detectability.csv   real output of src/detectability.py (60 replicates/cell)
data/                 gitignored - real runs and fake-data validation output go here
```

## Next steps (not started)

1. Verify tau2-bench's real API for the policy document / tool schemas / system
   instructions (WebFetch the pinned commit, same discipline simswap used for its
   evaluator crash guard - do not guess), then implement `apply_condition()`.
2. Decide the grid size (`n_tasks`) - see `ANALYSIS_PLAN.md`'s detectability table and
   cost estimate; not decided yet, operator call given the deadline.
3. Only then: push a smoke-test kernel, check it once, report back - never poll.
