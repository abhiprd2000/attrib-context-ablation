# Context ablation on tau2-bench: a failed experiment

Accepted at ATTRIB 2026 (NeurIPS 2026 Workshop on Attributing Model Behavior at Scale), idea track.

Code and data for the ATTRIB 2026 idea-track paper *Ablate-and-Measure Breaks When the
Measurement Depends on the Ablation*.

We set out to attribute tool-call failures to parts of a language agent's context, ran
a 144-episode ablation grid on tau2-bench, and could not interpret the result. The
reason is the finding: per-episode metrics on agentic benchmarks are defined only for
episodes that reach a scoreable state, and ablating the context changes how often
episodes get there. The comparison is then between outcome-selected subsets.

The sharpest symptom is in `tables/grounding_condition_contrasts.csv`. Because paired
testing keeps only cells defined in both arms, the control condition's own grounded
rate moves from 0.096 to 0.304 depending on which treatment it is paired against, even
though the control never changed.

## Result in one table

| condition | n pairs | control | condition | diff | p raw | p Holm |
|---|---|---|---|---|---|---|
| policy removed | 11 | 0.096 | 0.309 | -0.212 | 0.047 | 0.235 |
| policy truncated | 20 | 0.265 | 0.240 | +0.025 | 0.816 | 1.000 |
| schemas stripped | 18 | 0.198 | 0.261 | -0.063 | 0.485 | 1.000 |
| history truncated | 19 | 0.304 | 0.180 | +0.123 | 0.135 | 0.542 |
| system minimized | 11 | 0.299 | 0.208 | +0.090 | 0.517 | 1.000 |

No condition survives Holm correction. The grid had 87% power for a 20-point effect, so
large effects are excluded and smaller ones are not. See `RESULTS_FOR_PAPER.md` for
every number in the paper with its source file.

## Setup

```
pip install -r requirements.txt
```

tau2-bench is not on PyPI. Install from the pinned commit, which is the v1.0.1 release
commit and therefore includes the benchmark's task-quality corrections:

```
git clone https://github.com/sierra-research/tau2-bench.git /tmp/tau2-bench-src
cd /tmp/tau2-bench-src && git checkout fc0055dc4e0a316c3f83133267fbd6faaa770992
pip install /tmp/tau2-bench-src
export TAU2_DATA_DIR=/tmp/tau2-bench-src/data
```

## Reproducing the analysis

The 144 episode records are in this repository, so the tables can be regenerated on CPU
with no GPU and no model server. `analyze.py` reads one directory, so merge the two
sessions first:

```
mkdir -p /tmp/all_runs
cp data/grid_session1_20260817/runs/*.json /tmp/all_runs/
cp data/grid_session2_20260818/runs/*.json /tmp/all_runs/
cd src && python analyze.py --runs-dir /tmp/all_runs --tables-dir ../tables
python detectability_grounding.py
python -m pytest ../tests/test_grounding.py
```

Grounding rate is computed from the transcripts in `data/*/transcripts/`, which are
excluded from git for size. Without them the contrast tables in `tables/` are the
authoritative copy of those numbers.

## Reproducing the runs

Needs two T4-class GPUs and roughly 16 GPU-hours. `kernels/` holds the Kaggle notebooks
used for the pilot, the two grid sessions, and the rerun of failed cells. Both agent and
simulated user are Qwen2.5-7B-Instruct-AWQ served by vLLM 0.27.1 at temperature 0 with a
200-step cap.

## Layout

```
ANALYSIS_PLAN.md      pre-registered design; frozen and archived sections kept verbatim
PROGRESS.md           append-only session log, including decisions later reversed
RESULTS_FOR_PAPER.md  every number in the paper, tagged with its source file
SOURCE_NOTES.md       where each ablation intercepts tau2, with file and line references
src/
  schema.py                   RunRecord definition
  conditions.py               apply_condition(), the six context conditions
  sweep.py                    episode runner and task selection
  grounding.py                grounding-rate metric
  analyze.py                  contrasts, permutation test, Holm correction, attrition
  detectability.py            power simulation for action accuracy
  detectability_grounding.py  power simulation for grounding rate
  fake_runs.py                synthetic records for CPU-only testing
  pilot_core.py               pilot runner
  grid_core.py                session 1 runner
  session2_core.py            session 2 runner, cost-ordered with resume
  rerun_failed_core.py        rerun of failed cells
kernels/                Kaggle notebooks for each GPU session
tests/                  grounding regression tests
tables/                 analysis output, and the payload verification dump
data/                   144 episode records across pilot and two grid sessions
```

## Caveats worth knowing before you reuse this

- `policy_truncated` selects policy sections using each task's ground-truth actions. It
  bounds what perfect retrieval would give and does not simulate a retriever.
- 21 of 144 episodes lost their action-level score to a fault in the assertion judge on
  our side, unevenly across conditions. This affects only the action-level secondary
  metric, not grounding rate. Per-condition counts are in `RESULTS_FOR_PAPER.md`.
- `ANALYSIS_PLAN.md` records two disclosed deviations from the pre-registration and
  keeps the superseded accuracy-based plan in an archived block rather than deleting it.
- `PROGRESS.md` is a working log. Earlier entries describe decisions that later entries
  reverse; read it in order.

## License

MIT for code. See `LICENSE`.

## Citation

See `CITATION.cff`.
