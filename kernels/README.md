# Kaggle kernel - ATTRIB pilot

`pilot_run.py` is the kernel entry point; `../src/pilot_core.py` (uploaded as part of
the `attrib-code` dataset, alongside `schema.py`/`sweep.py`/`conditions.py`/etc, same
flat-layout pattern as simswap's `gate1_core.py` - read-only reference, that repo is
never edited) does the actual work: install pinned tau2 + vLLM 0.27.1, start one pair
of vLLM servers (Qwen2.5-7B-Instruct-AWQ, self-play, same model both sides per
`ANALYSIS_PLAN.md`'s fixed factors), then run **one episode per condition (6 total),
same task, same seed** through `sweep.run_cell()` - the exact function the real
240-episode sweep uses, just called once per condition here instead of for the full
grid. Purpose: confirm every condition runs end to end and produces a valid
`RunRecord`, and measure real per-condition wall-clock. This is NOT a results run -
n=1 per condition, no statistics, nothing here should be read as an experimental
finding.

## Deploy

```
kaggle datasets create -p ../src        # uploads src/'s contents (dataset-metadata.json lives there)
kaggle kernels push -p .                # uploads pilot_run.py + kernel-metadata.json, starts the run
```

`dataset-metadata.json` (in `../src/`) and `kernel-metadata.json` (here) already have
the real Kaggle username filled in (`abhiprd2000`) - not templates, ready to push as
committed. If the dataset already exists on a later push, use
`kaggle datasets version -p ../src -m "<message>"` instead of `create`.

## Config

- Model: `Qwen/Qwen2.5-7B-Instruct-AWQ`, both agent and user (self-play, matches
  `ANALYSIS_PLAN.md`'s fixed factors).
- Quantization: `awq` (not `awq_marlin` - T4/SM75 has no Marlin kernel; same
  constraint simswap's own real T4 runs verified, read-only reference).
- `max_model_len`: 32768 (simswap's real, working value for the same model).
- tau2-bench pinned commit `fc0055dc4e0a316c3f83133267fbd6faaa770992`.
- `max_steps`: `sweep.MAX_STEPS` = 200, identical across all 6 conditions - see
  `sweep.py`'s comment for provenance (tau2's own default + simswap's real observed
  episode range).
- `history_truncated`'s `condition_param` (k) for this pilot only: `"3"` - a pilot
  convenience value, NOT a decision about the real sweep's k, which `ANALYSIS_PLAN.md`
  does not yet pin.
- Task: the first task in the real sweep's actual 20-task selection
  (`sweep.select_tasks("retail", 20)[0]`), seed 0 - not an arbitrary task, one that
  will actually appear in the real grid.
- `machine_shape: "NvidiaTeslaT4"` in `kernel-metadata.json` - same value simswap's
  own metadata used for its real confirmed 2xT4 runs (read-only reference), reused
  verbatim rather than guessed.

## Output

`/kaggle/working/pilot_summary.json` - one dict, `conditions` keyed by condition name,
each with `status`, `termination_reason`, `n_turns`, `recorded_wall_clock_seconds`,
`measured_wall_clock_seconds`, `task_success`, `action_accuracy`, `error`. Also
per-condition `pilot_<task_id>__<condition>__seed0.json` (the actual `RunRecord`,
schema-validated), `transcripts/pilot_<task_id>__<condition>__seed0.json` (full
message list), `pilot_log.txt`, `vllm_gpu{0,1}_port{8001,8002}.log`,
`vram_timeline_pilot.csv`.
