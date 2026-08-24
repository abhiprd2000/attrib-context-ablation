"""ATTRIB grid, SESSION 2 - the final 10 cells (5 new task_ids x 2 seeds x 6
conditions = 60 episodes) completing the N=12 grid decided 2026-08-18 (see
ANALYSIS_PLAN.md's "Grid size: REVISED 2026-08-18" section). Session 1 already ran
the first 7 task_ids (84 episodes, 81 completed); this session covers the remaining
5, taken as the next slice of the same seeded prefix, not re-picked.

Same serving config as session 1, unchanged: vLLM 0.27.1, Qwen2.5-7B-Instruct-AWQ
self-play, `--quantization awq`, `--enable-auto-tool-choice --tool-call-parser
hermes`, temperature 0.0, MAX_STEPS=200, tau2 pinned
fc0055dc4e0a316c3f83133267fbd6faaa770992. Includes the session-1 crash fix
(`sweep.py::run_cell()` catches `litellm.exceptions.ContextWindowExceededError`
specifically, never retries it, records it as `termination_reason=
"context_window_exceeded"` rather than propagating) and `max_retries=1` (same
reasoning as session 1: a deterministic failure at temperature 0.0 cannot be fixed
by retrying). Each RunRecord is written to disk immediately after it completes, not
batched.

TASK SELECTION - prefix, not a pick (see ANALYSIS_PLAN.md for the full derivation):
`sweep.select_tasks("retail", 20)` at TASK_SAMPLE_SEED=20260814 returns, in order:
['93','43','98','77','66','94','4','46','107','64','102','110','82','73','7','74',
'55','89','15','13']. Session 1 took the first 7 (indices 0-6). Session 2 takes the
next 5 (indices 7-11): ['46','107','64','102','110']. Together these 12 are the
first 12 of the original 20 - the N=12 grid decided 2026-08-18, not a different
selection.

PROJECTED HOURS - real numbers, recomputed 2026-08-18 directly from
data/grid_session1_20260817/grid_session1_summary.json's real per-episode
wall_clock_seconds (the OUTER timer in grid_core.py, captured for every episode
regardless of status - unlike RunRecord.wall_clock_seconds, which is null for the 3
crashed pre-fix episodes and would silently understate their real cost). Per
condition, n=14 each:

| condition | real mean wall-clock |
|---|---|
| no_policy | 882.68s |
| schema_names_only | 594.17s |
| policy_truncated | 436.76s |
| full_context | 385.74s |
| system_minimized | 250.43s |
| history_truncated | 135.51s |

Sum = 2685.27s/cell (matches the 2685.3s figure already in PROGRESS.md's
scope-options table, confirming that number - NOT the earlier 6-episode pilot's
1698.24s figure, which diverged from real per-condition means by up to 32.5x for
some conditions).
10 cells x 2685.27s = 26852.7s = 7.459h, plus ~0.05h server startup ~= 7.51h
projected, leaving ~1.49h (17%) margin under the 9h Kaggle GPU session cap.
**Session 1 overran this same style of projection by 58%** (6.65h projected vs
10.50h real) - so 7.51h is optimistic, not a safe bound; see CONDITION RUN ORDER
below for how this session is hardened against running out of time mid-grid rather
than assuming the projection holds.

CONDITION RUN ORDER - cost-sorted, most expensive first, hardening 2026-08-18:
within each (task, seed) cell, the 6 conditions now run in real-cost order (highest
mean wall-clock first) instead of `schema.CONDITIONS`' declaration order:
`no_policy (882.68s) > schema_names_only (594.17s) > policy_truncated (436.76s) >
full_context (385.74s) > system_minimized (250.43s) > history_truncated (135.51s)`.
If the Kaggle time cap kills the box mid-cell, the two conditions NOT yet attempted
in that cell are the two cheapest (`system_minimized`, `history_truncated`) - cheap
to backfill in a later session - rather than the two most expensive
(`no_policy`, `schema_names_only`), which also happen to be the ones this project's
falsification criterion is built around. Completed cells (all 10, if the session
finishes) are unaffected by ordering; this only protects a session that gets cut
off partway through its LAST cell.

RESUMABILITY - hardened 2026-08-18: a run_path is only treated as "already done"
(skipped) if it exists AND parses as JSON AND has a non-null "status" field
(`_existing_valid_record()` below) - not on bare file existence. A truncated or
corrupt write (e.g. the process killed mid-`write_text()`) is re-run, not silently
treated as complete. CPU-tested against session 1's real 84 records (see
PROGRESS.md): all 84 correctly recognized as valid/skippable, none of the 60 new
(task_id, seed, condition) combinations for session 2's 5 tasks collide with them
(disjoint task_ids), so all 60 are correctly selected to run; a synthetic
truncated-JSON file is correctly NOT skipped.

CHECKPOINT DISCIPLINE - verified by reading the code, not assumed: `record.save
(run_path)` (line below, inside the innermost loop) is called synchronously
immediately after each `sweep.run_cell()` returns, before the loop proceeds to the
next episode - not batched. `RunRecord.save()` (`schema.py:179-181`) calls
`Path.write_text()`, which opens, writes, and closes the file synchronously within
that single call; the write is on disk (OS page cache, survives process kill) by
the time `save()` returns.

NOT PUSHED as of this file's creation (2026-08-18) - Kaggle API access was blocked
by persistent 403 errors as of the end of session 1; the operator has since
confirmed a new API token, verified separately before push.
"""

from __future__ import annotations

import json
import threading
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

from pilot_core import (  # noqa: F401 - AGENT_PORT/USER_PORT/MODEL re-exported for logging
    AGENT_PORT,
    MODEL,
    QUANT_LABEL,
    USER_PORT,
    WORK,
    install_deps,
    log,
    peak_vram_per_gpu,
    sh,
    start_vllm,
    vram_monitor,
    wait_for_server,
)

SESSION1_TASK_COUNT = 7  # tasks already run in session 1 (indices 0-6 of the prefix)
SESSION2_TASK_COUNT = 12  # end of the slice for session 2 (indices 7-11)
HISTORY_TRUNCATED_K = "3"  # same pilot-only convenience value as session 1

# Cost-sorted, most expensive first - real per-condition means from
# data/grid_session1_20260817/grid_session1_summary.json (n=14 each, see module
# docstring). If the Kaggle time cap kills the box mid-cell, this ordering ensures
# the conditions NOT yet attempted are the cheapest ones, not the most expensive /
# most load-bearing ones.
CONDITION_ORDER_BY_COST = [
    "no_policy",           # 882.68s
    "schema_names_only",   # 594.17s
    "policy_truncated",    # 436.76s
    "full_context",        # 385.74s
    "system_minimized",    # 250.43s
    "history_truncated",   # 135.51s
]


def _existing_valid_record(run_path: Path) -> bool:
    """True if run_path holds a parseable RunRecord (safe to skip); false if
    missing, truncated, or corrupt (must be (re-)run). Checked by content, not
    bare file existence, so a write killed mid-flush is retried, not lost."""
    if not run_path.exists():
        return False
    try:
        data = json.loads(run_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return isinstance(data, dict) and data.get("status") is not None


def run_grid_session2() -> dict:
    import schema  # noqa: F401
    import sweep
    from schema import CONDITIONS

    assert set(CONDITION_ORDER_BY_COST) == set(CONDITIONS), (
        "CONDITION_ORDER_BY_COST must be a reordering of schema.CONDITIONS, not a "
        "different set"
    )

    vram_csv = WORK / "vram_timeline_grid_session2.csv"
    log("=== ATTRIB GRID SESSION 2 ===")

    gpu_name = sweep.get_gpu_name()
    attrib_commit = sweep.get_attrib_commit()
    log(f"gpu_name={gpu_name} attrib_commit={attrib_commit}")

    stop_event = threading.Event()
    mon = threading.Thread(target=vram_monitor, args=(stop_event, vram_csv), daemon=True)
    mon.start()

    summary: dict = {
        "gpu_name": gpu_name, "attrib_commit": attrib_commit, "session": 2,
        "model": MODEL, "tau2_commit": sweep.TAU2_COMMIT_PINNED,
        "max_steps": sweep.MAX_STEPS,
        "n_completed": 0, "n_failed": 0, "n_skipped": 0, "episodes": [],
    }
    agent_proc = user_proc = None
    t_start = time.time()
    try:
        agent_proc = start_vllm(0, AGENT_PORT, log_prefix="agent_")
        user_proc = start_vllm(1, USER_PORT, log_prefix="user_")
        agent_up = wait_for_server(AGENT_PORT)
        user_up = wait_for_server(USER_PORT)
        summary["agent_server_up"] = agent_up
        summary["user_server_up"] = user_up
        if not (agent_up and user_up):
            for proc, name in ((agent_proc, "agent"), (user_proc, "user")):
                if proc.poll() is not None:
                    log(f"{name} server exited early with code {proc.returncode}")
            raise RuntimeError("vLLM server(s) failed to come up - see *vllm_gpu*.log for OOM/crash")

        from tau2.runner import get_tasks

        all_task_ids = sweep.select_tasks(sweep.DOMAIN, sweep.N_TASKS)
        session_task_ids = all_task_ids[SESSION1_TASK_COUNT:SESSION2_TASK_COUNT]
        tasks = {t.id: t for t in get_tasks(sweep.DOMAIN, task_ids=session_task_ids)}
        summary["session_task_ids"] = session_task_ids
        log(f"session 2 tasks ({len(session_task_ids)} of {sweep.N_TASKS}, "
            f"slice [{SESSION1_TASK_COUNT}:{SESSION2_TASK_COUNT}]): {session_task_ids}")

        transcripts_dir = WORK / "transcripts"
        transcripts_dir.mkdir(parents=True, exist_ok=True)
        out_dir = WORK / "runs"
        out_dir.mkdir(parents=True, exist_ok=True)

        agent_slug = MODEL.replace("/", "-")

        for task_id in session_task_ids:
            task = tasks[task_id]
            for seed in sweep.SEEDS:
                for condition in CONDITION_ORDER_BY_COST:
                    condition_param = HISTORY_TRUNCATED_K if condition == "history_truncated" else None
                    cond_slug = f"cond-{condition}" + (f"-{condition_param}" if condition_param else "")
                    run_id = f"{task_id}__{agent_slug}__{agent_slug}__{cond_slug}__seed{seed}"
                    run_path = out_dir / f"{run_id}.json"

                    if _existing_valid_record(run_path):
                        log(f"skip {run_id}: valid record already on disk")
                        summary["n_skipped"] += 1
                        continue

                    # max_retries=1: same reasoning as session 1 - a
                    # ContextWindowExceededError at temperature 0.0 is deterministic,
                    # sweep.py::run_cell() catches it specifically and never retries
                    # it regardless of this value; this just avoids burning time on
                    # any OTHER transient failure type.
                    args = SimpleNamespace(
                        domain=sweep.DOMAIN,
                        agent_model=MODEL, agent_revision="grid-session2",
                        agent_url=f"http://localhost:{AGENT_PORT}/v1", agent_quantization=QUANT_LABEL,
                        user_model=MODEL, user_revision="grid-session2",
                        user_url=f"http://localhost:{USER_PORT}/v1", user_quantization=QUANT_LABEL,
                        condition=condition, condition_param=condition_param,
                        max_steps=sweep.MAX_STEPS, max_retries=1, retry_sleep_seconds=5.0,
                        llm_timeout_seconds=300.0,
                    )
                    log(f"--- {run_id} ---")
                    t0 = time.time()
                    record = sweep.run_cell(args, task, seed, run_id, attrib_commit, gpu_name, transcripts_dir)
                    wall = time.time() - t0

                    # Written to disk IMMEDIATELY after each episode.
                    record.save(run_path)

                    log(f"[{run_id}] status={record.status} termination_reason={record.termination_reason} "
                        f"n_turns={record.n_turns} wall={wall:.1f}s")
                    summary["episodes"].append({
                        "run_id": run_id, "status": record.status,
                        "termination_reason": record.termination_reason,
                        "n_turns": record.n_turns, "wall_clock_seconds": wall,
                    })
                    if record.status == "completed":
                        summary["n_completed"] += 1
                    else:
                        summary["n_failed"] += 1

                    # Summary rewritten after EVERY episode too, not just at the end.
                    (WORK / "grid_session2_summary.json").write_text(
                        json.dumps(summary, indent=2, default=str)
                    )
    except Exception:
        log("EXCEPTION:\n" + traceback.format_exc())
        summary["fatal_error"] = traceback.format_exc()
    finally:
        summary["total_wall_clock_seconds"] = time.time() - t_start
        time.sleep(3)
        stop_event.set()
        mon.join(timeout=5)
        summary["peak_vram_mib_per_gpu"] = peak_vram_per_gpu(vram_csv)

        for proc in (agent_proc, user_proc):
            if proc is not None and proc.poll() is None:
                proc.terminate()
        for proc in (agent_proc, user_proc):
            if proc is not None:
                try:
                    proc.wait(timeout=20)
                except Exception:
                    proc.kill()

        sh("nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv")

        (WORK / "grid_session2_summary.json").write_text(json.dumps(summary, indent=2, default=str))
        log("=== GRID SESSION 2 DONE ===\n" + json.dumps(summary, indent=2, default=str))

    return summary
