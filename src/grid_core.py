"""ATTRIB grid, SESSION 1 - a real subset of the 240-episode grid, all 6 conditions
per (task, seed) cell, sized to fit inside one 9h Kaggle GPU session with margin.

Reuses `pilot_core.py`'s infra (install_deps, start_vllm, wait_for_server, logging,
VRAM monitoring, the evaluator crash guard) unchanged - same serving config as the
pilot that already ran clean: vLLM 0.27.1, Qwen2.5-7B-Instruct-AWQ self-play,
`--quantization awq`, `--enable-auto-tool-choice --tool-call-parser hermes`,
temperature 0.0, MAX_STEPS=200, tau2 pinned fc0055dc4e0a316c3f83133267fbd6faaa770992.
Only the run loop is new: iterate real (task, seed) cells x all 6 conditions instead
of the pilot's single task/seed, via `sweep.run_cell()` - the exact function the
real sweep uses - saving each RunRecord to disk immediately after it completes, not
batched at the end, so a session that dies mid-run does not lose completed work.

SESSION 1 SPLIT - real numbers, computed before pushing, not after:
Sum of the 6 real pilot per-condition wall-clocks (one full pass through all 6
conditions for one (task, seed) cell) = 1698.24s (262.41 full_context + 27.13
no_policy + 39.73 policy_truncated + 1093.18 schema_names_only + 174.35
history_truncated + 101.43 system_minimized - see PROGRESS.md/data/
pilot_run_20260817/pilot_summary.json for the source numbers).

SESSION_TASK_COUNT=7 tasks x 2 seeds = 14 cells x 6 conditions = 84 episodes.
14 x 1698.24s = 23775.4s = 6.60h + ~0.05h server startup ~= 6.65h projected,
leaving ~2.35h (26%) margin under the 9h Kaggle GPU session cap. The remaining 13
tasks (26 cells, 156 episodes) are deferred to session 2+ - NOT decided or launched
here; the full grid's real total (~18.87h, PROGRESS.md) needs at least 2 more
sessions after this one.
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

SESSION_TASK_COUNT = 7  # see module docstring for the real-number derivation
HISTORY_TRUNCATED_K = "3"  # SAME pilot-only convenience value - ANALYSIS_PLAN.md
# still does not pin a final k for the real grid; flagged there, not silently
# decided here either.


def run_grid_session1() -> dict:
    import schema  # noqa: F401
    import sweep
    from schema import CONDITIONS

    vram_csv = WORK / "vram_timeline_grid_session1.csv"
    log("=== ATTRIB GRID SESSION 1 ===")

    gpu_name = sweep.get_gpu_name()
    attrib_commit = sweep.get_attrib_commit()
    log(f"gpu_name={gpu_name} attrib_commit={attrib_commit}")

    stop_event = threading.Event()
    mon = threading.Thread(target=vram_monitor, args=(stop_event, vram_csv), daemon=True)
    mon.start()

    summary: dict = {
        "gpu_name": gpu_name, "attrib_commit": attrib_commit, "session": 1,
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
        session_task_ids = all_task_ids[:SESSION_TASK_COUNT]
        tasks = {t.id: t for t in get_tasks(sweep.DOMAIN, task_ids=session_task_ids)}
        summary["session_task_ids"] = session_task_ids
        log(f"session 1 tasks ({len(session_task_ids)} of {sweep.N_TASKS}): {session_task_ids}")

        transcripts_dir = WORK / "transcripts"
        transcripts_dir.mkdir(parents=True, exist_ok=True)
        out_dir = WORK / "runs"
        out_dir.mkdir(parents=True, exist_ok=True)

        agent_slug = MODEL.replace("/", "-")

        for task_id in session_task_ids:
            task = tasks[task_id]
            for seed in sweep.SEEDS:
                for condition in CONDITIONS:
                    condition_param = HISTORY_TRUNCATED_K if condition == "history_truncated" else None
                    cond_slug = f"cond-{condition}" + (f"-{condition_param}" if condition_param else "")
                    run_id = f"{task_id}__{agent_slug}__{agent_slug}__{cond_slug}__seed{seed}"
                    run_path = out_dir / f"{run_id}.json"

                    if run_path.exists():
                        log(f"skip {run_id}: already on disk")
                        summary["n_skipped"] += 1
                        continue

                    # max_retries=1 (changed 2026-08-17 from session 1's 3, after a
                    # real episode burned 73 minutes retrying a deterministic
                    # litellm.exceptions.ContextWindowExceededError 3x at temperature
                    # 0.0 before giving up - the SAME conversation hits the SAME token
                    # count every attempt, so a retry cannot succeed. sweep.py::
                    # run_cell() now catches that exception specifically and never
                    # retries it regardless of this value; max_retries=1 additionally
                    # avoids burning time retrying any OTHER transient failure type,
                    # matching pilot_core.py's original (deliberate, diagnostic)
                    # choice - see PROGRESS.md.
                    args = SimpleNamespace(
                        domain=sweep.DOMAIN,
                        agent_model=MODEL, agent_revision="grid-session1",
                        agent_url=f"http://localhost:{AGENT_PORT}/v1", agent_quantization=QUANT_LABEL,
                        user_model=MODEL, user_revision="grid-session1",
                        user_url=f"http://localhost:{USER_PORT}/v1", user_quantization=QUANT_LABEL,
                        condition=condition, condition_param=condition_param,
                        max_steps=sweep.MAX_STEPS, max_retries=1, retry_sleep_seconds=5.0,
                        llm_timeout_seconds=300.0,
                    )
                    log(f"--- {run_id} ---")
                    t0 = time.time()
                    record = sweep.run_cell(args, task, seed, run_id, attrib_commit, gpu_name, transcripts_dir)
                    wall = time.time() - t0

                    # Written to disk IMMEDIATELY after each episode - a session that
                    # dies mid-grid must not lose completed work.
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
                    (WORK / "grid_session1_summary.json").write_text(
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

        (WORK / "grid_session1_summary.json").write_text(json.dumps(summary, indent=2, default=str))
        log("=== GRID SESSION 1 DONE ===\n" + json.dumps(summary, indent=2, default=str))

    return summary
