"""ATTRIB grid - targeted rerun of exactly the 3 episodes that crashed in session 1
with litellm.exceptions.ContextWindowExceededError (uncaught harness_crash), now
that sweep.py::run_cell() catches that exception specifically and records it as its
own termination_reason instead of dying. NOT session 2 - this touches only the 3
named cells, nothing else. Reuses pilot_core.py's infra exactly like grid_core.py
does (install_deps, start_vllm, evaluator crash guard, same serving config).

The 3 failed cells, verified against grid_session1_20260817/grid_session1_summary
.json (2026-08-17):
  - task 4,  seed 0, condition no_policy         (wall=4393.1s before crashing)
  - task 4,  seed 1, condition no_policy         (wall=4278.7s before crashing)
  - task 94, seed 1, condition policy_truncated  (wall=1867.2s before crashing)
"""

from __future__ import annotations

import json
import threading
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

from pilot_core import (  # noqa: F401
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

# (task_id, seed, condition, condition_param) - the exact 3 failed cells, nothing else.
FAILED_CELLS = [
    ("4", 0, "no_policy", None),
    ("4", 1, "no_policy", None),
    ("94", 1, "policy_truncated", None),
]


def run_rerun_failed() -> dict:
    import schema  # noqa: F401
    import sweep

    vram_csv = WORK / "vram_timeline_rerun_failed.csv"
    log("=== ATTRIB RERUN: 3 session-1 cells that crashed on context_window_exceeded ===")

    gpu_name = sweep.get_gpu_name()
    attrib_commit = sweep.get_attrib_commit()
    log(f"gpu_name={gpu_name} attrib_commit={attrib_commit}")

    stop_event = threading.Event()
    mon = threading.Thread(target=vram_monitor, args=(stop_event, vram_csv), daemon=True)
    mon.start()

    summary: dict = {
        "gpu_name": gpu_name, "attrib_commit": attrib_commit, "purpose": "rerun_3_failed_cells",
        "model": MODEL, "tau2_commit": sweep.TAU2_COMMIT_PINNED, "max_steps": sweep.MAX_STEPS,
        "n_completed": 0, "n_failed": 0, "episodes": [],
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
            raise RuntimeError("vLLM server(s) failed to come up - see *vllm_gpu*.log")

        from tau2.runner import get_tasks

        task_ids = sorted({c[0] for c in FAILED_CELLS})
        tasks = {t.id: t for t in get_tasks(sweep.DOMAIN, task_ids=task_ids)}
        log(f"tasks needed: {task_ids}")

        transcripts_dir = WORK / "transcripts"
        transcripts_dir.mkdir(parents=True, exist_ok=True)
        out_dir = WORK / "runs"
        out_dir.mkdir(parents=True, exist_ok=True)

        agent_slug = MODEL.replace("/", "-")

        for task_id, seed, condition, condition_param in FAILED_CELLS:
            task = tasks[task_id]
            cond_slug = f"cond-{condition}" + (f"-{condition_param}" if condition_param else "")
            run_id = f"{task_id}__{agent_slug}__{agent_slug}__{cond_slug}__seed{seed}"
            run_path = out_dir / f"{run_id}.json"

            args = SimpleNamespace(
                domain=sweep.DOMAIN,
                agent_model=MODEL, agent_revision="rerun-failed",
                agent_url=f"http://localhost:{AGENT_PORT}/v1", agent_quantization=QUANT_LABEL,
                user_model=MODEL, user_revision="rerun-failed",
                user_url=f"http://localhost:{USER_PORT}/v1", user_quantization=QUANT_LABEL,
                condition=condition, condition_param=condition_param,
                max_steps=sweep.MAX_STEPS, max_retries=1, retry_sleep_seconds=5.0,
                llm_timeout_seconds=300.0,
            )
            log(f"--- {run_id} ---")
            t0 = time.time()
            record = sweep.run_cell(args, task, seed, run_id, attrib_commit, gpu_name, transcripts_dir)
            wall = time.time() - t0

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
            (WORK / "rerun_failed_summary.json").write_text(json.dumps(summary, indent=2, default=str))
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

        (WORK / "rerun_failed_summary.json").write_text(json.dumps(summary, indent=2, default=str))
        log("=== RERUN DONE ===\n" + json.dumps(summary, indent=2, default=str))

    return summary
