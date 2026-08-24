"""ATTRIB pilot: one episode per condition (6 total), same task, same seed.

Purpose: confirm every condition runs end to end against real vLLM-served
Qwen2.5-7B-Instruct-AWQ and produces a valid RunRecord, and measure real
per-condition wall-clock. NOT a results run - n=1 per condition, no statistics.
See PROGRESS.md and ANALYSIS_PLAN.md for the real 240-episode grid this pilot
checks plumbing for.

Adapted from F:/simswap/kernels/gate1_core.py's structure (read-only reference,
that repo is never edited) - same install/vLLM-startup pattern, since this
pilot uses the exact same tau2 pinned commit, vLLM version, and model
(Qwen2.5-7B-Instruct-AWQ self-play) simswap's own real runs already validated
on Kaggle 2xT4 hardware. Rewritten fresh for this repo's own sweep.py/
conditions.py/schema.py, not imported from simswap.
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import threading
import time
import traceback
import urllib.request
from pathlib import Path
from types import SimpleNamespace

WORK = Path("/kaggle/working")
AGENT_PORT, USER_PORT = 8001, 8002

TAU2_COMMIT_PINNED = "fc0055dc4e0a316c3f83133267fbd6faaa770992"
MODEL = "Qwen/Qwen2.5-7B-Instruct-AWQ"
PARSER = "hermes"
QUANTIZATION = "awq"  # NOT awq_marlin - T4/SM75 has no Marlin kernel, verified by
# simswap's own real T4 runs (kernels/gate1_core.py's comment, read-only reference).
QUANT_LABEL = "awq-int4"
MAX_MODEL_LEN = 32768  # simswap's real, working qwen self-play value on the same model.

# pilot-only choice, NOT the real sweep's k - ANALYSIS_PLAN.md does not yet pin a
# final k for history_truncated. Flagged, not silently decided here.
HISTORY_TRUNCATED_K_PILOT = "3"


def _log_path() -> Path:
    return WORK / "pilot_log.txt"


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(_log_path(), "a") as f:
        f.write(line + "\n")


def sh(cmd: str, timeout: int | None = None) -> subprocess.CompletedProcess:
    log(f"$ {cmd}")
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
    log(f"  rc={r.returncode}\nSTDOUT:\n{r.stdout[-4000:]}\nSTDERR:\n{r.stderr[-4000:]}")
    return r


def vram_monitor(stop_event: threading.Event, vram_csv: Path) -> None:
    with open(vram_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ts", "gpu_index", "mem_used_mib"])
        while not stop_event.is_set():
            r = subprocess.run(
                ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
                capture_output=True, text=True,
            )
            ts = time.time()
            for line in r.stdout.strip().splitlines():
                idx, used = [x.strip() for x in line.split(",")]
                w.writerow([ts, idx, used])
            f.flush()
            time.sleep(2)


def peak_vram_per_gpu(vram_csv: Path) -> dict:
    peaks: dict[str, int] = {}
    if not vram_csv.exists():
        return peaks
    with open(vram_csv) as f:
        for row in csv.DictReader(f):
            g, m = row["gpu_index"], int(row["mem_used_mib"])
            peaks[g] = max(peaks.get(g, 0), m)
    return peaks


def wait_for_server(port: int, timeout: int = 900) -> bool:
    url = f"http://localhost:{port}/v1/models"
    start = time.time()
    while time.time() - start < timeout:
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                if resp.status == 200:
                    log(f"server on port {port} is up after {time.time()-start:.0f}s")
                    return True
        except Exception:
            pass
        time.sleep(5)
    return False


def start_vllm(gpu: int, port: int, log_prefix: str = "") -> subprocess.Popen:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["VLLM_ATTENTION_BACKEND"] = "XFORMERS"
    args = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", MODEL, "--dtype", "float16", "--port", str(port),
        "--gpu-memory-utilization", "0.90",
        "--max-model-len", str(MAX_MODEL_LEN),
        "--quantization", QUANTIZATION,
        # tau2's agent sends tool_choice="auto" - vLLM 400s on that without these.
        "--enable-auto-tool-choice", "--tool-call-parser", PARSER,
    ]
    logfile = open(WORK / f"{log_prefix}vllm_gpu{gpu}_port{port}.log", "w")
    log(f"starting vllm: gpu={gpu} port={port} model={MODEL} quantization={QUANTIZATION} "
        f"parser={PARSER} cmd={' '.join(args)}")
    return subprocess.Popen(args, env=env, stdout=logfile, stderr=subprocess.STDOUT)


def install_deps() -> None:
    """One-time setup for a kernel session: pinned tau2 + vllm==0.27.1 + xformers,
    then the evaluator crash guard. Call once, then run_pilot() runs all 6
    conditions in the same session (servers start once - see ANALYSIS_PLAN.md's
    efficiency note: agent/user model never changes across conditions)."""
    WORK.mkdir(parents=True, exist_ok=True)
    sh("nvidia-smi --query-gpu=index,name,memory.total --format=csv")

    TAU2_SRC = "/tmp/tau2-bench-src"
    sh(f"rm -rf {TAU2_SRC}")
    sh(f"git clone --quiet https://github.com/sierra-research/tau2-bench.git {TAU2_SRC}")
    sh(f"cd {TAU2_SRC} && git checkout --quiet {TAU2_COMMIT_PINNED}")
    sh(f"{sys.executable} -m pip install -q {TAU2_SRC}")
    os.environ["TAU2_DATA_DIR"] = f"{TAU2_SRC}/data"
    sh(f"{sys.executable} -m pip show tau2")
    sh(f"{sys.executable} -c \"import tau2; print('tau2 module at', tau2.__file__)\"")

    # pinned, not -U: this exact version is what SOURCE_NOTES.md's CPU-only
    # verification ran against, and what simswap's own real T4 runs used
    # (kernels/gate1_core.py, read-only reference) - don't gamble on "latest".
    sh(f"{sys.executable} -m pip install -q vllm==0.27.1 xformers")

    patch_evaluator_crash_guard()


def patch_evaluator_crash_guard() -> None:
    """tau2.runner.simulation.run_simulation does `simulation = orchestrator.run()`
    (fully populates the transcript), THEN separately calls
    `evaluate_simulation(simulation=simulation, ...)` for the reward. If that second
    call raises - e.g. an nl_assertions check falls through litellm's default and
    hits a missing OPENAI_API_KEY - the exception propagates uncaught and the WHOLE
    simulation object, transcript included, is lost.

    Confirmed independently this session against the real cloned source at the
    pinned commit (tau2/runner/simulation.py, read in full while writing
    SOURCE_NOTES.md): run_simulation has no try/except around the
    evaluate_simulation call, and nothing catches it further up the
    run_single_task -> run_simulation chain either. Same finding simswap's own
    gate1_core.py already made and fixed (read-only reference) - reused here, not
    re-guessed, because it's the same tau2 version and the same code path.

    Must patch tau2.runner.simulation.evaluate_simulation specifically, NOT
    tau2.evaluator.evaluator.evaluate_simulation - run_simulation imported it via
    `from tau2.evaluator.evaluator import evaluate_simulation`, which bound a name
    in tau2.runner.simulation's OWN namespace at import time. Patching the origin
    module doesn't reach that already-bound reference (same bare-name lesson as
    conditions.py's own monkeypatches - see SOURCE_NOTES.md)."""
    import tau2.runner.simulation as tau2_sim
    from tau2.data_model.simulation import RewardInfo

    original = tau2_sim.evaluate_simulation

    def guarded(*args, **kwargs):
        try:
            return original(*args, **kwargs)
        except Exception as exc:
            log(f"WARNING: evaluate_simulation crashed, using stub RewardInfo "
                f"so the transcript isn't lost: {exc}")
            return RewardInfo(reward=0.0, info={"error": f"evaluation crashed: {exc}"})

    tau2_sim.evaluate_simulation = guarded
    log("patched tau2.runner.simulation.evaluate_simulation with crash guard")


def run_pilot() -> dict:
    """Run all 6 conditions, same task, same seed, against one pair of vLLM
    servers started once for this whole session."""
    import schema  # noqa: F401
    import sweep
    from schema import CONDITIONS

    vram_csv = WORK / "vram_timeline_pilot.csv"
    log("=== ATTRIB PILOT: 1 episode per condition (6 total), same task, same seed ===")

    gpu_name = sweep.get_gpu_name()
    log(f"gpu_name={gpu_name}")
    attrib_commit = sweep.get_attrib_commit()
    log(f"attrib_commit={attrib_commit}")

    stop_event = threading.Event()
    mon = threading.Thread(target=vram_monitor, args=(stop_event, vram_csv), daemon=True)
    mon.start()

    summary: dict = {
        "gpu_name": gpu_name, "attrib_commit": attrib_commit,
        "model": MODEL, "quantization": QUANTIZATION, "max_model_len": MAX_MODEL_LEN,
        "tau2_commit": TAU2_COMMIT_PINNED, "max_steps": sweep.MAX_STEPS,
        "conditions": {},
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

        task_id = sweep.select_tasks(sweep.DOMAIN, sweep.N_TASKS)[0]
        task = get_tasks(sweep.DOMAIN, task_ids=[task_id])[0]
        seed = 0
        summary["task_id"] = task_id
        summary["seed"] = seed
        log(f"pilot task_id={task_id} (first of the real 20-task grid, "
            f"sweep.select_tasks seed={sweep.TASK_SAMPLE_SEED}) seed={seed}")

        transcripts_dir = WORK / "transcripts"
        transcripts_dir.mkdir(parents=True, exist_ok=True)

        for condition in CONDITIONS:
            condition_param = HISTORY_TRUNCATED_K_PILOT if condition == "history_truncated" else None
            run_id = f"pilot_{task_id}__{condition}__seed{seed}"
            args = SimpleNamespace(
                domain=sweep.DOMAIN,
                agent_model=MODEL, agent_revision="pilot-live",
                agent_url=f"http://localhost:{AGENT_PORT}/v1", agent_quantization=QUANT_LABEL,
                user_model=MODEL, user_revision="pilot-live",
                user_url=f"http://localhost:{USER_PORT}/v1", user_quantization=QUANT_LABEL,
                condition=condition, condition_param=condition_param,
                max_steps=sweep.MAX_STEPS, max_retries=1, retry_sleep_seconds=5.0,
                llm_timeout_seconds=300.0,
            )
            log(f"--- condition={condition} condition_param={condition_param} ---")
            t0 = time.time()
            record = sweep.run_cell(args, task, seed, run_id, attrib_commit, gpu_name, transcripts_dir)
            measured_wall = time.time() - t0
            record.save(WORK / f"{run_id}.json")
            log(f"condition={condition} status={record.status} "
                f"termination_reason={record.termination_reason} n_turns={record.n_turns} "
                f"recorded_wall_clock_seconds={record.wall_clock_seconds} "
                f"measured_wall_clock_seconds={measured_wall:.1f}")
            summary["conditions"][condition] = {
                "run_id": run_id,
                "status": record.status,
                "termination_reason": record.termination_reason,
                "n_turns": record.n_turns,
                "recorded_wall_clock_seconds": record.wall_clock_seconds,
                "measured_wall_clock_seconds": measured_wall,
                "task_success": record.task_success,
                "action_accuracy": record.action_accuracy,
                "error": record.error,
            }
    except Exception:
        log("EXCEPTION:\n" + traceback.format_exc())
        summary["fatal_error"] = traceback.format_exc()
    finally:
        summary["total_wall_clock_seconds"] = time.time() - t_start
        time.sleep(3)  # let vram monitor grab post-load peak
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

        (WORK / "pilot_summary.json").write_text(json.dumps(summary, indent=2, default=str))
        log("=== PILOT DONE ===\n" + json.dumps(summary, indent=2, default=str))

    return summary
