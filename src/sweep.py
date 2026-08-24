"""Run one (agent, user model, condition) combination through tau2-bench retail,
N tasks x 2 seeds. Forked from simswap's src/sweep.py (2026-08-17) for the ATTRIB
context-ablation study: agent and user model are fixed across the whole study
(Qwen2.5-7B-Instruct-AWQ both sides) - only `condition` (which part of the context
was altered) and, for one condition, `condition_param` (the truncation k) vary.

Runs ONE (task-grid, condition) combination per invocation against vLLM servers you
already started - starting them is an outer loop, not this script's job. Because the
agent/user models never change across conditions, the SAME servers serve every
condition - no restart overhead between invocations, unlike simswap's original
agent-pair sweep. Resumable: re-running skips any (task, seed) cell already on disk
for that condition, checkpointed per episode.

Context ablation is implemented in conditions.py, not here - build_config() below
only builds the plain TextRunConfig (verified against real tau2 source: TextRunConfig
has no field for policy/tools/system-instructions/history, so there is nothing for
this function to mutate; see SOURCE_NOTES.md). run_cell() wraps the run_single_task()
call in conditions.apply_condition(), which installs condition-specific monkeypatches
for that one call and reverts them immediately after.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import random
import subprocess
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

from conditions import apply_condition
from schema import CONDITIONS, ActionTally, RunRecord

TAU2_VERSION_PINNED = "1.0.1"
TAU2_COMMIT_PINNED = "fc0055dc4e0a316c3f83133267fbd6faaa770992"

# 200, identical across all 6 conditions (no condition in conditions.py ever touches
# max_steps - only apply_condition()'s own patched functions run, and none of them
# reference it). Two independent sources agree on 200, not invented:
#   1. tau2.config.DEFAULT_MAX_STEPS at the pinned commit is 200 - verified directly
#      against the real cloned source (tau2/config.py:4), not assumed from docs.
#   2. simswap's own real observed episodes (F:/simswap/data/runs/*.json, 7 real
#      completed runs, read-only reference - never touch F:/simswap) actually
#      exercised this cap: n_turns ranged 30-201 across those 7, and one
#      (qwen_task2.json) genuinely ran to termination_reason="max_turns" at
#      n_turns=201 - so 200 is a real bound that was really hit, not a theoretical
#      ceiling nobody reaches.
MAX_STEPS = 200

GPU_NAME_DRY_RUN = "DRY_RUN_NO_GPU"

DOMAIN = "retail"
N_TASKS = 20
SEEDS = (0, 1)
TASK_SAMPLE_SEED = 20260814  # date this grid was pre-registered, ANALYSIS_PLAN.md

# tau2.data_model.simulation.TerminationReason -> our normalized set. Keyed by the
# enum's .value (lowercase snake_case), NOT its .name - inherited from simswap, where
# this was verified against real tau2 output at the pinned commit.
TERMINATION_REASON_MAP = {
    "user_stop": "normal",
    "agent_stop": "normal",
    "max_steps": "max_turns",
    "too_many_errors": "agent_error",
    "agent_error": "agent_error",
    "user_error": "harness_crash",
    "timeout": "harness_crash",
    "infrastructure_error": "harness_crash",
    # This branch is unreachable in practice (see convert_context_window_exceeded()'s
    # docstring: tau2's half-duplex orchestrator never actually sets this
    # termination_reason on a real SimulationRun - the raising exception escapes
    # run_single_task() uncaught first). Kept mapped to our own matching category,
    # not "harness_crash", in case a future tau2 version ever does set it gracefully.
    "context_window_exceeded": "context_window_exceeded",
    "unexpected_error": "harness_crash",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--agent-model", required=False)
    p.add_argument("--agent-revision", required=False)
    p.add_argument("--agent-url", required=False, help="vLLM OpenAI-compatible base_url, e.g. http://localhost:8001/v1")
    p.add_argument("--agent-quantization", required=False, help='e.g. "fp16", "awq-int4"')
    p.add_argument("--user-model", required=False)
    p.add_argument("--user-revision", required=False)
    p.add_argument("--user-url", required=False)
    p.add_argument("--user-quantization", required=False, help='e.g. "fp16", "awq-int4"')
    p.add_argument("--condition", required=False, choices=CONDITIONS)
    p.add_argument("--condition-param", required=False, default=None,
                    help='required (the k value) iff --condition=history_truncated')
    p.add_argument("--domain", default=DOMAIN)
    p.add_argument("--out", type=Path, default=Path("runs"))
    p.add_argument("--max-retries", type=int, default=3)
    p.add_argument("--retry-sleep-seconds", type=float, default=5.0)
    p.add_argument("--llm-timeout-seconds", type=float, default=120.0)
    p.add_argument("--max-steps", type=int, default=MAX_STEPS)
    p.add_argument("--print-tasks", action="store_true", help="print the selected task ids and exit")
    p.add_argument("--dry-run", action="store_true",
                    help="mock the LLM call, no GPU, exercise the real write path, one fake record to --out")
    args = p.parse_args()
    if args.print_tasks or args.dry_run:
        return args
    missing = [n for n in ("agent_model", "agent_revision", "agent_url", "agent_quantization",
                            "user_model", "user_revision", "user_url", "user_quantization", "condition")
               if getattr(args, n) is None]
    if missing:
        p.error(f"missing required args: {missing}")
    if args.condition == "history_truncated" and not args.condition_param:
        p.error("--condition-param (the k value) is required when --condition=history_truncated")
    if args.condition != "history_truncated" and args.condition_param is not None:
        p.error(f"--condition-param must be omitted for --condition={args.condition}")
    return args


def check_tau2_version() -> None:
    try:
        installed = importlib.metadata.version("tau2")
    except importlib.metadata.PackageNotFoundError:
        installed = importlib.metadata.version("tau2-bench")
    if installed != TAU2_VERSION_PINNED:
        raise RuntimeError(
            f"tau2 version mismatch: pinned {TAU2_VERSION_PINNED}, installed {installed}. "
            "Never upgrade mid-sweep - see ANALYSIS_PLAN.md."
        )


def get_attrib_commit() -> str:
    """Falls back to "no-commit-yet" rather than crashing: this repo is deliberately
    kept commit-free during validation (see PROGRESS.md's standing rule) - a bare
    `git rev-parse HEAD` has no ref to resolve until the operator commits."""
    repo_dir = Path(__file__).parent
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_dir, stderr=subprocess.DEVNULL
        ).decode().strip()
    except subprocess.CalledProcessError:
        return "no-commit-yet"


def get_gpu_name() -> str:
    """One GPU model for the box, via nvidia-smi. T4 vs P100 fp16 differ at temp=0."""
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"]
    ).decode()
    names = {line.strip() for line in out.splitlines() if line.strip()}
    if not names:
        raise RuntimeError("nvidia-smi reported no GPUs")
    if len(names) > 1:
        raise RuntimeError(f"mixed GPU models on this box: {sorted(names)} - sweep needs one model")
    return names.pop()


def check_gpu_consistency(out_dir: Path, gpu_name: str) -> None:
    """Every run file already on disk must match this session's GPU model."""
    for p in out_dir.glob("*.json"):
        if p.name == "ground_truth.json":
            continue
        record = RunRecord.load(p)
        if record.gpu_name == GPU_NAME_DRY_RUN:
            continue
        if record.gpu_name != gpu_name:
            raise RuntimeError(
                f"GPU mismatch: {p.name} was run on {record.gpu_name!r}, "
                f"this session has {gpu_name!r}. Do not mix GPU models in one sweep."
            )


def select_tasks(domain: str, n: int) -> list[str]:
    from tau2.runner import get_tasks

    tasks = get_tasks(domain, task_split_name="base")
    ids = sorted((t.id for t in tasks), key=lambda x: int(x) if x.isdigit() else x)
    return random.Random(TASK_SAMPLE_SEED).sample(ids, n)


def build_config(args: argparse.Namespace, seed: int):
    """Plain TextRunConfig - carries no condition state. Verified against real tau2
    source that TextRunConfig has no field for policy/tools/system-instructions/
    history (see SOURCE_NOTES.md); the actual context ablation happens in
    conditions.apply_condition(), wrapped around the run_single_task() call in
    run_cell() below, not here."""
    from tau2.data_model.simulation import TextRunConfig

    llm_args = {"temperature": 0.0, "timeout": args.llm_timeout_seconds}
    return TextRunConfig(
        domain=args.domain,
        agent="llm_agent",
        user="user_simulator",
        llm_agent=f"openai/{args.agent_model}",
        llm_args_agent={**llm_args, "api_base": args.agent_url, "api_key": "EMPTY"},
        llm_user=f"openai/{args.user_model}",
        llm_args_user={**llm_args, "api_base": args.user_url, "api_key": "EMPTY"},
        seed=seed,
        max_steps=args.max_steps,
    )


def action_tally(partial: dict | None, key: str) -> ActionTally:
    if partial is None:
        return ActionTally(0, 0)
    sub = partial.get(key)
    if sub is None:
        return ActionTally(0, 0)
    return ActionTally(count=sub["count"], correct=sub["correct"])


def convert_success(sim, run_id: str, task_id: str, args: argparse.Namespace, seed: int,
                     attrib_commit: str, gpu_name: str, transcript_path: str) -> RunRecord:
    reward_info = sim.reward_info
    if reward_info is None:
        raise RuntimeError(f"run {run_id}: completed simulation has no reward_info")
    partial = reward_info.partial_action_reward
    reason = TERMINATION_REASON_MAP[sim.termination_reason.value if hasattr(sim.termination_reason, "value") else sim.termination_reason]

    return RunRecord(
        schema_version=1,
        run_id=run_id,
        tau2_version=TAU2_VERSION_PINNED,
        tau2_commit=TAU2_COMMIT_PINNED,
        attrib_commit=attrib_commit,
        task_id=task_id,
        domain=args.domain,
        agent_model=args.agent_model,
        agent_model_revision=args.agent_revision,
        user_model=args.user_model,
        user_model_revision=args.user_revision,
        agent_quantization=args.agent_quantization,
        user_quantization=args.user_quantization,
        seed=seed,
        condition=args.condition,
        condition_param=args.condition_param,
        temperature=0.0,
        gpu_name=gpu_name,
        timestamp=str(sim.timestamp),
        max_steps=args.max_steps,
        status="completed",
        termination_reason=reason,
        error=None,
        task_success=bool(reward_info.reward >= 1.0 - 1e-6),
        reward=float(reward_info.reward),
        reward_info_note=getattr(reward_info, "info", None),
        n_turns=len(sim.get_messages()),
        read=action_tally(partial, "read"),
        write=action_tally(partial, "write"),
        wall_clock_seconds=float(sim.duration) if sim.duration is not None else None,
        agent_cost=float(sim.agent_cost) if sim.agent_cost is not None else None,
        user_cost=float(sim.user_cost) if sim.user_cost is not None else None,
        transcript_path=transcript_path,
    )


def convert_failure(run_id: str, task_id: str, args: argparse.Namespace, seed: int,
                     attrib_commit: str, gpu_name: str, transcript_path: str, exc: Exception) -> RunRecord:
    is_timeout = "timeout" in type(exc).__name__.lower() or "timeout" in str(exc).lower()
    return RunRecord(
        schema_version=1,
        run_id=run_id,
        tau2_version=TAU2_VERSION_PINNED,
        tau2_commit=TAU2_COMMIT_PINNED,
        attrib_commit=attrib_commit,
        task_id=task_id,
        domain=args.domain,
        agent_model=args.agent_model,
        agent_model_revision=args.agent_revision,
        user_model=args.user_model,
        user_model_revision=args.user_revision,
        agent_quantization=args.agent_quantization,
        user_quantization=args.user_quantization,
        seed=seed,
        condition=args.condition,
        condition_param=args.condition_param,
        temperature=0.0,
        gpu_name=gpu_name,
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        max_steps=args.max_steps,
        status="timeout" if is_timeout else "error",
        termination_reason="harness_crash",
        error="".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
        task_success=None,
        reward=None,
        reward_info_note=None,
        n_turns=0,
        read=ActionTally(0, 0),
        write=ActionTally(0, 0),
        wall_clock_seconds=None,
        agent_cost=None,
        user_cost=None,
        transcript_path=transcript_path,
    )


def convert_context_window_exceeded(run_id: str, task_id: str, args: argparse.Namespace, seed: int,
                                     attrib_commit: str, gpu_name: str, transcript_path: str,
                                     exc: Exception) -> RunRecord:
    """A prompt exceeded --max-model-len (litellm.exceptions.ContextWindowExceededError,
    raised by the served vLLM/OpenAI-compatible endpoint). Verified against the real
    tau2 source (grid session 1, 2026-08-17): tau2's own TerminationReason
    .CONTEXT_WINDOW_EXCEEDED enum value exists (data_model/simulation.py) but is never
    actually assigned anywhere in the half-duplex orchestrator - this exception
    propagates out of run_single_task() UNCAUGHT, before any SimulationRun is ever
    produced, so there is no sim.termination_reason to read. We record this category
    ourselves.

    status="completed" (a harness-level result exists, matching how max_turns/
    agent_error are already treated - see ANALYSIS_PLAN.md's "PRE-REGISTERED"
    sections), NOT status="error" - this is a real, recordable outcome, not a harness
    failure. reward=0.0, matching tau2's own evaluator gate's convention for every
    other premature termination (evaluator.py:119-129: any non-agent_stop/user_stop
    termination gets reward=0.0). n_turns=0/read=write=ActionTally(0,0): no
    SimulationRun means no transcript was ever produced - nothing to score, same as
    convert_failure()'s shape, but recorded as completed, not errored."""
    return RunRecord(
        schema_version=1,
        run_id=run_id,
        tau2_version=TAU2_VERSION_PINNED,
        tau2_commit=TAU2_COMMIT_PINNED,
        attrib_commit=attrib_commit,
        task_id=task_id,
        domain=args.domain,
        agent_model=args.agent_model,
        agent_model_revision=args.agent_revision,
        user_model=args.user_model,
        user_model_revision=args.user_revision,
        agent_quantization=args.agent_quantization,
        user_quantization=args.user_quantization,
        seed=seed,
        condition=args.condition,
        condition_param=args.condition_param,
        temperature=0.0,
        gpu_name=gpu_name,
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        max_steps=args.max_steps,
        status="completed",
        termination_reason="context_window_exceeded",
        error=None,
        task_success=False,
        reward=0.0,
        reward_info_note={"note": f"Context window exceeded (litellm.exceptions.ContextWindowExceededError, "
                                   f"uncaught by tau2): {str(exc)[:500]}"},
        n_turns=0,
        read=ActionTally(0, 0),
        write=ActionTally(0, 0),
        wall_clock_seconds=None,
        agent_cost=None,
        user_cost=None,
        transcript_path=transcript_path,
    )


def run_cell(args: argparse.Namespace, task, seed: int, run_id: str, attrib_commit: str,
             gpu_name: str, transcripts_dir: Path) -> RunRecord:
    from tau2.runner import run_single_task
    from litellm.exceptions import ContextWindowExceededError

    transcript_rel = f"transcripts/{run_id}.json"
    last_exc: Exception | None = None
    for attempt in range(1, args.max_retries + 1):
        try:
            config = build_config(args, seed)
            with apply_condition(args.condition, args.condition_param, task):
                sim = run_single_task(config, task, seed=seed)
            (transcripts_dir / f"{run_id}.json").write_text(
                json.dumps([m.model_dump(mode="json") for m in sim.get_messages()], indent=2)
            )
            return convert_success(sim, run_id, task.id, args, seed, attrib_commit, gpu_name, transcript_rel)
        except ContextWindowExceededError as exc:
            # Deterministic at temperature=0.0 - the SAME conversation will hit the
            # SAME token count on retry. Never retried, regardless of max_retries -
            # a real, one real episode (grid session 1) burned 73 minutes retrying
            # this exact failure 3 times before giving up. Fail fast, record it as
            # its own category, not "harness_crash".
            print(f"  {run_id}: context window exceeded (not retrying, deterministic at temp=0): {exc}")
            (transcripts_dir / f"{run_id}.json").write_text(json.dumps([]))
            return convert_context_window_exceeded(run_id, task.id, args, seed, attrib_commit, gpu_name,
                                                     transcript_rel, exc)
        except Exception as exc:  # noqa: BLE001 - retried, then recorded, never swallowed
            last_exc = exc
            print(f"  attempt {attempt}/{args.max_retries} failed for {run_id}: {exc}")
            if attempt < args.max_retries:
                time.sleep(args.retry_sleep_seconds)

    (transcripts_dir / f"{run_id}.json").write_text(json.dumps([]))
    return convert_failure(run_id, task.id, args, seed, attrib_commit, gpu_name, transcript_rel, last_exc)


class _FakeMessage:
    """Stands in for a tau2 message - just needs model_dump for the transcript writer."""

    def __init__(self, role: str, text: str) -> None:
        self.role = role
        self.text = text

    def model_dump(self, mode: str = "json") -> dict:
        return {"role": self.role, "text": self.text}


class _FakeRewardInfo:
    """Stands in for tau2 RewardInfo - one read miss, everything else clean."""

    def __init__(self) -> None:
        self.reward = 1.0
        self.partial_action_reward = {
            "read": {"count": 4, "correct": 3},
            "write": {"count": 2, "correct": 2},
        }
        self.info = None


class _FakeSim:
    """Stands in for a tau2 simulation result - no LLM, no GPU, real shape."""

    def __init__(self) -> None:
        self.reward_info = _FakeRewardInfo()
        self.termination_reason = "user_stop"
        self.timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self.duration = 1.23
        self.agent_cost = 0.0
        self.user_cost = 0.0

    def get_messages(self) -> list[_FakeMessage]:
        return [_FakeMessage("user", "mock turn"), _FakeMessage("assistant", "mock reply")]


def run_dry_run(out: Path, attrib_commit: str) -> None:
    """One fake cell through the real convert/save/log path, no tau2, no GPU. Does
    NOT go through apply_condition() - it exercises the write path, not the (not yet
    implemented) context-ablation logic."""
    gpu_name = GPU_NAME_DRY_RUN
    transcripts_dir = out / "transcripts"
    transcripts_dir.mkdir(parents=True, exist_ok=True)

    run_id = "dryrun_task00__dry-run-agent__dry-run-user__cond-full_context__seed0"
    run_path = out / f"{run_id}.json"
    transcript_rel = f"transcripts/{run_id}.json"
    fake_args = SimpleNamespace(agent_model="dry-run-agent", agent_revision="dry-run-rev",
                                 agent_quantization="dry-run-quant",
                                 user_model="dry-run-user", user_revision="dry-run-rev",
                                 user_quantization="dry-run-quant",
                                 condition="full_context", condition_param=None,
                                 domain="retail", max_steps=MAX_STEPS)

    if run_path.exists():
        print(f"DRY RUN: {run_path} already on disk, showing it")
        record = RunRecord.load(run_path)
    else:
        sim = _FakeSim()
        (transcripts_dir / f"{run_id}.json").write_text(
            json.dumps([m.model_dump(mode="json") for m in sim.get_messages()], indent=2)
        )
        record = convert_success(sim, run_id, "dryrun_task00", fake_args, 0, attrib_commit, gpu_name, transcript_rel)
        record.save(run_path)
        print(f"DRY RUN: wrote {run_path}")

    print(json.dumps(record.to_dict(), indent=2))
    log_experiment(fake_args, attrib_commit, gpu_name, 1, 0, 0,
                    tau2_version="DRY_RUN", tau2_commit="DRY_RUN")


def log_experiment(args: argparse.Namespace, attrib_commit: str, gpu_name: str,
                    n_completed: int, n_failed: int, n_skipped: int,
                    tau2_version: str = TAU2_VERSION_PINNED, tau2_commit: str = TAU2_COMMIT_PINNED) -> None:
    path = Path("EXPERIMENTS.md")
    header = "| date | attrib_commit | gpu_name | agent_model | user_model | condition | condition_param | tau2_version | tau2_commit | completed | failed | skipped |\n"
    sep = "|---|---|---|---|---|---|---|---|---|---|---|---|\n"
    if not path.exists():
        path.write_text("# Experiments\n\n" + header + sep)
    date = time.strftime("%Y-%m-%d")
    row = (f"| {date} | {attrib_commit[:12]} | {gpu_name} | {args.agent_model} | "
           f"{args.user_model} | {args.condition} | {args.condition_param} | {tau2_version} | {tau2_commit[:12]} | "
           f"{n_completed} | {n_failed} | {n_skipped} |\n")
    with open(path, "a") as f:
        f.write(row)


def main() -> None:
    args = parse_args()

    if args.print_tasks:
        print(select_tasks(DOMAIN, N_TASKS))
        return

    attrib_commit = get_attrib_commit()

    if args.dry_run:
        run_dry_run(args.out, attrib_commit)
        return

    check_tau2_version()
    gpu_name = get_gpu_name()

    args.out.mkdir(parents=True, exist_ok=True)
    check_gpu_consistency(args.out, gpu_name)

    from tau2.runner import get_tasks

    task_ids = select_tasks(args.domain, N_TASKS)
    tasks = {t.id: t for t in get_tasks(args.domain, task_ids=task_ids)}

    transcripts_dir = args.out / "transcripts"
    transcripts_dir.mkdir(parents=True, exist_ok=True)

    n_completed = n_failed = n_skipped = 0

    agent_slug = args.agent_model.replace("/", "-")
    user_slug = args.user_model.replace("/", "-")
    cond_slug = f"cond-{args.condition}" + (f"-{args.condition_param}" if args.condition_param else "")
    for task_id in task_ids:
        for seed in SEEDS:
            run_id = f"{task_id}__{agent_slug}__{user_slug}__{cond_slug}__seed{seed}"
            run_path = args.out / f"{run_id}.json"
            if run_path.exists():
                print(f"skip {run_id}: already on disk")
                n_skipped += 1
                continue

            print(f"running {run_id}")
            record = run_cell(args, tasks[task_id], seed, run_id, attrib_commit, gpu_name, transcripts_dir)
            record.save(run_path)
            if record.status == "completed":
                n_completed += 1
            else:
                n_failed += 1
                print(f"  RECORDED AS FAILED after {args.max_retries} attempts: {run_id} ({record.status})")

    print(f"\ndone: completed={n_completed} failed={n_failed} skipped={n_skipped}")
    log_experiment(args, attrib_commit, gpu_name, n_completed, n_failed, n_skipped)


if __name__ == "__main__":
    main()
