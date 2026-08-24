"""One run record: one (task, agent, user simulator, seed, condition) episode.

Forked from simswap's src/schema.py (2026-08-17) as a starting point for the ATTRIB
workshop project - context-ablation study, single fixed agent/user pair, `condition`
identifies which part of the context was altered. read/write tally shape mirrors
tau2-bench's RewardInfo.partial_action_reward (reward_info.partial_action_reward ->
{"read": {"count", "correct"}, "write": {...}}), verified field-by-field against real
tau2 output in the simswap project (see that repo's sweep.py docstring history).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

STATUSES = ("completed", "error", "timeout")

# our normalized categories, not tau2's raw enum - map tau2 -> this in sweep.py.
TERMINATION_REASONS = (
    "normal",
    "max_turns",
    "user_ended",
    "agent_error",
    "harness_crash",
    # Added 2026-08-17 (grid session 1): tau2's own TerminationReason
    # .CONTEXT_WINDOW_EXCEEDED enum value is defined (data_model/simulation.py) but
    # never actually assigned anywhere in the half-duplex orchestrator - a prompt
    # that exceeds --max-model-len raises litellm.exceptions
    # .ContextWindowExceededError UNCAUGHT, propagating out of run_single_task()
    # entirely, before any SimulationRun/termination_reason is ever produced.
    # sweep.py::run_cell() catches this specific exception itself and records this
    # category directly - deterministic at temperature=0, so retrying cannot help
    # (see convert_context_window_exceeded()).
    "context_window_exceeded",
)

# the 6 context-ablation conditions (ANALYSIS_PLAN.md), "full_context" is the control.
# "history_truncated" is the only one that takes a condition_param (the k value, as a
# string, e.g. "4") - every other condition leaves condition_param None.
CONDITIONS = (
    "full_context",
    "no_policy",
    "policy_truncated",
    "schema_names_only",
    "history_truncated",
    "system_minimized",
)


@dataclass
class ActionTally:
    """Count of correct vs total actions of one type (read or write)."""

    count: int = 0
    correct: int = 0

    def __post_init__(self) -> None:
        if self.count < 0 or self.correct < 0:
            raise ValueError(f"negative action tally: {self}")
        if self.correct > self.count:
            raise ValueError(f"correct > count: {self}")

    @property
    def accuracy(self) -> float | None:
        return self.correct / self.count if self.count > 0 else None


@dataclass
class RunRecord:
    schema_version: int
    run_id: str

    # provenance - tau2-bench changed grading across versions (v1.0.1), runs
    # across tau2 versions are not comparable. pin and record, never upgrade
    # mid-sweep.
    tau2_version: str
    tau2_commit: str
    attrib_commit: str  # this repo's commit for this run

    # identity
    task_id: str
    domain: str  # retail / telecom / etc
    agent_model: str
    agent_model_revision: str  # exact HF revision hash, not a tag
    user_model: str
    user_model_revision: str  # exact HF revision hash, not a tag
    agent_quantization: str  # e.g. "fp16", "awq-int4" - method actually served
    user_quantization: str
    seed: int
    condition: str  # one of CONDITIONS - which part of context was altered
    condition_param: str | None  # e.g. the k value for "history_truncated"; else None

    # config
    temperature: float
    gpu_name: str  # nvidia-smi name, whole sweep must be one GPU model
    timestamp: str  # ISO 8601, UTC
    max_steps: int  # tau2 TextRunConfig cap this run was given, not a default assumed

    # outcome
    status: str  # one of STATUSES - our harness-level result
    termination_reason: str  # one of TERMINATION_REASONS - why the episode ended
    error: str | None  # null on success, traceback string on failure
    task_success: bool | None  # binary metric, SECONDARY only
    reward: float | None  # raw tau2 reward
    reward_info_note: dict | None  # tau2 RewardInfo.info - free-text debug note, e.g. why reward is 0
    n_turns: int  # control variable - some simulators talk longer

    # action-level correctness - PRIMARY METRIC
    read: ActionTally
    write: ActionTally

    # extras
    wall_clock_seconds: float | None
    agent_cost: float | None
    user_cost: float | None
    transcript_path: str  # full conversation on disk, non-negotiable

    def __post_init__(self) -> None:
        if self.status not in STATUSES:
            raise ValueError(f"bad status {self.status!r}, must be one of {STATUSES}")
        if self.termination_reason not in TERMINATION_REASONS:
            raise ValueError(
                f"bad termination_reason {self.termination_reason!r}, "
                f"must be one of {TERMINATION_REASONS}"
            )
        if self.condition not in CONDITIONS:
            raise ValueError(f"bad condition {self.condition!r}, must be one of {CONDITIONS}")
        if self.condition == "history_truncated" and not self.condition_param:
            raise ValueError(f"run {self.run_id}: condition=history_truncated needs condition_param (the k value)")
        if self.condition != "history_truncated" and self.condition_param is not None:
            raise ValueError(
                f"run {self.run_id}: condition_param must be None for condition={self.condition!r}, "
                f"got {self.condition_param!r}"
            )
        if isinstance(self.read, dict):
            self.read = ActionTally(**self.read)
        if isinstance(self.write, dict):
            self.write = ActionTally(**self.write)
        if self.status == "completed":
            if self.reward is None:
                raise ValueError(f"completed run {self.run_id} has no reward")
            if self.error is not None:
                raise ValueError(f"completed run {self.run_id} has an error set")
        else:
            if not self.error:
                raise ValueError(f"non-completed run {self.run_id} has no error")
        if not self.transcript_path:
            raise ValueError(f"run {self.run_id} has no transcript_path")
        if not self.gpu_name:
            raise ValueError(f"run {self.run_id} has no gpu_name")
        if self.n_turns < 0:
            raise ValueError(f"run {self.run_id} has negative n_turns")
        if self.max_steps <= 0:
            raise ValueError(f"run {self.run_id} has non-positive max_steps")
        if not self.agent_quantization or not self.user_quantization:
            raise ValueError(f"run {self.run_id} missing quantization method")

    @property
    def action_total(self) -> int:
        return self.read.count + self.write.count

    @property
    def action_correct(self) -> int:
        return self.read.correct + self.write.correct

    @property
    def action_accuracy(self) -> float | None:
        total = self.action_total
        return self.action_correct / total if total > 0 else None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "RunRecord":
        return cls(**d)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: Path) -> "RunRecord":
        return cls.from_dict(json.loads(path.read_text()))
