"""Grounding-rate metric, per condition. Ported from simswap's failure_analysis.py
(the algorithm is unchanged and already validated there - 11 regression tests,
2 independent hand-traces - see that project's docs/AUDIT_REPORT.md); this copy
reads THIS project's transcript format (the bare message-list JSON sweep.py writes
to transcripts/, not simswap's raw_tau2_dump wrapper) and groups by `condition`
instead of by agent model, since agent/user are fixed here and condition is the
factor under test. Validated on hand-built fixtures (tests/test_grounding.py), not
on synthetic statistical data - the same validation strategy simswap used for this
exact algorithm, chosen because grounding is a deterministic per-conversation
property, not a statistical estimate needing a detectability sweep.

An ID-shaped tool-call argument (field name ending `_id`/`_ids`, or one of
`email`/`zip`/`first_name`/`last_name`) is grounded if its exact value appears as a
substring of an earlier successful tool result in the same conversation, checked
causally. Reports both the FLAT rate and the CONDITIONAL rate (restricted to calls
made after the grounding pool already held >=1 success) - see simswap's
docs/FAILURE_ANALYSIS.md for why the conditional rate is the honest headline.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

NARROW_FIELD_RE = re.compile(r"_id$|_ids$")
WIDE_FIELD_NAMES = {"email", "zip", "first_name", "last_name"}


def load_messages(transcript_path) -> list[dict]:
    """This project's sweep.py writes transcripts as a bare list of message dicts
    (role/tool_calls/content/error), not simswap's {"sim_dump": {"messages": ...}}
    wrapper - no policy text is captured in the transcript itself here (see
    ANALYSIS_PLAN.md's open item on capturing per-condition policy text for origin
    classification)."""
    return json.loads(Path(transcript_path).read_text())


def id_values(args: dict) -> list[tuple[str, str]]:
    """(field_name, value) pairs for arguments that look like identifiers."""
    out = []
    for k, v in args.items():
        is_id_field = bool(NARROW_FIELD_RE.search(k)) or (k in WIDE_FIELD_NAMES)
        if not is_id_field:
            continue
        if isinstance(v, str):
            out.append((k, v))
        elif isinstance(v, list):
            for item in v:
                if isinstance(item, str):
                    out.append((k, item))
    return out


def is_grounded_in(value: str, contents: list[str]) -> bool:
    """True if value appears as a substring of any prior successful result."""
    return any(value in c for c in contents)


def analyze_conversation(name: str, messages: list[dict]) -> dict:
    """Same causal-substring-match algorithm as simswap's failure_analysis.py -
    origin classification (user-stated vs. invented) is NOT included here, since it
    needs policy text and user utterances that this project's transcript format does
    not yet capture (see module docstring) - flagged as an open item, not silently
    dropped."""
    grounded_contents: list[str] = []
    tool_calls_log = []
    call_result_queue: list[int] = []

    for i, m in enumerate(messages):
        role = m.get("role")

        if role == "assistant" and m.get("tool_calls"):
            pool_nonempty = len(grounded_contents) > 0
            for tc in m["tool_calls"]:
                ids = id_values(tc.get("arguments") or {})
                id_records = []
                for field, value in ids:
                    is_grounded = is_grounded_in(value, grounded_contents)
                    id_records.append({
                        "field": field, "value": value, "grounded": is_grounded,
                        "pool_nonempty": pool_nonempty,
                    })
                call_index = len(tool_calls_log)
                tool_calls_log.append({
                    "turn": i, "name": tc.get("name"), "ids": id_records,
                    "pool_nonempty": pool_nonempty, "errored": None,
                })
                call_result_queue.append(call_index)

        elif role == "tool":
            content = m.get("content")
            is_error = bool(m.get("error"))
            content_str = content if isinstance(content, str) else json.dumps(content, default=str)

            if call_result_queue:
                call_idx = call_result_queue.pop(0)
                tool_calls_log[call_idx]["errored"] = is_error

            if not is_error:
                grounded_contents.append(content_str)

    total_ids = sum(len(c["ids"]) for c in tool_calls_log)
    ungrounded_ids = sum(1 for c in tool_calls_log for x in c["ids"] if not x["grounded"])
    calls_with_id = sum(1 for c in tool_calls_log if c["ids"])
    calls_with_ungrounded = sum(1 for c in tool_calls_log if any(not x["grounded"] for x in c["ids"]))

    ids_pool_nonempty = [x for c in tool_calls_log for x in c["ids"] if x["pool_nonempty"]]
    ungrounded_pool_nonempty = sum(1 for x in ids_pool_nonempty if not x["grounded"])

    return {
        "name": name,
        # RAW COUNTS below (total_tool_calls, total_id_arguments, ungrounded_id_
        # arguments, calls_with_any_id, calls_with_ungrounded_id, and their
        # _conditional counterparts) are bounded by how long THIS conversation ran
        # before it ended - normally, or via max_steps, or via max_errors=10
        # (tau2's too_many_errors cap, orchestrator.py:747-749 at the pinned
        # commit - see ANALYSIS_PLAN.md's agent_error section). A condition that
        # hits max_errors sooner gets fewer total tool calls and therefore a
        # smaller RAW ungrounded count, independent of its true underlying
        # error rate - comparing raw counts across conversations/conditions with
        # different lengths is invalid (censored by episode length). Only the
        # `ungrounded_fraction_*` fields (ungrounded / total, i.e. a rate over
        # what was actually attempted) are safe for cross-conversation or
        # cross-condition comparison; the raw counts are exposed for
        # transparency/debugging, not as comparable headline numbers.
        "total_tool_calls": len(tool_calls_log),
        "total_id_arguments": total_ids,
        "ungrounded_id_arguments": ungrounded_ids,
        "ungrounded_fraction_by_argument": (ungrounded_ids / total_ids) if total_ids else None,
        "calls_with_any_id": calls_with_id,
        "calls_with_ungrounded_id": calls_with_ungrounded,
        "ungrounded_fraction_by_call": (calls_with_ungrounded / calls_with_id) if calls_with_id else None,
        "total_id_arguments_conditional": len(ids_pool_nonempty),
        "ungrounded_id_arguments_conditional": ungrounded_pool_nonempty,
        "ungrounded_fraction_by_argument_conditional": (
            ungrounded_pool_nonempty / len(ids_pool_nonempty) if ids_pool_nonempty else None
        ),
        "pool_ever_nonempty": len(grounded_contents) > 0,
        "tool_calls_log": tool_calls_log,
    }


def pool_by_condition(per_conversation: dict[str, dict], condition_of: dict[str, str]) -> dict[str, dict]:
    """Aggregate analyze_conversation() results by condition - sums raw counts,
    recomputes fractions from the sums (not an average of per-conversation
    fractions - see simswap's test_matched_pool_aggregates_correctly for why that
    distinction matters)."""
    from collections import defaultdict

    by_cond: dict[str, list[dict]] = defaultdict(list)
    for name, result in per_conversation.items():
        by_cond[condition_of[name]].append(result)

    out = {}
    for condition, results in by_cond.items():
        total = sum(r["total_id_arguments"] for r in results)
        ungrounded = sum(r["ungrounded_id_arguments"] for r in results)
        total_c = sum(r["total_id_arguments_conditional"] for r in results)
        ungrounded_c = sum(r["ungrounded_id_arguments_conditional"] for r in results)
        out[condition] = {
            "n_conversations": len(results),
            # Same censoring caveat as analyze_conversation()'s raw counts, pooled:
            # a condition whose episodes hit max_errors/max_turns sooner (fewer
            # total tool calls per episode) will pool a smaller raw total/
            # ungrounded count even at an identical true error rate - the raw
            # counts here are transparency/debugging fields, NOT a comparable
            # headline number across conditions. Only ungrounded_fraction_* is.
            "total_id_arguments": total,
            "ungrounded_id_arguments": ungrounded,
            "ungrounded_fraction_by_argument": (ungrounded / total) if total else None,
            "total_id_arguments_conditional": total_c,
            "ungrounded_id_arguments_conditional": ungrounded_c,
            "ungrounded_fraction_by_argument_conditional": (ungrounded_c / total_c) if total_c else None,
        }
    return out


def grounding_rate_by_cell(records: list, messages_for) -> dict[str, dict[tuple[str, int], float]]:
    """condition -> {(task_id, seed): grounded_rate}, CONDITIONAL, one value per
    real episode (this grid has exactly one RunRecord per (task_id, seed, condition)
    cell - no averaging needed). Mirrors analyze.py::accuracy_by_cell()'s shape
    exactly so it feeds analyze.py::contrast() unchanged - built 2026-08-23 as
    plumbing for the grounding-rate significance test frozen in ANALYSIS_PLAN.md on
    2026-08-17 ("PRIMARY metric ... paired by (task_id, seed) against the
    full_context control"), which had never actually run because this per-cell
    shape did not exist yet (flagged as an open gap in ANALYSIS_PLAN.md since that
    date). Not a new metric, test, alpha, or Holm family - see PROGRESS.md's
    2026-08-23 entry disclosing that this function was written after this
    project's point estimates (pool_by_condition/average_by_condition aggregates)
    had already been seen, not blind.

    grounded_rate = 1 - ungrounded_fraction_by_argument_conditional (higher is
    better), so analyze.py::contrast()'s existing "positive diff = condition hurt"
    convention (written for action_accuracy, also higher-is-better) applies
    unchanged here and agrees in sign with ANALYSIS_PLAN.md's "ungrounded-rate
    increase" framing by construction: control_grounded - condition_grounded > 0
    iff the condition's ungrounded rate increased relative to control.

    `messages_for(record)` supplies one record's transcript message list (real
    caller: `lambda r: load_messages(transcripts_dir / f"{r.run_id}.json")`) - an
    injected callable rather than a hardcoded path convention, so this stays
    testable without touching disk.

    A cell is OMITTED (not recorded as 0.0/1.0) when the conditional rate is
    undefined - a crashed/context-window-exceeded episode (empty transcript) or one
    whose grounding pool never filled before it ended - same "undefined excluded,
    never treated as 0%/100%" discipline as average_by_condition()."""
    from schema import CONDITIONS

    out: dict[str, dict[tuple[str, int], float]] = {c: {} for c in CONDITIONS}
    for r in records:
        messages = messages_for(r)
        result = analyze_conversation(r.run_id, messages)
        rate = result["ungrounded_fraction_by_argument_conditional"]
        if rate is None:
            continue
        out[r.condition][(r.task_id, r.seed)] = 1.0 - rate
    return out


def average_by_condition(per_conversation: dict[str, dict], condition_of: dict[str, str]) -> dict[str, dict]:
    """Aggregate by condition via a per-conversation (macro) average instead of
    `pool_by_condition`'s pooled (micro) rate - each conversation contributes ONE
    equally-weighted rate, regardless of how many tool calls it made. Added
    2026-08-17 because `pool_by_condition` lets a long conversation (more tool
    calls, e.g. this project's real pilot: full_context ran 20 tool calls vs
    schema_names_only's 4) dominate the pooled denominator - real, verified
    disagreement on the pilot's 6 episodes, see PROGRESS.md. Report BOTH; neither
    is silently preferred.

    Conversations with an undefined rate (0 ID-bearing calls flat, or an
    empty-pool conditional rate) are EXCLUDED from that specific average, not
    counted as 0% or 100% - `n_conversations_in_average` records how many actually
    contributed, which can be less than the condition's total conversation count."""
    from collections import defaultdict

    by_cond: dict[str, list[dict]] = defaultdict(list)
    for name, result in per_conversation.items():
        by_cond[condition_of[name]].append(result)

    out = {}
    for condition, results in by_cond.items():
        flat_rates = [r["ungrounded_fraction_by_argument"] for r in results
                      if r["ungrounded_fraction_by_argument"] is not None]
        cond_rates = [r["ungrounded_fraction_by_argument_conditional"] for r in results
                      if r["ungrounded_fraction_by_argument_conditional"] is not None]
        out[condition] = {
            "n_conversations": len(results),
            "n_conversations_in_flat_average": len(flat_rates),
            "ungrounded_fraction_by_argument_avg": (
                sum(flat_rates) / len(flat_rates) if flat_rates else None
            ),
            "n_conversations_in_conditional_average": len(cond_rates),
            "ungrounded_fraction_by_argument_conditional_avg": (
                sum(cond_rates) / len(cond_rates) if cond_rates else None
            ),
        }
    return out
