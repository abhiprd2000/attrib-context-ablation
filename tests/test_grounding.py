"""Regression tests for grounding.py's ported algorithm - same fixture-based
approach as simswap's test_failure_analysis.py (the algorithm's correctness was
already established there; these tests confirm the port and the condition-pooling
addition behave identically on the same fixture shapes). Plain asserts, no pytest.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import grounding as g


def tool_call(name: str, arguments: dict) -> dict:
    return {"role": "assistant", "tool_calls": [{"id": "x", "name": name, "arguments": arguments}]}


def tool_result(content, error: bool = False) -> dict:
    return {"role": "tool", "content": content, "error": error}


def get_call_ids(result: dict, call_index: int) -> list[dict]:
    return result["tool_calls_log"][call_index]["ids"]


def find_id(ids: list[dict], field: str, value: str) -> dict:
    for x in ids:
        if x["field"] == field and x["value"] == value:
            return x
    raise AssertionError(f"no id record for {field}={value!r} in {ids}")


CASES = []


def case(fn):
    CASES.append(fn)
    return fn


@case
def test_grounded_via_non_suffixed_arg_tool():
    """find_user_id_by_name_zip has no _id-suffixed argument; its successful
    result must still ground a later get_user_details(user_id=...) call."""
    messages = [
        tool_call("find_user_id_by_name_zip", {"first_name": "Jane", "last_name": "Doe", "zip": "12345"}),
        tool_result("user_abc123", error=False),
        tool_call("get_user_details", {"user_id": "user_abc123"}),
        tool_result('{"user_id": "user_abc123", "name": "Jane Doe"}', error=False),
    ]
    result = g.analyze_conversation("t", messages)
    rec = find_id(get_call_ids(result, 1), "user_id", "user_abc123")
    assert rec["grounded"] is True, f"expected grounded: {rec}"


@case
def test_id_only_in_error_result_stays_ungrounded():
    """An ID that only ever appears in ERROR result content (echoed back) must
    not count as grounded."""
    messages = [
        tool_call("get_item_details", {"item_id": "FAKE-999"}),
        tool_result("Error: item FAKE-999 not found", error=True),
        tool_call("get_item_details", {"item_id": "FAKE-999"}),
        tool_result("Error: item FAKE-999 not found", error=True),
    ]
    result = g.analyze_conversation("t", messages)
    for i in (0, 1):
        rec = find_id(get_call_ids(result, i), "item_id", "FAKE-999")
        assert rec["grounded"] is False, f"error echo must not ground: {rec}"


@case
def test_causal_ordering_future_result_does_not_ground_past_use():
    """An ID used BEFORE any successful result confirms it must be ungrounded at
    the time of use, even if a LATER successful result contains that same value."""
    messages = [
        tool_call("get_order_details", {"order_id": "W123"}),
        tool_result("Error: not found", error=True),
        tool_call("find_user_id_by_email", {"email": "a@b.com"}),
        tool_result("order W123 belongs to user_xyz", error=False),
    ]
    result = g.analyze_conversation("t", messages)
    rec = find_id(get_call_ids(result, 0), "order_id", "W123")
    assert rec["grounded"] is False, f"future result must not retroactively ground: {rec}"


@case
def test_legitimate_grounding_same_and_later_turns():
    """A genuinely grounded ID is recognized both immediately after its grounding
    result and much later; an unrelated, never-grounded argument in that later
    call still shows up as ungrounded."""
    messages = [
        tool_call("get_order_details", {"order_id": "#W1"}),
        tool_result("order #W1: item ITEM-A", error=False),
        tool_call("get_item_details", {"item_id": "ITEM-A"}),
        tool_result("item ITEM-A details", error=False),
        tool_call("get_order_details", {"order_id": "#W1"}),  # later turn, same grounded id
        tool_result("order #W1 again", error=False),
        tool_call("get_item_details", {"item_id": "ITEM-NEVER-SEEN"}),
        tool_result("Error: not found", error=True),
    ]
    result = g.analyze_conversation("t", messages)
    rec_immediate = find_id(get_call_ids(result, 1), "item_id", "ITEM-A")
    assert rec_immediate["grounded"] is True
    rec_later = find_id(get_call_ids(result, 2), "order_id", "#W1")
    assert rec_later["grounded"] is True
    rec_unrelated = find_id(get_call_ids(result, 3), "item_id", "ITEM-NEVER-SEEN")
    assert rec_unrelated["grounded"] is False


@case
def test_empty_pool_conversation_has_no_conditional_evidence():
    """A conversation whose grounding pool never fills (every result is an error)
    must report conditional total 0, not a misleading 0% or 100%."""
    messages = [
        tool_call("get_order_details", {"order_id": "W1"}),
        tool_result("Error", error=True),
        tool_call("get_order_details", {"order_id": "W2"}),
        tool_result("Error", error=True),
    ]
    result = g.analyze_conversation("t", messages)
    assert result["pool_ever_nonempty"] is False
    assert result["total_id_arguments_conditional"] == 0
    assert result["ungrounded_fraction_by_argument_conditional"] is None


@case
def test_pool_by_condition_aggregates_correctly():
    """pool_by_condition must sum raw counts across conversations and recompute
    the fraction from the sums, not average per-conversation fractions - fixture
    built so the two approaches would visibly disagree (0.5 vs 0.625) if the bug
    were present."""
    # conversation A: 1/2 ungrounded (50%); conversation B: 3/4 ungrounded (75%)
    conv_a = {
        "name": "a", "total_id_arguments": 2, "ungrounded_id_arguments": 1,
        "ungrounded_fraction_by_argument": 0.5,
        "total_id_arguments_conditional": 0, "ungrounded_id_arguments_conditional": 0,
        "ungrounded_fraction_by_argument_conditional": None,
    }
    conv_b = {
        "name": "b", "total_id_arguments": 4, "ungrounded_id_arguments": 3,
        "ungrounded_fraction_by_argument": 0.75,
        "total_id_arguments_conditional": 0, "ungrounded_id_arguments_conditional": 0,
        "ungrounded_fraction_by_argument_conditional": None,
    }
    pooled = g.pool_by_condition(
        {"a": conv_a, "b": conv_b},
        {"a": "no_policy", "b": "no_policy"},
    )
    # correct (pooled): (1+3)/(2+4) = 0.6667. buggy (averaged): (0.5+0.75)/2 = 0.625
    assert abs(pooled["no_policy"]["ungrounded_fraction_by_argument"] - (4 / 6)) < 1e-9, pooled
    assert pooled["no_policy"]["n_conversations"] == 2


@case
def test_grounding_rate_by_cell_shape_and_exclusion():
    """Mirrors accuracy_by_cell's shape: one grounded_rate per (task_id, seed)
    cell, keyed by condition; a cell whose conditional rate is undefined (pool
    never filled) is OMITTED, not recorded as 0.0 or 1.0."""
    from types import SimpleNamespace

    grounded_messages = [
        tool_call("get_order_details", {"order_id": "W1"}),
        tool_result("order W1: item ITEM-A", error=False),
        tool_call("get_item_details", {"item_id": "ITEM-A"}),
        tool_result("item ITEM-A details", error=False),
    ]  # 1 conditional id (item_id=ITEM-A), grounded -> conditional ungrounded=0.0 -> grounded_rate=1.0

    empty_pool_messages = [
        tool_call("get_order_details", {"order_id": "W9"}),
        tool_result("Error", error=True),
    ]  # pool never fills -> conditional rate undefined -> cell omitted

    records = [
        SimpleNamespace(run_id="r1", condition="full_context", task_id="1", seed=0),
        SimpleNamespace(run_id="r2", condition="no_policy", task_id="1", seed=0),
    ]
    messages_by_run_id = {"r1": grounded_messages, "r2": empty_pool_messages}
    by_cell = g.grounding_rate_by_cell(records, lambda r: messages_by_run_id[r.run_id])

    assert by_cell["full_context"][("1", 0)] == 1.0, by_cell["full_context"]
    assert ("1", 0) not in by_cell["no_policy"], by_cell["no_policy"]
    assert by_cell["policy_truncated"] == {}, "untouched conditions must stay empty, not missing"


def main() -> None:
    failures = []
    for fn in CASES:
        try:
            fn()
            print(f"PASS: {fn.__name__}")
        except AssertionError as e:
            failures.append(fn.__name__)
            print(f"FAIL: {fn.__name__}: {e}")
        except Exception as e:  # noqa: BLE001
            failures.append(fn.__name__)
            print(f"ERROR: {fn.__name__}: {type(e).__name__}: {e}")

    print(f"\n{len(CASES) - len(failures)}/{len(CASES)} passed")
    if failures:
        print("FAILED:", failures)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
