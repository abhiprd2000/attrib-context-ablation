"""Context-ablation monkeypatches for tau2's agent-construction path.

`sweep.py::run_cell()` calls `tau2.runner.run_single_task(config, task, seed=seed)`
as a single black-box call - there is no TextRunConfig field that reaches the policy
text, tool schemas, system instructions, or conversation history (verified against a
real clone of tau2-bench at the pinned commit; see SOURCE_NOTES.md, which corrects an
earlier wrong premise in ANALYSIS_PLAN.md that assumed such a field existed). The real
interception is monkeypatching two tau2 functions and one module constant, installed
immediately before one `run_single_task()` call and reverted immediately after via
try/finally, so `full_context` (no patches installed at all) is a true no-op and a
crash never leaves a patched module state for the next task or condition.

Every object/field touched below, and its file:line in the pinned tau2 source, is
documented in SOURCE_NOTES.md - read that first if any of this looks surprising.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Optional

from pydantic import Field, create_model

from schema import CONDITIONS

# ---- policy_truncated: task action name -> policy.md "##" section header ----
# Verified against data/tau2/domains/retail/policy.md (title + preamble + 6 "##"
# sections) and every distinct action.name found across data/tau2/domains/retail/
# tasks.json (250+ matches, grepped in full - see SOURCE_NOTES.md). No unmapped
# action name exists in the real task set.
ACTION_TO_SECTION = {
    "cancel_pending_order": "Cancel pending order",
    "modify_pending_order_address": "Modify pending order",
    "modify_pending_order_items": "Modify pending order",
    "modify_pending_order_payment": "Modify pending order",
    "return_delivered_order_items": "Return delivered order",
    "exchange_delivered_order_items": "Exchange delivered order",
}

# Kept for every task regardless of which actions it needs: the preamble (lines 1-24
# of policy.md - authentication + confirm-before-mutating rules that gate every
# action) is kept implicitly by _split_policy_sections below; "Generic action rules"
# is the one named "##" section that applies to every mutating action, not just one.
ALWAYS_KEPT_SECTIONS = ("Generic action rules",)

# system_minimized: replacement for llm_agent.AGENT_INSTRUCTION. Design call (not a
# tau2-source fact - flagged in SOURCE_NOTES.md as needing one): keeps the tool-call/
# JSON-format mechanics intact so this condition isolates loss of the
# policy-following/persona framing, not loss of output-format comprehension - the two
# are different failure modes and conflating them would confound this condition with
# schema_names_only-style formatting breakage.
AGENT_INSTRUCTION_MINIMAL = (
    "In each turn you can either:\n"
    "- Send a message to the user.\n"
    "- Make a tool call.\n"
    "You cannot do both at the same time.\n\n"
    "Always make sure you generate valid JSON only."
).strip()


def split_policy_sections(policy_text: str) -> tuple[str, dict[str, str]]:
    """Split policy.md into (preamble_and_h1, {section_title: full_section_text}).

    A "##" section's text includes any "###" subsections under it (policy_truncated
    keeps whole "##" sections, never a partial subsection) up to the next "##" line.
    """
    lines = policy_text.splitlines(keepends=True)
    preamble_lines: list[str] = []
    sections: dict[str, list[str]] = {}
    current: Optional[str] = None
    for line in lines:
        if line.startswith("## "):
            current = line[3:].strip()
            sections[current] = [line]
        elif current is None:
            preamble_lines.append(line)
        else:
            sections[current].append(line)
    return "".join(preamble_lines), {k: "".join(v) for k, v in sections.items()}


def truncated_policy_for_task(policy_text: str, task) -> str:
    """Task-relevant policy.md slice: preamble + always-kept sections + the "##"
    section(s) matching the task's evaluation_criteria.actions[*].name."""
    preamble, sections = split_policy_sections(policy_text)
    action_names = {
        action.name
        for action in (getattr(task.evaluation_criteria, "actions", None) or [])
    }
    keep_titles = set(ALWAYS_KEPT_SECTIONS)
    for name in action_names:
        section_title = ACTION_TO_SECTION.get(name)
        if section_title:
            keep_titles.add(section_title)
    kept = [sections[title] for title in sections if title in keep_titles]
    return preamble + "".join(kept)


def strip_tool_schema(tool) -> None:
    """Mutate a tau2 Tool in place: drop all description text, keep name/params
    shape (names + types + required-ness) untouched - "schema_names_only".

    Two places description text hides in Tool.openai_schema (tool.py:139-149), both
    stripped here: short_desc/long_desc (function-level, tool.py:157-166) and each
    parameter's Field(description=...) inside `params` (tool.py:103, parsed from
    Google-style docstring :param: lines by docstring_parser) - verified real tau2
    tool docstrings put load-bearing hints there (e.g. RetailTools._get_order's
    order_id param: "such as '#W0000000'... careful there is a '#' symbol"), so
    stripping only the function-level description would leak content back in.
    """
    tool.short_desc = ""
    tool.long_desc = ""
    fields = {}
    for name, field in tool.params.model_fields.items():
        default = ... if field.is_required() else field.default
        fields[name] = (field.annotation, Field(default, description=None))
    tool.params = create_model("parameters", **fields)


@contextmanager
def apply_condition(condition: str, condition_param: str | None, task):
    """Install condition-specific monkeypatches for the duration of one
    run_single_task() call; revert them on exit (including on exception).

    `full_context` installs nothing - a true no-op, not a patch that happens to do
    nothing, so a bug in the patching machinery can never silently affect the control
    condition. Every other condition patches `tau2.runner.build.build_agent`
    (build.py:68-125) and/or `tau2.agent.llm_agent` module state; see SOURCE_NOTES.md
    for why those are the correct, verified interception points.
    """
    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition {condition!r}")

    if condition == "full_context":
        yield
        return

    if condition == "history_truncated" and not condition_param:
        raise ValueError("history_truncated requires condition_param (k)")

    import tau2.agent.llm_agent as tau2_llm_agent
    import tau2.runner.build as tau2_build
    from tau2.data_model.message import MultiToolMessage

    real_build_agent = tau2_build.build_agent
    real_generate_next_message = tau2_llm_agent.LLMAgent._generate_next_message
    real_agent_instruction = tau2_llm_agent.AGENT_INSTRUCTION

    def patched_build_agent(agent_name, environment, **kwargs):
        # environment.policy/get_tools() verified plain-attribute / list-of-Tool
        # mutation points - environment.py:56-83, SOURCE_NOTES.md "no_policy" /
        # "policy_truncated" / "schema_names_only".
        if condition == "no_policy":
            environment.policy = ""
        elif condition == "policy_truncated":
            environment.policy = truncated_policy_for_task(environment.policy, task)

        agent = real_build_agent(agent_name, environment, **kwargs)

        if condition == "schema_names_only":
            for tool in agent.tools:
                strip_tool_schema(tool)

        return agent

    def patched_generate_next_message(self, message, state):
        # Reimplements llm_agent.py:115-135 verbatim except the one line changed
        # below (marked). Only the messages sent to generate() are truncated;
        # state.messages keeps the full accumulated history untouched, since nothing
        # else in this condition should change (grading/transcript replay reads the
        # orchestrator's own message log, not this agent-internal state).
        from tau2.data_model.message import UserMessage
        from tau2.utils.llm_utils import generate

        if isinstance(message, UserMessage) and message.is_audio:
            raise ValueError("User message cannot be audio. Use VoiceLLMAgent instead.")
        if isinstance(message, MultiToolMessage):
            state.messages.extend(message.tool_messages)
        else:
            state.messages.append(message)
        k = int(condition_param)
        messages = state.system_messages + state.messages[-k:]  # <-- changed line
        assistant_message = generate(
            model=self.llm,
            tools=self.tools,
            messages=messages,
            call_name="agent_response",
            **self.llm_args,
        )
        return assistant_message

    if condition == "system_minimized":
        tau2_llm_agent.AGENT_INSTRUCTION = AGENT_INSTRUCTION_MINIMAL
    if condition == "history_truncated":
        tau2_llm_agent.LLMAgent._generate_next_message = patched_generate_next_message
    # no_policy / policy_truncated / schema_names_only all go through build_agent.
    tau2_build.build_agent = patched_build_agent

    try:
        yield
    finally:
        tau2_build.build_agent = real_build_agent
        tau2_llm_agent.LLMAgent._generate_next_message = real_generate_next_message
        tau2_llm_agent.AGENT_INSTRUCTION = real_agent_instruction
