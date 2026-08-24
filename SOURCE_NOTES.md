# Source notes - tau2-bench interception points for apply_condition()

Verified against a real clone of `sierra-research/tau2-bench` at the pinned commit
`fc0055dc4e0a316c3f83133267fbd6faaa770992` (confirmed via `git log -1`: "chore: prepare
release v1.0.1 - banking_knowledge grading fixes (#408)", 2026-07-16). Cloned to a
scratch dir outside this repo, read only, not committed here. All file:line references
below are against that clone. No guessing - every claim in this file was read directly
from source, not inferred from docs/README.

## Correction to ANALYSIS_PLAN.md's premise - read this first

ANALYSIS_PLAN.md's "Condition -> intervention mapping" section says
`apply_condition()` is meant to "mutate/rebuild the tau2 TextRunConfig... before it
reaches `run_single_task`". **This premise is wrong.** `TextRunConfig`
(`tau2/data_model/simulation.py:514-596`) has no field for policy text, tool schemas,
system instructions, or history length - it only carries model names, domain name,
seed, max_steps, etc. `sweep.py::run_cell()` calls `tau2.runner.run_single_task(config,
task, seed=seed)` (`tau2/runner/batch.py:341`) as a single black-box call: internally
it builds a fresh `Environment` + `Agent` + `Orchestrator` from `config` and `task` and
runs the whole simulation in one shot. There is no config field to set that reaches
into that internal construction.

**The actual interception point is monkeypatching**, applied immediately before each
`run_single_task()` call and reverted immediately after (so `full_context` stays a
true no-op and patches never leak across retries/conditions). This mirrors simswap's
`patch_evaluator_crash_guard` pattern (reimplement-and-replace a verified function),
not a config field. `apply_condition()`'s signature needs to change from
`(config, condition, condition_param) -> config` to something that also carries the
`task` (needed for `policy_truncated`'s per-task section selection) and returns a
context manager or install/uninstall pair, active only for the duration of one
`run_single_task()` call.

## Call chain (verified, file:line)

1. `sweep.py::run_cell()` -> `tau2.runner.run_single_task(config, task, seed=seed)`
   (`tau2/runner/batch.py:341-449`)
2. -> `build_orchestrator(config, task, ...)` -> `build_text_orchestrator(config, task,
   ...)` (`tau2/runner/build.py:356-434`, since `config` is `TextRunConfig` not
   `VoiceRunConfig`)
3. `build_text_orchestrator` line 393: `environment = build_environment(domain,
   solo_mode=solo_mode, env_kwargs=env_kwargs)` - builds a fresh `Environment` by
   calling the registered domain factory (`retail_domain_get_environment`, registered
   at `registry.py:323`).
4. `build_text_orchestrator` lines 395-402: `agent = build_agent(config.effective_agent,
   environment, llm=..., llm_args=..., task=task, solo_mode=solo_mode)`.
5. `build_agent()` (`tau2/runner/build.py:68-125`):
   - line 108: `tools = environment.get_tools()`
   - line 119: `domain_policy=environment.get_policy()`
   - lines 117-125: calls `agent_factory(tools=tools, domain_policy=..., llm=..., ...)`
     where `agent_factory` is `create_llm_agent` (registered `registry.py:297`) ->
     `LLMAgent(tools=tools, domain_policy=domain_policy, llm=llm, llm_args=llm_args)`
     (`tau2/agent/llm_agent.py:489-504`).
6. `LLMAgent.__init__` -> `HalfDuplexAgent.__init__` (`tau2/agent/base_agent.py:70-73`):
   `self.tools = tools` / `self.domain_policy = domain_policy` - **plain reference
   assignment, not a copy**. The exact list/string objects passed in are what every
   subsequent turn of the conversation reads.
7. Per turn, `LLMAgent._generate_next_message()` (`tau2/agent/llm_agent.py:115-135`):
   `messages = state.system_messages + state.messages`, then
   `generate(model=self.llm, tools=self.tools, messages=messages, ...)`.
8. `generate()` (`tau2/utils/llm_utils.py:355-416`): line 389
   `tools_schema = [tool.openai_schema for tool in tools] if tools else None`, then
   line 409-415 `completion(model=model, messages=litellm_messages,
   tools=tools_schema, tool_choice=tool_choice, **kwargs)` - **this is the literal
   point where policy text (embedded in the system message) and tool schemas enter
   the served chat-completion request.**

## Condition -> exact object/field to modify

### `no_policy`

- **Object**: `Environment.policy` (`tau2/environment/environment.py:56-57`, set in
  `__init__`; read via `get_policy()` at lines 71-75, a trivial `return self.policy`).
- **Loaded from**: `tau2/domains/retail/environment.py:17-32` - plain markdown file at
  `RETAIL_POLICY_PATH` = `data/tau2/domains/retail/policy.md` (verified: 137 lines,
  H1 title + unheaded preamble + 6 `##` sections, read below), read once per
  `get_environment()` call and stored verbatim as `Environment.policy`.
- **Where it enters the prompt**: `LLMAgent.domain_policy` -> `system_prompt` property
  (`tau2/agent/llm_agent.py:78-82`) -> `SYSTEM_PROMPT` template
  (`llm_agent.py:34-41`): `<policy>{domain_policy}</policy>`, wrapped in a
  `SystemMessage` in `get_init_state()` (`llm_agent.py:100-103`), prepended to every
  turn's message list (`llm_agent.py:127`).
- **Intervention**: after `environment = build_environment(...)` runs (step 3 above)
  and before `build_agent(...)` runs (step 4), set `environment.policy = ""`.
  Verified safe: `get_policy()` does no validation, just returns the attribute.

### `policy_truncated`

- **Same object/field** as `no_policy` (`Environment.policy`), but set to only the
  task-relevant section(s) instead of `""`.
- **Policy structure** (`data/tau2/domains/retail/policy.md`, read in full): H1 title
  (line 1) + unheaded preamble (lines 3-24, contains the authentication requirement
  and the "confirm before mutating" rule - these apply to every task and arguably
  can't be attributed to one action) + 6 `##`-level sections: `Domain basic` (26),
  `Generic action rules` (80), `Cancel pending order` (86), `Modify pending order`
  (94, with `###` subsections `Modify payment` / `Modify items`), `Return delivered
  order` (116), `Exchange delivered order` (128).
- **Task-relevance mapping (my design, not tau2's)**: `RetailTools` action methods
  (`tau2/domains/retail/tools.py`) map 1:1 to policy sections by name:
  `cancel_pending_order` -> "Cancel pending order"; `modify_pending_order_address` /
  `modify_pending_order_items` / `modify_pending_order_payment` -> "Modify pending
  order" (+ the matching `###` subsection where one exists); `return_delivered_order_items`
  -> "Return delivered order"; `exchange_delivered_order_items` -> "Exchange delivered
  order". Truncated policy = title + preamble (kept - it's the authentication/
  confirmation rule that gates every action, not truncatable without changing what
  "policy" means) + "Generic action rules" (kept, same reason) + the section(s)
  matching the *task's* `evaluation_criteria.actions[*].name`. **Confirmed against real
  data**: grepped every distinct `action.name` across all of `data/tau2/domains/
  retail/tasks.json` (250+ matches) - the only names that appear are
  `get_order_details` (read-only, no dedicated section, falls under the always-kept
  preamble/"Domain basic"), `cancel_pending_order`, `modify_pending_order_items`,
  `modify_pending_order_address`, `modify_pending_order_payment`,
  `return_delivered_order_items`, `exchange_delivered_order_items` - exactly the 6
  mutating actions the mapping above covers, no unmapped names found. Multi-action
  tasks (e.g. `get_order_details` + `cancel_pending_order` in the same task) resolve
  to the union of their sections, already the intended behavior.

### `schema_names_only`

- **Object**: `Tool.openai_schema` property (`tau2/environment/tool.py:139-149`) -
  `{"type": "function", "function": {"name": self.name, "description":
  self._get_description(), "parameters": self.params.model_json_schema()}}`.
- **Two places description text hides**, both need stripping:
  1. `self._get_description()` (`tool.py:157-166`) - `short_desc` + `long_desc`,
     parsed from the tool function's docstring by `Tool.parse_data()`
     (`tool.py:82-113`, `docstring_parser.parse`).
  2. Per-parameter `description` fields inside `self.params.model_json_schema()` -
     `parse_data()` line 103: `default = Field(default, description=doc_param[name]
     .description)` for every documented param. Verified these carry real policy-
     adjacent guidance, not just types - e.g. `RetailTools._get_order`'s docstring
     (`tau2/domains/retail/tools.py:29-43`): `order_id: The order id, such as
     '#W0000000'. Be careful there is a '#' symbol at the beginning of the order id.`
     Stripping only (1) and leaving (2) would leak meaningful content back in.
- **Fresh objects each call**: `ToolKitBase.get_tools()` (`tau2/environment/
  toolkit.py:144-174`) rebuilds `Tool` instances via `as_tool(tool)` every call (line
  161) - but `environment.get_tools()` is called exactly once per `run_single_task`
  (`build.py:108`), and that one list is what's stored as `LLMAgent.tools` and reused
  for every turn (verified step 6-8 above), so stripping needs to happen exactly once,
  on that one list, before `LLMAgent.__init__` runs.
- **Intervention**: after `tools = environment.get_tools()` (the equivalent step
  inside our reimplemented `build_agent` wrapper - see Implementation approach below),
  for each `Tool` in the list: rebuild `params` via `create_model` with each field's
  `Field(default, description=None)` (drop description, keep type/default/required-
  ness) and set `short_desc = long_desc = ""` so `_get_description()` falls back to
  `self.name` (`tool.py:158-160`, already handles the empty case with a
  logger.warning, no crash).

### `history_truncated`

- **Object**: `LLMAgentState.messages` (`tau2/agent/llm_agent.py:47-48`), read in
  `_generate_next_message()` (`llm_agent.py:115-135`): `messages =
  state.system_messages + state.messages` (line 127) - the full accumulated history,
  unconditionally concatenated after the system message(s).
- **`condition_param`**: `k`, the number of trailing turns to keep (already validated
  as required-iff-history_truncated in `schema.py`'s `RunRecord`, per PROGRESS.md).
- **Intervention**: reimplement `LLMAgent._generate_next_message` (verified body,
  `llm_agent.py:115-135`) with one inserted line - `messages = state.system_messages +
  state.messages[-k:]` instead of the full list - and monkeypatch
  `LLMAgent._generate_next_message` (or `generate_next_message`, its public wrapper at
  `llm_agent.py:105-113`, which calls the private one and appends to `state.messages`
  unchanged either way - patching the private method is enough and keeps state
  bookkeeping, i.e. the *stored* history, intact; only what's *sent* to the model is
  truncated) to this version for the duration of the run. Turn counting: `k` counts
  `Message` objects in `state.messages` (system messages excluded, already separate),
  not conversation turns - needs an explicit note in `condition_param`'s docs since
  "last k turns" (ANALYSIS_PLAN.md's phrasing) and "last k messages" are not the same
  thing once tool calls are counted (`MultiToolMessage` unpacks to multiple entries at
  line 124). Flagging, not resolving - decide before running any GPU time on this
  condition specifically, since it changes what k=5 means in practice.

### `system_minimized`

- **Object**: `AGENT_INSTRUCTION` module-level constant (`tau2/agent/llm_agent.py:
  24-32`), read directly by name (not `self.something`) inside `LLMAgent.system_prompt`
  (`llm_agent.py:78-82`): `SYSTEM_PROMPT.format(domain_policy=self.domain_policy,
  agent_instruction=AGENT_INSTRUCTION)`.
- **Full current text** (for reference, what "minimized" is a reduction of):
  ```
  You are a customer service agent that helps the user according to the <policy> provided below.
  In each turn you can either:
  - Send a message to the user.
  - Make a tool call.
  You cannot do both at the same time.

  Try to be helpful and always follow the policy. Always make sure you generate valid JSON only.
  ```
- **Intervention**: monkeypatch the module attribute `tau2.agent.llm_agent.
  AGENT_INSTRUCTION` to a minimal replacement string for the duration of the run, then
  restore. This is a plain global lookup at call time (Python resolves `AGENT_INSTRUCTION`
  in the `system_prompt` property's enclosing module namespace, not a bound reference
  captured at import time), so reassigning `tau2.agent.llm_agent.AGENT_INSTRUCTION`
  before `LLMAgent.system_prompt` is ever accessed (i.e. before `get_init_state()`
  runs, which happens inside `Orchestrator` construction/first turn - before
  `build_agent` even returns is safest) is sufficient. Minimal replacement text: TODO,
  needs an operator call on what "bare minimum" means concretely (e.g. just "You are a
  customer service agent. Follow the policy." vs. dropping the tool-call-format
  reminder too - dropping the JSON-validity reminder specifically risks conflating
  this condition with tool-call format failures, a confound worth avoiding).

## Implementation approach (all 5 conditions, one mechanism)

Since every condition needs code to run strictly between `build_environment()` and
`build_agent()` (or, for `system_minimized`, any time before `LLMAgent.system_prompt`
is first read) - a point `run_single_task()` never exposes as a hook - the concrete
plan is to monkeypatch `tau2.runner.build.build_agent` itself: replace the module
attribute with a wrapper that (a) applies the condition's mutation to `environment`
and/or the `tools` list it returns, (b) calls the real, unmodified `build_agent` (kept
as a reference before patching) with the mutated inputs, (c) is installed immediately
before `run_single_task()` and removed immediately after (try/finally), so a crash
never leaves a patched module state for the next task/condition. `system_minimized`
patches `tau2.agent.llm_agent.AGENT_INSTRUCTION` the same way, same install/finally
scope. `history_truncated` patches `LLMAgent._generate_next_message` the same way.
`full_context` installs no patches at all - a true no-op, not a patch that happens to
do nothing, so there is no way for a bug in the patching machinery to silently affect
the control condition.

This is a source-verified plan, not yet code. Next step per the operator's brief:
implement `apply_condition()` (now understood to mean "install condition-specific
monkeypatches for the duration of one `run_single_task()` call", not "mutate a
config object") against this document, then verify by dumping the actual request
payload per condition (intercept at `tau2/utils/llm_utils.py:389`,
`tools_schema = [tool.openai_schema for tool in tools]`, and the `messages` list
passed into `completion()` at the same call) and diffing against `full_context` -
confirm the intended text is gone and nothing else changed, before any GPU time.
