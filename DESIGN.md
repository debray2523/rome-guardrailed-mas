# Design decisions

## 1. The failure being designed against

ROME's failure was not a bad answer. It was **unbounded autonomous execution**: a tool loop with no deterministic ceiling, which let the agent keep acting until it found and used capabilities nobody intended it to have. The design principle here is that **every loop has a ceiling enforced by code the model cannot influence**.

## 2. Why a loop cap of N ≤ 3

**Exact semantics.** The Executor may run at most 3 times per task. Before every Executor attempt, the deterministic `loop_guard` node increments `execution_count`. The conditional edge `route_after_guard` sends the run to `fallback` as soon as `execution_count > 3`. The 4th request therefore makes the counter 4, and the Executor never runs a 4th time. The test `test_loop_cap_terminates_after_three_executions` asserts exactly this: 3 executor calls, `execution_count == 4`, and `status == "terminated_loop_cap"`.

**Why 3:**
- **Self-correction gains flatten quickly.** One attempt plus two reviewer-guided revisions captures most of the benefit of reflection loops. If an answer is still rejected after two concrete rounds of feedback, the task is usually underspecified or beyond the model, so more iterations mostly add cost.
- **Worst case is bounded and known in advance.** Maximum LLM calls per task = 1 planner + 3 executor + 3 reviewer = 7. Each call is further capped by `UsageLimits` (3 requests, 6,000 tokens), so cost and exposure can be calculated before deployment.
- **Small damage window.** Every iteration is another chance for an agent to take an unintended action. Three keeps that window small.
- **It is a policy, not a magic number.** `MAX_EXECUTIONS` lives in `config.py` next to every other ceiling, so it can be tuned per risk tier and audited in one place.

**Defence in depth.** A single counter can be bypassed by a routing bug, so four independent layers exist:

| Layer | Mechanism | Catches |
|---|---|---|
| 1 | `execution_count` + conditional edge | normal reject loops |
| 2 | Kill-switch (`MAS_KILL_SWITCH=1` or `KILL_SWITCH` file), checked by `loop_guard` and by a CrewAI `step_callback` on every executor step | operator emergency stop, including mid-execution |
| 3 | LangGraph `recursion_limit=20` (the worst legitimate run needs 14 steps) | graph wiring bugs |
| 4 | Loops inside nodes are bounded too: PydanticAI `retries=1`, CrewAI `max_iter=3` (default 20), `max_execution_time=60s`, `allow_delegation=False` | hidden loops inside frameworks |

Layer 4 matters. A graph-level cap is misleading if a node hides its own loop. PydanticAI retries schema failures and CrewAI agents iterate internally, and both defaults are generous.

**Every exit is safe and typed.** Loop cap, kill-switch, planner/executor/reviewer exceptions and schema failures all route to one `fallback` node. It returns a `FinalResponse` with a closed `status`, never a raw string or a stack trace. Unapproved output is never written to memory.

## 3. Why this PydanticAI schema design

Schemas are the contract between agents, so their job is to make the next step's input **impossible to misread**.

- **`Literal` verdicts drive routing.** `ReviewerOutput.verdict: Literal["approve","reject"]` means the router branches on a closed enum, not on parsing prose ("looks mostly fine?"). `FinalResponse.status` is also closed, so callers can switch on it safely.
- **`extra="forbid"` everywhere.** A hallucinated field such as `"shell_command": "ssh -R ..."` is a validation error, not silently ignored data. For an incident about unauthorised network actions, this is the relevant guarantee.
- **Bounded sizes.** Plans have 1–5 sequentially numbered steps (checked by a model validator), text fields have maximum lengths, and `confidence` is in [0, 1]. No agent can return an unbounded plan that would itself become a long-running loop.
- **Validation at the boundary, not afterwards.** Planner and Reviewer use `pydantic_ai.Agent(output_type=...)`, so the model is constrained to the schema via tool calling and validated on receipt. The CrewAI Executor uses `output_pydantic=ExecutorOutput` and is then re-validated with `model_validate`. Invalid output never reaches state: `test_invalid_planner_output_is_rejected_not_passed_on` shows the Executor is never called.
- **Retries = 1, deliberately.** One repair attempt handles occasional formatting slips. Persistent failure is treated as a signal and sent to the fallback, not retried indefinitely.

## 4. Memory without runaway tokens

Mem0 stores facts persistently: SQLite holds the history/audit log, Chroma the vectors, and local HF embeddings avoid API cost. Each run injects only the **top-3** relevant facts into the prompt, never the raw transcript, so prompt size stays flat as history grows. Memories are keyed by `user_id` (isolation is tested). Memory is best-effort: if it is unavailable, the run proceeds without it rather than failing. Trade-off: `add(infer=True)` costs one LLM call to condense facts.

## 5. What this does *not* solve (scope)

Loop caps alone would not have stopped ROME. Reverse SSH tunnels and GPU hijacking are **capability** failures, and they need controls outside the agent graph:
- sandboxed execution (containers or gVisor, no host access)
- default-deny network egress
- explicit tool allow-lists (here the Executor has `tools=[]`)
- compute quotas and billing alerts
- audit logging of every tool call

This system implements the orchestration-layer controls the brief asks for (deterministic bounds, typed contracts, safe termination) and treats the rest as required platform controls.
