"""LangGraph orchestration: Planner -> Executor -> Reviewer with a hard loop cap.

    START -> load_memory -> planner -> loop_guard --(count<=3)--> executor -> reviewer
                                           |                                   |
                                           +--(count>3 / kill)--> fallback     +--approve--> finalize
                                                                     |         +--reject---> loop_guard
                                                                     v
                                                         save_memory -> END

Only planner, executor and reviewer call an LLM. Every other node is
deterministic Python, so the control flow itself cannot be talked out of the cap.
"""
from __future__ import annotations

import asyncio
import sys
from dataclasses import dataclass
from typing import Optional

from langgraph.graph import END, START, StateGraph
from pydantic_ai import Agent

from . import config
from .agents import USAGE_LIMITS, ExecutorCallable
from .guardrails import (
    KillSwitchEngaged,
    kill_switch_engaged,
    fallback,
    loop_guard,
    route_after_guard,
    route_after_review,
    route_on_halt,
)
from .memory import MemoryStore
from .schemas import FinalResponse, PlannerOutput, ReviewerOutput
from .state import GraphState, initial_state


@dataclass
class MASDeps:
    planner: Agent[None, PlannerOutput]
    reviewer: Agent[None, ReviewerOutput]
    executor: ExecutorCallable
    memory: Optional[MemoryStore] = None


def _halt(exc: Exception, node: str) -> dict:
    if isinstance(exc, KillSwitchEngaged):
        return {"halt_reason": "kill_switch", "trace": [f"{node}: kill-switch"]}
    return {
        "halt_reason": "error",
        "error": f"{node}: {type(exc).__name__}: {exc}",
        "trace": [f"{node}: error {type(exc).__name__}"],
    }


def _is_rate_limited(exc: BaseException) -> bool:
    text = f"{type(exc).__name__} {exc}".lower()
    return "429" in text or "rate_limit" in text or "ratelimit" in text


async def _with_rate_limit_retry(make_call, node: str):
    """Bounded retry for HTTP 429 only: at most RATE_LIMIT_RETRIES extra attempts with a
    fixed wait. Any other error, or a 429 after the last retry, propagates to the node's
    fallback handling. This is deliberately not an open-ended loop."""
    for attempt in range(config.RATE_LIMIT_RETRIES + 1):
        try:
            return await make_call()
        except Exception as exc:
            if attempt == config.RATE_LIMIT_RETRIES or not _is_rate_limited(exc) or kill_switch_engaged():
                raise
            print(f"[{node}] rate limited; waiting {config.RATE_LIMIT_WAIT_S:.0f}s "
                  f"(retry {attempt + 1}/{config.RATE_LIMIT_RETRIES})", file=sys.stderr)
            await asyncio.sleep(config.RATE_LIMIT_WAIT_S)


def build_graph(deps: MASDeps):
    async def load_memory(state: GraphState) -> dict:
        if deps.memory is None:
            return {"memories": [], "trace": ["load_memory: disabled"]}
        try:
            mems = await deps.memory.recall(state["user_id"], state["task"])
        except Exception as exc:  # memory is best-effort, never blocks the run
            return {"memories": [], "trace": [f"load_memory: unavailable ({type(exc).__name__})"]}
        return {"memories": mems, "trace": [f"load_memory: {len(mems)} facts"]}

    async def planner(state: GraphState) -> dict:
        if kill_switch_engaged():  # stop before the first LLM call, not only at loop_guard
            return {"halt_reason": "kill_switch", "trace": ["planner: kill-switch"]}
        mem = "\n".join(f"- {m}" for m in state.get("memories", [])) or "- (none)"
        prompt = f"Task: {state['task']}\n\nRelevant user memories:\n{mem}"
        try:
            res = await _with_rate_limit_retry(
                lambda: deps.planner.run(prompt, usage_limits=USAGE_LIMITS), "planner")
        except Exception as exc:
            return _halt(exc, "planner")
        return {"plan": res.output, "trace": [f"planner: {len(res.output.steps)} steps"]}

    async def executor(state: GraphState) -> dict:
        review = state.get("review")
        feedback = review.feedback if review and review.verdict == "reject" else None
        try:
            out = await _with_rate_limit_retry(
                lambda: deps.executor(
                    task=state["task"],
                    plan=state["plan"],
                    feedback=feedback,
                    memories=state.get("memories", []),
                ),
                "executor",
            )
        except Exception as exc:
            return _halt(exc, "executor")
        return {"execution": out, "trace": [f"executor: attempt {state['execution_count']}"]}

    async def reviewer(state: GraphState) -> dict:
        plan = state["plan"]
        steps = "\n".join(f"{s.step_number}. {s.action}" for s in plan.steps)
        mem = "\n".join(f"- {m}" for m in state.get("memories", [])) or "- (none)"
        prompt = (
            f"Task: {state['task']}\n\nPlan:\n{steps}\n\nUser memories:\n{mem}\n\n"
            f"Executor result:\n{state['execution'].result}"
        )
        try:
            res = await _with_rate_limit_retry(
                lambda: deps.reviewer.run(prompt, usage_limits=USAGE_LIMITS), "reviewer")
        except Exception as exc:
            return _halt(exc, "reviewer")
        return {"review": res.output, "trace": [f"reviewer: {res.output.verdict}"]}

    def finalize(state: GraphState) -> dict:
        final = FinalResponse(
            status="completed",
            answer=state["execution"].result,
            execution_count=state["execution_count"],
            reason="approved by reviewer",
        )
        return {"final": final, "trace": ["finalize: completed"]}

    async def save_memory(state: GraphState) -> dict:
        if deps.memory is None:
            return {"trace": ["save_memory: disabled"]}
        final = state["final"]
        messages = [{"role": "user", "content": state["task"]}]
        # Never persist unapproved output. Assistant answers are only stored when Mem0
        # condenses them into facts (infer=True); raw answers would bloat later prompts.
        if final.status == "completed" and deps.memory.infer:
            messages.append({"role": "assistant", "content": final.answer})
        try:
            await deps.memory.remember(state["user_id"], messages)
        except Exception as exc:
            return {"trace": [f"save_memory: failed ({type(exc).__name__})"]}
        return {"trace": [f"save_memory: stored {len(messages)} message(s)"]}

    g = StateGraph(GraphState)
    g.add_node("load_memory", load_memory)
    g.add_node("planner", planner)
    g.add_node("loop_guard", loop_guard)
    g.add_node("executor", executor)
    g.add_node("reviewer", reviewer)
    g.add_node("finalize", finalize)
    g.add_node("fallback", fallback)
    g.add_node("save_memory", save_memory)

    g.add_edge(START, "load_memory")
    g.add_edge("load_memory", "planner")
    g.add_conditional_edges("planner", route_on_halt("loop_guard"), ["loop_guard", "fallback"])
    # THE loop-cap edge: execution_count > MAX_EXECUTIONS -> fallback
    g.add_conditional_edges("loop_guard", route_after_guard, ["executor", "fallback"])
    g.add_conditional_edges("executor", route_on_halt("reviewer"), ["reviewer", "fallback"])
    g.add_conditional_edges("reviewer", route_after_review, ["finalize", "loop_guard", "fallback"])
    g.add_edge("finalize", "save_memory")
    g.add_edge("fallback", "save_memory")
    g.add_edge("save_memory", END)
    return g.compile()


async def run_task(app, user_id: str, task: str) -> GraphState:
    return await app.ainvoke(
        initial_state(user_id, task), config={"recursion_limit": config.RECURSION_LIMIT}
    )
