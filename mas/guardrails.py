"""Loop cap, kill-switch and the safe fallback response.

Three independent layers stop a runaway run:
1. execution_count + conditional edge (route_after_guard)  -> primary cap, N <= 3
2. external kill-switch (env var or KILL_SWITCH file)       -> operator override
3. LangGraph recursion_limit                                -> backstop for routing bugs
"""
from __future__ import annotations

import os
from typing import Literal

from . import config
from .schemas import FinalResponse
from .state import GraphState


class KillSwitchEngaged(RuntimeError):
    """Raised inside long-running work (e.g. CrewAI steps) when an operator halts the system."""


def kill_switch_engaged() -> bool:
    flag = os.getenv(config.KILL_SWITCH_ENV, "").strip().lower() in {"1", "true", "yes", "on"}
    return flag or config.KILL_SWITCH_FILE.exists()


def check_kill_switch() -> None:
    if kill_switch_engaged():
        raise KillSwitchEngaged("Kill-switch engaged by operator")


# ---- nodes / routers -------------------------------------------------------

def loop_guard(state: GraphState) -> dict:
    """Deterministic control node (no LLM). Reserves one Executor attempt.

    Increments execution_count *before* the Executor can run, so the counter
    reflects attempts requested. The 4th request makes it 4 (> 3) and the
    conditional edge below sends the run to the fallback instead.
    """
    count = state.get("execution_count", 0) + 1
    update: dict = {"execution_count": count, "trace": [f"loop_guard: execution_count={count}"]}
    if kill_switch_engaged():
        update["halt_reason"] = "kill_switch"
    elif count > config.MAX_EXECUTIONS:
        update["halt_reason"] = "loop_cap"
    return update


def route_after_guard(state: GraphState) -> Literal["executor", "fallback"]:
    """Conditional edge that enforces the cap: execution_count > 3 -> terminate."""
    if state["execution_count"] > config.MAX_EXECUTIONS or state.get("halt_reason"):
        return "fallback"
    return "executor"


def route_after_review(state: GraphState) -> Literal["finalize", "loop_guard", "fallback"]:
    if state.get("halt_reason"):
        return "fallback"
    review = state.get("review")
    if review is not None and review.verdict == "approve":
        return "finalize"
    return "loop_guard"  # every retry must pass the guard again


def route_on_halt(next_node: str):
    """Factory for edges after LLM nodes: any halt_reason short-circuits to fallback."""

    def _route(state: GraphState) -> str:
        return "fallback" if state.get("halt_reason") else next_node

    _route.__name__ = f"route_to_{next_node}_or_fallback"
    return _route


def fallback(state: GraphState) -> dict:
    """Kill-switch / safe fallback. Returns a schema-valid FinalResponse, never raw text."""
    reason = state.get("halt_reason") or "error"
    count = state.get("execution_count", 0)
    if reason == "loop_cap":
        final = FinalResponse(
            status="terminated_loop_cap",
            answer=(
                "I could not produce a reviewer-approved answer within the safety budget "
                f"of {config.MAX_EXECUTIONS} execution attempts, so execution was stopped. "
                "No further actions were taken. Please refine the request or escalate to a human."
            ),
            execution_count=count,
            reason=f"execution_count={count} exceeded MAX_EXECUTIONS={config.MAX_EXECUTIONS}",
        )
    elif reason == "kill_switch":
        final = FinalResponse(
            status="killed",
            answer="Execution was halted by the system kill-switch. No further actions were taken.",
            execution_count=count,
            reason="operator kill-switch engaged",
        )
    else:
        final = FinalResponse(
            status="error",
            answer="The request could not be completed safely. No further actions were taken.",
            execution_count=count,
            reason=(state.get("error") or "unknown error")[:500],
        )
    return {"final": final, "trace": [f"fallback: status={final.status}"]}
