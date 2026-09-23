"""LangGraph state. `execution_count` is the loop-guard counter."""
from __future__ import annotations

import operator
from typing import Annotated, Literal, Optional, TypedDict

from .schemas import ExecutorOutput, FinalResponse, PlannerOutput, ReviewerOutput

HaltReason = Literal["loop_cap", "kill_switch", "error"]


class GraphState(TypedDict, total=False):
    user_id: str
    task: str
    memories: list[str]
    plan: Optional[PlannerOutput]
    execution: Optional[ExecutorOutput]
    review: Optional[ReviewerOutput]
    execution_count: int                 # Executor attempts requested so far
    halt_reason: Optional[HaltReason]    # set by any node that must stop the run
    error: Optional[str]
    trace: Annotated[list[str], operator.add]  # append-only audit trail
    final: Optional[FinalResponse]


def initial_state(user_id: str, task: str) -> GraphState:
    return GraphState(
        user_id=user_id,
        task=task,
        memories=[],
        plan=None,
        execution=None,
        review=None,
        execution_count=0,
        halt_reason=None,
        error=None,
        trace=[],
        final=None,
    )
