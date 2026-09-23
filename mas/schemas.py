"""Strict output contracts for every agent.

Design rules applied to every model:
* extra="forbid"            -> hallucinated / unexpected keys are rejected
* Literal enums              -> routing decisions are a closed set, never free text
* bounded lengths and lists  -> no model can return an unbounded plan or payload
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class PlanStep(StrictModel):
    step_number: int = Field(ge=1, le=5)
    action: str = Field(min_length=3, max_length=300)


class PlannerOutput(StrictModel):
    """Planner -> Executor contract."""

    goal: str = Field(min_length=3, max_length=300)
    steps: list[PlanStep] = Field(min_length=1, max_length=5)

    @model_validator(mode="after")
    def steps_are_sequential(self) -> "PlannerOutput":
        expected = list(range(1, len(self.steps) + 1))
        if [s.step_number for s in self.steps] != expected:
            raise ValueError(f"step_number must run 1..n in order, expected {expected}")
        return self


class ExecutorOutput(StrictModel):
    """Executor -> Reviewer contract."""

    result: str = Field(min_length=1, max_length=4000)
    steps_completed: list[int] = Field(default_factory=list, max_length=5)
    confidence: float = Field(ge=0.0, le=1.0)


class ReviewerOutput(StrictModel):
    """Reviewer -> router contract. `verdict` alone decides the next edge."""

    verdict: Literal["approve", "reject"]
    feedback: str = Field(max_length=1000)
    issues: list[str] = Field(default_factory=list, max_length=5)


class FinalResponse(StrictModel):
    """The only shape that ever leaves the graph, success or failure."""

    status: Literal["completed", "terminated_loop_cap", "killed", "error"]
    answer: str
    execution_count: int = Field(ge=0)
    reason: str
