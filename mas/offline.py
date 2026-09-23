"""Deterministic stand-ins for the LLMs, used by tests and `--offline` demos.

They exercise the exact same graph, guardrails, schemas and Mem0 storage;
only the model responses are scripted. Output from these runs is labelled
[OFFLINE STUB] so it is never mistaken for real model output.
"""
from __future__ import annotations

from pydantic_ai.models.test import TestModel

from .agents import build_planner, build_reviewer
from .schemas import ExecutorOutput


def stub_planner():
    return build_planner(
        TestModel(
            custom_output_args={
                "goal": "Answer the user's request",
                "steps": [
                    {"step_number": 1, "action": "Recall relevant user preferences"},
                    {"step_number": 2, "action": "Draft the answer following the plan"},
                ],
            }
        )
    )


def stub_reviewer(always: str = "approve"):
    return build_reviewer(
        TestModel(
            custom_output_args={
                "verdict": always,
                "feedback": "Looks complete." if always == "approve" else "Rejected by forced-reject demo.",
                "issues": [] if always == "approve" else ["forced rejection"],
            }
        )
    )


class StubExecutor:
    def __init__(self):
        self.calls = 0

    async def __call__(self, task, plan, feedback, memories) -> ExecutorOutput:
        self.calls += 1
        recalled = "; ".join(memories) if memories else "none"
        return ExecutorOutput(
            result=f"[OFFLINE STUB] attempt {self.calls} for task '{task}'. Recalled memories: {recalled}",
            steps_completed=[s.step_number for s in plan.steps],
            confidence=0.9,
        )
