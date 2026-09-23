"""Deterministic guardrail tests - no API key or network required."""
import asyncio

import pytest
from pydantic_ai.messages import ModelResponse, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel

from mas import config
from mas.agents import build_planner, build_reviewer
from mas.graph import MASDeps, build_graph, run_task
from mas.offline import StubExecutor, stub_planner, stub_reviewer
from mas.schemas import FinalResponse


def run(deps, task="demo task"):
    return asyncio.run(run_task(build_graph(deps), "tester", task))


@pytest.fixture(autouse=True)
def no_kill_switch(monkeypatch, tmp_path):
    monkeypatch.delenv(config.KILL_SWITCH_ENV, raising=False)
    monkeypatch.setattr(config, "KILL_SWITCH_FILE", tmp_path / "KILL_SWITCH")


def test_loop_cap_terminates_after_three_executions():
    executor = StubExecutor()
    state = run(MASDeps(stub_planner(), stub_reviewer("reject"), executor))

    assert executor.calls == config.MAX_EXECUTIONS == 3      # the Executor never runs a 4th time
    assert state["execution_count"] == 4                     # 4th request tripped the cap (> 3)
    final = state["final"]
    assert final.status == "terminated_loop_cap"
    FinalResponse.model_validate(final.model_dump())          # fallback is schema-valid
    assert state["trace"][-2] == "fallback: status=terminated_loop_cap"


def test_happy_path_completes_in_one_execution():
    executor = StubExecutor()
    state = run(MASDeps(stub_planner(), stub_reviewer("approve"), executor))
    assert executor.calls == 1
    assert state["final"].status == "completed"
    assert state["final"].execution_count == 1


def test_reject_then_approve_completes_within_cap():
    verdicts = iter(["reject", "approve"])

    def scripted(messages, info: AgentInfo) -> ModelResponse:
        v = next(verdicts)
        args = {"verdict": v, "feedback": f"scripted {v}", "issues": []}
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, args)])

    executor = StubExecutor()
    state = run(MASDeps(stub_planner(), build_reviewer(FunctionModel(scripted)), executor))
    assert executor.calls == 2
    assert state["final"].status == "completed"
    assert state["execution_count"] == 2


def test_kill_switch_env_halts_before_any_execution(monkeypatch):
    monkeypatch.setenv(config.KILL_SWITCH_ENV, "1")
    executor = StubExecutor()
    state = run(MASDeps(stub_planner(), stub_reviewer("approve"), executor))
    assert executor.calls == 0
    assert state["final"].status == "killed"


def test_kill_switch_file_halts_mid_loop(tmp_path):
    kill_file = config.KILL_SWITCH_FILE

    class TripAfterFirst(StubExecutor):
        async def __call__(self, *a, **kw):
            out = await super().__call__(*a, **kw)
            kill_file.write_text("stop")  # operator pulls the switch during attempt 1
            return out

    executor = TripAfterFirst()
    state = run(MASDeps(stub_planner(), stub_reviewer("reject"), executor))
    assert executor.calls == 1
    assert state["final"].status == "killed"


def test_executor_exception_routes_to_safe_fallback():
    async def boom(**_):
        raise RuntimeError("tool crashed")

    state = run(MASDeps(stub_planner(), stub_reviewer(), boom))
    assert state["final"].status == "error"
    assert "tool crashed" in state["final"].reason


def test_invalid_planner_output_is_rejected_not_passed_on():
    # steps must have 1..5 items; the model keeps returning an empty list
    bad = build_planner(TestModel(custom_output_args={"goal": "x" * 10, "steps": []}))
    executor = StubExecutor()
    state = run(MASDeps(bad, stub_reviewer(), executor))
    assert executor.calls == 0
    assert state["final"].status == "error"
    assert state["plan"] is None


def test_recursion_limit_is_a_real_backstop():
    # A worst-case run (3 rejections) fits inside the limit, but not with much slack.
    assert 14 <= config.RECURSION_LIMIT <= 25
