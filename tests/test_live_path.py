"""Live-path checks against a mocked Groq HTTP API (no key, no network)."""
import asyncio
import json

import httpx
from groq import AsyncGroq
from pydantic_ai.models.groq import GroqModel
from pydantic_ai.providers.groq import GroqProvider

from mas import agents, config
from mas.graph import MASDeps, build_graph, run_task
from mas.offline import StubExecutor, stub_planner, stub_reviewer
from mas.schemas import PlannerOutput


def test_live_planner_uses_native_json_schema_and_token_caps():
    seen = {}

    def handler(req: httpx.Request):
        body = json.loads(req.content)
        seen.update(body)
        content = json.dumps({"goal": "Migrate CRM", "steps": [{"step_number": 1, "action": "Inventory tables"}]})
        return httpx.Response(200, json={
            "id": "x", "object": "chat.completion", "created": 0, "model": body["model"],
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": content}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20}})

    client = AsyncGroq(api_key="test", http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    model = GroqModel(config.GROQ_MODEL, provider=GroqProvider(groq_client=client))
    planner = agents.build_planner()  # live configuration

    async def go():
        with planner.override(model=model):
            return (await planner.run("Task: migrate", usage_limits=agents.USAGE_LIMITS)).output

    out = asyncio.run(go())
    assert isinstance(out, PlannerOutput)
    assert seen["response_format"]["type"] == "json_schema"
    assert seen["response_format"]["json_schema"]["schema"]["additionalProperties"] is False
    assert seen.get("max_tokens", seen.get("max_completion_tokens")) == config.LLM_MAX_OUTPUT_TOKENS
    assert seen["reasoning_effort"] == config.LLM_REASONING_EFFORT


def test_rate_limit_retry_is_bounded(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "RATE_LIMIT_WAIT_S", 0)
    monkeypatch.setattr(config, "KILL_SWITCH_FILE", tmp_path / "KILL_SWITCH")
    monkeypatch.delenv(config.KILL_SWITCH_ENV, raising=False)

    class AlwaysRateLimited(StubExecutor):
        async def __call__(self, *a, **kw):
            self.calls += 1
            raise RuntimeError("Error code: 429 - rate_limit_exceeded")

    ex = AlwaysRateLimited()
    state = asyncio.run(run_task(build_graph(MASDeps(stub_planner(), stub_reviewer(), ex)), "t", "task"))
    assert ex.calls == 1 + config.RATE_LIMIT_RETRIES   # 3 attempts, then stop
    assert state["final"].status == "error"


def test_rate_limit_then_success(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "RATE_LIMIT_WAIT_S", 0)
    monkeypatch.setattr(config, "KILL_SWITCH_FILE", tmp_path / "KILL_SWITCH")
    monkeypatch.delenv(config.KILL_SWITCH_ENV, raising=False)

    class OnceRateLimited(StubExecutor):
        async def __call__(self, *a, **kw):
            if self.calls == 0:
                self.calls += 1
                raise RuntimeError("429 Too Many Requests")
            return await super().__call__(*a, **kw)

    ex = OnceRateLimited()
    state = asyncio.run(run_task(build_graph(MASDeps(stub_planner(), stub_reviewer(), ex)), "t", "task"))
    assert state["final"].status == "completed"
    assert state["execution_count"] == 1   # a retried 429 is not a new loop iteration
