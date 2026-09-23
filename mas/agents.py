"""The three agents.

* Planner  - PydanticAI Agent, output_type=PlannerOutput
* Executor - CrewAI Agent/Task/Crew, output_pydantic=ExecutorOutput,
             then re-validated strictly with the same Pydantic schema
* Reviewer - PydanticAI Agent, output_type=ReviewerOutput
"""
from __future__ import annotations

from typing import Awaitable, Callable, Optional, Protocol

from pydantic_ai import Agent, NativeOutput
from pydantic_ai.models import Model
from pydantic_ai.usage import UsageLimits

from . import config
from .guardrails import check_kill_switch
from .schemas import ExecutorOutput, PlannerOutput, ReviewerOutput

USAGE_LIMITS = UsageLimits(
    request_limit=config.REQUEST_LIMIT_PER_CALL,
    total_tokens_limit=config.TOKEN_LIMIT_PER_CALL,
)

PLANNER_INSTRUCTIONS = (
    "You are the Planner in a guardrailed multi-agent system. Break the user's task into "
    "1-5 short, ordered, concrete steps numbered from 1. Use the provided user memories "
    "only when they are relevant (e.g. stated preferences). Do not plan any action that "
    "requires network access, credentials, shell commands or spending compute."
)

REVIEWER_INSTRUCTIONS = (
    "You are the Reviewer. Approve the executor's result only if it fully and correctly "
    "addresses the task and follows the plan and any relevant user preferences. Otherwise "
    "reject it with specific, actionable feedback. Return verdict 'approve' or 'reject'."
)


def groq_model_name() -> str:
    return f"groq:{config.GROQ_MODEL}"


def _live_settings() -> dict:
    return {"max_tokens": config.LLM_MAX_OUTPUT_TOKENS,
            "groq_reasoning_effort": config.LLM_REASONING_EFFORT}


def _agent(model, schema, instructions: str, name: str):
    """Live (model=None): Groq with native JSON-schema structured output, which the
    gpt-oss models support, low reasoning effort and capped output. Tests pass a
    scripted model and use PydanticAI's default tool-call output mode."""
    live = model is None
    return Agent(
        groq_model_name() if live else model,
        output_type=NativeOutput(schema) if live else schema,
        instructions=instructions,
        retries=config.PYDANTIC_AI_RETRIES,
        model_settings=_live_settings() if live else None,
        name=name,
        defer_model_check=True,
    )


def build_planner(model: Model | str | None = None) -> Agent[None, PlannerOutput]:
    return _agent(model, PlannerOutput, PLANNER_INSTRUCTIONS, "planner")


def build_reviewer(model: Model | str | None = None) -> Agent[None, ReviewerOutput]:
    return _agent(model, ReviewerOutput, REVIEWER_INSTRUCTIONS, "reviewer")


class ExecutorFn(Protocol):
    def __call__(
        self, task: str, plan: PlannerOutput, feedback: Optional[str], memories: list[str]
    ) -> Awaitable[ExecutorOutput]: ...


def _no_braces(text: str) -> str:
    # CrewAI treats {name} as template placeholders; neutralise user-supplied braces.
    return text.replace("{", "(").replace("}", ")")


class CrewAIExecutor:
    """Executor node backed by a single-agent CrewAI crew with hard ceilings:
    no tools, no delegation, max_iter=3, wall-clock limit, kill-switch checked every step."""

    def __init__(self, model: str | None = None):
        from crewai import LLM

        self.llm = LLM(
            model=model or f"groq/{config.GROQ_MODEL}",
            temperature=0.0,
            max_tokens=config.LLM_MAX_OUTPUT_TOKENS,
            reasoning_effort=config.LLM_REASONING_EFFORT,
        )

    async def __call__(self, task, plan, feedback, memories) -> ExecutorOutput:
        from crewai import Agent as CrewAgent
        from crewai import Crew, Process, Task

        check_kill_switch()
        worker = CrewAgent(
            role="Executor",
            goal="Carry out the approved plan and produce the final answer for the user.",
            backstory="A careful analyst who follows plans exactly and never takes external actions.",
            llm=self.llm,
            tools=[],                       # explicit empty tool allow-list
            allow_delegation=False,
            max_iter=config.CREW_MAX_ITER,  # CrewAI default is 20
            max_retry_limit=1,
            max_execution_time=config.CREW_MAX_EXECUTION_TIME_S,
            step_callback=lambda _step: check_kill_switch(),
            verbose=False,
        )
        steps = "\n".join(f"{s.step_number}. {s.action}" for s in plan.steps)
        mem = "\n".join(f"- {m}" for m in memories) or "- (none)"
        fb = f"\nReviewer feedback on your previous attempt (fix these):\n{feedback}" if feedback else ""
        description = _no_braces(
            f"Task: {task}\n\nPlan:\n{steps}\n\nRelevant user memories:\n{mem}{fb}\n\n"
            "Execute the plan and write the answer."
        )
        crew_task = Task(
            description=description,
            expected_output=(
                "JSON with keys: result (the full answer), steps_completed (list of step numbers), "
                "confidence (0-1)."
            ),
            agent=worker,
            output_pydantic=ExecutorOutput,
        )
        crew = Crew(agents=[worker], tasks=[crew_task], process=Process.sequential, verbose=False)
        out = await crew.kickoff_async()
        if out.pydantic is not None:
            return ExecutorOutput.model_validate(out.pydantic.model_dump())
        # Strict re-parse of the raw text; raises ValidationError on anything malformed.
        return ExecutorOutput.model_validate_json(out.raw)


ExecutorCallable = Callable[..., Awaitable[ExecutorOutput]]
