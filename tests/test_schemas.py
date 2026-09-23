import pytest
from pydantic import ValidationError

from mas.schemas import ExecutorOutput, FinalResponse, PlannerOutput, ReviewerOutput

GOOD_PLAN = {"goal": "Migrate data", "steps": [{"step_number": 1, "action": "Inventory tables"}]}


def test_valid_plan_parses():
    assert PlannerOutput.model_validate(GOOD_PLAN).steps[0].step_number == 1


@pytest.mark.parametrize(
    "payload",
    [
        {**GOOD_PLAN, "shell_command": "ssh -R ..."},                            # hallucinated key
        {"goal": "g" * 5, "steps": []},                                          # empty plan
        {"goal": "g" * 5, "steps": [{"step_number": i, "action": "do it"} for i in range(1, 7)]},  # >5
        {"goal": "g" * 5, "steps": [{"step_number": 2, "action": "skip"}]},      # not sequential
    ],
)
def test_bad_plans_rejected(payload):
    with pytest.raises(ValidationError):
        PlannerOutput.model_validate(payload)


def test_reviewer_verdict_is_closed_set():
    with pytest.raises(ValidationError):
        ReviewerOutput.model_validate({"verdict": "maybe", "feedback": "?"})


def test_executor_confidence_bounded():
    with pytest.raises(ValidationError):
        ExecutorOutput.model_validate({"result": "x", "confidence": 1.5})


def test_final_response_status_closed_set():
    with pytest.raises(ValidationError):
        FinalResponse.model_validate(
            {"status": "mining_crypto", "answer": "", "execution_count": 1, "reason": ""}
        )
