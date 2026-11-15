from __future__ import annotations

from collections.abc import Sequence

from actionrankwm.branches import ValueConfig, evaluate_simulated_branch
from actionrankwm.metrics.text import critical_field_accuracy, structured_f1, token_f1
from actionrankwm.semantics import canonicalize_predicted_observation, normalize_observation
from actionrankwm.types import BranchRecord, PredictionRecord, ScoredCandidate, TaskSpec


FAIL_CLOSED_SIMULATED_VALUE = -1.01


def score_candidates(
    tasks: Sequence[TaskSpec],
    branches: Sequence[BranchRecord],
    predictions: Sequence[PredictionRecord],
    value_config: ValueConfig,
) -> list[ScoredCandidate]:
    task_lookup = {task.task_id: task for task in tasks}
    branch_lookup = {(item.task_id, item.action.action_id): item for item in branches}
    prediction_lookup = {(item.task_id, item.action_id): item for item in predictions}
    if len(branch_lookup) != len(branches):
        raise ValueError("Duplicate real branch keys")
    if len(prediction_lookup) != len(predictions):
        raise ValueError("Duplicate prediction keys")
    expected = {
        (task.task_id, action.action_id)
        for task in tasks
        for action in task.candidates
    }
    if set(branch_lookup) != expected:
        missing = sorted(expected - set(branch_lookup))[:10]
        extra = sorted(set(branch_lookup) - expected)[:10]
        raise ValueError(f"Real branch key mismatch; missing={missing}, extra={extra}")
    if set(prediction_lookup) != expected:
        missing = sorted(expected - set(prediction_lookup))[:10]
        extra = sorted(set(prediction_lookup) - expected)[:10]
        raise ValueError(f"Prediction key mismatch; missing={missing}, extra={extra}")

    scored: list[ScoredCandidate] = []
    for key in sorted(expected):
        task = task_lookup[key[0]]
        branch = branch_lookup[key]
        prediction = prediction_lookup[key]
        parsed = normalize_observation(prediction.parsed_observation)
        canonical, patch_repair = canonicalize_predicted_observation(task, parsed)
        simulated_value, predicted_success, _, simulation_error = evaluate_simulated_branch(
            task=task,
            action=branch.action,
            predicted=canonical,
            config=value_config,
        )
        combined_error = prediction.error
        if simulation_error:
            combined_error = (
                f"{combined_error};{simulation_error}" if combined_error else simulation_error
            )
        prediction_failed = not prediction.parse_ok or combined_error is not None
        if prediction_failed:
            simulated_value = FAIL_CLOSED_SIMULATED_VALUE
            predicted_success = False
        scored.append(
            ScoredCandidate(
                task_id=task.task_id,
                domain=task.domain,
                action_id=branch.action.action_id,
                action_role=branch.action.role,
                action_risk=branch.action.risk,
                real_value=branch.real_value,
                simulated_value=simulated_value,
                token_f1=token_f1(branch.real_observation, parsed),
                structured_f1=structured_f1(branch.real_observation, parsed),
                critical_field_accuracy=critical_field_accuracy(
                    branch.real_observation,
                    parsed,
                    [str(path) for path in task.metadata.get("critical_fields", [])],
                ),
                predicted_success=predicted_success,
                parse_ok=prediction.parse_ok,
                real_terminal_success=branch.terminal_success,
                real_violations=branch.violations,
                patch_repair=patch_repair,
                prediction_error=combined_error,
            )
        )
    return scored
