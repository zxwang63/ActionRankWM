from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any


EPSILON = 1e-12
DELTAS = (0.10, 0.20, 0.30, 0.40, 0.50)
GAMMAS = (0.50, 0.60, 0.70, 0.75, 0.80, 0.90)
FROZEN_CELL = (0.20, 0.75)


def _candidate_index(
    candidates: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Mapping[str, Any]]]:
    grouped: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in candidates:
        task_id = str(row["task_id"])
        action_id = str(row["action_id"])
        if action_id in grouped[task_id]:
            raise ValueError(f"Duplicate candidate key: {task_id}/{action_id}")
        grouped[task_id][action_id] = row
    return dict(grouped)


def _validated_inputs(
    decisions: Sequence[Mapping[str, Any]],
    candidates: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Mapping[str, Any]], dict[str, dict[str, Mapping[str, Any]]]]:
    decision_by_task: dict[str, Mapping[str, Any]] = {}
    for row in decisions:
        task_id = str(row["task_id"])
        if task_id in decision_by_task:
            raise ValueError(f"Duplicate decision task: {task_id}")
        decision_by_task[task_id] = row
    candidate_by_task = _candidate_index(candidates)
    if not decision_by_task or set(decision_by_task) != set(candidate_by_task):
        raise ValueError("Decision and candidate task inventories differ")

    for task_id, decision in decision_by_task.items():
        task_candidates = candidate_by_task[task_id]
        if len(task_candidates) < 2:
            raise ValueError(f"Task has fewer than two candidates: {task_id}")
        top_ids = [str(value) for value in decision["simulated_top_action_ids"]]
        if not top_ids or len(top_ids) != len(set(top_ids)):
            raise ValueError(f"Malformed simulated top set: {task_id}")
        if not set(top_ids).issubset(task_candidates):
            raise ValueError(f"Simulated top set has an unknown candidate: {task_id}")
        best_real = max(float(row["real_value"]) for row in task_candidates.values())
        if abs(best_real - float(decision["best_real_value"])) > EPSILON:
            raise ValueError(f"Decision best-real value mismatch: {task_id}")
        if not decision.get("template_id") or not decision.get("world_id"):
            raise ValueError(f"Task lacks mechanism/world metadata: {task_id}")
    return decision_by_task, candidate_by_task


def _task_event(
    decision: Mapping[str, Any],
    task_candidates: Mapping[str, Mapping[str, Any]],
    delta: float,
    gamma: float,
) -> bool:
    best_real = float(decision["best_real_value"])
    return any(
        best_real - float(task_candidates[action_id]["real_value"]) + EPSILON
        >= delta
        and float(task_candidates[action_id]["token_f1"]) >= gamma
        for action_id in map(str, decision["simulated_top_action_ids"])
    )


def _cell(
    decision_by_task: Mapping[str, Mapping[str, Any]],
    candidate_by_task: Mapping[str, Mapping[str, Mapping[str, Any]]],
    delta: float,
    gamma: float,
    *,
    excluded_mechanism: str | None = None,
) -> dict[str, Any]:
    retained = [
        task_id
        for task_id, row in decision_by_task.items()
        if str(row["template_id"]) != excluded_mechanism
    ]
    events = [
        task_id
        for task_id in retained
        if _task_event(
            decision_by_task[task_id], candidate_by_task[task_id], delta, gamma
        )
    ]
    mechanisms = sorted(
        {str(decision_by_task[task_id]["template_id"]) for task_id in events}
    )
    return {
        "delta": delta,
        "gamma": gamma,
        "task_count": len(retained),
        "event_count": len(events),
        "event_rate": len(events) / len(retained),
        "contributing_mechanism_count": len(mechanisms),
        "contributing_mechanisms": mechanisms,
    }


def analyze_model(
    decisions: Sequence[Mapping[str, Any]],
    candidates: Sequence[Mapping[str, Any]],
    *,
    deltas: Sequence[float] = DELTAS,
    gammas: Sequence[float] = GAMMAS,
) -> dict[str, Any]:
    decision_by_task, candidate_by_task = _validated_inputs(decisions, candidates)
    delta_grid = tuple(sorted(float(value) for value in deltas))
    gamma_grid = tuple(sorted(float(value) for value in gammas))
    if (
        not delta_grid
        or not gamma_grid
        or len(delta_grid) != len(set(delta_grid))
        or len(gamma_grid) != len(set(gamma_grid))
        or any(value < 0.0 for value in delta_grid)
        or any(not 0.0 <= value <= 1.0 for value in gamma_grid)
    ):
        raise ValueError("Invalid threshold grid")

    cells = [
        _cell(decision_by_task, candidate_by_task, delta, gamma)
        for delta in delta_grid
        for gamma in gamma_grid
    ]
    by_key = {
        (float(row["delta"]), float(row["gamma"])): row for row in cells
    }
    for gamma in gamma_grid:
        counts = [int(by_key[(delta, gamma)]["event_count"]) for delta in delta_grid]
        if counts != sorted(counts, reverse=True):
            raise ValueError("Event counts increase with the regret threshold")
    for delta in delta_grid:
        counts = [int(by_key[(delta, gamma)]["event_count"]) for gamma in gamma_grid]
        if counts != sorted(counts, reverse=True):
            raise ValueError("Event counts increase with the fidelity threshold")

    if FROZEN_CELL not in by_key:
        raise ValueError("Grid omits the frozen threshold cell")
    mechanisms = sorted(
        {str(row["template_id"]) for row in decision_by_task.values()}
    )
    leave_one_out = [
        {
            "excluded_mechanism": mechanism,
            **_cell(
                decision_by_task,
                candidate_by_task,
                FROZEN_CELL[0],
                FROZEN_CELL[1],
                excluded_mechanism=mechanism,
            ),
        }
        for mechanism in mechanisms
    ]
    return {
        "task_count": len(decision_by_task),
        "candidate_count": sum(len(rows) for rows in candidate_by_task.values()),
        "mechanisms": mechanisms,
        "grid": cells,
        "frozen_cell": by_key[FROZEN_CELL],
        "leave_one_mechanism_out_at_frozen_cell": leave_one_out,
        "robust_cell_count_ge_10_events_ge_3_mechanisms": sum(
            int(row["event_count"]) >= 10
            and int(row["contributing_mechanism_count"]) >= 3
            for row in cells
        ),
        "grid_cell_count": len(cells),
    }


def analyze_panel(
    models: Mapping[
        str,
        tuple[Sequence[Mapping[str, Any]], Sequence[Mapping[str, Any]]],
    ],
) -> dict[str, Any]:
    if len(models) < 2:
        raise ValueError("Threshold sensitivity requires at least two models")
    analyzed: dict[str, dict[str, Any]] = {}
    reference_tasks: dict[str, tuple[str, str]] | None = None
    reference_values: dict[tuple[str, str], float] | None = None
    for model, (decisions, candidates) in models.items():
        decision_by_task, candidate_by_task = _validated_inputs(decisions, candidates)
        task_metadata = {
            task_id: (str(row["template_id"]), str(row["world_id"]))
            for task_id, row in decision_by_task.items()
        }
        values = {
            (task_id, action_id): float(row["real_value"])
            for task_id, task_candidates in candidate_by_task.items()
            for action_id, row in task_candidates.items()
        }
        if reference_tasks is None:
            reference_tasks = task_metadata
            reference_values = values
        elif task_metadata != reference_tasks or values != reference_values:
            raise ValueError("Models do not share an identical task/real-value panel")
        analyzed[model] = analyze_model(decisions, candidates)

    pooled_grid: list[dict[str, Any]] = []
    labels = sorted(analyzed)
    for delta in DELTAS:
        for gamma in GAMMAS:
            rows = [
                next(
                    row
                    for row in analyzed[label]["grid"]
                    if row["delta"] == delta and row["gamma"] == gamma
                )
                for label in labels
            ]
            event_count = sum(int(row["event_count"]) for row in rows)
            model_task_count = sum(int(row["task_count"]) for row in rows)
            mechanisms = sorted(
                {
                    mechanism
                    for row in rows
                    for mechanism in row["contributing_mechanisms"]
                }
            )
            pooled_grid.append(
                {
                    "delta": delta,
                    "gamma": gamma,
                    "model_task_count": model_task_count,
                    "event_count": event_count,
                    "event_rate": event_count / model_task_count,
                    "contributing_mechanism_count": len(mechanisms),
                    "contributing_mechanisms": mechanisms,
                }
            )
    return {
        "models": analyzed,
        "pooled_grid": pooled_grid,
        "pooled_model_task_count": sum(
            int(row["task_count"]) for row in analyzed.values()
        ),
        "thresholds": {
            "deltas": list(DELTAS),
            "gammas": list(GAMMAS),
            "frozen_cell": {"delta": FROZEN_CELL[0], "gamma": FROZEN_CELL[1]},
        },
    }
