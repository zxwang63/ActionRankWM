from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from statistics import fmean
from typing import Any

from actionrankwm.types import BranchRecord, TaskSpec


EPSILON = 1e-12


def _clip(value: float) -> float:
    return max(-1.0, min(1.0, value))


def _number_key(value: float) -> str:
    rounded = round(float(value), 12)
    if abs(rounded) <= EPSILON:
        rounded = 0.0
    rendered = f"{rounded:.12f}".rstrip("0").rstrip(".")
    return rendered or "0"


def _counter(values: Sequence[Any]) -> dict[str, int]:
    counts = Counter(str(value) for value in values)
    return {key: counts[key] for key in sorted(counts)}


def _numeric_counter(values: Sequence[float]) -> dict[str, int]:
    counts = Counter(_number_key(value) for value in values)
    return {
        key: counts[key]
        for key in sorted(counts, key=lambda value: float(value))
    }


def _percentile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValueError("Cannot summarize an empty finite suite")
    ordered = sorted(float(value) for value in values)
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _gap_summary(values: Sequence[float]) -> dict[str, float]:
    return {
        "min": min(values),
        "p25": _percentile(values, 0.25),
        "median": _percentile(values, 0.50),
        "p75": _percentile(values, 0.75),
        "max": max(values),
        "mean": fmean(values),
    }


def _summarize(
    task_rows: Sequence[Mapping[str, Any]],
    branch_rows: Sequence[Mapping[str, Any]],
    *,
    harmful_regret_threshold: float,
) -> dict[str, Any]:
    raw_values = [float(row["raw_value"]) for row in branch_rows]
    clipped_values = [float(row["real_value"]) for row in branch_rows]
    gaps = [float(row["top_gap"]) for row in task_rows]
    role_counts = Counter(
        str(role)
        for row in task_rows
        for role, count in row["candidate_role_counts"].items()
        for _ in range(int(count))
    )
    worlds = {int(row["world_id"]) for row in task_rows}
    combinations = [
        "success={}|invalid={}|violation={}".format(
            int(bool(row["terminal_success"])),
            int(bool(row["candidate_invalid"])),
            int(bool(row["has_violation"])),
        )
        for row in branch_rows
    ]
    return {
        "task_count": len(task_rows),
        "candidate_count": len(branch_rows),
        "world_count": len(worlds),
        "candidate_role_counts": {
            key: role_counts[key] for key in sorted(role_counts)
        },
        "executed_step_counts": _counter(
            [int(row["executed_steps"]) for row in branch_rows]
        ),
        "terminal_success_candidate_count": sum(
            bool(row["terminal_success"]) for row in branch_rows
        ),
        "candidate_invalid_count": sum(
            bool(row["candidate_invalid"]) for row in branch_rows
        ),
        "candidate_with_violation_count": sum(
            bool(row["has_violation"]) for row in branch_rows
        ),
        "success_invalid_violation_combination_counts": _counter(combinations),
        "unique_violation_cardinality_counts": _counter(
            [int(row["unique_violation_count"]) for row in branch_rows]
        ),
        "raw_value_counts": _numeric_counter(raw_values),
        "raw_value_min": min(raw_values),
        "raw_value_max": max(raw_values),
        "clipped_value_counts": _numeric_counter(clipped_values),
        "clipped_value_min": min(clipped_values),
        "clipped_value_max": max(clipped_values),
        "lower_clipping_count": sum(value < -1.0 - EPSILON for value in raw_values),
        "upper_clipping_count": sum(value > 1.0 + EPSILON for value in raw_values),
        "utility_level_count_distribution": _counter(
            [int(row["utility_level_count"]) for row in task_rows]
        ),
        "real_top_set_size_distribution": _counter(
            [int(row["real_top_set_size"]) for row in task_rows]
        ),
        "tasks_with_tied_real_top_count": sum(
            int(row["real_top_set_size"]) > 1 for row in task_rows
        ),
        "tasks_with_all_candidates_tied_count": sum(
            int(row["utility_level_count"]) == 1 for row in task_rows
        ),
        "top_gap_summary": _gap_summary(gaps),
        "candidate_regret_ge_threshold": {
            "threshold": harmful_regret_threshold,
            "count": sum(
                float(row["real_regret"]) + EPSILON >= harmful_regret_threshold
                for row in branch_rows
            ),
        },
    }


def characterize_true_branches(
    tasks: Sequence[TaskSpec],
    branches: Sequence[BranchRecord],
    *,
    action_cost: float,
    violation_penalty: float,
    invalid_penalty: float,
    harmful_regret_threshold: float,
    expected_worlds: Sequence[int],
    expected_mechanism_count: int = 6,
) -> dict[str, Any]:
    if len(tasks) != 120 or len(branches) != 600:
        raise ValueError("E4b characterization requires exactly 120 tasks/600 branches")
    task_by_id = {task.task_id: task for task in tasks}
    if len(task_by_id) != len(tasks):
        raise ValueError("Duplicate E4b task ID")

    branch_by_key: dict[tuple[str, str], BranchRecord] = {}
    branches_by_task: dict[str, list[BranchRecord]] = defaultdict(list)
    for branch in branches:
        key = (branch.task_id, branch.action.action_id)
        if key in branch_by_key:
            raise ValueError(f"Duplicate E4b branch: {key}")
        if branch.task_id not in task_by_id:
            raise ValueError(f"Branch refers to unknown task: {branch.task_id}")
        branch_by_key[key] = branch
        branches_by_task[branch.task_id].append(branch)

    expected_world_set = {int(world) for world in expected_worlds}
    worlds = {int(task.metadata["world_id"]) for task in tasks}
    if worlds != expected_world_set:
        raise ValueError("Unexpected E4b world inventory")
    mechanisms = {str(task.metadata["mechanism"]) for task in tasks}
    if len(mechanisms) != expected_mechanism_count:
        raise ValueError("Unexpected E4b mechanism inventory")
    mechanisms_by_world: dict[int, set[str]] = defaultdict(set)
    tasks_by_mechanism = Counter()

    task_rows: list[dict[str, Any]] = []
    branch_rows: list[dict[str, Any]] = []
    for task in sorted(tasks, key=lambda item: item.task_id):
        world_id = int(task.metadata["world_id"])
        mechanism = str(task.metadata["mechanism"])
        mechanisms_by_world[world_id].add(mechanism)
        tasks_by_mechanism[mechanism] += 1
        task_branches = sorted(
            branches_by_task[task.task_id], key=lambda item: item.action.action_id
        )
        expected_actions = {action.action_id: action for action in task.candidates}
        if len(expected_actions) != 5 or len(task_branches) != 5:
            raise ValueError(f"Task does not have five distinct candidates: {task.task_id}")
        if {branch.action.action_id for branch in task_branches} != set(expected_actions):
            raise ValueError(f"Task/branch candidate mismatch: {task.task_id}")

        analyzed: list[dict[str, Any]] = []
        for branch in task_branches:
            if branch.action.role != expected_actions[branch.action.action_id].role:
                raise ValueError(f"Candidate role mismatch: {task.task_id}")
            violations = sorted(set(str(value) for value in branch.violations))
            raw_value = (
                float(bool(branch.terminal_success))
                - action_cost * int(branch.executed_steps)
                - violation_penalty * len(violations)
                - invalid_penalty * float(not branch.candidate_ok)
            )
            clipped = _clip(raw_value)
            if abs(clipped - float(branch.real_value)) > EPSILON:
                raise ValueError(
                    f"Stored real value does not match frozen value contract: "
                    f"{task.task_id}/{branch.action.action_id}"
                )
            row = {
                "task_id": task.task_id,
                "action_id": branch.action.action_id,
                "mechanism": mechanism,
                "world_id": world_id,
                "candidate_role": branch.action.role,
                "executed_steps": int(branch.executed_steps),
                "terminal_success": bool(branch.terminal_success),
                "candidate_invalid": not bool(branch.candidate_ok),
                "has_violation": bool(violations),
                "unique_violation_count": len(violations),
                "raw_value": raw_value,
                "real_value": float(branch.real_value),
                "lower_clipped": raw_value < -1.0 - EPSILON,
                "upper_clipped": raw_value > 1.0 + EPSILON,
            }
            analyzed.append(row)

        best_value = max(float(row["real_value"]) for row in analyzed)
        distinct_values = sorted(
            {_number_key(float(row["real_value"])) for row in analyzed},
            key=float,
            reverse=True,
        )
        second_distinct = float(distinct_values[1]) if len(distinct_values) > 1 else None
        top_gap = best_value - second_distinct if second_distinct is not None else 0.0
        for row in analyzed:
            row["real_regret"] = max(0.0, best_value - float(row["real_value"]))
        branch_rows.extend(analyzed)

        role_counts = Counter(str(row["candidate_role"]) for row in analyzed)
        step_counts = Counter(str(int(row["executed_steps"])) for row in analyzed)
        task_rows.append(
            {
                "task_id": task.task_id,
                "mechanism": mechanism,
                "world_id": world_id,
                "candidate_count": len(analyzed),
                "candidate_role_counts": {
                    key: role_counts[key] for key in sorted(role_counts)
                },
                "utility_level_count": len(distinct_values),
                "real_top_set_size": sum(
                    abs(float(row["real_value"]) - best_value) <= EPSILON
                    for row in analyzed
                ),
                "best_value": best_value,
                "second_distinct_value": second_distinct,
                "top_gap": top_gap,
                "clipping_count": sum(
                    bool(row["lower_clipped"] or row["upper_clipped"])
                    for row in analyzed
                ),
                "invalid_candidate_count": sum(
                    bool(row["candidate_invalid"]) for row in analyzed
                ),
                "violation_candidate_count": sum(
                    bool(row["has_violation"]) for row in analyzed
                ),
                "terminal_success_candidate_count": sum(
                    bool(row["terminal_success"]) for row in analyzed
                ),
                "executed_step_counts": {
                    key: step_counts[key] for key in sorted(step_counts)
                },
            }
        )

    if any(mechanisms_by_world[world] != mechanisms for world in worlds):
        raise ValueError("Every E4b world must contain all six mechanisms")
    if set(tasks_by_mechanism.values()) != {20}:
        raise ValueError("Every E4b mechanism must contain exactly 20 tasks")

    overall = _summarize(
        task_rows,
        branch_rows,
        harmful_regret_threshold=harmful_regret_threshold,
    )
    by_mechanism = {}
    for mechanism in sorted(mechanisms):
        by_mechanism[mechanism] = _summarize(
            [row for row in task_rows if row["mechanism"] == mechanism],
            [row for row in branch_rows if row["mechanism"] == mechanism],
            harmful_regret_threshold=harmful_regret_threshold,
        )
    return {
        "status": "E4B_TRUE_BRANCH_SUITE_CHARACTERIZATION_COMPLETE_POSTHOC_SPECIFICATION",
        "scientific_use": False,
        "model_output_read": False,
        "gate_neutral": True,
        "value_contract": {
            "action_cost": action_cost,
            "violation_penalty": violation_penalty,
            "invalid_penalty": invalid_penalty,
            "clipping": [-1.0, 1.0],
            "tie_tolerance": EPSILON,
            "harmful_regret_threshold_reused_descriptively": harmful_regret_threshold,
        },
        "overall": overall,
        "by_mechanism": by_mechanism,
        "task_rows": task_rows,
    }
