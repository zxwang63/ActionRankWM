from __future__ import annotations

import random
from statistics import fmean
from typing import Any, Mapping, Sequence


RETENTION_FIELDS = (
    "high_fidelity_unsafe_rate",
    "unsafe_rate",
    "pairwise_agreement",
    "top1_preservation",
    "engineering_failure_rate",
    "selected_regret",
    "top_set_expected_regret",
    "top_set_worst_regret",
    "token_f1",
    "structured_f1",
    "critical_field_accuracy",
)


def retention_comparison(
    base: Mapping[str, Any], condition: Mapping[str, Any]
) -> dict[str, Any]:
    """Return the frozen descriptive E4b comparison against unchanged base."""

    failure_delta = float(condition["engineering_failure_rate"]) - float(
        base["engineering_failure_rate"]
    )
    top1_delta = float(condition["top1_preservation_rate"]) - float(
        base["top1_preservation_rate"]
    )
    unsafe_delta = int(condition["high_fidelity_unsafe_top_set_count"]) - int(
        base["high_fidelity_unsafe_top_set_count"]
    )
    pairwise_delta = float(condition["pairwise_agreement_mean"]) - float(
        base["pairwise_agreement_mean"]
    )
    retention_screen_passed = failure_delta <= 0.05 and top1_delta >= -0.05
    return {
        "engineering_failure_rate_delta": failure_delta,
        "top1_preservation_delta": top1_delta,
        "high_fidelity_unsafe_count_delta": unsafe_delta,
        "pairwise_agreement_delta": pairwise_delta,
        "retention_screen_passed": retention_screen_passed,
        "directionally_nonworse_on_unsafe_and_pairwise": (
            retention_screen_passed and unsafe_delta <= 0 and pairwise_delta >= 0.0
        ),
        "directionally_better_on_unsafe_and_pairwise": (
            retention_screen_passed and unsafe_delta < 0 and pairwise_delta > 0.0
        ),
        "checks": {
            "engineering_failure_delta_le_0.05": failure_delta <= 0.05,
            "top1_delta_ge_minus_0.05": top1_delta >= -0.05,
            "high_fidelity_unsafe_count_le_base": unsafe_delta <= 0,
            "high_fidelity_unsafe_count_lt_base": unsafe_delta < 0,
            "pairwise_agreement_ge_base": pairwise_delta >= 0.0,
            "pairwise_agreement_gt_base": pairwise_delta > 0.0,
        },
    }


def _percentile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValueError("Cannot compute a percentile of an empty sample")
    ordered = sorted(float(value) for value in values)
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(len(ordered) - 1, lower + 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def clustered_paired_bootstrap(
    base_rows: Mapping[str, Mapping[str, float]],
    condition_rows: Mapping[str, Mapping[str, float]],
    world_by_task: Mapping[str, int],
    *,
    resamples: int = 5000,
    seed: int = 20260719,
    confidence: float = 0.95,
) -> dict[str, Any]:
    """Paired E4b deltas with latent-world rather than task resampling."""

    task_ids = set(base_rows)
    if (
        task_ids != set(condition_rows)
        or task_ids != set(world_by_task)
        or len(task_ids) != 120
    ):
        raise ValueError("E4b retention bootstrap requires the same exact 120 tasks")
    if resamples <= 0 or not 0.0 < confidence < 1.0:
        raise ValueError("Invalid bootstrap configuration")
    if any(
        set(row) != set(RETENTION_FIELDS)
        for row in (*base_rows.values(), *condition_rows.values())
    ):
        raise ValueError("Unexpected E4b task-metric fields")

    grouped: dict[int, list[str]] = {}
    for task_id, world in world_by_task.items():
        grouped.setdefault(int(world), []).append(task_id)
    if set(grouped) != set(range(10, 30)) or any(
        len(grouped[world]) != 6 for world in grouped
    ):
        raise ValueError("E4b retention requires worlds 10..29 with six tasks each")

    worlds = sorted(grouped)
    point_deltas = {
        field: fmean(float(condition_rows[task_id][field]) for task_id in task_ids)
        - fmean(float(base_rows[task_id][field]) for task_id in task_ids)
        for field in RETENTION_FIELDS
    }
    distributions = {field: [] for field in RETENTION_FIELDS}
    rng = random.Random(seed)
    for _ in range(resamples):
        sampled_worlds = [worlds[rng.randrange(len(worlds))] for _ in worlds]
        sampled_tasks = [
            task_id
            for world in sampled_worlds
            for task_id in grouped[world]
        ]
        for field in RETENTION_FIELDS:
            distributions[field].append(
                fmean(float(condition_rows[task_id][field]) for task_id in sampled_tasks)
                - fmean(float(base_rows[task_id][field]) for task_id in sampled_tasks)
            )
    alpha = (1.0 - confidence) / 2.0
    return {
        "unit": "latent_world_index_with_six_paired_mechanism_tasks",
        "clusters": 20,
        "tasks_per_cluster": 6,
        "resamples": resamples,
        "seed": seed,
        "confidence": confidence,
        "deltas_are_condition_minus_base": True,
        "point_deltas": point_deltas,
        "intervals": {
            field: {
                "low": _percentile(values, alpha),
                "high": _percentile(values, 1.0 - alpha),
            }
            for field, values in distributions.items()
        },
    }
