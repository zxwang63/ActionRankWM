from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from statistics import fmean
from typing import Any


EPSILON = 1e-12
VALUE_RANGE = 2.01


def _sign(value: float) -> int:
    if value > EPSILON:
        return 1
    if value < -EPSILON:
        return -1
    return 0


def simulated_top(values: Mapping[str, float]) -> tuple[str, ...]:
    if not values:
        raise ValueError("A value vector cannot be empty")
    best = max(float(value) for value in values.values())
    return tuple(
        sorted(
            action_id
            for action_id, value in values.items()
            if abs(float(value) - best) <= EPSILON
        )
    )


def pairwise_order_disagreement(
    left: Mapping[str, float], right: Mapping[str, float]
) -> float:
    """Symmetric disagreement between two complete candidate rankings.

    An opposite strict order receives one point, a strict-order/tie mismatch
    receives half a point, and matching orders or matching ties receive zero.
    The score is averaged over all candidate pairs and never reads real values.
    """

    if set(left) != set(right) or len(left) < 2:
        raise ValueError("Value vectors must contain the same two or more actions")
    action_ids = sorted(left)
    disagreements: list[float] = []
    for left_index, first in enumerate(action_ids):
        for second in action_ids[left_index + 1 :]:
            left_sign = _sign(float(left[first]) - float(left[second]))
            right_sign = _sign(float(right[first]) - float(right[second]))
            if left_sign == right_sign:
                disagreements.append(0.0)
            elif left_sign == 0 or right_sign == 0:
                disagreements.append(0.5)
            else:
                disagreements.append(1.0)
    return fmean(disagreements)


def top_set_jaccard_disagreement(
    left: Mapping[str, float], right: Mapping[str, float]
) -> float:
    left_top = set(simulated_top(left))
    right_top = set(simulated_top(right))
    return 1.0 - len(left_top & right_top) / len(left_top | right_top)


def normalized_value_disagreement(
    left: Mapping[str, float], right: Mapping[str, float]
) -> float:
    if set(left) != set(right) or not left:
        raise ValueError("Value vectors must contain the same actions")
    return min(
        1.0,
        fmean(abs(float(left[action]) - float(right[action])) for action in left)
        / VALUE_RANGE,
    )


def disagreement_features(
    left: Mapping[str, float],
    right: Mapping[str, float],
    *,
    left_invalid: Sequence[str] = (),
    right_invalid: Sequence[str] = (),
) -> dict[str, float]:
    if set(left) != set(right):
        raise ValueError("Value vectors must contain the same actions")
    left_top = set(simulated_top(left))
    right_top = set(simulated_top(right))
    left_invalid_set = set(left_invalid)
    right_invalid_set = set(right_invalid)
    if not left_invalid_set.issubset(left) or not right_invalid_set.issubset(right):
        raise ValueError("Invalid-action IDs must belong to their value vectors")
    return {
        "pairwise_order_disagreement": pairwise_order_disagreement(left, right),
        "top_set_jaccard_disagreement": 1.0
        - len(left_top & right_top) / len(left_top | right_top),
        "top_set_exact_disagreement": float(left_top != right_top),
        "normalized_value_disagreement": normalized_value_disagreement(left, right),
        "invalid_set_jaccard_disagreement": (
            1.0
            - len(left_invalid_set & right_invalid_set)
            / len(left_invalid_set | right_invalid_set)
            if left_invalid_set or right_invalid_set
            else 0.0
        ),
    }


def expected_tie_aware_routing(
    rows: Sequence[Mapping[str, Any]],
    scores: Sequence[float],
    *,
    budget_fraction: float,
    harmful_key: str,
) -> dict[str, float | int | None]:
    """Expected routing outcome with fractional inclusion at a score tie.

    The boundary tie is averaged rather than resolved with task identifiers.
    This prevents an arbitrary deterministic key from making a risk signal look
    better or worse when many tasks share the same score.
    """

    if len(rows) != len(scores):
        raise ValueError("rows and scores must have equal length")
    if not 0.0 <= budget_fraction <= 1.0:
        raise ValueError("budget_fraction must lie in [0, 1]")
    if not rows:
        return {
            "task_count": 0,
            "budget_fraction": budget_fraction,
            "expected_routed_task_count": 0,
            "baseline_harmful_count": 0,
            "expected_remaining_harmful_count": 0,
            "expected_harmful_reduction": None,
            "baseline_terminal_success_rate": None,
            "expected_routed_terminal_success_rate": None,
            "expected_terminal_success_gain": None,
        }

    route_count = min(len(rows), max(0, math.ceil(len(rows) * budget_fraction)))
    if route_count == 0:
        probabilities = [0.0] * len(rows)
    elif route_count == len(rows):
        probabilities = [1.0] * len(rows)
    else:
        cutoff = sorted((float(score) for score in scores), reverse=True)[
            route_count - 1
        ]
        above = [index for index, score in enumerate(scores) if float(score) > cutoff]
        tied = [
            index
            for index, score in enumerate(scores)
            if abs(float(score) - cutoff) <= EPSILON
        ]
        tied_probability = (route_count - len(above)) / len(tied)
        probabilities = [
            1.0
            if float(score) > cutoff
            else tied_probability
            if abs(float(score) - cutoff) <= EPSILON
            else 0.0
            for score in scores
        ]

    harmful = [float(bool(row[harmful_key])) for row in rows]
    baseline_harmful = sum(harmful)
    remaining_harmful = sum(
        label * (1.0 - probability)
        for label, probability in zip(harmful, probabilities)
    )
    baseline_success = fmean(
        float(row["selected_real_terminal_success"]) for row in rows
    )
    routed_success = fmean(
        (1.0 - probability) * float(row["selected_real_terminal_success"])
        + probability * float(row["best_real_terminal_success"])
        for row, probability in zip(rows, probabilities)
    )
    return {
        "task_count": len(rows),
        "budget_fraction": budget_fraction,
        "expected_routed_task_count": sum(probabilities),
        "baseline_harmful_count": int(baseline_harmful),
        "expected_remaining_harmful_count": remaining_harmful,
        "expected_harmful_reduction": (
            (baseline_harmful - remaining_harmful) / baseline_harmful
            if baseline_harmful
            else None
        ),
        "baseline_terminal_success_rate": baseline_success,
        "expected_routed_terminal_success_rate": routed_success,
        "expected_terminal_success_gain": routed_success - baseline_success,
    }
