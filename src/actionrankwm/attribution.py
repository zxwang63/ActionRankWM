from __future__ import annotations

import copy
import hashlib
from collections.abc import Mapping, Sequence
from statistics import fmean
from typing import Any


EPSILON = 1e-12


def stable_tie_key(task_id: str, action_id: str) -> str:
    return hashlib.sha256(
        f"actionrankwm-v1|{task_id}|{action_id}".encode("utf-8")
    ).hexdigest()


def _segments(path: str) -> list[str]:
    if not path or path == "<root>":
        return []
    return path.split(".")


def lookup_path(value: Any, path: str) -> tuple[bool, Any]:
    current = value
    for segment in _segments(path):
        if isinstance(current, Mapping):
            if segment not in current:
                return False, None
            current = current[segment]
        elif isinstance(current, list) and segment.isdigit():
            index = int(segment)
            if index >= len(current):
                return False, None
            current = current[index]
        else:
            return False, None
    return True, current


def paths_differ(left: Any, right: Any, path: str) -> bool:
    left_exists, left_value = lookup_path(left, path)
    right_exists, right_value = lookup_path(right, path)
    return left_exists != right_exists or left_value != right_value


def differing_leaf_paths(left: Any, right: Any, prefix: str = "") -> list[str]:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        paths: list[str] = []
        for key in sorted(set(left) | set(right), key=str):
            path = f"{prefix}.{key}" if prefix else str(key)
            if key not in left or key not in right:
                paths.append(path)
            else:
                paths.extend(differing_leaf_paths(left[key], right[key], path))
        return paths
    if isinstance(left, list) and isinstance(right, list):
        paths = []
        for index in range(max(len(left), len(right))):
            path = f"{prefix}.{index}" if prefix else str(index)
            if index >= len(left) or index >= len(right):
                paths.append(path)
            else:
                paths.extend(differing_leaf_paths(left[index], right[index], path))
        return paths
    return [] if left == right else [prefix or "<root>"]


def paths_overlap(left: str, right: str) -> bool:
    return (
        left == right
        or left.startswith(f"{right}.")
        or right.startswith(f"{left}.")
    )


def _new_container(next_segment: str) -> Any:
    return [] if next_segment.isdigit() else {}


def _assign_path(root: Any, path: str, exists: bool, source_value: Any) -> Any:
    segments = _segments(path)
    if not segments:
        return copy.deepcopy(source_value) if exists else None
    result = copy.deepcopy(root)
    if not isinstance(result, (dict, list)):
        result = _new_container(segments[0])
    current = result
    for index, segment in enumerate(segments[:-1]):
        next_segment = segments[index + 1]
        if isinstance(current, dict):
            child = current.get(segment)
            if not isinstance(child, (dict, list)):
                child = _new_container(next_segment)
                current[segment] = child
            current = child
        elif isinstance(current, list) and segment.isdigit():
            position = int(segment)
            while len(current) <= position:
                current.append(None)
            child = current[position]
            if not isinstance(child, (dict, list)):
                child = _new_container(next_segment)
                current[position] = child
            current = child
        else:
            return result
    last = segments[-1]
    if isinstance(current, dict):
        if exists:
            current[last] = copy.deepcopy(source_value)
        else:
            current.pop(last, None)
    elif isinstance(current, list) and last.isdigit():
        position = int(last)
        if exists:
            while len(current) <= position:
                current.append(None)
            current[position] = copy.deepcopy(source_value)
        elif position < len(current):
            del current[position]
    return result


def restore_paths(predicted: Any, real: Any, paths: Sequence[str]) -> Any:
    restored = copy.deepcopy(predicted)
    for path in sorted(set(paths), key=lambda item: (item.count("."), item)):
        exists, value = lookup_path(real, path)
        restored = _assign_path(restored, path, exists, value)
    return restored


def pairwise_agreement(real_values: Mapping[str, float], simulated_values: Mapping[str, float]) -> float:
    action_ids = sorted(real_values)
    score = 0.0
    comparable = 0
    for left_index, left in enumerate(action_ids):
        for right in action_ids[left_index + 1 :]:
            real_delta = real_values[left] - real_values[right]
            if abs(real_delta) <= EPSILON:
                continue
            comparable += 1
            simulated_delta = simulated_values[left] - simulated_values[right]
            if abs(simulated_delta) <= EPSILON:
                score += 0.5
            elif (real_delta > 0) == (simulated_delta > 0):
                score += 1.0
    return score / comparable if comparable else 1.0


def summarize_value_vectors(
    task_id: str,
    real_values: Mapping[str, float],
    simulated_values: Mapping[str, float],
    harmful_regret_threshold: float,
) -> dict[str, Any]:
    if set(real_values) != set(simulated_values) or not real_values:
        raise ValueError("Real and simulated value vectors must have the same actions")
    best_real_value = max(real_values.values())
    real_best = sorted(
        action_id
        for action_id, value in real_values.items()
        if abs(value - best_real_value) <= EPSILON
    )
    best_simulated_value = max(simulated_values.values())
    simulated_top = sorted(
        action_id
        for action_id, value in simulated_values.items()
        if abs(value - best_simulated_value) <= EPSILON
    )
    selected = min(simulated_top, key=lambda action_id: stable_tie_key(task_id, action_id))
    top_regrets = [max(0.0, best_real_value - real_values[action_id]) for action_id in simulated_top]
    harmful_top = [
        action_id
        for action_id, regret in zip(simulated_top, top_regrets)
        if regret + EPSILON >= harmful_regret_threshold
    ]
    return {
        "real_best_action_ids": real_best,
        "simulated_top_action_ids": simulated_top,
        "selected_action_id": selected,
        "top1_preserved": selected in real_best,
        "top_set_intersects_real_best": bool(set(simulated_top) & set(real_best)),
        "top_set_subset_of_real_best": set(simulated_top).issubset(real_best),
        "harmful_top_action_ids": harmful_top,
        "contains_harmful_top": bool(harmful_top),
        "selected_regret": max(0.0, best_real_value - real_values[selected]),
        "expected_top_regret": fmean(top_regrets),
        "worst_top_regret": max(top_regrets),
        "pairwise_agreement": pairwise_agreement(real_values, simulated_values),
    }


def decompose_harmful_top_pairs(
    decision: Mapping[str, Any],
    scored_by_action: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    best_ids = list(decision["real_best_action_ids"])
    if not best_ids:
        raise ValueError("Decision has no real-best action")
    best_id = sorted(
        best_ids,
        key=lambda action_id: (
            -float(scored_by_action[action_id]["simulated_value"]),
            stable_tie_key(str(decision["task_id"]), action_id),
        ),
    )[0]
    best = scored_by_action[best_id]
    high_fidelity = set(decision["simulated_top_high_fidelity_harmful_action_ids"])
    rows = []
    for harmful_id in decision["simulated_top_harmful_action_ids"]:
        harmful = scored_by_action[harmful_id]
        gap = float(best["real_value"]) - float(harmful["real_value"])
        over_raw = float(harmful["simulated_value"]) - float(harmful["real_value"])
        under_raw = float(best["real_value"]) - float(best["simulated_value"])
        distortion = over_raw + under_raw
        simulated_margin = float(harmful["simulated_value"]) - float(best["simulated_value"])
        residual = distortion - (gap + simulated_margin)
        over_positive = over_raw > EPSILON
        under_positive = under_raw > EPSILON
        if over_positive and under_positive:
            category = "both"
        elif over_positive:
            category = "harmful_overprediction_only"
        elif under_positive:
            category = "real_best_underprediction_only"
        else:
            category = "residual_or_numerical"
        if abs(residual) > 1e-9:
            category = "residual_or_numerical"
        rows.append(
            {
                "task_id": decision["task_id"],
                "real_best_action_id": best_id,
                "harmful_action_id": harmful_id,
                "high_fidelity_harmful": harmful_id in high_fidelity,
                "real_action_gap": gap,
                "harmful_overprediction_raw": over_raw,
                "real_best_underprediction_raw": under_raw,
                "harmful_overprediction_positive": max(0.0, over_raw),
                "real_best_underprediction_positive": max(0.0, under_raw),
                "simulated_harmful_minus_best_margin": simulated_margin,
                "total_ranking_distortion": distortion,
                "algebraic_residual": residual,
                "category": category,
            }
        )
    return rows
