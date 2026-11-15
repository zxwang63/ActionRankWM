from __future__ import annotations

import math
import hashlib
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from statistics import fmean
from typing import Any

from actionrankwm.types import ScoredCandidate, TaskSpec


_EPSILON = 1e-12


def _stable_tie_key(task_id: str, action_id: str) -> str:
    """Role-blind deterministic tie break; candidate order never decides."""

    return hashlib.sha256(f"actionrankwm-v1|{task_id}|{action_id}".encode("utf-8")).hexdigest()


def _mean(values: Iterable[float]) -> float | None:
    materialized = list(values)
    return fmean(materialized) if materialized else None


def _average_ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(order):
        end = cursor + 1
        while end < len(order) and abs(values[order[end]] - values[order[cursor]]) <= _EPSILON:
            end += 1
        average_rank = (cursor + end - 1) / 2.0
        for position in range(cursor, end):
            ranks[order[position]] = average_rank
        cursor = end
    return ranks


def spearman(values_x: Sequence[float], values_y: Sequence[float]) -> float | None:
    if len(values_x) != len(values_y) or len(values_x) < 3:
        return None
    ranks_x = _average_ranks(values_x)
    ranks_y = _average_ranks(values_y)
    mean_x = fmean(ranks_x)
    mean_y = fmean(ranks_y)
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(ranks_x, ranks_y))
    denominator_x = sum((x - mean_x) ** 2 for x in ranks_x)
    denominator_y = sum((y - mean_y) ** 2 for y in ranks_y)
    denominator = math.sqrt(denominator_x * denominator_y)
    if denominator <= _EPSILON:
        return None
    return numerator / denominator


def _pairwise_agreement(candidates: Sequence[ScoredCandidate]) -> tuple[float, int]:
    score = 0.0
    comparable = 0
    for left_index, left in enumerate(candidates):
        for right in candidates[left_index + 1 :]:
            real_delta = left.real_value - right.real_value
            if abs(real_delta) <= _EPSILON:
                continue
            comparable += 1
            simulated_delta = left.simulated_value - right.simulated_value
            if abs(simulated_delta) <= _EPSILON:
                score += 0.5
            elif (real_delta > 0) == (simulated_delta > 0):
                score += 1.0
    return (score / comparable if comparable else 1.0), comparable


def score_task_decisions(
    scored_candidates: Sequence[ScoredCandidate],
    tasks: Sequence[TaskSpec],
    harmful_regret_threshold: float,
    high_text_fidelity_threshold: float,
) -> list[dict[str, Any]]:
    """Reduce candidate outcomes to independent latent-task decision units."""

    task_lookup = {task.task_id: task for task in tasks}
    grouped: dict[str, list[ScoredCandidate]] = defaultdict(list)
    for candidate in scored_candidates:
        grouped[candidate.task_id].append(candidate)

    decisions: list[dict[str, Any]] = []
    for task_id in sorted(grouped):
        candidates = sorted(grouped[task_id], key=lambda item: item.action_id)
        task = task_lookup[task_id]
        best_real_value = max(item.real_value for item in candidates)
        best_real = [item for item in candidates if abs(item.real_value - best_real_value) <= _EPSILON]
        simulated_order = sorted(
            candidates,
            key=lambda item: (
                -item.simulated_value,
                _stable_tie_key(task_id, item.action_id),
            ),
        )
        selected = simulated_order[0]
        second_value = simulated_order[1].simulated_value if len(simulated_order) > 1 else selected.simulated_value
        simulated_best_value = selected.simulated_value
        simulated_top = [
            item
            for item in candidates
            if abs(item.simulated_value - simulated_best_value) <= _EPSILON
        ]
        regret = max(0.0, best_real_value - selected.real_value)
        simulated_top_regrets = [
            max(0.0, best_real_value - item.real_value) for item in simulated_top
        ]
        harmful_top = [
            item
            for item, item_regret in zip(simulated_top, simulated_top_regrets)
            if item_regret + _EPSILON >= harmful_regret_threshold
        ]
        high_fidelity_harmful_top = [
            item
            for item in harmful_top
            if item.token_f1 >= high_text_fidelity_threshold
        ]
        expected_top_regret = fmean(simulated_top_regrets)
        worst_top_regret = max(simulated_top_regrets)
        harmful = regret + _EPSILON >= harmful_regret_threshold
        pairwise, comparable_pairs = _pairwise_agreement(candidates)
        valid_predictions = [
            item for item in candidates if item.parse_ok and item.prediction_error is None
        ]
        within_task_spearman = spearman(
            [item.real_value for item in candidates],
            [item.simulated_value for item in candidates],
        )
        metadata = task.metadata
        decisions.append(
            {
                "task_id": task_id,
                "domain": task.domain,
                "template_id": metadata.get("template_id"),
                "world_id": metadata.get("world_id"),
                "group_id": f"{task.domain}|{metadata.get('template_id')}|{metadata.get('world_id')}",
                "candidate_count": len(candidates),
                "real_best_action_ids": [item.action_id for item in best_real],
                "best_real_value": best_real_value,
                "best_real_terminal_success": any(item.real_terminal_success for item in best_real),
                "selected_action_id": selected.action_id,
                "selected_action_role": selected.action_role,
                "selected_action_risk": selected.action_risk,
                "tie_break_rule": "sha256(actionrankwm-v1|task_id|action_id)",
                "simulated_top_action_ids": sorted(item.action_id for item in simulated_top),
                "simulated_top_tie_count": len(simulated_top),
                "simulated_top_is_tied": len(simulated_top) > 1,
                "simulated_top_harmful_action_ids": sorted(
                    item.action_id for item in harmful_top
                ),
                "simulated_top_harmful_count": len(harmful_top),
                "simulated_top_contains_harmful": bool(harmful_top),
                "simulated_top_high_fidelity_harmful_action_ids": sorted(
                    item.action_id for item in high_fidelity_harmful_top
                ),
                "simulated_top_contains_high_fidelity_harmful": bool(
                    high_fidelity_harmful_top
                ),
                "simulated_top_harmful_token_f1_max": max(
                    (item.token_f1 for item in harmful_top), default=None
                ),
                "selected_real_value": selected.real_value,
                "selected_simulated_value": selected.simulated_value,
                "selected_real_terminal_success": selected.real_terminal_success,
                "selected_token_f1": selected.token_f1,
                "selected_structured_f1": selected.structured_f1,
                "selected_critical_field_accuracy": selected.critical_field_accuracy,
                "simulated_margin": selected.simulated_value - second_value,
                "simulated_spread": max(item.simulated_value for item in candidates)
                - min(item.simulated_value for item in candidates),
                "predicted_success_fraction": fmean(float(item.predicted_success) for item in candidates),
                "parse_failure_fraction": fmean(float(not item.parse_ok) for item in candidates),
                "prediction_error_fraction": fmean(float(item.prediction_error is not None) for item in candidates),
                "valid_prediction_count": len(valid_predictions),
                "all_predictions_invalid": not valid_predictions,
                "pairwise_agreement": pairwise,
                "comparable_pairs": comparable_pairs,
                "within_task_spearman": within_task_spearman,
                "top1_preserved": selected.action_id in {item.action_id for item in best_real},
                "top1_set_intersects_real_best": bool(
                    {item.action_id for item in simulated_top}
                    & {item.action_id for item in best_real}
                ),
                "top1_set_subset_of_real_best": {
                    item.action_id for item in simulated_top
                }.issubset({item.action_id for item in best_real}),
                "regret": regret,
                "simulated_top_expected_regret": expected_top_regret,
                "simulated_top_worst_regret": worst_top_regret,
                "simulated_top_expected_regret_is_harmful": (
                    expected_top_regret + _EPSILON >= harmful_regret_threshold
                ),
                "simulated_top_worst_regret_is_harmful": (
                    worst_top_regret + _EPSILON >= harmful_regret_threshold
                ),
                "harmful_flip": harmful,
                "high_text_fidelity_harmful_flip": harmful
                and selected.token_f1 >= high_text_fidelity_threshold,
            }
        )
    return decisions


def _aggregate_slice(
    decisions: Sequence[Mapping[str, Any]], candidates: Sequence[ScoredCandidate]
) -> dict[str, Any]:
    low_regret = [-float(item["regret"]) for item in decisions]
    pairwise = [float(item["pairwise_agreement"]) for item in decisions]
    fidelity_series = {
        "selected_token_f1": [float(item["selected_token_f1"]) for item in decisions],
        "selected_structured_f1": [float(item["selected_structured_f1"]) for item in decisions],
        "selected_critical_field_accuracy": [
            float(item["selected_critical_field_accuracy"]) for item in decisions
        ],
    }
    associations = {
        "pairwise_agreement_vs_low_regret_spearman": spearman(pairwise, low_regret),
        **{
            f"{name}_vs_low_regret_spearman": spearman(values, low_regret)
            for name, values in fidelity_series.items()
        },
    }
    text_correlations = [
        value
        for key, value in associations.items()
        if key != "pairwise_agreement_vs_low_regret_spearman" and value is not None
    ]
    decision_correlation = associations["pairwise_agreement_vs_low_regret_spearman"]
    best_text_correlation = max(text_correlations) if text_correlations else None
    advantage = (
        decision_correlation - best_text_correlation
        if decision_correlation is not None and best_text_correlation is not None
        else None
    )
    return {
        "task_count": len(decisions),
        "candidate_count": len(candidates),
        "parse_rate": _mean(float(item.parse_ok) for item in candidates),
        "candidate_token_f1_mean": _mean(item.token_f1 for item in candidates),
        "candidate_structured_f1_mean": _mean(item.structured_f1 for item in candidates),
        "candidate_critical_field_accuracy_mean": _mean(
            item.critical_field_accuracy for item in candidates
        ),
        "candidate_patch_repair_rate": _mean(float(item.patch_repair is not None) for item in candidates),
        "pairwise_agreement_mean": _mean(float(item["pairwise_agreement"]) for item in decisions),
        "top1_preservation_rate": _mean(float(item["top1_preserved"]) for item in decisions),
        "top1_set_intersection_rate": _mean(
            float(item["top1_set_intersects_real_best"]) for item in decisions
        ),
        "top1_set_strict_subset_rate": _mean(
            float(item["top1_set_subset_of_real_best"]) for item in decisions
        ),
        "simulated_top_tie_rate": _mean(
            float(item["simulated_top_is_tied"]) for item in decisions
        ),
        "simulated_top_expected_regret_mean": _mean(
            float(item["simulated_top_expected_regret"]) for item in decisions
        ),
        "simulated_top_worst_regret_mean": _mean(
            float(item["simulated_top_worst_regret"]) for item in decisions
        ),
        "unsafe_top_set_rate": _mean(
            float(item["simulated_top_contains_harmful"]) for item in decisions
        ),
        "unsafe_top_set_count": sum(
            bool(item["simulated_top_contains_harmful"]) for item in decisions
        ),
        "high_text_fidelity_unsafe_top_set_count": sum(
            bool(item["simulated_top_contains_high_fidelity_harmful"])
            for item in decisions
        ),
        "all_predictions_invalid_task_rate": _mean(
            float(item["all_predictions_invalid"]) for item in decisions
        ),
        "all_predictions_invalid_task_count": sum(
            bool(item["all_predictions_invalid"]) for item in decisions
        ),
        "expected_harmful_top_set_rate": _mean(
            float(item["simulated_top_expected_regret_is_harmful"])
            for item in decisions
        ),
        "worst_harmful_top_set_rate": _mean(
            float(item["simulated_top_worst_regret_is_harmful"])
            for item in decisions
        ),
        "mean_real_regret": _mean(float(item["regret"]) for item in decisions),
        "harmful_flip_rate": _mean(float(item["harmful_flip"]) for item in decisions),
        "harmful_flip_count": sum(bool(item["harmful_flip"]) for item in decisions),
        "high_text_fidelity_harmful_flip_count": sum(
            bool(item["high_text_fidelity_harmful_flip"]) for item in decisions
        ),
        "selected_terminal_success_rate": _mean(
            float(item["selected_real_terminal_success"]) for item in decisions
        ),
        "associations": associations,
        "best_text_fidelity_correlation": best_text_correlation,
        "decision_correlation_advantage": advantage,
    }


def aggregate_decision_metrics(
    decisions: Sequence[Mapping[str, Any]], scored_candidates: Sequence[ScoredCandidate]
) -> dict[str, Any]:
    domains = sorted({str(item["domain"]) for item in decisions})
    roles = sorted({item.action_role for item in scored_candidates})
    templates = sorted(
        {str(item["template_id"]) for item in decisions if item.get("template_id") is not None}
    )
    template_by_task = {
        str(item["task_id"]): str(item["template_id"])
        for item in decisions
        if item.get("template_id") is not None
    }

    def candidate_slice(candidates: Sequence[ScoredCandidate]) -> dict[str, Any]:
        return {
            "candidate_count": len(candidates),
            "parse_rate": _mean(float(item.parse_ok) for item in candidates),
            "token_f1_mean": _mean(item.token_f1 for item in candidates),
            "structured_f1_mean": _mean(item.structured_f1 for item in candidates),
            "critical_field_accuracy_mean": _mean(
                item.critical_field_accuracy for item in candidates
            ),
            "patch_repair_rate": _mean(float(item.patch_repair is not None) for item in candidates),
            "value_mae": _mean(
                abs(item.simulated_value - item.real_value) for item in candidates
            ),
            "value_signed_error_simulated_minus_real": _mean(
                item.simulated_value - item.real_value for item in candidates
            ),
            "simulated_success_rate": _mean(float(item.predicted_success) for item in candidates),
            "real_terminal_success_rate": _mean(
                float(item.real_terminal_success) for item in candidates
            ),
        }

    return {
        "overall": _aggregate_slice(decisions, scored_candidates),
        "by_domain": {
            domain: _aggregate_slice(
                [item for item in decisions if item["domain"] == domain],
                [item for item in scored_candidates if item.domain == domain],
            )
            for domain in domains
        },
        "by_action_role": {
            role: candidate_slice(
                [item for item in scored_candidates if item.action_role == role]
            )
            for role in roles
        },
        "by_template": {
            template: _aggregate_slice(
                [item for item in decisions if item.get("template_id") == template],
                [
                    item
                    for item in scored_candidates
                    if template_by_task.get(item.task_id) == template
                ],
            )
            for template in templates
        },
        "selected_action_role_counts": {
            role: sum(item["selected_action_role"] == role for item in decisions)
            for role in sorted({str(item["selected_action_role"]) for item in decisions})
        },
    }
