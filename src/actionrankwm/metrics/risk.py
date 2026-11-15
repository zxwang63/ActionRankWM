from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from statistics import fmean
from typing import Any


FEATURE_NAMES = (
    "parse_failure_fraction",
    "prediction_error_fraction",
    "simulated_margin",
    "simulated_spread",
    "predicted_success_fraction",
    "selected_simulated_value",
    "selected_action_high_risk",
)


def _features(decision: Mapping[str, Any]) -> list[float]:
    return [
        float(decision["parse_failure_fraction"]),
        float(decision["prediction_error_fraction"]),
        float(decision["simulated_margin"]),
        float(decision["simulated_spread"]),
        float(decision["predicted_success_fraction"]),
        float(decision["selected_simulated_value"]),
        float(decision["selected_action_risk"] == "high"),
    ]


def _sigmoid(value: float) -> float:
    if value >= 0:
        exponential = math.exp(-value)
        return 1.0 / (1.0 + exponential)
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


def _fit_logistic(features: Sequence[Sequence[float]], labels: Sequence[int]) -> dict[str, Any]:
    dimensions = len(FEATURE_NAMES)
    means = [fmean(row[column] for row in features) for column in range(dimensions)]
    scales = []
    for column in range(dimensions):
        variance = fmean((row[column] - means[column]) ** 2 for row in features)
        scales.append(max(math.sqrt(variance), 1e-6))
    standardized = [
        [(row[column] - means[column]) / scales[column] for column in range(dimensions)]
        for row in features
    ]
    weights = [0.0] * dimensions
    positive_rate = min(1.0 - 1e-5, max(1e-5, fmean(labels)))
    intercept = math.log(positive_rate / (1.0 - positive_rate))
    learning_rate = 0.08
    regularization = 0.02
    for _ in range(500):
        probabilities = [
            _sigmoid(intercept + sum(weight * value for weight, value in zip(weights, row)))
            for row in standardized
        ]
        errors = [probability - label for probability, label in zip(probabilities, labels)]
        intercept -= learning_rate * fmean(errors)
        for column in range(dimensions):
            gradient = fmean(
                error * row[column] for error, row in zip(errors, standardized)
            ) + regularization * weights[column]
            weights[column] -= learning_rate * gradient
    return {"means": means, "scales": scales, "weights": weights, "intercept": intercept}


def _predict(model: Mapping[str, Any], features: Sequence[float]) -> float:
    standardized = [
        (features[index] - model["means"][index]) / model["scales"][index]
        for index in range(len(FEATURE_NAMES))
    ]
    return _sigmoid(
        float(model["intercept"])
        + sum(weight * value for weight, value in zip(model["weights"], standardized))
    )


def _fold(group_id: str, folds: int) -> int:
    digest = hashlib.sha256(group_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % folds


def auroc(labels: Sequence[int], scores: Sequence[float]) -> float | None:
    positives = [score for label, score in zip(labels, scores) if label == 1]
    negatives = [score for label, score in zip(labels, scores) if label == 0]
    if not positives or not negatives:
        return None
    wins = 0.0
    for positive in positives:
        for negative in negatives:
            if positive > negative:
                wins += 1.0
            elif positive == negative:
                wins += 0.5
    return wins / (len(positives) * len(negatives))


def _routing_metrics(
    decisions: Sequence[Mapping[str, Any]], scores: Sequence[float], budget: float
) -> dict[str, Any]:
    route_count = min(len(decisions), max(0, math.ceil(len(decisions) * budget)))
    routed = set(
        sorted(
            range(len(decisions)),
            key=lambda index: (-scores[index], str(decisions[index]["task_id"])),
        )[:route_count]
    )
    baseline_harmful = sum(bool(item["harmful_flip"]) for item in decisions)
    remaining_harmful = sum(
        bool(item["harmful_flip"]) for index, item in enumerate(decisions) if index not in routed
    )
    harmful_reduction = (
        (baseline_harmful - remaining_harmful) / baseline_harmful if baseline_harmful else None
    )
    baseline_success = fmean(
        float(item["selected_real_terminal_success"]) for item in decisions
    ) if decisions else 0.0
    routed_success = fmean(
        float(
            item["best_real_terminal_success"]
            if index in routed
            else item["selected_real_terminal_success"]
        )
        for index, item in enumerate(decisions)
    ) if decisions else 0.0
    return {
        "budget_fraction": budget,
        "routed_task_count": route_count,
        "baseline_harmful_count": baseline_harmful,
        "remaining_harmful_count": remaining_harmful,
        "harmful_reduction": harmful_reduction,
        "baseline_terminal_success_rate": baseline_success,
        "routed_terminal_success_rate": routed_success,
        "terminal_success_gain": routed_success - baseline_success,
    }


def evaluate_cross_fitted_risk(
    decisions: Sequence[Mapping[str, Any]], routing_budget: float, folds: int = 5
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Cross-fit a frozen lightweight risk head on inference-time features only.

    Splits are grouped by the task's latent world. Text/structured fidelity is
    evaluated only as a retrospective baseline and never enters the proposed
    risk head.
    """

    if not decisions:
        return {"status": "not_evaluated", "reason": "no_tasks"}, []
    labels = [int(bool(item["harmful_flip"])) for item in decisions]
    proposed_scores: list[float | None] = [None] * len(decisions)
    fold_records: list[dict[str, Any]] = []
    for held_out in range(folds):
        test_indices = [
            index for index, item in enumerate(decisions) if _fold(str(item["group_id"]), folds) == held_out
        ]
        train_indices = [index for index in range(len(decisions)) if index not in test_indices]
        if not test_indices:
            fold_records.append({"fold": held_out, "status": "empty_test"})
            continue
        train_labels = [labels[index] for index in train_indices]
        if not train_indices or len(set(train_labels)) < 2:
            prior = fmean(train_labels) if train_labels else fmean(labels)
            for index in test_indices:
                proposed_scores[index] = prior
            fold_records.append(
                {
                    "fold": held_out,
                    "status": "prior_only",
                    "train_tasks": len(train_indices),
                    "test_tasks": len(test_indices),
                }
            )
            continue
        model = _fit_logistic([_features(decisions[index]) for index in train_indices], train_labels)
        for index in test_indices:
            proposed_scores[index] = _predict(model, _features(decisions[index]))
        fold_records.append(
            {
                "fold": held_out,
                "status": "fit",
                "train_tasks": len(train_indices),
                "test_tasks": len(test_indices),
                "train_positive_rate": fmean(train_labels),
            }
        )
    fallback_prior = fmean(labels)
    final_scores = [fallback_prior if score is None else score for score in proposed_scores]
    baselines = {
        "token_fidelity_risk": [1.0 - float(item["selected_token_f1"]) for item in decisions],
        "structured_fidelity_risk": [
            1.0 - float(item["selected_structured_f1"]) for item in decisions
        ],
        "critical_field_risk": [
            1.0 - float(item["selected_critical_field_accuracy"]) for item in decisions
        ],
    }
    inference_baselines = {
        "action_risk_heuristic": [
            float(item["selected_action_risk"] == "high") for item in decisions
        ],
        "low_margin_risk": [-float(item["simulated_margin"]) for item in decisions],
        "parse_or_prediction_error_risk": [
            max(
                float(item["parse_failure_fraction"]),
                float(item["prediction_error_fraction"]),
            )
            for item in decisions
        ],
        "low_predicted_value_risk": [
            -float(item["selected_simulated_value"]) for item in decisions
        ],
    }
    proposed_auroc = auroc(labels, final_scores)
    baseline_aurocs = {name: auroc(labels, scores) for name, scores in baselines.items()}
    inference_baseline_aurocs = {
        name: auroc(labels, scores) for name, scores in inference_baselines.items()
    }
    all_baseline_aurocs = {**baseline_aurocs, **inference_baseline_aurocs}
    valid_baselines = {
        name: value for name, value in all_baseline_aurocs.items() if value is not None
    }
    best_baseline_name = max(valid_baselines, key=valid_baselines.get) if valid_baselines else None
    best_baseline = valid_baselines[best_baseline_name] if best_baseline_name else None
    advantage = (
        proposed_auroc - best_baseline
        if proposed_auroc is not None and best_baseline is not None
        else None
    )
    per_task = [
        {
            "task_id": item["task_id"],
            "domain": item["domain"],
            "group_id": item["group_id"],
            "fold": _fold(str(item["group_id"]), folds),
            "harmful_flip": bool(item["harmful_flip"]),
            "proposed_risk": final_scores[index],
            **{name: scores[index] for name, scores in baselines.items()},
            **{name: scores[index] for name, scores in inference_baselines.items()},
        }
        for index, item in enumerate(decisions)
    ]
    status = "evaluated" if proposed_auroc is not None else "not_evaluated"
    reason = None if status == "evaluated" else "both_harmful_and_non_harmful_tasks_required"
    return (
        {
            "status": status,
            "reason": reason,
            "feature_names": list(FEATURE_NAMES),
            "split_unit": "domain|template_id|world_id",
            "folds": folds,
            "fold_records": fold_records,
            "positive_count": sum(labels),
            "negative_count": len(labels) - sum(labels),
            "proposed_auroc": proposed_auroc,
            "retrospective_fidelity_baseline_aurocs": baseline_aurocs,
            "inference_baseline_aurocs": inference_baseline_aurocs,
            "strongest_baseline_name": best_baseline_name,
            "strongest_baseline_auroc": best_baseline,
            "proposed_auroc_advantage": advantage,
            "routing": _routing_metrics(decisions, final_scores, routing_budget),
            "baseline_routing": {
                name: _routing_metrics(decisions, scores, routing_budget)
                for name, scores in {**baselines, **inference_baselines}.items()
            },
        },
        per_task,
    )
