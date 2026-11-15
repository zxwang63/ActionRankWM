from __future__ import annotations

import hashlib
import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from statistics import fmean
from typing import Any

from actionrankwm.metrics.decision import spearman


def _statistics(decisions: Sequence[Mapping[str, Any]]) -> dict[str, float | None]:
    if not decisions:
        return {}
    low_regret = [-float(item["regret"]) for item in decisions]
    pairwise = [float(item["pairwise_agreement"]) for item in decisions]
    decision_correlation = spearman(pairwise, low_regret)
    text_correlations = {
        "token_fidelity_vs_low_regret_spearman": spearman(
            [float(item["selected_token_f1"]) for item in decisions], low_regret
        ),
        "structured_fidelity_vs_low_regret_spearman": spearman(
            [float(item["selected_structured_f1"]) for item in decisions], low_regret
        ),
        "critical_field_fidelity_vs_low_regret_spearman": spearman(
            [float(item["selected_critical_field_accuracy"]) for item in decisions],
            low_regret,
        ),
    }
    valid_text = [value for value in text_correlations.values() if value is not None]
    best_text = max(valid_text) if valid_text else None
    advantage = (
        decision_correlation - best_text
        if decision_correlation is not None and best_text is not None
        else None
    )
    return {
        "pairwise_agreement_mean": fmean(
            float(item["pairwise_agreement"]) for item in decisions
        ),
        "top1_preservation_rate": fmean(
            float(item["top1_preserved"]) for item in decisions
        ),
        "mean_real_regret": fmean(float(item["regret"]) for item in decisions),
        "harmful_flip_rate": fmean(
            float(item["harmful_flip"]) for item in decisions
        ),
        "unsafe_top_set_rate": fmean(
            float(item["simulated_top_contains_harmful"]) for item in decisions
        ),
        "high_text_fidelity_unsafe_top_set_count": float(
            sum(
                bool(item["simulated_top_contains_high_fidelity_harmful"])
                for item in decisions
            )
        ),
        "pairwise_agreement_vs_low_regret_spearman": decision_correlation,
        **text_correlations,
        "best_text_fidelity_correlation": best_text,
        "decision_correlation_advantage": advantage,
    }


def _percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("percentile requires at least one value")
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _seed_for(seed: int, label: str) -> int:
    digest = hashlib.sha256(f"{seed}|{label}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def _bootstrap_slice(
    decisions: Sequence[Mapping[str, Any]],
    *,
    resamples: int,
    confidence: float,
    seed: int,
) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for decision in decisions:
        grouped[str(decision["group_id"])].append(decision)
    groups = sorted(grouped)
    estimates = _statistics(decisions)
    if len(groups) < 2:
        return {
            "status": "not_evaluated",
            "reason": "at_least_two_latent_world_groups_required",
            "task_count": len(decisions),
            "group_count": len(groups),
            "estimates": estimates,
        }

    rng = random.Random(seed)
    samples: dict[str, list[float]] = {name: [] for name in estimates}
    for _ in range(resamples):
        sampled: list[Mapping[str, Any]] = []
        for _ in groups:
            sampled.extend(grouped[rng.choice(groups)])
        values = _statistics(sampled)
        for name, value in values.items():
            if value is not None:
                samples[name].append(float(value))

    alpha = (1.0 - confidence) / 2.0
    minimum_valid = max(20, resamples // 2)
    intervals: dict[str, Any] = {}
    for name, estimate in estimates.items():
        valid = samples[name]
        intervals[name] = {
            "estimate": estimate,
            "ci_low": _percentile(valid, alpha) if len(valid) >= minimum_valid else None,
            "ci_high": _percentile(valid, 1.0 - alpha)
            if len(valid) >= minimum_valid
            else None,
            "valid_resamples": len(valid),
            "valid_fraction": len(valid) / resamples,
        }
    return {
        "status": "evaluated",
        "task_count": len(decisions),
        "group_count": len(groups),
        "intervals": intervals,
    }


def bootstrap_decision_uncertainty(
    decisions: Sequence[Mapping[str, Any]],
    *,
    resamples: int = 2000,
    confidence: float = 0.95,
    seed: int = 20260718,
) -> dict[str, Any]:
    """Grouped percentile bootstrap over prespecified latent-world clusters."""

    if resamples <= 0:
        raise ValueError("bootstrap resamples must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("bootstrap confidence must lie strictly between zero and one")
    domains = sorted({str(item["domain"]) for item in decisions})
    return {
        "method": "percentile_cluster_bootstrap",
        "cluster_unit": "domain|template_id|world_id",
        "resamples": resamples,
        "confidence": confidence,
        "seed": seed,
        "overall": _bootstrap_slice(
            decisions,
            resamples=resamples,
            confidence=confidence,
            seed=_seed_for(seed, "overall"),
        ),
        "by_domain": {
            domain: _bootstrap_slice(
                [item for item in decisions if item["domain"] == domain],
                resamples=resamples,
                confidence=confidence,
                seed=_seed_for(seed, domain),
            )
            for domain in domains
        },
    }
