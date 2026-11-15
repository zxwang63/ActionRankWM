from __future__ import annotations

import hashlib
import itertools
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from actionrankwm.attribution import EPSILON, lookup_path, paths_differ
from actionrankwm.io import canonical_json, digest_json


MAX_CARDINALITY = 3
FIXED_ATOM_PATHS = (
    "ok",
    "result",
    "error_type",
    "error",
    "state_patch.domain",
    "state_patch.operation",
)


@dataclass(frozen=True, order=True)
class RepairAtom:
    action_id: str
    path: str

    def to_dict(self) -> dict[str, str]:
        return {"action_id": self.action_id, "path": self.path}


def _namespace_names(observation: Any) -> set[str]:
    exists, namespaces = lookup_path(observation, "state_patch.namespaces")
    if not exists or not isinstance(namespaces, Mapping):
        return set()
    return {str(name) for name in namespaces}


def schema_repair_atoms(
    action_id: str,
    predicted_observation: Mapping[str, Any],
    real_observation: Mapping[str, Any],
) -> list[RepairAtom]:
    """Return differing atoms from the protocol's fixed schema partition."""

    candidate_paths = list(FIXED_ATOM_PATHS)
    namespaces = sorted(
        _namespace_names(predicted_observation) | _namespace_names(real_observation)
    )
    candidate_paths.extend(
        f"state_patch.namespaces.{namespace}" for namespace in namespaces
    )
    return [
        RepairAtom(action_id=action_id, path=path)
        for path in candidate_paths
        if paths_differ(predicted_observation, real_observation, path)
    ]


def group_atom_paths(atoms: Sequence[RepairAtom]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for atom in sorted(set(atoms)):
        grouped.setdefault(atom.action_id, []).append(atom.path)
    return grouped


def classify_value_change(
    baseline_summary: Mapping[str, Any],
    baseline_values: Mapping[str, float],
    repaired_values: Mapping[str, float],
) -> str:
    harmful_decreased = any(
        float(repaired_values[action_id])
        < float(baseline_values[action_id]) - EPSILON
        for action_id in baseline_summary["harmful_top_action_ids"]
    )
    best_increased = any(
        float(repaired_values[action_id])
        > float(baseline_values[action_id]) + EPSILON
        for action_id in baseline_summary["real_best_action_ids"]
    )
    if harmful_decreased and best_increased:
        return "harmful_overprediction_and_real_best_underprediction"
    if harmful_decreased:
        return "harmful_overprediction"
    if best_increased:
        return "real_best_underprediction"
    return "other_or_tie_structure"


def exact_minimal_search(
    atoms: Sequence[RepairAtom],
    evaluate: Callable[[tuple[RepairAtom, ...]], Mapping[str, Any]],
    max_cardinality: int = MAX_CARDINALITY,
) -> dict[str, Any]:
    """Search exact subsets and bind every attempted outcome into one digest."""

    if max_cardinality != MAX_CARDINALITY:
        raise ValueError(f"Protocol v1 requires max_cardinality={MAX_CARDINALITY}")
    ordered = tuple(sorted(set(atoms)))
    attempt_hasher = hashlib.sha256()
    attempted_by_cardinality: dict[str, int] = {}
    successful: list[dict[str, Any]] = []
    minimal_cardinality: int | None = None
    total_attempts = 0

    for cardinality in range(1, max_cardinality + 1):
        count = 0
        level_successes: list[dict[str, Any]] = []
        for subset in itertools.combinations(ordered, cardinality):
            result = dict(evaluate(subset))
            summary = result["summary"]
            failures = list(result.get("replay_failures", []))
            attempt_record = {
                "atoms": [atom.to_dict() for atom in subset],
                "contains_harmful_top": bool(summary["contains_harmful_top"]),
                "replay_failure_count": len(failures),
                "simulated_values_sha256": digest_json(result["simulated_values"]),
            }
            attempt_hasher.update(canonical_json(attempt_record).encode("utf-8"))
            attempt_hasher.update(b"\n")
            count += 1
            total_attempts += 1
            if not summary["contains_harmful_top"]:
                level_successes.append(
                    {
                        "atoms": [atom.to_dict() for atom in subset],
                        "summary": summary,
                        "simulated_values": result["simulated_values"],
                        "replay_failures": failures,
                    }
                )
        attempted_by_cardinality[str(cardinality)] = count
        if level_successes:
            minimal_cardinality = cardinality
            successful = level_successes
            break

    return {
        "atom_count": len(ordered),
        "max_cardinality": max_cardinality,
        "attempt_count": total_attempts,
        "attempted_by_cardinality": attempted_by_cardinality,
        "attempt_digest_sha256": attempt_hasher.hexdigest(),
        "minimal_cardinality": minimal_cardinality,
        "successful_subsets": successful,
    }
