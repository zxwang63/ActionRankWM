from __future__ import annotations

import copy
import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from actionrankwm.attribution import (
    EPSILON,
    lookup_path,
    paths_overlap,
    restore_paths,
    summarize_value_vectors,
)
from actionrankwm.branches import ValueConfig, evaluate_simulated_branch
from actionrankwm.io import canonical_json, digest_json
from actionrankwm.metrics.text import token_f1
from actionrankwm.types import Action, BranchRecord, TaskSpec


FIXED_SCHEMA_PATHS = (
    "ok",
    "result",
    "error_type",
    "error",
    "state_patch.operation",
)


@dataclass(frozen=True)
class Corruption:
    task_id: str
    action_id: str
    generator: str
    changed_paths: tuple[str, ...]
    observation: dict[str, Any]
    simulated_value: float
    token_f1: float
    relative_length_delta: float
    pairwise_agreement: float
    harmful_top: bool
    top_set_changed: bool
    decision_critical: bool

    @property
    def stable_id(self) -> str:
        payload = {
            "task_id": self.task_id,
            "action_id": self.action_id,
            "generator": self.generator,
            "changed_paths": list(self.changed_paths),
            "observation": self.observation,
        }
        return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "corruption_id": self.stable_id,
            "task_id": self.task_id,
            "action_id": self.action_id,
            "generator": self.generator,
            "changed_paths": list(self.changed_paths),
            "observation": copy.deepcopy(self.observation),
            "simulated_value": self.simulated_value,
            "token_f1": self.token_f1,
            "relative_length_delta": self.relative_length_delta,
            "pairwise_agreement": self.pairwise_agreement,
            "harmful_top": self.harmful_top,
            "top_set_changed": self.top_set_changed,
            "decision_critical": self.decision_critical,
        }


def _namespace_paths(observations: Sequence[Mapping[str, Any]]) -> list[str]:
    names: set[str] = set()
    for observation in observations:
        exists, namespaces = lookup_path(observation, "state_patch.namespaces")
        if exists and isinstance(namespaces, Mapping):
            names.update(str(name) for name in namespaces)
    return [f"state_patch.namespaces.{name}" for name in sorted(names)]


def _length_delta(reference: Mapping[str, Any], candidate: Mapping[str, Any]) -> float:
    reference_length = len(canonical_json(reference))
    candidate_length = len(canonical_json(candidate))
    return abs(candidate_length - reference_length) / max(reference_length, 1)


def _replace_path(
    target: Mapping[str, Any], source: Mapping[str, Any], path: str
) -> dict[str, Any]:
    restored = restore_paths(target, source, (path,))
    if not isinstance(restored, Mapping):
        raise TypeError("A corruption must remain a JSON object")
    return copy.deepcopy(dict(restored))


def _replace_paths(
    target: Mapping[str, Any], source: Mapping[str, Any], paths: Sequence[str]
) -> dict[str, Any]:
    restored = restore_paths(target, source, paths)
    if not isinstance(restored, Mapping):
        raise TypeError("A corruption must remain a JSON object")
    return copy.deepcopy(dict(restored))


def valid_training_observation(
    task: TaskSpec, observation: Mapping[str, Any]
) -> bool:
    if task.domain != "toolsandbox":
        return isinstance(observation.get("ok"), bool) and isinstance(
            observation.get("state_patch"), Mapping
        )
    if set(observation) != {"ok", "result", "error_type", "error", "state_patch"}:
        return False
    ok = observation.get("ok")
    error_type = observation.get("error_type")
    error = observation.get("error")
    if not isinstance(ok, bool):
        return False
    if ok:
        if error_type is not None or error is not None:
            return False
    else:
        if not isinstance(error_type, str) or not error_type:
            return False
        if not isinstance(error, str) or not error or observation.get("result") is not None:
            return False
    patch = observation.get("state_patch")
    if not isinstance(patch, Mapping) or set(patch) != {
        "domain",
        "operation",
        "namespaces",
    }:
        return False
    if patch.get("domain") != "toolsandbox":
        return False
    operation = patch.get("operation")
    namespaces = patch.get("namespaces")
    if not isinstance(namespaces, Mapping):
        return False
    if operation == "none":
        return not namespaces
    if operation != "replace_namespaces" or not namespaces or not ok:
        return False
    if set(namespaces) - {"setting", "contact", "messaging", "reminder"}:
        return False
    return all(
        isinstance(rows, list)
        and all(isinstance(row, Mapping) for row in rows)
        for rows in namespaces.values()
    )


def _mutated_scalar(value: Any) -> Any:
    if isinstance(value, bool):
        return not value
    if value is None:
        return "unknown"
    if isinstance(value, int) and not isinstance(value, bool):
        return value + 1
    if isinstance(value, float):
        return value + 1e-3
    if isinstance(value, str):
        if not value:
            return "x"
        last = value[-1]
        replacement = "x" if last != "x" else "y"
        return value[:-1] + replacement
    return None


def _leaf_paths(value: Any, prefix: str = "") -> list[str]:
    if isinstance(value, Mapping):
        paths: list[str] = []
        for key in sorted(value, key=str):
            path = f"{prefix}.{key}" if prefix else str(key)
            paths.extend(_leaf_paths(value[key], path))
        return paths
    if isinstance(value, list):
        paths = []
        for index, item in enumerate(value):
            path = f"{prefix}.{index}" if prefix else str(index)
            paths.extend(_leaf_paths(item, path))
        return paths
    return [prefix or "<root>"]


def _assign_existing_scalar(
    observation: Mapping[str, Any], path: str, replacement: Any
) -> dict[str, Any] | None:
    exists, current = lookup_path(observation, path)
    if not exists or isinstance(current, (Mapping, list)) or current == replacement:
        return None
    source: Any = {}
    segments = path.split(".") if path else []
    cursor = source
    for index, segment in enumerate(segments):
        is_last = index == len(segments) - 1
        next_is_index = not is_last and segments[index + 1].isdigit()
        if isinstance(cursor, dict):
            if segment.isdigit():
                return None
            if is_last:
                cursor[segment] = copy.deepcopy(replacement)
            else:
                cursor[segment] = [] if next_is_index else {}
                cursor = cursor[segment]
        elif isinstance(cursor, list):
            if not segment.isdigit():
                return None
            position = int(segment)
            while len(cursor) <= position:
                cursor.append(None)
            if is_last:
                cursor[position] = copy.deepcopy(replacement)
            else:
                cursor[position] = [] if next_is_index else {}
                cursor = cursor[position]
        else:
            return None
    return _replace_path(observation, source, path)


def candidate_observation_corruptions(
    positive: Mapping[str, Any],
    sibling_observations: Sequence[Mapping[str, Any]],
    critical_paths: Sequence[str],
) -> list[tuple[str, tuple[str, ...], dict[str, Any]]]:
    """Enumerate deterministic, label-free JSON corruptions for one branch.

    The returned candidates have not been replayed and are not yet training
    examples.  Decision-critical versus value-neutral status is determined only
    after executable replay by :func:`classify_corruptions`.
    """

    observations = [positive, *sibling_observations]
    atomic_paths = sorted(
        set(FIXED_SCHEMA_PATHS)
        | set(_namespace_paths(observations))
        | {str(path) for path in critical_paths}
    )
    candidates: dict[str, tuple[str, tuple[str, ...], dict[str, Any]]] = {}

    for sibling_index, sibling in enumerate(sibling_observations):
        for path in atomic_paths:
            if not lookup_path(positive, path)[0] and not lookup_path(sibling, path)[0]:
                continue
            changed = _replace_path(positive, sibling, path)
            if changed == positive:
                continue
            key = digest_json(changed)
            candidates.setdefault(
                key,
                (f"sibling_copy:{sibling_index}:{path}", (path,), changed),
            )

        coherent_groups = (
            ("status", ("ok", "result", "error_type", "error")),
            ("patch", ("state_patch",)),
            (
                "complete_observation",
                ("ok", "result", "error_type", "error", "state_patch"),
            ),
        )
        for group_name, paths in coherent_groups:
            changed = _replace_paths(positive, sibling, paths)
            if changed == positive:
                continue
            key = digest_json(changed)
            candidates.setdefault(
                key,
                (
                    f"sibling_group:{sibling_index}:{group_name}",
                    tuple(paths),
                    changed,
                ),
            )

    for path in _leaf_paths(positive):
        exists, value = lookup_path(positive, path)
        if not exists or isinstance(value, (Mapping, list)):
            continue
        replacement = _mutated_scalar(value)
        if replacement is None:
            continue
        changed = _assign_existing_scalar(positive, path, replacement)
        if changed is None or changed == positive:
            continue
        key = digest_json(changed)
        candidates.setdefault(key, (f"scalar_mutation:{path}", (path,), changed))

    return [candidates[key] for key in sorted(candidates)]


def classify_corruptions(
    task: TaskSpec,
    action: Action,
    positive: Mapping[str, Any],
    sibling_observations: Sequence[Mapping[str, Any]],
    real_values: Mapping[str, float],
    value_config: ValueConfig,
    *,
    min_token_f1: float = 0.75,
    max_relative_length_delta: float = 0.10,
    harmful_regret_threshold: float = 0.20,
) -> list[Corruption]:
    """Replay and classify all deterministic corruptions for one action."""

    if action.action_id not in real_values:
        raise ValueError(f"Missing real value for {task.task_id}/{action.action_id}")
    baseline = summarize_value_vectors(
        task.task_id, real_values, real_values, harmful_regret_threshold
    )
    critical_paths = [str(path) for path in task.metadata.get("critical_fields", [])]
    results: list[Corruption] = []
    for generator, changed_paths, observation in candidate_observation_corruptions(
        positive, sibling_observations, critical_paths
    ):
        fidelity = token_f1(positive, observation)
        relative_length_delta = _length_delta(positive, observation)
        if fidelity + EPSILON < min_token_f1:
            continue
        if relative_length_delta > max_relative_length_delta + EPSILON:
            continue
        if not valid_training_observation(task, observation):
            continue
        simulated_value, _, _, error = evaluate_simulated_branch(
            task, action, observation, value_config
        )
        if error is not None:
            continue
        simulated_values = dict(real_values)
        simulated_values[action.action_id] = float(simulated_value)
        summary = summarize_value_vectors(
            task.task_id,
            real_values,
            simulated_values,
            harmful_regret_threshold,
        )
        pairwise_changed = summary["pairwise_agreement"] < 1.0 - EPSILON
        top_set_changed = (
            summary["simulated_top_action_ids"] != baseline["simulated_top_action_ids"]
        )
        decision_critical = pairwise_changed or top_set_changed
        results.append(
            Corruption(
                task_id=task.task_id,
                action_id=action.action_id,
                generator=generator,
                changed_paths=changed_paths,
                observation=observation,
                simulated_value=float(simulated_value),
                token_f1=fidelity,
                relative_length_delta=relative_length_delta,
                pairwise_agreement=float(summary["pairwise_agreement"]),
                harmful_top=bool(summary["contains_harmful_top"]),
                top_set_changed=top_set_changed,
                decision_critical=decision_critical,
            )
        )
    return sorted(results, key=lambda item: item.stable_id)


def _is_noncritical(corruption: Corruption, critical_paths: Sequence[str]) -> bool:
    return all(
        not any(paths_overlap(path, critical) for critical in critical_paths)
        for path in corruption.changed_paths
    )


def _eligible_decision_corruption(
    corruption: Corruption, critical_paths: Sequence[str]
) -> bool:
    if corruption.generator.startswith("sibling_group:"):
        return True
    for path in corruption.changed_paths:
        if path in FIXED_SCHEMA_PATHS:
            return True
        if path.startswith("state_patch.namespaces.") and len(path.split(".")) == 3:
            return True
        exists, changed_value = lookup_path(corruption.observation, path)
        if (
            exists
            and isinstance(changed_value, bool)
            and path.startswith("state_patch.namespaces.")
        ):
            return True
        if any(paths_overlap(path, critical) for critical in critical_paths):
            return True
    return False


def select_matched_pair(
    corruptions: Sequence[Corruption],
    real_value: float,
    critical_paths: Sequence[str],
) -> tuple[Corruption, Corruption] | None:
    """Select one decision-critical and one same-prompt value-neutral negative.

    The neutral control is matched as closely as possible on token fidelity and
    serialized-length change.  Non-overlap with predeclared critical paths is
    preferred, but executable value neutrality is the operative control when a
    task marks an entire namespace as critical. Returning ``None`` keeps an
    unmatched prompt out of all trainable conditions rather than giving one
    objective extra inputs.
    """

    decision = [
        item
        for item in corruptions
        if item.decision_critical
        and _eligible_decision_corruption(item, critical_paths)
    ]
    neutral = [
        item
        for item in corruptions
        if not item.decision_critical
        and abs(item.simulated_value - real_value) <= EPSILON
    ]
    if not decision or not neutral:
        return None
    chosen_decision = sorted(
        decision,
        key=lambda item: (
            not item.harmful_top,
            item.pairwise_agreement,
            -abs(item.simulated_value - real_value),
            -item.token_f1,
            item.stable_id,
        ),
    )[0]
    chosen_neutral = sorted(
        neutral,
        key=lambda item: (
            not _is_noncritical(item, critical_paths),
            not item.generator.startswith("scalar_mutation:state_patch."),
            abs(item.token_f1 - chosen_decision.token_f1),
            abs(item.relative_length_delta - chosen_decision.relative_length_delta),
            item.stable_id,
        ),
    )[0]
    return chosen_decision, chosen_neutral


def branch_lookup(branches: Sequence[BranchRecord]) -> dict[tuple[str, str], BranchRecord]:
    lookup: dict[tuple[str, str], BranchRecord] = {}
    for branch in branches:
        key = (branch.task_id, branch.action.action_id)
        if key in lookup:
            raise ValueError(f"Duplicate branch: {key}")
        lookup[key] = branch
    return lookup
