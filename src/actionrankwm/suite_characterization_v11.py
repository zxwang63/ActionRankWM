from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

from actionrankwm.suite_characterization import characterize_true_branches
from actionrankwm.types import BranchRecord, TaskSpec


WORLD_ID_PATTERN = re.compile(
    r"^(?P<prefix>[a-z0-9_]+)_world_(?P<index>[0-9]{4})$"
)


def normalize_external_world_ids(
    tasks: Sequence[TaskSpec], *, expected_worlds: Sequence[int]
) -> tuple[list[TaskSpec], dict[str, Any]]:
    """Return task copies with fail-closed integer world indices.

    The immutable external task schema names a world separately within each
    mechanism. The characterization core needs the shared numeric suffix in
    order to verify that each of the six mechanisms occurs in every world.
    """

    expected = {int(world) for world in expected_worlds}
    normalized: list[TaskSpec] = []
    observed: set[int] = set()
    raw_identifier_count = 0
    for task in tasks:
        mechanism = str(task.metadata.get("mechanism", ""))
        raw_world = task.metadata.get("world_id")
        if not isinstance(raw_world, str):
            raise ValueError(f"E4b world_id is not a string: {task.task_id}")
        match = WORLD_ID_PATTERN.fullmatch(raw_world)
        if match is None:
            raise ValueError(f"Malformed E4b world_id: {raw_world}")
        if match.group("prefix") != mechanism:
            raise ValueError(
                f"E4b world_id/mechanism mismatch: {task.task_id}/{raw_world}/{mechanism}"
            )
        world_index = int(match.group("index"))
        if world_index not in expected:
            raise ValueError(f"Unexpected E4b world index: {world_index}")

        row = task.to_dict()
        row["metadata"] = dict(row["metadata"])
        row["metadata"]["world_id"] = world_index
        normalized.append(TaskSpec.from_dict(row))
        observed.add(world_index)
        raw_identifier_count += 1

    if observed != expected:
        raise ValueError("Normalized E4b worlds do not match the frozen inventory")
    return normalized, {
        "schema": "mechanism_world_####",
        "pattern": WORLD_ID_PATTERN.pattern,
        "task_count": raw_identifier_count,
        "parsed_world_indices": sorted(observed),
        "source_tasks_mutated": False,
    }


def characterize_external_true_branches_v11(
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
    normalized, normalization = normalize_external_world_ids(
        tasks, expected_worlds=expected_worlds
    )
    result = characterize_true_branches(
        normalized,
        branches,
        action_cost=action_cost,
        violation_penalty=violation_penalty,
        invalid_penalty=invalid_penalty,
        harmful_regret_threshold=harmful_regret_threshold,
        expected_worlds=expected_worlds,
        expected_mechanism_count=expected_mechanism_count,
    )
    result["world_id_normalization"] = normalization
    return result
