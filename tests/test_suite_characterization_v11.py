from __future__ import annotations

import unittest

from actionrankwm.suite_characterization_v11 import normalize_external_world_ids
from actionrankwm.types import TaskSpec


def _task(*, mechanism: str, world_id: object, task_id: str = "task") -> TaskSpec:
    return TaskSpec(
        task_id=task_id,
        domain="toolsandbox",
        instruction="synthetic",
        initial_state={},
        target={},
        constraints=[],
        oracle_plan=[],
        candidates=[],
        metadata={"mechanism": mechanism, "world_id": world_id},
    )


class SuiteCharacterizationV11Tests(unittest.TestCase):
    def test_external_world_ids_are_normalized_without_mutating_source(self) -> None:
        tasks = [
            _task(
                mechanism="low_battery_cascade",
                world_id=f"low_battery_cascade_world_{world:04d}",
                task_id=f"task-{world}",
            )
            for world in range(10, 30)
        ]
        normalized, audit = normalize_external_world_ids(
            tasks, expected_worlds=range(10, 30)
        )
        self.assertEqual([task.metadata["world_id"] for task in normalized], list(range(10, 30)))
        self.assertEqual(tasks[0].metadata["world_id"], "low_battery_cascade_world_0010")
        self.assertFalse(audit["source_tasks_mutated"])

    def test_mechanism_prefix_mismatch_fails_closed(self) -> None:
        tasks = [
            _task(
                mechanism="contact_defaulting",
                world_id=f"wrong_prefix_world_{world:04d}",
                task_id=f"task-{world}",
            )
            for world in range(10, 30)
        ]
        with self.assertRaisesRegex(ValueError, "world_id/mechanism mismatch"):
            normalize_external_world_ids(tasks, expected_worlds=range(10, 30))

    def test_malformed_world_id_fails_closed(self) -> None:
        tasks = [
            _task(
                mechanism="contact_defaulting",
                world_id=f"contact_defaulting_world_{world}",
                task_id=f"task-{world}",
            )
            for world in range(10, 30)
        ]
        with self.assertRaisesRegex(ValueError, "Malformed E4b world_id"):
            normalize_external_world_ids(tasks, expected_worlds=range(10, 30))


if __name__ == "__main__":
    unittest.main()
