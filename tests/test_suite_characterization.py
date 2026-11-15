from __future__ import annotations

import copy
import unittest

from actionrankwm.suite_characterization import characterize_true_branches
from actionrankwm.types import Action, BranchRecord, TaskSpec


ROLES = ("oracle", "harmless", "violation", "invalid", "multi")


def _fixture() -> tuple[list[TaskSpec], list[BranchRecord]]:
    tasks: list[TaskSpec] = []
    branches: list[BranchRecord] = []
    for world in range(10, 30):
        for mechanism_index in range(6):
            task_id = f"task-{world}-{mechanism_index}"
            actions = [
                Action(
                    action_id=f"a-{role}",
                    tool="synthetic",
                    arguments={"role": role},
                    role=role,
                )
                for role in ROLES
            ]
            tasks.append(
                TaskSpec(
                    task_id=task_id,
                    domain="toolsandbox",
                    instruction="synthetic",
                    initial_state={},
                    target={},
                    constraints=[],
                    oracle_plan=[],
                    candidates=actions,
                    metadata={
                        "world_id": world,
                        "mechanism": f"mechanism-{mechanism_index}",
                    },
                )
            )
            specifications = {
                "oracle": (True, True, 1, [], 0.92),
                "harmless": (False, True, 1, [], -0.08),
                "violation": (False, True, 1, ["v1"], -0.48),
                "invalid": (False, False, 1, [], -0.43),
                "multi": (False, True, 3, ["v1", "v2"], -1.0),
            }
            for action in actions:
                success, ok, steps, violations, value = specifications[action.role]
                branches.append(
                    BranchRecord(
                        task_id=task_id,
                        domain="toolsandbox",
                        action=action,
                        initial_state_digest="synthetic",
                        real_observation={},
                        real_state_patch={},
                        candidate_ok=ok,
                        immediate_success=success,
                        terminal_success=success,
                        violations=violations,
                        executed_steps=steps,
                        real_value=value,
                    )
                )
    return tasks, branches


def _run(
    tasks: list[TaskSpec], branches: list[BranchRecord]
) -> dict[str, object]:
    return characterize_true_branches(
        tasks,
        branches,
        action_cost=0.08,
        violation_penalty=0.40,
        invalid_penalty=0.35,
        harmful_regret_threshold=0.20,
        expected_worlds=range(10, 30),
    )


class SuiteCharacterizationTests(unittest.TestCase):
    def test_complete_finite_suite_summary_is_exact(self) -> None:
        tasks, branches = _fixture()
        result = _run(tasks, branches)
        overall = result["overall"]
        self.assertEqual(overall["task_count"], 120)
        self.assertEqual(overall["candidate_count"], 600)
        self.assertEqual(overall["candidate_role_counts"], {role: 120 for role in ROLES})
        self.assertEqual(overall["executed_step_counts"], {"1": 480, "3": 120})
        self.assertEqual(overall["terminal_success_candidate_count"], 120)
        self.assertEqual(overall["candidate_invalid_count"], 120)
        self.assertEqual(overall["candidate_with_violation_count"], 240)
        self.assertEqual(overall["lower_clipping_count"], 120)
        self.assertEqual(overall["upper_clipping_count"], 0)
        self.assertEqual(overall["utility_level_count_distribution"], {"5": 120})
        self.assertEqual(overall["real_top_set_size_distribution"], {"1": 120})
        self.assertEqual(overall["top_gap_summary"]["median"], 1.0)
        self.assertEqual(overall["candidate_regret_ge_threshold"]["count"], 480)
        self.assertEqual(set(result["by_mechanism"]), {f"mechanism-{i}" for i in range(6)})
        self.assertTrue(result["gate_neutral"])
        self.assertFalse(result["model_output_read"])

    def test_changed_stored_value_fails_closed(self) -> None:
        tasks, branches = _fixture()
        branches = copy.deepcopy(branches)
        branches[0].real_value = 0.91
        with self.assertRaisesRegex(ValueError, "frozen value contract"):
            _run(tasks, branches)

    def test_duplicate_branch_fails_closed(self) -> None:
        tasks, branches = _fixture()
        branches = branches[:-1] + [copy.deepcopy(branches[0])]
        with self.assertRaisesRegex(ValueError, "Duplicate E4b branch"):
            _run(tasks, branches)


if __name__ == "__main__":
    unittest.main()
