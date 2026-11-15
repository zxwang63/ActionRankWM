from __future__ import annotations

import copy
import random
import unittest

from actionrankwm.attribution import paths_overlap
from actionrankwm.branches import ValueConfig, evaluate_real_branch
from actionrankwm.dccp import (
    candidate_observation_corruptions,
    classify_corruptions,
    select_matched_pair,
    valid_training_observation,
)
from actionrankwm.tasks import generate_tasks
from actionrankwm.toolsandbox_tasks import low_battery_cascade


class DCCPTests(unittest.TestCase):
    def setUp(self) -> None:
        self.value_config = ValueConfig()
        self.task = generate_tasks(
            ["sqlite_mcp"],
            tasks_per_template=1,
            seed=19,
            template_ids=["sqlite.rule.audit_suffix"],
        )[0]
        self.branches = [
            evaluate_real_branch(self.task, action, self.value_config)
            for action in self.task.candidates
        ]
        self.real_values = {
            branch.action.action_id: branch.real_value for branch in self.branches
        }

    def test_corruption_enumeration_is_deterministic_and_unique(self) -> None:
        branch = self.branches[0]
        siblings = [item.real_observation for item in self.branches[1:]]
        first = candidate_observation_corruptions(
            branch.real_observation,
            siblings,
            self.task.metadata["critical_fields"],
        )
        second = candidate_observation_corruptions(
            branch.real_observation,
            siblings,
            self.task.metadata["critical_fields"],
        )
        self.assertEqual(first, second)
        serialized = [str(item[2]) for item in first]
        self.assertEqual(len(serialized), len(set(serialized)))

    def test_matched_pair_uses_same_positive_and_executable_negatives(self) -> None:
        branch = self.branches[0]
        siblings = [
            item.real_observation
            for item in self.branches
            if item.action.action_id != branch.action.action_id
        ]
        corruptions = classify_corruptions(
            self.task,
            branch.action,
            branch.real_observation,
            siblings,
            self.real_values,
            self.value_config,
        )
        selected = select_matched_pair(
            corruptions,
            branch.real_value,
            self.task.metadata["critical_fields"],
        )
        self.assertIsNotNone(selected)
        decision, neutral = selected or (None, None)
        assert decision is not None and neutral is not None
        self.assertTrue(decision.decision_critical)
        self.assertFalse(neutral.decision_critical)
        self.assertAlmostEqual(neutral.simulated_value, branch.real_value)
        self.assertGreaterEqual(decision.token_f1, 0.75)
        self.assertLessEqual(decision.relative_length_delta, 0.10)
        self.assertTrue(
            all(
                not any(
                    paths_overlap(path, critical)
                    for critical in self.task.metadata["critical_fields"]
                )
                for path in neutral.changed_paths
            )
        )

    def test_toolsandbox_training_contract_rejects_inconsistent_patches(self) -> None:
        task = low_battery_cascade(30, random.Random(7))
        successful = {
            "ok": True,
            "result": None,
            "error_type": None,
            "error": None,
            "state_patch": {
                "domain": "toolsandbox",
                "operation": "none",
                "namespaces": {},
            },
        }
        self.assertTrue(valid_training_observation(task, successful))
        inconsistent = copy.deepcopy(successful)
        inconsistent["state_patch"]["namespaces"] = {"setting": []}
        self.assertFalse(valid_training_observation(task, inconsistent))
        failed = copy.deepcopy(successful)
        failed.update(
            {
                "ok": False,
                "result": None,
                "error_type": "ValueError",
                "error": "already equal",
            }
        )
        self.assertTrue(valid_training_observation(task, failed))
        failed["state_patch"]["operation"] = "replace_namespaces"
        failed["state_patch"]["namespaces"] = {"setting": []}
        self.assertFalse(valid_training_observation(task, failed))


if __name__ == "__main__":
    unittest.main()
