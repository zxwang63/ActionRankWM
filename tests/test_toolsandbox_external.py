from __future__ import annotations

import importlib.util
import json
import unittest

from actionrankwm.branches import ValueConfig, evaluate_real_branch, evaluate_simulated_branch
from actionrankwm.providers.prompts import build_messages
from actionrankwm.semantics import apply_state_patch, target_success_for_state
from actionrankwm.tasks import generate_tasks


TEMPLATE_IDS = [
    "toolsandbox.low_battery_cascade",
    "toolsandbox.low_battery_permission",
    "toolsandbox.cellular_message_dependency",
    "toolsandbox.self_identity_cardinality",
    "toolsandbox.duplicate_contact_semantics",
    "toolsandbox.idempotent_setting_error",
]


class ToolSandboxTaskTests(unittest.TestCase):
    def test_frozen_inventory_is_pure_and_grouped(self) -> None:
        tasks = generate_tasks(
            ["toolsandbox"],
            tasks_per_template=2,
            seed=20260718,
            template_ids=TEMPLATE_IDS,
            task_indices=[0, 1],
        )
        self.assertEqual(len(tasks), 12)
        self.assertEqual(len({task.task_id for task in tasks}), 12)
        self.assertTrue(all(len(task.candidates) == 5 for task in tasks))
        self.assertTrue(all(task.candidates[0].role == "oracle" for task in tasks))
        self.assertTrue(
            all(task.metadata["suite"] == "external_validity_v09" for task in tasks)
        )
        self.assertEqual(
            {task.metadata["mechanism"] for task in tasks},
            {template_id.removeprefix("toolsandbox.") for template_id in TEMPLATE_IDS},
        )

    def test_patch_and_target_use_complete_namespace_snapshots(self) -> None:
        task = generate_tasks(
            ["toolsandbox"],
            1,
            20260718,
            ["toolsandbox.low_battery_cascade"],
        )[0]
        setting = dict(task.initial_state["namespaces"]["setting"][0])
        setting["wifi"] = False
        projected = apply_state_patch(
            task.initial_state,
            {
                "domain": "toolsandbox",
                "operation": "replace_namespaces",
                "namespaces": {"setting": [setting]},
            },
        )
        self.assertTrue(target_success_for_state(task, projected))
        with self.assertRaises(ValueError):
            apply_state_patch(
                task.initial_state,
                {
                    "domain": "toolsandbox",
                    "operation": "replace_namespaces",
                    "namespaces": {"setting": {"wifi": False}},
                },
            )

    def test_prompt_exposes_semantics_but_not_candidate_roles(self) -> None:
        task = generate_tasks(
            ["toolsandbox"],
            1,
            20260718,
            ["toolsandbox.low_battery_cascade"],
        )[0]
        messages = build_messages(task, task.candidates[2])
        user = messages[-1]["content"]
        self.assertIn("tool_docs", user)
        self.assertIn("set_low_battery_mode_status", user)
        self.assertNotIn("mechanism_critical", user)
        self.assertNotIn(task.task_id, user)
        self.assertNotIn("action_id", json.loads(user)["action"])
        self.assertNotIn("real_value", user)
        self.assertNotIn("oracle_plan", user)


@unittest.skipUnless(
    importlib.util.find_spec("tool_sandbox") is not None,
    "optional Apple ToolSandbox runtime is unavailable",
)
class ToolSandboxRuntimeTests(unittest.TestCase):
    def test_real_branches_are_roundtrip_exact_and_decision_relevant(self) -> None:
        tasks = generate_tasks(
            ["toolsandbox"],
            1,
            20260718,
            TEMPLATE_IDS,
        )
        value_config = ValueConfig()
        for task in tasks:
            branches = [
                evaluate_real_branch(task, action, value_config)
                for action in task.candidates
            ]
            oracle = next(branch for branch in branches if branch.action.role == "oracle")
            critical = next(
                branch
                for branch in branches
                if branch.action.role == "mechanism_critical"
            )
            self.assertAlmostEqual(
                oracle.real_value,
                max(branch.real_value for branch in branches),
                places=12,
            )
            self.assertGreaterEqual(oracle.real_value - critical.real_value, 0.20)
            for branch in branches:
                simulated, _, _, error = evaluate_simulated_branch(
                    task, branch.action, branch.real_observation, value_config
                )
                self.assertIsNone(error)
                self.assertAlmostEqual(simulated, branch.real_value, places=12)


if __name__ == "__main__":
    unittest.main()
