from __future__ import annotations

import json
import unittest

from actionrankwm.bundle import BUNDLE_SCHEMA, build_model_inputs, opaque_join_id
from actionrankwm.tasks import generate_tasks


TEMPLATE_IDS = [
    "toolsandbox.low_battery_cascade",
    "toolsandbox.low_battery_permission",
    "toolsandbox.cellular_message_dependency",
    "toolsandbox.self_identity_cardinality",
    "toolsandbox.duplicate_contact_semantics",
    "toolsandbox.idempotent_setting_error",
]


class LabelBlindBundleTests(unittest.TestCase):
    def test_model_inputs_exclude_all_join_labels_and_supervision(self) -> None:
        tasks = generate_tasks(
            ["toolsandbox"],
            tasks_per_template=2,
            seed=20260718,
            template_ids=TEMPLATE_IDS,
            task_indices=[0, 1],
        )
        records = build_model_inputs(tasks)
        self.assertEqual(len(records), 60)
        self.assertEqual(len({record["join_id"] for record in records}), 60)
        for task, task_records in zip(tasks, (records[index : index + 5] for index in range(0, 60, 5))):
            for action, record in zip(task.candidates, task_records):
                self.assertEqual(record["schema"], BUNDLE_SCHEMA)
                self.assertEqual(record["join_id"], opaque_join_id(task.task_id, action.action_id))
                visible = json.dumps(record["messages"], ensure_ascii=False, sort_keys=True)
                self.assertNotIn(task.task_id, visible)
                user = json.loads(record["messages"][1]["content"])
                self.assertNotIn("task_id", user)
                self.assertNotIn("action_id", user["action"])
                self.assertNotIn("target", user)
                self.assertNotIn("constraints", user)
                self.assertNotIn("oracle_plan", user)
                self.assertNotIn("role", user["action"])
                self.assertNotIn("risk", user["action"])

    def test_opaque_join_is_stable_and_key_specific(self) -> None:
        first = opaque_join_id("task", "c0")
        self.assertEqual(first, opaque_join_id("task", "c0"))
        self.assertNotEqual(first, opaque_join_id("task", "c1"))
        self.assertNotEqual(first, opaque_join_id("other", "c0"))
        self.assertEqual(len(first), 64)


if __name__ == "__main__":
    unittest.main()
