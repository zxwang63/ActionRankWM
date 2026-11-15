from __future__ import annotations

import importlib.util
import unittest
from collections import Counter

from actionrankwm.official_scenarios import deterministic_topological_order


class OfficialScenarioPureTests(unittest.TestCase):
    def test_deterministic_topological_order_breaks_ties_by_node(self) -> None:
        self.assertEqual(
            deterministic_topological_order(5, [(0, 2), (1, 2), (2, 4), (3, 4)]),
            [0, 1, 2, 3, 4],
        )

    def test_topological_order_rejects_cycles(self) -> None:
        with self.assertRaises(ValueError):
            deterministic_topological_order(2, [(0, 1), (1, 0)])


@unittest.skipUnless(
    importlib.util.find_spec("tool_sandbox") is not None,
    "optional Apple ToolSandbox runtime is unavailable",
)
class OfficialScenarioRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from actionrankwm.official_scenarios import load_unaugmented_scenarios

        cls.scenarios = load_unaugmented_scenarios()

    def test_exact_unaugmented_and_local_inventories(self) -> None:
        from actionrankwm.official_scenarios import scenario_inventory_row

        rows = [
            scenario_inventory_row(name, family, scenario)
            for name, (family, scenario) in self.scenarios.items()
        ]
        self.assertEqual(len(rows), 129)
        self.assertEqual(
            Counter(row["family"] for row in rows),
            {
                "single_tool_call": 19,
                "multiple_tool_call": 54,
                "multiple_user_turn": 28,
                "insufficient_information": 28,
            },
        )
        self.assertEqual(sum(bool(row["module_local"]) for row in rows), 38)
        self.assertEqual(sum(bool(row["v0_narrow_local"]) for row in rows), 32)
        self.assertTrue(
            all(row["template_group"] != row["scenario_name"] for row in rows if row["is_alt_view"])
        )

    def test_three_prototypes_are_replay_exact_and_decision_relevant(self) -> None:
        from actionrankwm.official_scenarios import PROTOTYPE_SCENARIOS, audit_prototype

        expected_levels = {
            "remove_contact_by_phone": [0.666666666667, 1.0],
            "update_contact_relationship_with_relationship": [0.666666666667, 1.0],
            "send_message_with_contact_content_cellular_off": [0.75, 1.0],
        }
        for name in PROTOTYPE_SCENARIOS:
            family, scenario = self.scenarios[name]
            result = audit_prototype(name, family, scenario)
            self.assertTrue(result["passed"], result["blockers"])
            self.assertTrue(result["determinism_passed"])
            self.assertEqual(result["utility_levels"], expected_levels[name])
            self.assertGreaterEqual(result["real_utility_gap"], 0.20)
            canonical = next(
                branch
                for branch in result["branches"]
                if branch["first_action_key"] == result["canonical_first_action_key"]
            )
            self.assertEqual(canonical["similarity"], 1.0)
            for branch in result["branches"]:
                sandbox_rows = [
                    record
                    for record in branch["execution_records"]
                    if record["kind"] == "action"
                ]
                self.assertTrue(sandbox_rows)


if __name__ == "__main__":
    unittest.main()
