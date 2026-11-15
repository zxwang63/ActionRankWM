from __future__ import annotations

import importlib.util
import unittest


@unittest.skipUnless(
    importlib.util.find_spec("tool_sandbox") is not None,
    "optional Apple ToolSandbox runtime is unavailable",
)
class OfficialScenarioReplayV11Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from actionrankwm.official_scenarios import load_unaugmented_scenarios

        cls.scenarios = load_unaugmented_scenarios()

    def test_prototypes_and_terminal_state_alt_use_two_message_transactions(self) -> None:
        from actionrankwm.official_scenarios import PROTOTYPE_SCENARIOS
        from actionrankwm.official_scenarios_v11 import audit_official_scenario_v11

        names = [
            *PROTOTYPE_SCENARIOS,
            "update_contact_relationship_with_relationship_alt",
        ]
        for name in names:
            family, scenario = self.scenarios[name]
            result = audit_official_scenario_v11(name, family, scenario)
            self.assertTrue(result["passed"], (name, result["blockers"]))
            self.assertTrue(result["transaction_contract_passed"])
            self.assertTrue(result["determinism_passed"])
            canonical = next(
                branch
                for branch in result["branches"]
                if branch["first_action_key"] == result["canonical_first_action_key"]
            )
            self.assertEqual(canonical["similarity"], 1.0)
            for branch in result["branches"]:
                for record in branch["execution_records"]:
                    if record["kind"] != "action":
                        continue
                    self.assertEqual(
                        record["response_index"], record["request_index"] + 1
                    )
                    self.assertEqual(record["trace_count"], int(record["ok"]))


if __name__ == "__main__":
    unittest.main()
