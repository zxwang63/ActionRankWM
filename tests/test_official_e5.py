from __future__ import annotations

import importlib.util
import json
import unittest
import uuid

from actionrankwm.official_e5 import (
    OfficialE5PredictionError,
    PLACEHOLDER,
    _validate_observation,
    audit_label_blind_messages,
)
from actionrankwm.official_scenarios import OfficialAction


class OfficialE5PureTests(unittest.TestCase):
    def test_trace_is_reconstructed_from_predicted_result(self) -> None:
        action = OfficialAction(
            key="hidden-action",
            milestone_index=0,
            tool="search_contacts",
            arguments={"phone_number": "+15555550123"},
            source="test",
        )
        observation = {
            "ok": True,
            "result": [{"person_id": "predicted"}],
            "error_type": None,
            "error": None,
            "state_patch": {
                "domain": "toolsandbox",
                "operation": "none",
                "namespaces": {},
            },
        }
        _, result, error, traces = _validate_observation(
            observation, action, uuid.UUID(int=1)
        )
        self.assertIsNone(error)
        self.assertEqual(result, observation["result"])
        self.assertEqual(
            json.loads(traces[0]),
            {
                "tool_name": action.tool,
                "arguments": action.arguments,
                "result": observation["result"],
            },
        )

    def test_generated_id_requires_exact_placeholder(self) -> None:
        action = OfficialAction(
            key="hidden-action",
            milestone_index=0,
            tool="send_message_with_phone_number",
            arguments={"phone_number": "+15555550123", "content": "hello"},
            source="test",
        )
        base = {
            "ok": True,
            "result": PLACEHOLDER,
            "error_type": None,
            "error": None,
            "state_patch": {
                "domain": "toolsandbox",
                "operation": "none",
                "namespaces": {},
            },
        }
        _, result, _, traces = _validate_observation(base, action, uuid.UUID(int=1))
        self.assertEqual(result, str(uuid.UUID(int=1)))
        self.assertEqual(json.loads(traces[0])["result"], str(uuid.UUID(int=1)))
        invalid = dict(base)
        invalid["result"] = "invented-id"
        with self.assertRaises(OfficialE5PredictionError):
            _validate_observation(invalid, action, uuid.UUID(int=1))

    def test_label_blind_audit_rejects_hidden_identifier(self) -> None:
        user = {
            "domain": "toolsandbox",
            "current_state": {"namespaces": {}},
            "action": {"tool": "search_contacts", "arguments": {}},
            "tool_docs": {"search_contacts": {}},
            "benchmark_context": {"current_timestamp": 1.0},
        }
        messages = [
            {"role": "system", "content": "transition"},
            {"role": "user", "content": json.dumps(user)},
        ]
        audit_label_blind_messages(
            messages,
            task_id="private-task",
            action_id="private-action",
            scenario_names=["private-scenario"],
            semantic_groups=["private-group"],
            action_keys=["private-key"],
        )
        user["scenario_name"] = "private-scenario"
        messages[1]["content"] = json.dumps(user)
        with self.assertRaises(ValueError):
            audit_label_blind_messages(
                messages,
                task_id="private-task",
                action_id="private-action",
                scenario_names=["private-scenario"],
                semantic_groups=["private-group"],
                action_keys=["private-key"],
            )


@unittest.skipUnless(
    importlib.util.find_spec("tool_sandbox") is not None,
    "optional Apple ToolSandbox runtime is unavailable",
)
class OfficialE5RuntimeTests(unittest.TestCase):
    def test_truth_observation_roundtrip_matches_locked_v11(self) -> None:
        from actionrankwm.official_e5 import (
            comparable_result,
            execute_official_e5_branch,
        )
        from actionrankwm.official_scenarios import (
            extract_official_plan,
            load_unaugmented_scenarios,
        )
        from actionrankwm.official_scenarios_v11 import evaluate_official_candidate_v11

        name = "send_message_with_contact_content_cellular_off"
        family, scenario = load_unaugmented_scenarios()[name]
        plan = extract_official_plan(name, family, scenario)
        for action in plan.actions:
            actual = execute_official_e5_branch(scenario, plan, action.key)
            locked = evaluate_official_candidate_v11(scenario, plan, action.key)
            self.assertEqual(comparable_result(actual), comparable_result(locked))
            injected = execute_official_e5_branch(
                scenario,
                plan,
                action.key,
                injected_observation=actual["first_observation"],
            )
            self.assertEqual(comparable_result(actual), comparable_result(injected))
            self.assertEqual(actual["first_trace"], injected["first_trace"])


if __name__ == "__main__":
    unittest.main()
