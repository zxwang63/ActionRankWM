from __future__ import annotations

import unittest

from actionrankwm.threshold_sensitivity import analyze_model, analyze_panel


def _fixture(token_f1: float = 0.8):
    decisions = [
        {
            "task_id": "task_a",
            "template_id": "mechanism_a",
            "world_id": "world_1",
            "best_real_value": 1.0,
            "simulated_top_action_ids": ["good", "bad"],
        },
        {
            "task_id": "task_b",
            "template_id": "mechanism_b",
            "world_id": "world_1",
            "best_real_value": 0.9,
            "simulated_top_action_ids": ["good"],
        },
    ]
    candidates = [
        {"task_id": "task_a", "action_id": "good", "real_value": 1.0, "token_f1": 1.0},
        {"task_id": "task_a", "action_id": "bad", "real_value": 0.7, "token_f1": token_f1},
        {"task_id": "task_b", "action_id": "good", "real_value": 0.9, "token_f1": 0.5},
        {"task_id": "task_b", "action_id": "bad", "real_value": 0.0, "token_f1": 1.0},
    ]
    return decisions, candidates


class ThresholdSensitivityTests(unittest.TestCase):
    def test_grid_uses_every_top_member_and_is_monotone(self) -> None:
        decisions, candidates = _fixture()
        result = analyze_model(decisions, candidates)
        self.assertEqual(result["frozen_cell"]["event_count"], 1)
        self.assertEqual(result["frozen_cell"]["contributing_mechanisms"], ["mechanism_a"])
        by_key = {(row["delta"], row["gamma"]): row for row in result["grid"]}
        self.assertEqual(by_key[(0.30, 0.80)]["event_count"], 1)
        self.assertEqual(by_key[(0.30, 0.90)]["event_count"], 0)
        self.assertEqual(by_key[(0.40, 0.50)]["event_count"], 0)

    def test_leave_one_mechanism_out_has_the_correct_denominator(self) -> None:
        decisions, candidates = _fixture()
        result = analyze_model(decisions, candidates)
        rows = {
            row["excluded_mechanism"]: row
            for row in result["leave_one_mechanism_out_at_frozen_cell"]
        }
        self.assertEqual(rows["mechanism_a"]["task_count"], 1)
        self.assertEqual(rows["mechanism_a"]["event_count"], 0)
        self.assertEqual(rows["mechanism_b"]["event_count"], 1)

    def test_panel_rejects_changed_real_values(self) -> None:
        left = _fixture()
        right_decisions, right_candidates = _fixture()
        right_candidates = [dict(row) for row in right_candidates]
        right_candidates[1]["real_value"] = 0.69
        with self.assertRaisesRegex(ValueError, "identical task/real-value panel"):
            analyze_panel({"left": left, "right": (right_decisions, right_candidates)})


if __name__ == "__main__":
    unittest.main()
