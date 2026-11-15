from __future__ import annotations

import unittest

from actionrankwm.attribution import (
    decompose_harmful_top_pairs,
    differing_leaf_paths,
    restore_paths,
    summarize_value_vectors,
)


class AttributionTests(unittest.TestCase):
    def test_restore_paths_adds_replaces_and_removes_real_fields(self) -> None:
        real = {
            "ok": True,
            "state_patch": {"rows": [{"enabled": False, "name": "real"}]},
        }
        predicted = {
            "ok": False,
            "extra": "remove",
            "state_patch": {"rows": [{"enabled": True}]},
        }

        restored = restore_paths(
            predicted,
            real,
            ("ok", "extra", "state_patch.rows.0.enabled", "state_patch.rows.0.name"),
        )

        self.assertEqual(restored, real)
        self.assertNotEqual(predicted, restored)
        self.assertEqual(
            differing_leaf_paths(predicted, real),
            ["extra", "ok", "state_patch.rows.0.enabled", "state_patch.rows.0.name"],
        )

    def test_value_vector_summary_keeps_harmful_tied_top_member(self) -> None:
        summary = summarize_value_vectors(
            "task-a",
            {"oracle": 1.0, "harmful": 0.5, "other": 0.0},
            {"oracle": 0.4, "harmful": 0.4, "other": 0.1},
            harmful_regret_threshold=0.2,
        )

        self.assertEqual(summary["simulated_top_action_ids"], ["harmful", "oracle"])
        self.assertTrue(summary["contains_harmful_top"])
        self.assertEqual(summary["harmful_top_action_ids"], ["harmful"])
        self.assertEqual(summary["worst_top_regret"], 0.5)
        self.assertFalse(summary["top_set_subset_of_real_best"])

    def test_pair_decomposition_separates_under_and_over_prediction(self) -> None:
        decision = {
            "task_id": "task-b",
            "real_best_action_ids": ["best"],
            "simulated_top_harmful_action_ids": ["over", "under", "both"],
            "simulated_top_high_fidelity_harmful_action_ids": ["under"],
        }
        scored = {
            "best": {"real_value": 1.0, "simulated_value": 0.5},
            "over": {"real_value": 0.0, "simulated_value": 1.0},
            "under": {"real_value": 0.5, "simulated_value": 0.5},
            "both": {"real_value": 0.2, "simulated_value": 0.9},
        }

        rows = decompose_harmful_top_pairs(decision, scored)
        by_id = {row["harmful_action_id"]: row for row in rows}

        self.assertEqual(by_id["over"]["category"], "both")
        self.assertEqual(by_id["under"]["category"], "real_best_underprediction_only")
        self.assertEqual(by_id["both"]["category"], "both")
        self.assertTrue(by_id["under"]["high_fidelity_harmful"])
        self.assertTrue(all(abs(row["algebraic_residual"]) < 1e-12 for row in rows))


if __name__ == "__main__":
    unittest.main()
