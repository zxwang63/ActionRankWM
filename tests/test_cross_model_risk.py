from __future__ import annotations

import unittest

from actionrankwm.metrics.disagreement import (
    disagreement_features,
    expected_tie_aware_routing,
    normalized_value_disagreement,
    pairwise_order_disagreement,
    simulated_top,
    top_set_jaccard_disagreement,
)


class CrossModelRiskTests(unittest.TestCase):
    def test_pairwise_disagreement_is_tie_aware_and_symmetric(self) -> None:
        strict = {"a": 1.0, "b": 0.0, "c": -1.0}
        reverse = {"a": -1.0, "b": 0.0, "c": 1.0}
        tied = {"a": 0.0, "b": 0.0, "c": 0.0}
        self.assertEqual(pairwise_order_disagreement(strict, strict), 0.0)
        self.assertEqual(pairwise_order_disagreement(strict, reverse), 1.0)
        self.assertEqual(pairwise_order_disagreement(strict, tied), 0.5)
        self.assertEqual(
            pairwise_order_disagreement(strict, reverse),
            pairwise_order_disagreement(reverse, strict),
        )

    def test_top_and_value_disagreement(self) -> None:
        left = {"a": 1.0, "b": 1.0, "c": 0.0}
        right = {"a": 1.0, "b": 0.0, "c": 0.0}
        self.assertEqual(simulated_top(left), ("a", "b"))
        self.assertEqual(top_set_jaccard_disagreement(left, right), 0.5)
        self.assertGreater(normalized_value_disagreement(left, right), 0.0)
        features = disagreement_features(
            left, right, left_invalid=("c",), right_invalid=()
        )
        self.assertEqual(features["top_set_exact_disagreement"], 1.0)
        self.assertEqual(features["invalid_set_jaccard_disagreement"], 1.0)

    def test_tie_aware_routing_averages_boundary(self) -> None:
        rows = [
            {
                "unsafe": index < 2,
                "selected_real_terminal_success": 0.0,
                "best_real_terminal_success": 1.0,
            }
            for index in range(4)
        ]
        result = expected_tie_aware_routing(
            rows,
            [1.0, 1.0, 1.0, 0.0],
            budget_fraction=0.5,
            harmful_key="unsafe",
        )
        self.assertEqual(result["expected_routed_task_count"], 2.0)
        self.assertAlmostEqual(result["expected_remaining_harmful_count"], 2.0 / 3.0)
        self.assertAlmostEqual(result["expected_harmful_reduction"], 2.0 / 3.0)
        self.assertAlmostEqual(result["expected_terminal_success_gain"], 0.5)

    def test_invalid_inputs_fail(self) -> None:
        with self.assertRaises(ValueError):
            pairwise_order_disagreement({"a": 1.0}, {"a": 1.0})
        with self.assertRaises(ValueError):
            disagreement_features(
                {"a": 1.0, "b": 0.0},
                {"a": 1.0, "b": 0.0},
                left_invalid=("missing",),
            )


if __name__ == "__main__":
    unittest.main()
