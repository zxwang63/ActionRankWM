from __future__ import annotations

import unittest

from actionrankwm.dccp_retention import (
    RETENTION_FIELDS,
    clustered_paired_bootstrap,
    retention_comparison,
)


def summary(unsafe: int, pairwise: float, top1: float, failure: float) -> dict[str, float | int]:
    return {
        "high_fidelity_unsafe_top_set_count": unsafe,
        "pairwise_agreement_mean": pairwise,
        "top1_preservation_rate": top1,
        "engineering_failure_rate": failure,
    }


class DCCPRetentionTests(unittest.TestCase):
    def test_strict_directional_label_does_not_change_retention_screen(self) -> None:
        base = summary(20, 0.80, 0.70, 0.02)
        better = retention_comparison(base, summary(15, 0.82, 0.66, 0.07))
        self.assertTrue(better["retention_screen_passed"])
        self.assertTrue(better["directionally_better_on_unsafe_and_pairwise"])
        tied = retention_comparison(base, summary(20, 0.80, 0.70, 0.02))
        self.assertTrue(tied["directionally_nonworse_on_unsafe_and_pairwise"])
        self.assertFalse(tied["directionally_better_on_unsafe_and_pairwise"])

    def test_failure_or_top1_regression_fails_screen(self) -> None:
        base = summary(20, 0.80, 0.70, 0.02)
        high_failure = retention_comparison(base, summary(10, 0.90, 0.70, 0.071))
        low_top1 = retention_comparison(base, summary(10, 0.90, 0.649, 0.02))
        self.assertFalse(high_failure["retention_screen_passed"])
        self.assertFalse(low_top1["retention_screen_passed"])

    def test_clustered_bootstrap_preserves_twenty_worlds_and_pairing(self) -> None:
        base = {}
        condition = {}
        worlds = {}
        for world in range(10, 30):
            for mechanism in range(6):
                task_id = f"task_{mechanism}_{world:04d}"
                worlds[task_id] = world
                base[task_id] = {field: 0.2 for field in RETENTION_FIELDS}
                condition[task_id] = {
                    field: (0.1 if field == "high_fidelity_unsafe_rate" else 0.3)
                    for field in RETENTION_FIELDS
                }
        result = clustered_paired_bootstrap(
            base, condition, worlds, resamples=100, seed=7
        )
        self.assertEqual(result["clusters"], 20)
        self.assertEqual(result["tasks_per_cluster"], 6)
        self.assertAlmostEqual(
            result["point_deltas"]["high_fidelity_unsafe_rate"], -0.1
        )
        self.assertAlmostEqual(result["intervals"]["pairwise_agreement"]["low"], 0.1)

    def test_cluster_inventory_is_strict(self) -> None:
        rows = {
            f"t{i}": {field: 0.0 for field in RETENTION_FIELDS}
            for i in range(120)
        }
        worlds = {task_id: 10 + index // 6 for index, task_id in enumerate(rows)}
        worlds["t0"] = 99
        with self.assertRaisesRegex(ValueError, "worlds 10..29"):
            clustered_paired_bootstrap(rows, rows, worlds, resamples=2)


if __name__ == "__main__":
    unittest.main()
