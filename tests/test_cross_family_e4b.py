from __future__ import annotations

import unittest

from actionrankwm.cross_family_e4b import (
    clustered_unsafe_interval,
    family_positive,
    panel_decision,
)


class CrossFamilyE4bTests(unittest.TestCase):
    def test_clustered_interval_resamples_twenty_latent_worlds(self) -> None:
        unsafe = {}
        worlds = {}
        for world in range(10, 30):
            for mechanism in range(6):
                task_id = f"task_{mechanism}_{world:04d}"
                worlds[task_id] = world
                unsafe[task_id] = world < 15
        result = clustered_unsafe_interval(unsafe, worlds, resamples=200, seed=3)
        self.assertEqual(result["clusters"], 20)
        self.assertEqual(result["tasks_per_cluster"], 6)
        self.assertAlmostEqual(result["point"], 0.25)
        self.assertGreaterEqual(result["low"], 0.0)
        self.assertLessEqual(result["high"], 1.0)

    def test_family_gate_requires_every_check(self) -> None:
        passing = family_positive(
            integrity_passed=True,
            unsafe_count=10,
            contributing_mechanism_count=3,
            interval_low=0.001,
        )
        self.assertTrue(passing["positive"])
        tied_zero = family_positive(
            integrity_passed=True,
            unsafe_count=10,
            contributing_mechanism_count=3,
            interval_low=0.0,
        )
        self.assertFalse(tied_zero["positive"])

    def test_panel_requires_two_of_three_families(self) -> None:
        supported = panel_decision(
            {
                "meta": {"positive": True},
                "google": {"positive": True},
                "allenai": {"positive": False},
            }
        )
        self.assertTrue(supported["supported"])
        not_supported = panel_decision(
            {
                "meta": {"positive": True},
                "google": {"positive": False},
                "allenai": {"positive": False},
            }
        )
        self.assertFalse(not_supported["supported"])
        self.assertEqual(
            not_supported["status"], "CROSS_FAMILY_E4B_FAILURE_NOT_SUPPORTED"
        )


if __name__ == "__main__":
    unittest.main()
