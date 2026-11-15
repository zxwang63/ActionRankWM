from __future__ import annotations

import unittest

from actionrankwm.mr_sft_confirmation import (
    CONFIRMATION_FIELDS,
    CONFIRMATION_MODELS,
    CONFIRMATION_MODEL_KEYS,
    CONFIRMATION_PREFIXES,
    CONFIRMATION_SEEDS,
    aggregate_seed_panel,
    confirmation_served_model_names,
    evaluate_scale,
    joint_decision,
    retention_launch_screen,
)


def _rows(*, improved: bool) -> tuple[dict, dict, dict]:
    base = {}
    seeds = {seed: {} for seed in CONFIRMATION_SEEDS}
    worlds = {}
    for world in range(100, 150):
        for mechanism in range(6):
            task_id = f"task_{mechanism}_{world}"
            worlds[task_id] = world
            base[task_id] = {
                "high_fidelity_unsafe_rate": 1.0,
                "unsafe_rate": 1.0,
                "pairwise_agreement": 0.70,
                "top1_preservation": 0.70,
                "engineering_failure_rate": 0.10,
                "selected_regret": 0.30,
                "top_set_expected_regret": 0.35,
                "top_set_worst_regret": 0.45,
                "token_f1": 0.60,
                "structured_f1": 0.50,
                "critical_field_accuracy": 0.55,
            }
            for index, seed in enumerate(CONFIRMATION_SEEDS):
                delta = 0.01 * index
                seeds[seed][task_id] = {
                    "high_fidelity_unsafe_rate": 0.0 if improved else 1.0,
                    "unsafe_rate": 0.2 if improved else 1.0,
                    "pairwise_agreement": 0.80 + delta if improved else 0.70,
                    "top1_preservation": 0.75 if improved else 0.60,
                    "engineering_failure_rate": 0.05 if improved else 0.20,
                    "selected_regret": 0.20 if improved else 0.30,
                    "top_set_expected_regret": 0.20 if improved else 0.35,
                    "top_set_worst_regret": 0.30 if improved else 0.45,
                    "token_f1": 0.80 if improved else 0.60,
                    "structured_f1": 0.70 if improved else 0.50,
                    "critical_field_accuracy": 0.75 if improved else 0.55,
                }
    return base, seeds, worlds


class TestMrSftConfirmation(unittest.TestCase):
    def test_amended_condition_inventory_is_exact_and_unique(self) -> None:
        self.assertEqual(tuple(CONFIRMATION_MODELS), CONFIRMATION_MODEL_KEYS)
        self.assertEqual(set(CONFIRMATION_PREFIXES), set(CONFIRMATION_MODEL_KEYS))
        served = confirmation_served_model_names()
        self.assertEqual(set(served), set(CONFIRMATION_MODEL_KEYS))
        names = [
            name
            for model_names in served.values()
            for name in model_names.values()
        ]
        self.assertEqual(len(names), 12)
        self.assertEqual(len(set(names)), 12)
        self.assertTrue(
            all(
                set(row) == {"base", *map(str, CONFIRMATION_SEEDS)}
                for row in served.values()
            )
        )

    def test_retention_launch_screen_is_fail_closed(self) -> None:
        result = retention_launch_screen(
            {
                "status": "DCCP_E4B_RETENTION_DESCRIPTIVE_COMPLETE",
                "primary_validation": {
                    "status": "DCCP_METHOD_STOP_CONTROL_CANDIDATE"
                },
                "primary_status_unchanged": True,
                "can_rescue_primary": False,
                "e4c_authorized": False,
                "multi_seed_authorized": False,
                "comparisons_to_base": {
                    "uniform_sft": {
                        "retention_screen_passed": True,
                        "directionally_nonworse_on_unsafe_and_pairwise": True,
                    }
                },
            }
        )
        self.assertTrue(result["passed"])
        self.assertFalse(result["e4c_authorized_by_this_screen_alone"])

        failed = retention_launch_screen(
            {
                "status": "DCCP_E4B_RETENTION_DESCRIPTIVE_COMPLETE",
                "primary_validation": {
                    "status": "DCCP_METHOD_STOP_CONTROL_CANDIDATE"
                },
                "primary_status_unchanged": True,
                "can_rescue_primary": False,
                "e4c_authorized": False,
                "multi_seed_authorized": False,
                "comparisons_to_base": {
                    "uniform_sft": {
                        "retention_screen_passed": True,
                        "directionally_nonworse_on_unsafe_and_pairwise": False,
                    }
                },
            }
        )
        self.assertFalse(failed["passed"])

    def test_positive_panel_passes_all_checks_deterministically(self) -> None:
        base, seeds, worlds = _rows(improved=True)
        first = aggregate_seed_panel(base, seeds, worlds, resamples=200)
        second = aggregate_seed_panel(base, seeds, worlds, resamples=200)
        self.assertEqual(first, second)
        self.assertEqual(
            set(first["base"]),
            set(CONFIRMATION_FIELDS),
        )
        result = evaluate_scale(first)
        self.assertTrue(result["confirmed"])
        self.assertTrue(all(result["checks"].values()))

    def test_negative_panel_and_joint_status_are_retained(self) -> None:
        base, seeds, worlds = _rows(improved=False)
        panel = aggregate_seed_panel(base, seeds, worlds, resamples=100)
        negative = evaluate_scale(panel)
        self.assertFalse(negative["confirmed"])

        base2, seeds2, worlds2 = _rows(improved=True)
        positive = evaluate_scale(
            aggregate_seed_panel(base2, seeds2, worlds2, resamples=100)
        )
        mixed = joint_decision(
            {
                "qwen3.5-4b": positive,
                "qwen3.5-9b": negative,
                "llama3.1-8b": positive,
            }
        )
        self.assertEqual(mixed["status"], "MR_SFT_CONFIRMED_CROSS_FAMILY")
        self.assertTrue(mixed["cross_family_transfer_supported"])
        neither = joint_decision(
            {
                "qwen3.5-4b": negative,
                "qwen3.5-9b": negative,
                "llama3.1-8b": negative,
            }
        )
        self.assertEqual(neither["status"], "MR_SFT_CONFIRMATION_NOT_SUPPORTED")

    def test_amended_joint_statuses_are_exhaustive(self) -> None:
        base_bad, seeds_bad, worlds_bad = _rows(improved=False)
        negative = evaluate_scale(
            aggregate_seed_panel(base_bad, seeds_bad, worlds_bad, resamples=20)
        )
        base_good, seeds_good, worlds_good = _rows(improved=True)
        positive = evaluate_scale(
            aggregate_seed_panel(base_good, seeds_good, worlds_good, resamples=20)
        )

        all_three = joint_decision(
            {
                "qwen3.5-4b": positive,
                "qwen3.5-9b": positive,
                "llama3.1-8b": positive,
            }
        )
        self.assertEqual(
            all_three["status"],
            "MR_SFT_CONFIRMED_QWEN_SCALES_AND_META_FAMILY",
        )
        self.assertTrue(all_three["qwen_cross_scale_supported"])
        self.assertTrue(all_three["cross_family_transfer_supported"])

        qwen_only = joint_decision(
            {
                "qwen3.5-4b": positive,
                "qwen3.5-9b": positive,
                "llama3.1-8b": negative,
            }
        )
        self.assertEqual(
            qwen_only["status"], "MR_SFT_CONFIRMED_QWEN_BOTH_SCALES_ONLY"
        )
        self.assertTrue(qwen_only["qwen_cross_scale_supported"])
        self.assertFalse(qwen_only["cross_family_transfer_supported"])

        model_specific = joint_decision(
            {
                "qwen3.5-4b": negative,
                "qwen3.5-9b": positive,
                "llama3.1-8b": negative,
            }
        )
        self.assertEqual(
            model_specific["status"], "MR_SFT_CONFIRMED_MODEL_SPECIFIC"
        )
        self.assertFalse(model_specific["qwen_cross_scale_supported"])
        self.assertFalse(model_specific["cross_family_transfer_supported"])

        invalid = {
            **negative,
            "checks": {**negative["checks"], "panel_integrity_passed": False},
        }
        engineering_invalid = joint_decision(
            {
                "qwen3.5-4b": positive,
                "qwen3.5-9b": positive,
                "llama3.1-8b": invalid,
            }
        )
        self.assertEqual(
            engineering_invalid["status"],
            "MR_SFT_CONFIRMATION_ENGINEERING_INVALID",
        )

        with self.assertRaises(ValueError):
            joint_decision(
                {"qwen3.5-4b": positive, "qwen3.5-9b": positive}
            )

    def test_panel_requires_exact_three_seed_world_inventory(self) -> None:
        base, seeds, worlds = _rows(improved=True)
        seeds.pop(CONFIRMATION_SEEDS[-1])
        with self.assertRaises(ValueError):
            aggregate_seed_panel(base, seeds, worlds, resamples=10)


if __name__ == "__main__":
    unittest.main()
