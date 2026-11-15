from __future__ import annotations

import unittest

from actionrankwm.dccp_evaluation import (
    engineering_failure,
    evaluate_panel,
    paired_task_bootstrap,
)


def row(unsafe: int, pairwise: float, top1: float, failure: float) -> dict[str, float | int]:
    return {
        "high_fidelity_unsafe_top_set_count": unsafe,
        "pairwise_agreement_mean": pairwise,
        "top1_preservation_rate": top1,
        "engineering_failure_rate": failure,
    }


class DCCPEvaluationTests(unittest.TestCase):
    def test_engineering_failure_is_conservative_union(self) -> None:
        self.assertFalse(
            engineering_failure(
                {"parse_ok": True, "patch_repair": None, "prediction_error": None}
            )
        )
        self.assertTrue(
            engineering_failure(
                {"parse_ok": True, "patch_repair": {"kind": "repair"}, "prediction_error": None}
            )
        )
        self.assertTrue(
            engineering_failure(
                {"parse_ok": False, "patch_repair": None, "prediction_error": None}
            )
        )
        self.assertTrue(
            engineering_failure(
                {"parse_ok": True, "patch_repair": None, "prediction_error": "simulation"}
            )
        )

    def test_dccp_advances_only_when_all_strict_checks_pass(self) -> None:
        result = evaluate_panel(
            {
                "base": row(10, 0.50, 0.80, 0.02),
                "uniform_sft": row(8, 0.54, 0.78, 0.03),
                "value_neutral_preference": row(7, 0.55, 0.79, 0.04),
                "dccp": row(6, 0.60, 0.80, 0.03),
            }
        )
        self.assertEqual(result["status"], "DCCP_PILOT_ADVANCE_TO_NEW_FULL_PROTOCOL")
        self.assertTrue(result["dccp_advances"])
        self.assertFalse(result["e4c_authorized"])

    def test_control_tie_stops_dccp_and_names_control_candidate(self) -> None:
        result = evaluate_panel(
            {
                "base": row(10, 0.50, 0.80, 0.02),
                "uniform_sft": row(6, 0.60, 0.80, 0.03),
                "value_neutral_preference": row(9, 0.52, 0.80, 0.03),
                "dccp": row(6, 0.60, 0.80, 0.03),
            }
        )
        self.assertEqual(result["status"], "DCCP_METHOD_STOP_CONTROL_CANDIDATE")
        self.assertFalse(result["dccp_advances"])
        self.assertEqual(result["qualifying_control_candidates"], ["uniform_sft"])
        self.assertFalse(result["dccp_checks"]["strictly_fewer_unsafe_than_both_controls"])

    def test_zero_base_unsafe_cannot_pass_relative_reduction(self) -> None:
        result = evaluate_panel(
            {
                "base": row(0, 0.50, 0.80, 0.02),
                "uniform_sft": row(0, 0.55, 0.80, 0.02),
                "value_neutral_preference": row(0, 0.55, 0.80, 0.02),
                "dccp": row(0, 0.60, 0.80, 0.02),
            }
        )
        self.assertEqual(result["status"], "DCCP_POSTTRAINING_PILOT_STOP")
        self.assertIsNone(result["comparisons_to_base"]["dccp"]["unsafe_reduction"])

    def test_paired_bootstrap_is_deterministic_and_paired(self) -> None:
        base = {
            f"t{i:02d}": {
                "high_fidelity_unsafe_rate": float(i % 2),
                "pairwise_agreement": 0.5,
                "top1_preservation": 0.8,
                "engineering_failure_rate": 0.1,
            }
            for i in range(30)
        }
        condition = {
            task_id: {
                "high_fidelity_unsafe_rate": values["high_fidelity_unsafe_rate"],
                "pairwise_agreement": values["pairwise_agreement"] + 0.1,
                "top1_preservation": values["top1_preservation"],
                "engineering_failure_rate": values["engineering_failure_rate"] - 0.05,
            }
            for task_id, values in base.items()
        }
        first = paired_task_bootstrap(base, condition, resamples=100, seed=7)
        second = paired_task_bootstrap(base, condition, resamples=100, seed=7)
        self.assertEqual(first, second)
        self.assertAlmostEqual(first["intervals"]["pairwise_agreement"]["low"], 0.1)
        self.assertAlmostEqual(first["intervals"]["engineering_failure_rate"]["high"], -0.05)


if __name__ == "__main__":
    unittest.main()
