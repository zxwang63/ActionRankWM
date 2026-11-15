import unittest

from actionrankwm.failure_stratification import (
    candidate_failure_flags,
    stratify_model_panel,
)


def candidate(task, action, *, parse=True, repair=None, error=None):
    return {
        "task_id": task,
        "action_id": action,
        "parse_ok": parse,
        "patch_repair": repair,
        "prediction_error": error,
    }


def decision(task, *, real_best, harmful=(), high_fidelity=()):
    return {
        "task_id": task,
        "candidate_count": 3,
        "template_id": "toolsandbox.synthetic",
        "real_best_action_ids": list(real_best),
        "simulated_top_harmful_action_ids": list(harmful),
        "simulated_top_high_fidelity_harmful_action_ids": list(high_fidelity),
        "simulated_top_contains_harmful": bool(harmful),
        "simulated_top_contains_high_fidelity_harmful": bool(high_fidelity),
    }


class FailureStratificationTests(unittest.TestCase):
    def test_candidate_failure_union_is_strict(self):
        self.assertFalse(
            candidate_failure_flags(candidate("t", "a"))["engineering_failure"]
        )
        for row in (
            candidate("t", "a", parse=False),
            candidate("t", "a", repair="allowed"),
            candidate("t", "a", error="patch_error"),
        ):
            self.assertTrue(candidate_failure_flags(row)["engineering_failure"])

    def test_clean_witness_does_not_require_unrelated_candidate_clean(self):
        result = stratify_model_panel(
            model_key="m",
            decisions=[
                decision(
                    "t1",
                    real_best=("best",),
                    harmful=("harm",),
                    high_fidelity=("harm",),
                ),
                decision(
                    "t2",
                    real_best=("best",),
                    harmful=("harm",),
                    high_fidelity=("harm",),
                ),
            ],
            candidates=[
                candidate("t1", "best"),
                candidate("t1", "harm"),
                candidate("t1", "other", error="unrelated_failure"),
                candidate("t2", "best", error="competitor_failure"),
                candidate("t2", "harm"),
                candidate("t2", "other"),
            ],
        )
        unsafe = result["summary"]["events"]["unsafe"]
        high_fidelity = result["summary"]["events"]["high_fidelity_unsafe"]
        self.assertEqual(unsafe["event_count"], 2)
        self.assertEqual(unsafe["counts"]["all_candidates_clean"], 0)
        self.assertEqual(unsafe["counts"]["clean_semantic_witness"], 1)
        self.assertEqual(high_fidelity["counts"]["clean_semantic_witness"], 1)
        self.assertEqual(unsafe["counts"]["any_real_best_failure"], 1)

    def test_repair_on_harmful_witness_fails_relevant_clean_check(self):
        result = stratify_model_panel(
            model_key="m",
            decisions=[
                decision(
                    "t",
                    real_best=("best",),
                    harmful=("harm",),
                    high_fidelity=("harm",),
                )
            ],
            candidates=[
                candidate("t", "best"),
                candidate("t", "harm", repair="allowed"),
                candidate("t", "other"),
            ],
        )
        summary = result["summary"]["events"]["high_fidelity_unsafe"]
        self.assertEqual(summary["counts"]["all_real_best_clean"], 1)
        self.assertEqual(summary["counts"]["all_relevant_candidates_clean"], 0)
        self.assertEqual(summary["counts"]["clean_semantic_witness"], 0)

    def test_duplicate_candidate_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Duplicate candidate"):
            stratify_model_panel(
                model_key="m",
                decisions=[decision("t", real_best=("a",))],
                candidates=[candidate("t", "a"), candidate("t", "a")],
            )

    def test_missing_engineering_field_is_rejected(self):
        row = candidate("t", "a")
        del row["patch_repair"]
        with self.assertRaisesRegex(ValueError, "required fields"):
            candidate_failure_flags(row)

    def test_pooled_component_counts_retain_zero_denominators(self):
        from actionrankwm.failure_stratification import stratify_panels

        panel = stratify_panels(
            {
                "m": (
                    [decision("t", real_best=("best",))],
                    [
                        candidate("t", "best"),
                        candidate("t", "harm", parse=False),
                        candidate("t", "other", repair="allowed"),
                    ],
                )
            }
        )
        components = panel["pooled"]["failure_components"]
        self.assertEqual(components["all_candidates"]["candidate_count"], 3)
        self.assertEqual(
            components["all_candidates"]["counts"]["engineering_failure"], 2
        )
        self.assertEqual(
            components["real_best_candidates"]["counts"]["engineering_failure"],
            0,
        )
        self.assertEqual(components["harmful_top_candidates"]["candidate_count"], 0)
        self.assertIsNone(
            components["harmful_top_candidates"]["rates"]["parse_failure"]
        )


if __name__ == "__main__":
    unittest.main()
