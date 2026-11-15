from __future__ import annotations

import unittest

from actionrankwm.single_branch_audit import analyze_task, summarize_rows


def _candidate(action_id: str, real: float, simulated: float, token_f1: float = 1.0):
    return {
        "action_id": action_id,
        "real_value": real,
        "simulated_value": simulated,
        "token_f1": token_f1,
        "parse_ok": True,
        "prediction_error": None,
    }


def _decision(*, best=None, harmful=None, high_fidelity=True, tied=True):
    best = best or ["best"]
    harmful = harmful or ["bad"]
    return {
        "task_id": "task-1",
        "group_id": "group-1",
        "candidate_count": 2,
        "real_best_action_ids": best,
        "simulated_top_contains_harmful": bool(harmful),
        "simulated_top_contains_high_fidelity_harmful": bool(harmful)
        and high_fidelity,
        "simulated_top_harmful_action_ids": harmful,
        "simulated_top_is_tied": tied,
    }


class SingleBranchAuditTests(unittest.TestCase):
    def test_exact_best_branch_can_miss_harmful_overprediction(self) -> None:
        row = analyze_task(
            model="model",
            mechanism="mechanism",
            decision=_decision(),
            candidates=[_candidate("best", 1.0, 1.0), _candidate("bad", 0.5, 1.0)],
        )
        self.assertTrue(row["unsafe_despite_exact_real_best"])
        self.assertTrue(row["high_fidelity_unsafe_despite_exact_real_best"])
        summary = summarize_rows([row])
        self.assertEqual(summary["unsafe_despite_exact_real_best_count"], 1)
        self.assertEqual(
            summary[
                "unsafe_despite_exact_real_best_fraction_of_corresponding_unsafe"
            ],
            1.0,
        )

    def test_all_tied_real_best_branches_must_pass(self) -> None:
        decision = _decision(best=["best-a", "best-b"])
        decision["candidate_count"] = 3
        row = analyze_task(
            model="model",
            mechanism="mechanism",
            decision=decision,
            candidates=[
                _candidate("best-a", 1.0, 1.0),
                _candidate("best-b", 1.0, 0.9),
                _candidate("bad", 0.5, 1.0),
            ],
        )
        self.assertFalse(row["all_real_best_value_exact"])
        self.assertFalse(row["unsafe_despite_exact_real_best"])

    def test_exact_branch_identity_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "overprediction identity"):
            analyze_task(
                model="model",
                mechanism="mechanism",
                decision=_decision(),
                candidates=[
                    _candidate("best", 1.0, 1.0),
                    _candidate("bad", 0.5, 0.9),
                ],
            )

    def test_exact_high_fidelity_best_is_stricter(self) -> None:
        row = analyze_task(
            model="model",
            mechanism="mechanism",
            decision=_decision(),
            candidates=[
                _candidate("best", 1.0, 1.0, token_f1=0.7),
                _candidate("bad", 0.5, 1.0),
            ],
        )
        self.assertTrue(row["all_real_best_value_exact"])
        self.assertFalse(row["all_real_best_value_exact_and_high_fidelity"])


if __name__ == "__main__":
    unittest.main()
