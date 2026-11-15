from __future__ import annotations

import unittest

from actionrankwm.field_agnostic import (
    RepairAtom,
    classify_value_change,
    exact_minimal_search,
    schema_repair_atoms,
)


class FieldAgnosticRepairTests(unittest.TestCase):
    def test_schema_atoms_use_fixed_fields_and_namespace_subtrees(self) -> None:
        predicted = {
            "ok": True,
            "result": "same",
            "ignored_extra": "not an atom",
            "state_patch": {
                "domain": "toolsandbox",
                "operation": "none",
                "namespaces": {"setting": [{"wifi": True}]},
            },
        }
        real = {
            "ok": False,
            "result": "same",
            "error_type": "permission",
            "state_patch": {
                "domain": "toolsandbox",
                "operation": "replace_namespaces",
                "namespaces": {
                    "contact": [],
                    "setting": [{"wifi": False, "cellular": True}],
                },
            },
        }

        atoms = schema_repair_atoms("c2", predicted, real)

        self.assertEqual(
            [atom.path for atom in atoms],
            [
                "ok",
                "error_type",
                "state_patch.operation",
                "state_patch.namespaces.contact",
                "state_patch.namespaces.setting",
            ],
        )
        self.assertNotIn("ignored_extra", [atom.path for atom in atoms])

    def test_exact_search_saves_all_minimal_pairs_and_stops(self) -> None:
        atoms = [
            RepairAtom("c0", "ok"),
            RepairAtom("c1", "result"),
            RepairAtom("c2", "error"),
        ]

        def evaluate(subset: tuple[RepairAtom, ...]) -> dict[str, object]:
            paths = {atom.path for atom in subset}
            recovered = "ok" in paths and bool(paths & {"result", "error"})
            return {
                "summary": {"contains_harmful_top": not recovered},
                "replay_failures": [],
                "simulated_values": {"c0": float(recovered)},
            }

        first = exact_minimal_search(atoms, evaluate)
        second = exact_minimal_search(list(reversed(atoms)), evaluate)

        self.assertEqual(first["minimal_cardinality"], 2)
        self.assertEqual(first["attempted_by_cardinality"], {"1": 3, "2": 3})
        self.assertEqual(len(first["successful_subsets"]), 2)
        self.assertEqual(first["attempt_digest_sha256"], second["attempt_digest_sha256"])

    def test_value_change_classification_is_bidirectional(self) -> None:
        baseline_summary = {
            "harmful_top_action_ids": ["harmful"],
            "real_best_action_ids": ["best"],
        }
        baseline = {"harmful": 0.8, "best": 0.2}

        self.assertEqual(
            classify_value_change(
                baseline_summary,
                baseline,
                {"harmful": 0.4, "best": 0.7},
            ),
            "harmful_overprediction_and_real_best_underprediction",
        )
        self.assertEqual(
            classify_value_change(
                baseline_summary,
                baseline,
                {"harmful": 0.8, "best": 0.7},
            ),
            "real_best_underprediction",
        )


if __name__ == "__main__":
    unittest.main()
