from __future__ import annotations

import unittest

from actionrankwm.branches import ValueConfig, evaluate_real_branch, evaluate_simulated_branch
from actionrankwm.envs.base import make_environment
from actionrankwm.semantics import canonicalize_predicted_observation
from actionrankwm.tasks import generate_tasks
from actionrankwm.metrics.text import token_f1
from actionrankwm.types import Action


class TaskAndEnvironmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tasks = generate_tasks(["sqlite_mcp", "terminal_fs"], tasks_per_template=1, seed=7)
        cls.value_config = ValueConfig()

    def test_task_inventory_is_balanced_and_unique(self) -> None:
        self.assertEqual(len(self.tasks), 8)
        self.assertEqual(len({task.task_id for task in self.tasks}), 8)
        self.assertEqual(sum(task.domain == "sqlite_mcp" for task in self.tasks), 4)
        self.assertEqual(sum(task.domain == "terminal_fs" for task in self.tasks), 4)
        self.assertTrue(all(len(task.candidates) == 5 for task in self.tasks))
        self.assertTrue(all(task.candidates[0].role == "oracle" for task in self.tasks))
        self.assertTrue(all(any(action.role == "destructive" for action in task.candidates) for task in self.tasks))

    def test_rule_rich_templates_have_real_counterfactual_value_gaps(self) -> None:
        rule_tasks = generate_tasks(
            ["sqlite_mcp", "terminal_fs"],
            tasks_per_template=1,
            seed=17,
            template_ids=[
                "sqlite.rule.approval_guard",
                "sqlite.rule.version_trigger",
                "sqlite.rule.audit_suffix",
                "terminal.rule.json_validation",
                "terminal.rule.write_token",
                "terminal.rule.post_write_hook",
            ],
        )
        self.assertEqual(len(rule_tasks), 6)
        self.assertTrue(all(task.metadata["suite"] == "rule_rich" for task in rule_tasks))
        for task in rule_tasks:
            branches = [
                evaluate_real_branch(task, action, self.value_config)
                for action in task.candidates
            ]
            oracle = next(item for item in branches if item.action.role == "oracle")
            wrong_value = next(item for item in branches if item.action.role == "wrong_value")
            self.assertEqual(oracle.real_value, max(item.real_value for item in branches), task.task_id)
            self.assertGreater(oracle.real_value - wrong_value.real_value, 0.20, task.task_id)
            for branch in branches:
                simulated, _, _, error = evaluate_simulated_branch(
                    task, branch.action, branch.real_observation, self.value_config
                )
                self.assertIsNone(error, (task.task_id, branch.action.action_id, error))
                self.assertAlmostEqual(simulated, branch.real_value, places=10)

    def test_successful_side_effect_tasks_expose_high_fidelity_value_errors(self) -> None:
        tasks = generate_tasks(
            ["sqlite_mcp", "terminal_fs"],
            tasks_per_template=1,
            seed=19,
            template_ids=[
                "sqlite.rule.audit_suffix",
                "terminal.rule.post_write_hook",
            ],
        )
        self.assertEqual(len(tasks), 2)
        for task in tasks:
            action = next(item for item in task.candidates if item.role == "wrong_value")
            real = evaluate_real_branch(task, action, self.value_config)
            if task.domain == "sqlite_mcp":
                table = action.arguments["table"]
                source = next(
                    row
                    for row in task.initial_state["tables"][table]
                    if row["id"] == action.arguments["where"]["id"]
                )
                predicted_row = dict(source)
                predicted_row.update(action.arguments["set"])
                prediction = {
                    "ok": True,
                    "affected_rows": 1,
                    "rows": [predicted_row],
                    "state_patch": {
                        "domain": "sqlite_mcp",
                        "operation": "update",
                        "table": table,
                        "where": action.arguments["where"],
                        "rows": [predicted_row],
                    },
                }
            else:
                prediction = {
                    "ok": True,
                    "path": action.arguments["path"],
                    "bytes": len(action.arguments["content"].encode()),
                    "state_patch": {
                        "domain": "terminal_fs",
                        "operation": "write",
                        "path": action.arguments["path"],
                        "exists": True,
                        "content": action.arguments["content"],
                    },
                }
            simulated, _, _, error = evaluate_simulated_branch(
                task, action, prediction, self.value_config
            )
            self.assertIsNone(error)
            self.assertGreater(simulated - real.real_value, 1.0, task.task_id)
            self.assertGreater(token_f1(real.real_observation, prediction), 0.75, task.task_id)

    def test_mechanism_holdout_tasks_are_distinct_and_decision_relevant(self) -> None:
        template_ids = [
            "sqlite.rule.formula_recompute",
            "sqlite.rule.lookup_derive",
            "terminal.rule.json_canonicalization",
            "terminal.rule.write_rewrite",
        ]
        tasks = generate_tasks(
            ["sqlite_mcp", "terminal_fs"],
            tasks_per_template=3,
            seed=20260718,
            template_ids=template_ids,
            task_indices=[25, 30, 35],
        )
        self.assertEqual(len(tasks), 12)
        self.assertEqual(len({task.task_id for task in tasks}), 12)
        self.assertEqual(len({task.metadata["world_id"] for task in tasks}), 12)
        self.assertTrue(
            all(task.metadata["suite"] == "mechanism_holdout_v08" for task in tasks)
        )
        for task in tasks:
            branches = [
                evaluate_real_branch(task, action, self.value_config)
                for action in task.candidates
            ]
            oracle = next(item for item in branches if item.action.role == "oracle")
            wrong_value = next(
                item for item in branches if item.action.role == "wrong_value"
            )
            self.assertEqual(
                oracle.real_value, max(item.real_value for item in branches), task.task_id
            )
            self.assertGreater(
                oracle.real_value - wrong_value.real_value, 1.0, task.task_id
            )
            for branch in branches:
                simulated, _, _, error = evaluate_simulated_branch(
                    task, branch.action, branch.real_observation, self.value_config
                )
                self.assertIsNone(error, (task.task_id, branch.action.action_id, error))
                self.assertAlmostEqual(simulated, branch.real_value, places=10)

    def test_mechanism_holdout_naive_direct_effect_is_high_fidelity_and_unsafe(self) -> None:
        template_ids = [
            "sqlite.rule.formula_recompute",
            "sqlite.rule.lookup_derive",
            "terminal.rule.json_canonicalization",
            "terminal.rule.write_rewrite",
        ]
        tasks = generate_tasks(
            ["sqlite_mcp", "terminal_fs"],
            3,
            20260718,
            template_ids,
            task_indices=[25, 30, 35],
        )
        for task in tasks:
            action = next(item for item in task.candidates if item.role == "wrong_value")
            real = evaluate_real_branch(task, action, self.value_config)
            if task.domain == "sqlite_mcp":
                table = action.arguments["table"]
                predicted_row = next(
                    dict(row)
                    for row in task.initial_state["tables"][table]
                    if row["id"] == action.arguments["where"]["id"]
                )
                predicted_row.update(action.arguments["set"])
                prediction = {
                    "ok": True,
                    "affected_rows": 1,
                    "rows": [predicted_row],
                    "state_patch": {
                        "domain": "sqlite_mcp",
                        "operation": "update",
                        "table": table,
                        "where": action.arguments["where"],
                        "rows": [predicted_row],
                    },
                }
            else:
                content = action.arguments["content"]
                prediction = {
                    "ok": True,
                    "path": action.arguments["path"],
                    "bytes": len(content.encode()),
                    "state_patch": {
                        "domain": "terminal_fs",
                        "operation": "write",
                        "path": action.arguments["path"],
                        "exists": True,
                        "content": content,
                    },
                }
            simulated, _, _, error = evaluate_simulated_branch(
                task, action, prediction, self.value_config
            )
            self.assertIsNone(error)
            self.assertGreater(simulated - real.real_value, 1.0, task.task_id)
            self.assertGreater(token_f1(real.real_observation, prediction), 0.90)

    def test_rule_world_holdout_changes_schema_and_task_ids(self) -> None:
        templates = [
            "sqlite.rule.approval_guard",
            "sqlite.rule.version_trigger",
            "terminal.rule.json_validation",
            "terminal.rule.write_token",
        ]
        first = generate_tasks(
            ["sqlite_mcp", "terminal_fs"], 1, 23, templates, start_index=0
        )
        held_out = generate_tasks(
            ["sqlite_mcp", "terminal_fs"], 1, 23, templates, start_index=5
        )
        self.assertTrue({item.task_id for item in first}.isdisjoint(item.task_id for item in held_out))
        for left, right in zip(first, held_out):
            self.assertNotEqual(left.metadata["world_id"], right.metadata["world_id"])
            self.assertNotEqual(left.initial_state, right.initial_state)

        sparse = generate_tasks(
            ["sqlite_mcp", "terminal_fs"],
            3,
            23,
            templates,
            task_indices=[10, 15, 20],
        )
        self.assertEqual(len(sparse), 12)
        self.assertEqual(
            {int(item.task_id.rsplit("_", 1)[1]) for item in sparse},
            {10, 15, 20},
        )
        with self.assertRaises(ValueError):
            generate_tasks(
                ["sqlite_mcp"],
                2,
                23,
                ["sqlite.rule.version_trigger"],
                task_indices=[10],
            )

    def test_clones_are_isolated(self) -> None:
        for task in self.tasks:
            environment = make_environment(task)
            clone = environment.clone()
            try:
                original_state = environment.state_dict()
                result = clone.apply(task.oracle_plan[0])
                self.assertTrue(result.ok, task.task_id)
                self.assertEqual(environment.state_dict(), original_state, task.task_id)
                self.assertNotEqual(clone.state_dict(), original_state, task.task_id)
            finally:
                clone.close()
                environment.close()

    def test_oracle_is_real_best_and_destruction_is_costly(self) -> None:
        for task in self.tasks:
            branches = [
                evaluate_real_branch(task, action, self.value_config) for action in task.candidates
            ]
            oracle = next(branch for branch in branches if branch.action.role == "oracle")
            destructive = next(branch for branch in branches if branch.action.role == "destructive")
            self.assertEqual(oracle.real_value, max(branch.real_value for branch in branches), task.task_id)
            self.assertGreater(oracle.real_value - destructive.real_value, 0.20, task.task_id)

    def test_true_observations_roundtrip_to_identical_value(self) -> None:
        for task in self.tasks:
            for action in task.candidates:
                branch = evaluate_real_branch(task, action, self.value_config)
                simulated_value, _, _, error = evaluate_simulated_branch(
                    task, action, branch.real_observation, self.value_config
                )
                self.assertIsNone(error, (task.task_id, action.action_id, error))
                self.assertAlmostEqual(simulated_value, branch.real_value, places=10)

    def test_sqlite_set_where_shorthand_is_repaired_without_real_outcomes(self) -> None:
        task = next(task for task in self.tasks if task.task_id.startswith("sqlite_ticket"))
        values = []
        for action in task.candidates[:2]:
            observation = {
                "ok": True,
                "state_patch": {
                    "domain": "sqlite_mcp",
                    "operation": "update",
                    "table": action.arguments["table"],
                    "where": action.arguments["where"],
                    "set": action.arguments["set"],
                },
            }
            canonical, repair = canonicalize_predicted_observation(task, observation)
            self.assertEqual(repair, "sqlite_set_where_to_complete_rows")
            self.assertIn("rows", canonical["state_patch"])
            value, _, _, error = evaluate_simulated_branch(
                task, action, observation, self.value_config
            )
            self.assertIsNone(error)
            values.append(value)
        self.assertGreater(values[0], values[1])

    def test_sqlite_update_without_rows_or_set_fails_closed(self) -> None:
        task = next(task for task in self.tasks if task.task_id.startswith("sqlite_ticket"))
        action = task.candidates[0]
        observation = {
            "ok": True,
            "state_patch": {
                "domain": "sqlite_mcp",
                "operation": "update",
                "table": "tickets",
                "where": {"id": 1000},
            },
        }
        _, _, _, error = evaluate_simulated_branch(task, action, observation, self.value_config)
        self.assertIn("requires explicit complete rows", error)

    def test_result_row_primary_key_takes_priority_over_action_where(self) -> None:
        task = next(task for task in self.tasks if task.task_id.startswith("sqlite_ticket"))
        action = task.candidates[0]
        other_row = dict(task.initial_state["tables"]["tickets"][1])
        observation = {
            "ok": True,
            "state_patch": {
                "domain": "sqlite_mcp",
                "operation": "update",
                "table": "tickets",
                "where": action.arguments["where"],
                "rows": [other_row],
            },
        }
        value, success, violations, error = evaluate_simulated_branch(
            task, action, observation, self.value_config
        )
        self.assertFalse(success)
        self.assertEqual(violations, [])
        self.assertAlmostEqual(value, 0.84)
        self.assertIsNone(error)

    def test_terminal_full_state_snapshot_repairs_one_changed_file(self) -> None:
        task = next(task for task in self.tasks if task.task_id.startswith("terminal_timeout"))
        values = []
        for action in task.candidates[:2]:
            files = dict(task.initial_state["files"])
            files[action.arguments["path"]] = action.arguments["content"]
            observation = {"ok": True, "state_patch": {"files": files}}
            canonical, repair = canonicalize_predicted_observation(task, observation)
            self.assertEqual(repair, "terminal_full_files_to_single_patch")
            self.assertEqual(canonical["state_patch"]["path"], action.arguments["path"])
            value, _, _, error = evaluate_simulated_branch(
                task, action, observation, self.value_config
            )
            self.assertIsNone(error)
            values.append(value)
        self.assertGreater(values[0], values[1])

    def test_terminal_multifile_snapshot_is_not_repaired(self) -> None:
        task = next(task for task in self.tasks if task.task_id.startswith("terminal_timeout"))
        action = task.candidates[0]
        files = dict(task.initial_state["files"])
        files["config/service.json"] = action.arguments["content"]
        files["README.md"] = "also changed\n"
        observation = {"ok": True, "state_patch": {"files": files}}
        _, repair = canonicalize_predicted_observation(task, observation)
        self.assertIsNone(repair)
        _, _, _, error = evaluate_simulated_branch(task, action, observation, self.value_config)
        self.assertIn("Unsupported or missing patch domain", error)

    def test_terminal_single_file_fields_repair_missing_domain_and_operation(self) -> None:
        task = next(task for task in self.tasks if task.task_id.startswith("terminal_timeout"))
        values = []
        for action in task.candidates[:2]:
            observation = {
                "ok": True,
                "state_patch": {
                    "path": action.arguments["path"],
                    "exists": True,
                    "content": action.arguments["content"],
                },
            }
            canonical, repair = canonicalize_predicted_observation(task, observation)
            self.assertEqual(repair, "terminal_single_file_fields_to_patch")
            self.assertEqual(canonical["state_patch"]["domain"], "terminal_fs")
            self.assertEqual(canonical["state_patch"]["operation"], "write")
            value, _, _, error = evaluate_simulated_branch(
                task, action, observation, self.value_config
            )
            self.assertIsNone(error)
            values.append(value)
        self.assertGreater(values[0], values[1])

    def test_terminal_path_escape_fails_without_mutating_state(self) -> None:
        task = next(task for task in self.tasks if task.domain == "terminal_fs")
        environment = make_environment(task)
        try:
            before = environment.state_dict()
            result = environment.apply(
                Action("escape", "fs.write", {"path": "../escape.txt", "content": "bad"})
            )
            self.assertFalse(result.ok)
            self.assertEqual(environment.state_dict(), before)
        finally:
            environment.close()


if __name__ == "__main__":
    unittest.main()
