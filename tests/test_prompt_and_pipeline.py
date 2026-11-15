from __future__ import annotations

import json
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

from actionrankwm.collection import analyze_collection
from actionrankwm.intervention import (
    compare_prompt_runs,
    evaluate_intervention_pilot,
)
from actionrankwm.pipeline import run_gate0
from actionrankwm.providers.openai_compatible import LocalHTTPProvider
from actionrankwm.providers.prompts import build_messages
from actionrankwm.rescore import rescore_run
from actionrankwm.tasks import generate_tasks
from actionrankwm.trace import aggregate_trace_audits, audit_rule_trace
from actionrankwm.types import PredictionRecord


class PromptAndPipelineTests(unittest.TestCase):
    def test_world_model_prompt_excludes_goal_and_real_outcome(self) -> None:
        task = generate_tasks(["sqlite_mcp"], tasks_per_template=1, seed=3)[0]
        messages = build_messages(task, task.candidates[0])
        user_payload = json.loads(messages[1]["content"])
        self.assertEqual(set(user_payload), {"domain", "current_state", "action"})
        self.assertNotIn("instruction", user_payload)
        self.assertNotIn("target", user_payload)
        self.assertNotIn("real_observation", user_payload)
        self.assertNotIn(task.instruction, messages[0]["content"])
        self.assertIn("Never put `set` inside state_patch", messages[0]["content"])

    def test_rule_trace_prompt_is_generic_and_keeps_the_blind_payload(self) -> None:
        task = generate_tasks(
            ["sqlite_mcp"],
            1,
            20260718,
            ["sqlite.rule.formula_recompute"],
            task_indices=[25],
        )[0]
        action = task.candidates[2]
        messages = build_messages(task, action, prompt_style="rule_trace")
        user_payload = json.loads(messages[1]["content"])
        self.assertEqual(set(user_payload), {"domain", "current_state", "action"})
        self.assertEqual(
            set(user_payload["action"]),
            {"tool", "arguments"},
        )
        for forbidden in (
            task.instruction,
            task.candidates[2].description,
            task.candidates[2].role,
        ):
            self.assertNotIn(forbidden, messages[0]["content"])
            self.assertNotIn(forbidden, messages[1]["content"])
        self.assertIn("direct_effect", messages[0]["content"])
        self.assertIn("rule_trace", messages[0]["content"])
        self.assertIn("one entry for every visible environment rule", messages[0]["content"])
        with self.assertRaises(ValueError):
            build_messages(task, action, prompt_style="unknown")

    def test_rule_trace_audit_checks_count_order_and_match_semantics(self) -> None:
        task = generate_tasks(
            ["sqlite_mcp"],
            1,
            20260718,
            ["sqlite.rule.formula_recompute"],
            task_indices=[25],
        )[0]
        action = task.candidates[2]
        prediction = PredictionRecord(
            task_id=task.task_id,
            domain=task.domain,
            action_id=action.action_id,
            provider="test",
            model="test",
            raw_text="",
            parsed_observation={
                "direct_effect": {},
                "rule_trace": [
                    {
                        "rule_index": 0,
                        "matched": True,
                        "derived_effect": {},
                    }
                ],
                "observation": {},
            },
            parse_ok=True,
            latency_seconds=0.0,
        )
        audit = audit_rule_trace(task, prediction)
        self.assertTrue(audit["wrapper_valid"])
        self.assertTrue(audit["rule_count_covered"])
        self.assertTrue(audit["original_order_valid"])
        self.assertEqual(audit["match_correct_count"], 1)
        aggregate = aggregate_trace_audits([audit], "rule_trace")
        self.assertEqual(aggregate["wrapper_valid_rate"], 1.0)
        self.assertEqual(aggregate["match_accuracy"], 1.0)

    def test_local_compatible_client_needs_no_openai_sdk(self) -> None:
        task = generate_tasks(["sqlite_mcp"], tasks_per_template=1, seed=5)[0]
        observation = {
            "ok": True,
            "state_patch": {"domain": "sqlite_mcp", "operation": "none"},
        }
        captured: dict[str, object] = {}

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802 - stdlib callback name
                length = int(self.headers["Content-Length"])
                request_payload = json.loads(self.rfile.read(length))
                captured["request"] = request_payload
                response_payload = {
                    "choices": [{"message": {"content": json.dumps(observation)}}],
                    "usage": {"total_tokens": 17},
                    "model": "served-test-revision",
                    "system_fingerprint": "fp_test_123",
                    "service_tier": "default",
                }
                body = json.dumps(response_payload).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format: str, *args: object) -> None:
                del format, args

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            provider = LocalHTTPProvider(
                model="test-model",
                base_url=f"http://127.0.0.1:{server.server_port}/v1",
                api_key="local",
                config={"max_retries": 0, "timeout_seconds": 2, "temperature": 0.0},
            )
            result = provider.predict(task, task.candidates[0])
            self.assertEqual(result.parsed_observation, observation)
            self.assertEqual(result.usage["total_tokens"], 17)
            self.assertEqual(result.usage["response_model"], "served-test-revision")
            self.assertEqual(result.usage["system_fingerprint"], "fp_test_123")
            self.assertEqual(result.usage["service_tier"], "default")
            request_payload = captured["request"]
            self.assertIsInstance(request_payload, dict)
            self.assertEqual(request_payload["model"], "test-model")
            self.assertEqual(
                set(json.loads(request_payload["messages"][1]["content"])),
                {"domain", "current_state", "action"},
            )
            self.assertNotIn("api_key", result.usage)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

    def test_pipeline_writes_auditable_matched_artifacts(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        config = json.loads((project_root / "configs" / "smoke.json").read_text())
        config["tasks_per_template"] = 1
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            run_dir = root / "run"
            result = run_gate0(config_path, run_dir, "replay-perfect")
            manifest = result["manifest"]
            self.assertEqual(result["gate_decision"]["status"], "CONTROL_COMPLETE")
            self.assertEqual(manifest["roundtrip_rate"], 1.0)
            self.assertEqual(manifest["counts"]["tasks"], 8)
            self.assertEqual(manifest["counts"]["real_branches"], 40)
            self.assertEqual(manifest["counts"]["real_branches"], manifest["counts"]["predictions"])
            self.assertEqual(manifest["counts"]["predictions"], manifest["counts"]["scored_candidates"])
            self.assertFalse((run_dir / "predictions.partial.jsonl").exists())
            for name in (
                "manifest.json",
                "tasks.jsonl",
                "real_branches.jsonl",
                "predictions.jsonl",
                "scored_candidates.jsonl",
                "task_decisions.jsonl",
                "trace_audits.jsonl",
                "metrics.json",
                "gate_decision.json",
                "gate_decision.md",
                "failures.jsonl",
                "repairs.jsonl",
            ):
                self.assertTrue((run_dir / name).is_file(), name)
            with self.assertRaises(FileExistsError):
                run_gate0(config_path, run_dir, "replay-perfect")
            rescore_dir = root / "rescore"
            rescored = rescore_run(run_dir, rescore_dir)
            self.assertEqual(rescored["manifest"]["run_type"], "offline_rescore")
            self.assertEqual(rescored["manifest"]["counts"]["predictions"], 40)
            self.assertEqual(
                rescored["metrics"]["decision"]["overall"]["top1_preservation_rate"],
                1.0,
            )
            self.assertTrue((rescore_dir / "source_artifacts.json").exists() is False)
            self.assertEqual(manifest["counts"]["trace_audits"], 40)
            self.assertEqual(result["metrics"]["trace"]["prompt_style"], "direct")

    def test_collection_validates_sources_and_recomputes_frozen_runs(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        config = json.loads(
            (project_root / "configs" / "rule_holdout_pilot.json").read_text()
        )
        config.update(
            {
                "bootstrap_resamples": 20,
                "tasks_per_template": 1,
                "template_ids": [
                    "sqlite.rule.approval_guard",
                    "terminal.rule.json_validation",
                ],
            }
        )
        config["gate"].update(
            {
                "min_domains": 2,
                "min_tasks_per_domain": 1,
            }
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runs = []
            for offset in (5, 10):
                current = dict(config)
                current["provider"] = dict(config["provider"])
                current["gate"] = dict(config["gate"])
                current["start_index"] = offset
                current["bootstrap_resamples"] = 20 if offset == 5 else 30
                config_path = root / f"config_{offset}.json"
                config_path.write_text(json.dumps(current), encoding="utf-8")
                run_dir = root / f"run_{offset}"
                with patch.dict(
                    "os.environ",
                    {
                        "WORLD_MODEL_REVISION": "test-revision",
                        "WORLD_MODEL_CONFIG_SHA256": "a" * 64,
                        "SLURM_JOB_ID": "12345",
                        "SLURM_JOB_NAME": "test-job",
                    },
                    clear=False,
                ):
                    result = run_gate0(config_path, run_dir, "replay-perfect")
                self.assertEqual(
                    result["manifest"]["provider"]["model_provenance"],
                    {"revision": "test-revision", "config_sha256": "a" * 64},
                )
                self.assertEqual(
                    result["manifest"]["runtime"]["scheduler"]["job_id"], "12345"
                )
                self.assertFalse("api_key" in result["manifest"]["provider"])
                runs.append(run_dir)

            collection_dir = root / "collection"
            combined = analyze_collection(
                runs, collection_dir, model_revision="test-revision"
            )
            self.assertEqual(combined["manifest"]["run_type"], "collection_rescore")
            self.assertEqual(combined["manifest"]["counts"]["tasks"], 4)
            self.assertEqual(combined["manifest"]["counts"]["predictions"], 20)
            self.assertEqual(
                combined["metrics"]["decision"]["overall"]["top1_preservation_rate"],
                1.0,
            )
            self.assertEqual(len(combined["manifest"]["source_runs"]), 2)
            collection_config = json.loads(
                (collection_dir / "config.snapshot.json").read_text()
            )
            self.assertEqual(collection_config["bootstrap_resamples"], 30)
            self.assertTrue((collection_dir / "manifest.json").is_file())
            case_analysis = json.loads(
                (collection_dir / "case_analysis.json").read_text()
            )
            self.assertEqual(case_analysis["overall"]["task_count"], 4)
            self.assertEqual(
                case_analysis["overall"][
                    "high_text_fidelity_unsafe_top_set_count"
                ],
                0,
            )
            self.assertTrue((collection_dir / "case_analysis.md").is_file())

            with self.assertRaisesRegex(ValueError, "duplicate task IDs"):
                analyze_collection(
                    [runs[0], runs[0]],
                    root / "duplicate_collection",
                    model_revision="test-revision",
                )

            with (runs[1] / "predictions.jsonl").open("a", encoding="utf-8") as handle:
                handle.write("\n")
            with self.assertRaisesRegex(ValueError, "integrity mismatch"):
                analyze_collection(
                    runs,
                    root / "tampered_collection",
                    model_revision="test-revision",
                )

    def test_paired_intervention_comparison_enforces_the_frozen_contract(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runs = {}
            for style in ("direct", "trace"):
                source = project_root / "configs" / f"mechanism_holdout_{style}_pilot.json"
                config = json.loads(source.read_text())
                config["bootstrap_resamples"] = 20
                config_path = root / f"{style}.json"
                config_path.write_text(json.dumps(config), encoding="utf-8")
                run_dir = root / f"{style}_run"
                run_gate0(config_path, run_dir, "replay-perfect")
                runs[style] = run_dir
            comparison_dir = root / "comparison"
            comparison = compare_prompt_runs(
                runs["direct"], runs["trace"], comparison_dir
            )["comparison"]
            self.assertEqual(comparison["overall"]["direct_top1"], 1.0)
            self.assertEqual(comparison["overall"]["trace_top1"], 1.0)
            self.assertEqual(comparison["bootstrap"]["unit"], "latent_environment_group")
            self.assertTrue((comparison_dir / "paired_tasks.jsonl").is_file())

            decision = evaluate_intervention_pilot(
                comparison_dir,
                comparison_dir,
                comparison_dir,
                root / "pilot_decision",
            )
            self.assertEqual(decision["decision"], "NON_DIAGNOSTIC")
            self.assertFalse(decision["diagnostic"]["passed"])

            trace_config = json.loads((root / "trace.json").read_text())
            trace_config["provider"]["max_tokens"] += 1
            mismatched_path = root / "trace_mismatched.json"
            mismatched_path.write_text(json.dumps(trace_config), encoding="utf-8")
            mismatched_run = root / "trace_mismatched_run"
            run_gate0(mismatched_path, mismatched_run, "replay-perfect")
            with self.assertRaisesRegex(ValueError, "differ beyond"):
                compare_prompt_runs(
                    runs["direct"],
                    mismatched_run,
                    root / "invalid_comparison",
                )


if __name__ == "__main__":
    unittest.main()
