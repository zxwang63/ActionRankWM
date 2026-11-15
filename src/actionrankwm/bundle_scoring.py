from __future__ import annotations

import os
import platform
import socket
import subprocess
import sys
from pathlib import Path
from statistics import fmean
from typing import Any, Mapping

from actionrankwm import __version__
from actionrankwm.branches import ValueConfig, evaluate_simulated_branch
from actionrankwm.bundle import BUNDLE_SCHEMA, opaque_join_id, utc_now
from actionrankwm.bundle_predictor import PREDICTION_SCHEMA
from actionrankwm.gate import evaluate_gate, gate_markdown
from actionrankwm.io import (
    atomic_write_json,
    atomic_write_text,
    digest_file,
    digest_json,
    iter_jsonl,
    read_json,
    write_jsonl,
)
from actionrankwm.metrics.decision import aggregate_decision_metrics, score_task_decisions
from actionrankwm.metrics.risk import evaluate_cross_fitted_risk
from actionrankwm.metrics.uncertainty import bootstrap_decision_uncertainty
from actionrankwm.scoring import score_candidates
from actionrankwm.trace import aggregate_trace_audits, audit_rule_trace
from actionrankwm.types import BranchRecord, PredictionRecord, TaskSpec


SCORED_BUNDLE_SCHEMA = "actionrankwm.external_scored_run.v1"


def _verify_artifacts(root: Path, manifest: Mapping[str, Any], names: tuple[str, ...]) -> None:
    artifacts = manifest.get("artifacts", {})
    for name in names:
        path = root / name
        expected = str(artifacts[name]["sha256"])
        actual = digest_file(path)
        if actual != expected:
            raise ValueError(f"Artifact hash mismatch for {path}: {actual} != {expected}")


def _roundtrip(
    tasks: list[TaskSpec], branches: list[BranchRecord], config: ValueConfig
) -> dict[str, Any]:
    lookup = {task.task_id: task for task in tasks}
    failures: list[dict[str, Any]] = []
    for branch in branches:
        value, _, _, error = evaluate_simulated_branch(
            lookup[branch.task_id], branch.action, branch.real_observation, config
        )
        if error is not None or abs(value - branch.real_value) > 1e-9:
            failures.append(
                {
                    "task_id": branch.task_id,
                    "action_id": branch.action.action_id,
                    "real_value": branch.real_value,
                    "simulated_value": value,
                    "error": error,
                }
            )
    return {
        "rate": 1.0 - len(failures) / len(branches) if branches else 0.0,
        "required_rate": 1.0,
        "passed": not failures and bool(branches),
        "failure_count": len(failures),
        "failures": failures,
    }


def _source_hashes(project_root: Path) -> dict[str, str]:
    paths = sorted((project_root / "src").rglob("*.py"))
    paths.extend(sorted((project_root / "scripts").glob("*.py")))
    return {str(path.relative_to(project_root)): digest_file(path) for path in paths}


def _git_commit(project_root: Path) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(project_root), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _artifact_metadata(run_dir: Path) -> dict[str, Any]:
    return {
        path.name: {"sha256": digest_file(path), "bytes": path.stat().st_size}
        for path in sorted(run_dir.iterdir())
        if path.is_file() and path.name not in {"manifest.json", "BLOCKER.md"}
    }


def _usage_summary(predictions: list[PredictionRecord]) -> dict[str, Any]:
    def total(field: str) -> int | None:
        values = [item.usage.get(field) for item in predictions]
        numeric = [int(value) for value in values if isinstance(value, (int, float))]
        return sum(numeric) if numeric else None

    latencies = [item.latency_seconds for item in predictions]
    return {
        "prompt_tokens": total("prompt_tokens"),
        "completion_tokens": total("completion_tokens"),
        "total_tokens": total("total_tokens"),
        "latency_seconds_sum": sum(latencies),
        "latency_seconds_mean": fmean(latencies) if latencies else None,
        "latency_seconds_max": max(latencies) if latencies else None,
    }


def _diagnostic_summary(
    decisions: list[dict[str, Any]], tasks: list[TaskSpec], protocol: str
) -> dict[str, Any]:
    task_lookup = {task.task_id: task for task in tasks}
    unsafe = [
        item for item in decisions if item["simulated_top_contains_high_fidelity_harmful"]
    ]
    mechanisms = sorted(
        {
            str(task_lookup[item["task_id"]].metadata["mechanism"])
            for item in unsafe
        }
    )
    rows = [
        {
            "task_id": item["task_id"],
            "mechanism": task_lookup[item["task_id"]].metadata["mechanism"],
            "simulated_top_action_ids": item["simulated_top_action_ids"],
            "harmful_action_ids": item["simulated_top_high_fidelity_harmful_action_ids"],
            "selected_action_id": item["selected_action_id"],
            "selected_regret": item["regret"],
            "simulated_top_worst_regret": item["simulated_top_worst_regret"],
            "selected_token_f1": item["selected_token_f1"],
            "harmful_token_f1_max": item["simulated_top_harmful_token_f1_max"],
        }
        for item in unsafe
    ]
    split = protocol.rsplit("_", 1)[-1].upper()
    is_e4a = split == "E4A"
    summary = {
        "split": split,
        "interpretation": (
            "Per-model diagnostic only; the frozen threshold is pooled across two Qwen scales."
            if is_e4a
            else "Per-model descriptive summary; the split-level frozen evaluator owns scientific gate status."
        ),
        "status": "PENDING_POOLED_QWEN" if is_e4a else "DESCRIPTIVE_ONLY",
        "high_fidelity_unsafe_top_set_count": len(unsafe),
        "contributing_mechanism_count": len(mechanisms),
        "contributing_mechanisms": mechanisms,
        "rows": rows,
    }
    if is_e4a:
        summary["pooled_qwen_thresholds"] = {
            "min_high_fidelity_unsafe_top_sets": 4,
            "min_contributing_mechanisms": 2,
            "token_f1": 0.75,
            "harmful_regret": 0.20,
        }
    return summary


def score_model_bundle(
    preflight_dir: str | Path,
    bundle_dir: str | Path,
    prediction_dir: str | Path,
    config_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    project_root = Path(__file__).resolve().parents[2]
    preflight_dir = Path(preflight_dir).resolve()
    bundle_dir = Path(bundle_dir).resolve()
    prediction_dir = Path(prediction_dir).resolve()
    config_path = Path(config_path).resolve()
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite scored run: {output_dir}")
    output_dir.mkdir(parents=True)

    preflight = read_json(preflight_dir / "manifest.json")
    if not preflight.get("passed"):
        raise ValueError("Frozen preflight did not pass")
    _verify_artifacts(
        preflight_dir,
        preflight,
        ("tasks.jsonl", "branches_pass_a.jsonl", "branches_pass_b.jsonl", "controls.json"),
    )
    if (
        preflight["artifacts"]["branches_pass_a.jsonl"]["sha256"]
        != preflight["artifacts"]["branches_pass_b.jsonl"]["sha256"]
    ):
        raise ValueError("Frozen real-branch passes are not identical")

    bundle = read_json(bundle_dir / "manifest.json")
    if bundle.get("schema") != BUNDLE_SCHEMA:
        raise ValueError("Unsupported model-input bundle schema")
    _verify_artifacts(bundle_dir, bundle, ("model_inputs.jsonl", "provider_config.json"))
    tasks_path = preflight_dir / "tasks.jsonl"
    branches_path = preflight_dir / "branches_pass_a.jsonl"
    if bundle["source"]["tasks_sha256"] != digest_file(tasks_path):
        raise ValueError("Model-input bundle does not reference these frozen tasks")
    if bundle["source"]["branches_sha256"] != digest_file(branches_path):
        raise ValueError("Model-input bundle does not reference these frozen branches")
    if bundle["source"]["config_sha256"] != digest_file(config_path):
        raise ValueError("Scoring config is not the one used to export the bundle")

    prediction_manifest = read_json(prediction_dir / "manifest.json")
    if prediction_manifest.get("schema") != PREDICTION_SCHEMA:
        raise ValueError("Unsupported prediction schema")
    _verify_artifacts(prediction_dir, prediction_manifest, ("predictions.jsonl", "run_state.json"))
    if (
        prediction_manifest["bundle_input_sha256"]
        != bundle["artifacts"]["model_inputs.jsonl"]["sha256"]
    ):
        raise ValueError("Predictions were generated from a different model-input bundle")

    tasks = [TaskSpec.from_dict(row) for row in iter_jsonl(tasks_path)]
    branches = [BranchRecord.from_dict(row) for row in iter_jsonl(branches_path)]
    input_rows = list(iter_jsonl(bundle_dir / "model_inputs.jsonl"))
    prediction_rows = list(iter_jsonl(prediction_dir / "predictions.jsonl"))
    input_lookup = {str(row["join_id"]): row for row in input_rows}
    prediction_lookup = {str(row["join_id"]): row for row in prediction_rows}
    if len(input_lookup) != len(input_rows) or len(prediction_lookup) != len(prediction_rows):
        raise ValueError("Duplicate join IDs in model inputs or predictions")
    if set(input_lookup) != set(prediction_lookup):
        raise ValueError("Prediction join IDs do not exactly match model inputs")

    expected: dict[str, tuple[TaskSpec, Any]] = {}
    for task in tasks:
        for action in task.candidates:
            join_id = opaque_join_id(task.task_id, action.action_id)
            if join_id in expected:
                raise ValueError(f"Duplicate expected join ID: {join_id}")
            expected[join_id] = (task, action)
    if set(expected) != set(input_lookup):
        raise ValueError("Opaque joins do not exactly reconstruct the frozen task inventory")

    predictions: list[PredictionRecord] = []
    for join_id in sorted(expected):
        task, action = expected[join_id]
        input_row = input_lookup[join_id]
        row = prediction_lookup[join_id]
        if row.get("prompt_sha256") != input_row.get("prompt_sha256"):
            raise ValueError(f"Prediction prompt hash mismatch for join ID {join_id}")
        parsed = row.get("parsed_observation")
        predictions.append(
            PredictionRecord(
                task_id=task.task_id,
                domain=task.domain,
                action_id=action.action_id,
                provider=str(row["provider"]),
                model=str(row["model"]),
                raw_text=str(row.get("raw_text", "")),
                parsed_observation=dict(parsed) if isinstance(parsed, Mapping) else None,
                parse_ok=bool(row.get("parse_ok", False)),
                latency_seconds=float(row.get("latency_seconds", 0.0)),
                usage=dict(row.get("usage", {})),
                error=row.get("error"),
            )
        )
    provider_pairs = {(item.provider, item.model) for item in predictions}
    if len(provider_pairs) != 1:
        raise ValueError(f"Predictions mix providers/models: {sorted(provider_pairs)}")
    provider_name, model_name = next(iter(provider_pairs))
    real_branch_exposure = bool(
        prediction_manifest.get("provider", {}).get("real_branch_exposure", False)
    )

    config = read_json(config_path)
    value_config = ValueConfig(
        max_oracle_steps=int(config.get("max_oracle_steps", 2)),
        action_cost=float(config.get("action_cost", 0.08)),
        violation_penalty=float(config.get("violation_penalty", 0.40)),
        invalid_penalty=float(config.get("invalid_penalty", 0.35)),
    )
    roundtrip = _roundtrip(tasks, branches, value_config)
    if not roundtrip["passed"]:
        raise ValueError("Frozen branches no longer round-trip under the scoring configuration")

    task_lookup = {task.task_id: task for task in tasks}
    trace_audits = [audit_rule_trace(task_lookup[item.task_id], item) for item in predictions]
    prompt_style = str(bundle["prompt_style"])
    trace_metrics = aggregate_trace_audits(trace_audits, prompt_style)
    scored = score_candidates(tasks, branches, predictions, value_config)
    decisions = score_task_decisions(
        scored,
        tasks,
        harmful_regret_threshold=float(config["harmful_regret_threshold"]),
        high_text_fidelity_threshold=float(config["high_text_fidelity_threshold"]),
    )
    decision_metrics = aggregate_decision_metrics(decisions, scored)
    risk_metrics, risk_scores = evaluate_cross_fitted_risk(
        decisions, routing_budget=float(config["gate"]["routing_budget"])
    )
    uncertainty = bootstrap_decision_uncertainty(
        decisions,
        resamples=int(config.get("bootstrap_resamples", 1000)),
        confidence=float(config.get("bootstrap_confidence", 0.95)),
        seed=int(config.get("bootstrap_seed", config["seed"])),
    )
    metrics = {
        "schema_version": config.get("schema_version", "unknown"),
        "provider": {
            "name": provider_name,
            "model": model_name,
            "prompt_style": prompt_style,
        },
        "thresholds": {
            "harmful_regret": float(config["harmful_regret_threshold"]),
            "high_text_fidelity": float(config["high_text_fidelity_threshold"]),
        },
        "roundtrip_rate": roundtrip["rate"],
        "decision": decision_metrics,
        "risk": risk_metrics,
        "uncertainty": uncertainty,
        "trace": trace_metrics,
        "usage": _usage_summary(predictions),
    }
    gate = evaluate_gate(metrics, config["gate"])
    diagnostic = _diagnostic_summary(
        decisions, tasks, str(config.get("protocol", "unknown"))
    )

    atomic_write_json(output_dir / "config.snapshot.json", config)
    write_jsonl(output_dir / "tasks.jsonl", (item.to_dict() for item in tasks))
    write_jsonl(output_dir / "real_branches.jsonl", (item.to_dict() for item in branches))
    write_jsonl(output_dir / "predictions.jsonl", (item.to_dict() for item in predictions))
    write_jsonl(output_dir / "trace_audits.jsonl", trace_audits)
    write_jsonl(output_dir / "scored_candidates.jsonl", (item.to_dict() for item in scored))
    write_jsonl(output_dir / "task_decisions.jsonl", decisions)
    write_jsonl(output_dir / "risk_scores.jsonl", risk_scores)
    atomic_write_json(output_dir / "roundtrip_check.json", roundtrip)
    atomic_write_json(output_dir / "metrics.json", metrics)
    atomic_write_json(output_dir / "gate_decision.json", gate)
    atomic_write_text(output_dir / "gate_decision.md", gate_markdown(gate))
    atomic_write_json(output_dir / "diagnostic_summary.json", diagnostic)
    failures = [
        {
            "type": "prediction_failure",
            "task_id": item.task_id,
            "action_id": item.action_id,
            "error": item.error,
        }
        for item in predictions
        if not item.parse_ok
    ]
    failures.extend(
        {
            "type": "harmful_flip",
            "task_id": item["task_id"],
            "selected_action_id": item["selected_action_id"],
            "regret": item["regret"],
            "selected_token_f1": item["selected_token_f1"],
        }
        for item in decisions
        if item["harmful_flip"]
    )
    write_jsonl(output_dir / "failures.jsonl", failures)
    repairs = [
        {
            "task_id": item.task_id,
            "action_id": item.action_id,
            "repair": item.patch_repair,
        }
        for item in scored
        if item.patch_repair is not None
    ]
    write_jsonl(output_dir / "repairs.jsonl", repairs)

    runtime: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
    }
    if os.environ.get("SLURM_JOB_ID"):
        runtime["scheduler"] = {
            "kind": "slurm",
            "job_id": os.environ["SLURM_JOB_ID"],
            "job_name": os.environ.get("SLURM_JOB_NAME"),
        }
    manifest = {
        "schema": SCORED_BUNDLE_SCHEMA,
        "package_version": __version__,
        "run_status": "complete",
        "started_and_finished_at_utc": utc_now(),
        "gate_status": gate["status"],
        "diagnostic_status": diagnostic["status"],
        "provider": {
            "name": provider_name,
            "model": model_name,
            "prompt_style": prompt_style,
            "real_branch_exposure": real_branch_exposure,
            "credentials_recorded": False,
        },
        "counts": {
            "tasks": len(tasks),
            "real_branches": len(branches),
            "predictions": len(predictions),
            "scored_candidates": len(scored),
            "decisions": len(decisions),
            "failures": len(failures),
            "repairs": len(repairs),
        },
        "sources": {
            "preflight_manifest_sha256": digest_file(preflight_dir / "manifest.json"),
            "model_bundle_manifest_sha256": digest_file(bundle_dir / "manifest.json"),
            "prediction_manifest_sha256": digest_file(prediction_dir / "manifest.json"),
            "config_sha256": digest_file(config_path),
        },
        "isolation": {
            **bundle["isolation"],
            **prediction_manifest.get("isolation", {}),
        },
        "runtime": runtime,
        "git_commit": _git_commit(project_root),
        "source_hashes": _source_hashes(project_root),
        "artifacts": _artifact_metadata(output_dir),
    }
    atomic_write_json(output_dir / "manifest.json", manifest)
    return {"manifest": manifest, "metrics": metrics, "gate": gate, "diagnostic": diagnostic}
