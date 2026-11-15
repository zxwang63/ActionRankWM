from __future__ import annotations

import copy
import os
import platform
import socket
import subprocess
import sys
import traceback
from collections import defaultdict
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from actionrankwm import __version__
from actionrankwm.branches import ValueConfig, evaluate_real_branch, evaluate_simulated_branch
from actionrankwm.gate import evaluate_gate, gate_markdown
from actionrankwm.io import (
    atomic_write_json,
    atomic_write_text,
    canonical_json,
    digest_file,
    digest_json,
    read_json,
    write_jsonl,
)
from actionrankwm.metrics.decision import aggregate_decision_metrics, score_task_decisions
from actionrankwm.metrics.risk import evaluate_cross_fitted_risk
from actionrankwm.metrics.uncertainty import bootstrap_decision_uncertainty
from actionrankwm.providers.base import create_provider
from actionrankwm.scoring import score_candidates
from actionrankwm.tasks import generate_tasks
from actionrankwm.trace import aggregate_trace_audits, audit_rule_trace
from actionrankwm.types import BranchRecord, PredictionRecord, TaskSpec


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _balanced_limit(
    tasks: Sequence[TaskSpec], limit: int | None, domain_order: Sequence[str]
) -> list[TaskSpec]:
    if limit is None or limit >= len(tasks):
        return list(tasks)
    if limit <= 0:
        raise ValueError("task_limit must be positive")
    buckets: dict[str, list[TaskSpec]] = defaultdict(list)
    for task in tasks:
        buckets[task.domain].append(task)
    selected: list[TaskSpec] = []
    offsets = {domain: 0 for domain in domain_order}
    while len(selected) < limit:
        progress = False
        for domain in domain_order:
            offset = offsets[domain]
            if offset < len(buckets[domain]):
                selected.append(buckets[domain][offset])
                offsets[domain] += 1
                progress = True
                if len(selected) == limit:
                    break
        if not progress:
            break
    return selected


def _limit_candidates(tasks: Sequence[TaskSpec], limit: int | None) -> list[TaskSpec]:
    limited = copy.deepcopy(list(tasks))
    if limit is None:
        return limited
    if limit <= 0:
        raise ValueError("candidate_limit_per_task must be positive")
    for task in limited:
        task.candidates = task.candidates[:limit]
        if not task.candidates:
            raise ValueError(f"No candidates remain for {task.task_id}")
    return limited


def _git_commit(project_root: Path) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(project_root), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _source_hashes(project_root: Path) -> dict[str, str]:
    paths = sorted((project_root / "src").rglob("*.py"))
    paths.extend(sorted((project_root / "scripts").glob("*.py")))
    return {str(path.relative_to(project_root)): digest_file(path) for path in paths}


def _artifact_metadata(run_dir: Path) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for path in sorted(run_dir.iterdir()):
        if path.is_file() and path.name not in {"manifest.json", "BLOCKER.md"}:
            metadata[path.name] = {"sha256": digest_file(path), "bytes": path.stat().st_size}
    return metadata


def _model_provenance_from_environment() -> dict[str, str]:
    provenance = {}
    for output_key, environment_key in (
        ("revision", "WORLD_MODEL_REVISION"),
        ("config_sha256", "WORLD_MODEL_CONFIG_SHA256"),
    ):
        value = os.environ.get(environment_key, "").strip()
        if value:
            provenance[output_key] = value
    return provenance


def _scheduler_provenance_from_environment() -> dict[str, str] | None:
    job_id = os.environ.get("SLURM_JOB_ID", "").strip()
    if not job_id:
        return None
    scheduler = {"kind": "slurm", "job_id": job_id}
    job_name = os.environ.get("SLURM_JOB_NAME", "").strip()
    if job_name:
        scheduler["job_name"] = job_name
    return scheduler


def _roundtrip_rate(
    tasks: Sequence[TaskSpec], branches: Sequence[BranchRecord], value_config: ValueConfig
) -> tuple[float, list[dict[str, Any]]]:
    task_lookup = {task.task_id: task for task in tasks}
    failures: list[dict[str, Any]] = []
    passed = 0
    for branch in branches:
        task = task_lookup[branch.task_id]
        simulated_value, _, _, error = evaluate_simulated_branch(
            task, branch.action, branch.real_observation, value_config
        )
        matches = error is None and abs(simulated_value - branch.real_value) <= 1e-9
        if matches:
            passed += 1
        else:
            failures.append(
                {
                    "task_id": branch.task_id,
                    "action_id": branch.action.action_id,
                    "real_value": branch.real_value,
                    "roundtrip_value": simulated_value,
                    "error": error,
                }
            )
    return passed / len(branches) if branches else 0.0, failures


def _prediction_record(
    task: TaskSpec,
    branch: BranchRecord,
    provider: Any,
    expose_real_branch: bool,
) -> PredictionRecord:
    try:
        result = provider.predict(
            task=task,
            action=branch.action,
            real_branch=branch if expose_real_branch else None,
        )
        return PredictionRecord(
            task_id=task.task_id,
            domain=task.domain,
            action_id=branch.action.action_id,
            provider=provider.name,
            model=provider.model,
            raw_text=result.raw_text,
            parsed_observation=result.parsed_observation,
            parse_ok=result.parsed_observation is not None and result.error is None,
            latency_seconds=result.latency_seconds,
            usage=result.usage,
            error=result.error,
        )
    except Exception as exc:
        return PredictionRecord(
            task_id=task.task_id,
            domain=task.domain,
            action_id=branch.action.action_id,
            provider=provider.name,
            model=provider.model,
            raw_text="",
            parsed_observation=None,
            parse_ok=False,
            latency_seconds=0.0,
            error=f"provider:{type(exc).__name__}: {exc}",
        )


def run_gate0(config_path: str | Path, run_dir: str | Path, provider_name: str) -> dict[str, Any]:
    project_root = Path(__file__).resolve().parents[2]
    config_path = Path(config_path).resolve()
    run_dir = Path(run_dir).resolve()
    if run_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing run directory: {run_dir}")
    run_dir.mkdir(parents=True)
    stage = "initialization"
    started_at = _utc_now()
    try:
        config = read_json(config_path)
        atomic_write_json(run_dir / "config.snapshot.json", config)
        domains = [str(domain) for domain in config["domains"]]
        value_config = ValueConfig(
            max_oracle_steps=int(config.get("max_oracle_steps", 2)),
            action_cost=float(config.get("action_cost", 0.08)),
            violation_penalty=float(config.get("violation_penalty", 0.40)),
            invalid_penalty=float(config.get("invalid_penalty", 0.35)),
        )

        stage = "task_generation"
        tasks = generate_tasks(
            domains=domains,
            tasks_per_template=int(config["tasks_per_template"]),
            seed=int(config["seed"]),
            template_ids=config.get("template_ids"),
            start_index=int(config.get("start_index", 0)),
            task_indices=config.get("task_indices"),
        )
        tasks = _balanced_limit(tasks, config.get("task_limit"), domains)
        tasks = _limit_candidates(tasks, config.get("candidate_limit_per_task"))
        task_count = write_jsonl(run_dir / "tasks.jsonl", (task.to_dict() for task in tasks))

        stage = "real_branch_execution"
        branches = [
            evaluate_real_branch(task, action, value_config)
            for task in tasks
            for action in task.candidates
        ]
        branch_count = write_jsonl(
            run_dir / "real_branches.jsonl", (branch.to_dict() for branch in branches)
        )
        roundtrip_rate, roundtrip_failures = _roundtrip_rate(tasks, branches, value_config)
        atomic_write_json(
            run_dir / "roundtrip_check.json",
            {
                "rate": roundtrip_rate,
                "required_rate": 0.95,
                "passed": roundtrip_rate >= 0.95,
                "failure_count": len(roundtrip_failures),
                "failures": roundtrip_failures[:100],
            },
        )
        if roundtrip_rate < 0.95:
            raise RuntimeError(
                f"True-observation round-trip rate {roundtrip_rate:.3f} is below 0.95"
            )

        stage = "provider_initialization"
        provider = create_provider(provider_name, dict(config.get("provider", {})))
        expose_real_branch = provider.name.startswith("replay-")
        prompt_style = str(
            getattr(
                provider,
                "prompt_style",
                config.get("provider", {}).get("prompt_style", "direct"),
            )
        )

        stage = "world_model_prediction"
        task_lookup = {task.task_id: task for task in tasks}
        prediction_workers = int(config.get("provider", {}).get("concurrency", 1))
        if prediction_workers <= 0:
            raise ValueError("provider.concurrency must be positive")
        def predict_branch(branch: BranchRecord) -> PredictionRecord:
            return _prediction_record(
                task_lookup[branch.task_id], branch, provider, expose_real_branch
            )

        prediction_partial_path = run_dir / "predictions.partial.jsonl"
        prediction_final_path = run_dir / "predictions.jsonl"
        predictions: list[PredictionRecord] = []
        prediction_partial_handle = prediction_partial_path.open("x", encoding="utf-8")
        def journal_prediction(prediction: PredictionRecord) -> None:
            predictions.append(prediction)
            prediction_partial_handle.write(canonical_json(prediction.to_dict()))
            prediction_partial_handle.write("\n")
            if len(predictions) % 10 == 0:
                prediction_partial_handle.flush()
                os.fsync(prediction_partial_handle.fileno())

        if prediction_workers == 1:
            try:
                for branch in branches:
                    journal_prediction(predict_branch(branch))
            finally:
                prediction_partial_handle.flush()
                os.fsync(prediction_partial_handle.fileno())
                prediction_partial_handle.close()
        else:
            try:
                with ThreadPoolExecutor(max_workers=prediction_workers) as executor:
                    for prediction in executor.map(predict_branch, branches):
                        journal_prediction(prediction)
            finally:
                prediction_partial_handle.flush()
                os.fsync(prediction_partial_handle.fileno())
                prediction_partial_handle.close()
        os.replace(prediction_partial_path, prediction_final_path)
        prediction_count = len(predictions)

        stage = "rule_trace_audit"
        trace_audits = [
            audit_rule_trace(task_lookup[prediction.task_id], prediction)
            for prediction in predictions
        ]
        trace_audit_count = write_jsonl(run_dir / "trace_audits.jsonl", trace_audits)
        trace_metrics = aggregate_trace_audits(trace_audits, prompt_style)

        stage = "candidate_scoring"
        scored = score_candidates(tasks, branches, predictions, value_config)
        scored_count = write_jsonl(
            run_dir / "scored_candidates.jsonl", (item.to_dict() for item in scored)
        )

        stage = "task_decision_scoring"
        decisions = score_task_decisions(
            scored,
            tasks,
            harmful_regret_threshold=float(config["harmful_regret_threshold"]),
            high_text_fidelity_threshold=float(config["high_text_fidelity_threshold"]),
        )
        write_jsonl(run_dir / "task_decisions.jsonl", decisions)
        decision_metrics = aggregate_decision_metrics(decisions, scored)
        risk_metrics, risk_scores = evaluate_cross_fitted_risk(
            decisions,
            routing_budget=float(config["gate"]["routing_budget"]),
        )
        uncertainty = bootstrap_decision_uncertainty(
            decisions,
            resamples=int(config.get("bootstrap_resamples", 2000)),
            confidence=float(config.get("bootstrap_confidence", 0.95)),
            seed=int(config.get("bootstrap_seed", config["seed"])),
        )
        write_jsonl(run_dir / "risk_scores.jsonl", risk_scores)
        metrics = {
            "schema_version": config.get("schema_version", "unknown"),
            "provider": {
                "name": provider.name,
                "model": provider.model,
                "prompt_style": prompt_style,
            },
            "thresholds": {
                "harmful_regret": float(config["harmful_regret_threshold"]),
                "high_text_fidelity": float(config["high_text_fidelity_threshold"]),
            },
            "roundtrip_rate": roundtrip_rate,
            "decision": decision_metrics,
            "risk": risk_metrics,
            "uncertainty": uncertainty,
            "trace": trace_metrics,
        }
        atomic_write_json(run_dir / "metrics.json", metrics)

        stage = "gate_evaluation"
        gate_decision = evaluate_gate(metrics, config["gate"])
        atomic_write_json(run_dir / "gate_decision.json", gate_decision)
        atomic_write_text(run_dir / "gate_decision.md", gate_markdown(gate_decision))

        failures: list[dict[str, Any]] = []
        failures.extend(
            {
                "type": "prediction_failure",
                "task_id": item.task_id,
                "action_id": item.action_id,
                "error": item.error,
            }
            for item in predictions
            if not item.parse_ok
        )
        failures.extend(
            {
                "type": "harmful_flip",
                "task_id": item["task_id"],
                "domain": item["domain"],
                "selected_action_id": item["selected_action_id"],
                "selected_action_role": item["selected_action_role"],
                "regret": item["regret"],
                "selected_token_f1": item["selected_token_f1"],
            }
            for item in decisions
            if item["harmful_flip"]
        )
        write_jsonl(run_dir / "failures.jsonl", failures)
        repairs = [
            {
                "task_id": item.task_id,
                "domain": item.domain,
                "action_id": item.action_id,
                "repair": item.patch_repair,
            }
            for item in scored
            if item.patch_repair is not None
        ]
        write_jsonl(run_dir / "repairs.jsonl", repairs)

        stage = "manifest"
        finished_at = _utc_now()
        provider_manifest = {
            "name": provider.name,
            "model": provider.model,
            "prompt_style": prompt_style,
            "real_branch_exposure": expose_real_branch,
            "prediction_concurrency": prediction_workers,
            "credentials_recorded": False,
        }
        model_provenance = _model_provenance_from_environment()
        if model_provenance:
            provider_manifest["model_provenance"] = model_provenance
        runtime = {
            "python": sys.version,
            "platform": platform.platform(),
            "hostname": socket.gethostname(),
            "pid": os.getpid(),
        }
        scheduler = _scheduler_provenance_from_environment()
        if scheduler is not None:
            runtime["scheduler"] = scheduler
        manifest = {
            "schema_version": config.get("schema_version", "unknown"),
            "package_version": __version__,
            "run_status": "complete",
            "gate_status": gate_decision["status"],
            "started_at_utc": started_at,
            "finished_at_utc": finished_at,
            "config_path": str(config_path),
            "config_sha256": digest_file(config_path),
            "config_digest": digest_json(config),
            "run_dir": str(run_dir),
            "provider": provider_manifest,
            "counts": {
                "tasks": task_count,
                "real_branches": branch_count,
                "predictions": prediction_count,
                "trace_audits": trace_audit_count,
                "scored_candidates": scored_count,
                "failures": len(failures),
                "repairs": len(repairs),
            },
            "expected_candidate_count": sum(len(task.candidates) for task in tasks),
            "roundtrip_rate": roundtrip_rate,
            "runtime": runtime,
            "git_commit": _git_commit(project_root),
            "source_hashes": _source_hashes(project_root),
            "artifacts": _artifact_metadata(run_dir),
        }
        counts = manifest["counts"]
        expected = manifest["expected_candidate_count"]
        if not (
            counts["real_branches"]
            == counts["predictions"]
            == counts["trace_audits"]
            == counts["scored_candidates"]
            == expected
        ):
            raise RuntimeError(f"Artifact count mismatch: {counts}, expected={expected}")
        atomic_write_json(run_dir / "manifest.json", manifest)
        return {"manifest": manifest, "metrics": metrics, "gate_decision": gate_decision}
    except Exception as exc:
        blocker = "\n".join(
            [
                "# BLOCKER",
                "",
                f"- Time (UTC): `{_utc_now()}`",
                f"- Failed stage: `{stage}`",
                f"- Error: `{type(exc).__name__}: {exc}`",
                "",
                "The run stopped without changing any frozen threshold. Inspect existing artifacts before retrying in a new run directory.",
                "",
                "## Traceback",
                "",
                "```text",
                traceback.format_exc(),
                "```",
                "",
            ]
        )
        atomic_write_text(run_dir / "BLOCKER.md", blocker)
        raise
