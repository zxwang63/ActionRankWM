from __future__ import annotations

import os
import platform
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from actionrankwm import __version__
from actionrankwm.branches import ValueConfig
from actionrankwm.gate import evaluate_gate, gate_markdown
from actionrankwm.io import (
    atomic_write_json,
    atomic_write_text,
    digest_file,
    iter_jsonl,
    read_json,
    write_jsonl,
)
from actionrankwm.metrics.decision import aggregate_decision_metrics, score_task_decisions
from actionrankwm.metrics.risk import evaluate_cross_fitted_risk
from actionrankwm.metrics.uncertainty import bootstrap_decision_uncertainty
from actionrankwm.pipeline import _artifact_metadata, _git_commit, _source_hashes
from actionrankwm.scoring import score_candidates
from actionrankwm.trace import aggregate_trace_audits, audit_rule_trace
from actionrankwm.types import BranchRecord, PredictionRecord, TaskSpec


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def rescore_run(source_run: str | Path, output_dir: str | Path) -> dict[str, Any]:
    source_run = Path(source_run).resolve()
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite rescore directory: {output_dir}")
    required = (
        "config.snapshot.json",
        "tasks.jsonl",
        "real_branches.jsonl",
        "predictions.jsonl",
    )
    missing = [name for name in required if not (source_run / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Source run is missing artifacts: {missing}")
    output_dir.mkdir(parents=True)
    config = read_json(source_run / "config.snapshot.json")
    source_manifest = (
        read_json(source_run / "manifest.json")
        if (source_run / "manifest.json").is_file()
        else {"run_status": "blocked", "provider": None}
    )
    tasks = [TaskSpec.from_dict(value) for value in iter_jsonl(source_run / "tasks.jsonl")]
    branches = [
        BranchRecord.from_dict(value) for value in iter_jsonl(source_run / "real_branches.jsonl")
    ]
    predictions = [
        PredictionRecord.from_dict(value) for value in iter_jsonl(source_run / "predictions.jsonl")
    ]
    task_lookup = {task.task_id: task for task in tasks}
    prompt_style = str(
        config.get("provider", {}).get("prompt_style", "direct")
    )
    trace_audits = [
        audit_rule_trace(task_lookup[prediction.task_id], prediction)
        for prediction in predictions
    ]
    trace_metrics = aggregate_trace_audits(trace_audits, prompt_style)
    source_provider = source_manifest.get("provider")
    if source_provider is None and predictions:
        source_provider = {
            "name": predictions[0].provider,
            "model": predictions[0].model,
            "recovered_from_blocked_run": True,
        }
    value_config = ValueConfig(
        max_oracle_steps=int(config.get("max_oracle_steps", 2)),
        action_cost=float(config.get("action_cost", 0.08)),
        violation_penalty=float(config.get("violation_penalty", 0.40)),
        invalid_penalty=float(config.get("invalid_penalty", 0.35)),
    )
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
        resamples=int(config.get("bootstrap_resamples", 2000)),
        confidence=float(config.get("bootstrap_confidence", 0.95)),
        seed=int(config.get("bootstrap_seed", config["seed"])),
    )
    metrics = {
        "schema_version": config.get("schema_version", "unknown"),
        "run_type": "offline_rescore",
        "source_run": str(source_run),
        "source_provider": source_provider,
        "thresholds": {
            "harmful_regret": float(config["harmful_regret_threshold"]),
            "high_text_fidelity": float(config["high_text_fidelity_threshold"]),
        },
        "decision": decision_metrics,
        "risk": risk_metrics,
        "uncertainty": uncertainty,
        "trace": trace_metrics,
    }
    gate_decision = evaluate_gate(metrics, config["gate"])
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
    failures = [
        {
            "type": "prediction_failure",
            "task_id": item.task_id,
            "action_id": item.action_id,
            "error": item.prediction_error,
        }
        for item in scored
        if not item.parse_ok or item.prediction_error is not None
    ]
    failures.extend(
        {
            "type": "harmful_flip",
            "task_id": item["task_id"],
            "domain": item["domain"],
            "selected_action_id": item["selected_action_id"],
            "regret": item["regret"],
        }
        for item in decisions
        if item["harmful_flip"]
    )
    atomic_write_json(output_dir / "config.snapshot.json", config)
    write_jsonl(output_dir / "scored_candidates.jsonl", (item.to_dict() for item in scored))
    write_jsonl(output_dir / "task_decisions.jsonl", decisions)
    write_jsonl(output_dir / "trace_audits.jsonl", trace_audits)
    write_jsonl(output_dir / "risk_scores.jsonl", risk_scores)
    write_jsonl(output_dir / "repairs.jsonl", repairs)
    write_jsonl(output_dir / "failures.jsonl", failures)
    atomic_write_json(output_dir / "metrics.json", metrics)
    atomic_write_json(output_dir / "gate_decision.json", gate_decision)
    atomic_write_text(output_dir / "gate_decision.md", gate_markdown(gate_decision))

    project_root = Path(__file__).resolve().parents[2]
    manifest = {
        "schema_version": config.get("schema_version", "unknown"),
        "package_version": __version__,
        "run_status": "complete",
        "run_type": "offline_rescore",
        "created_at_utc": _utc_now(),
        "output_dir": str(output_dir),
        "source_run": str(source_run),
        "source_manifest_sha256": (
            digest_file(source_run / "manifest.json")
            if (source_run / "manifest.json").is_file()
            else None
        ),
        "source_artifacts": {
            name: {
                "sha256": digest_file(source_run / name),
                "bytes": (source_run / name).stat().st_size,
            }
            for name in required
        },
        "source_blocker_sha256": (
            digest_file(source_run / "BLOCKER.md")
            if (source_run / "BLOCKER.md").is_file()
            else None
        ),
        "source_provider": source_provider,
        "gate_status": gate_decision["status"],
        "counts": {
            "tasks": len(tasks),
            "real_branches": len(branches),
            "predictions": len(predictions),
            "trace_audits": len(trace_audits),
            "scored_candidates": len(scored),
            "repairs": len(repairs),
            "failures": len(failures),
        },
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "hostname": socket.gethostname(),
            "pid": os.getpid(),
        },
        "git_commit": _git_commit(project_root),
        "rescorer_source_hashes": _source_hashes(project_root),
        "artifacts": _artifact_metadata(output_dir),
    }
    expected = len(branches)
    if len(predictions) != expected or len(scored) != expected:
        raise RuntimeError(
            f"Rescore count mismatch: branches={len(branches)}, predictions={len(predictions)}, scored={len(scored)}"
        )
    atomic_write_json(output_dir / "manifest.json", manifest)
    return {"manifest": manifest, "metrics": metrics, "gate_decision": gate_decision}
