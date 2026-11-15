from __future__ import annotations

import platform
import socket
import sys
from collections.abc import Mapping
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
    digest_json,
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
from actionrankwm.types import BranchRecord, PredictionRecord, ScoredCandidate, TaskSpec


_REQUIRED = (
    "config.snapshot.json",
    "tasks.jsonl",
    "real_branches.jsonl",
    "predictions.jsonl",
    "manifest.json",
)

_SEMANTIC_CONFIG_KEYS = (
    "schema_version",
    "seed",
    "domains",
    "template_ids",
    "max_oracle_steps",
    "action_cost",
    "violation_penalty",
    "invalid_penalty",
    "harmful_regret_threshold",
    "high_text_fidelity_threshold",
    "bootstrap_confidence",
    "bootstrap_seed",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _semantic_contract(config: dict[str, Any]) -> dict[str, Any]:
    contract = {key: config.get(key) for key in _SEMANTIC_CONFIG_KEYS}
    contract["routing_budget"] = config.get("gate", {}).get("routing_budget")
    provider = config.get("provider", {})
    contract["provider_generation"] = {
        "temperature": provider.get("temperature"),
        "max_tokens": provider.get("max_tokens"),
        "prompt_style": provider.get("prompt_style", "direct"),
    }
    return contract


def _verify_source_artifacts(source: Path, manifest: Mapping[str, Any]) -> None:
    if manifest.get("run_status") != "complete":
        raise ValueError(f"Source run is not complete: {source}")
    artifact_records = manifest.get("artifacts")
    if not isinstance(artifact_records, Mapping):
        raise ValueError(f"Source manifest has no artifact records: {source}")
    for name in _REQUIRED[:-1]:
        record = artifact_records.get(name)
        if not isinstance(record, Mapping):
            raise ValueError(f"Source manifest is missing the {name} digest: {source}")
        expected_digest = record.get("sha256")
        expected_bytes = record.get("bytes")
        actual_path = source / name
        actual_digest = digest_file(actual_path)
        actual_bytes = actual_path.stat().st_size
        if expected_digest != actual_digest or expected_bytes != actual_bytes:
            raise ValueError(
                f"Source artifact integrity mismatch for {actual_path}; "
                f"manifest_sha256={expected_digest}, actual_sha256={actual_digest}, "
                f"manifest_bytes={expected_bytes}, actual_bytes={actual_bytes}"
            )


def _decision_counts(decisions: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "task_count": len(decisions),
        "unsafe_top_set_count": sum(
            bool(item["simulated_top_contains_harmful"]) for item in decisions
        ),
        "high_text_fidelity_unsafe_top_set_count": sum(
            bool(item["simulated_top_contains_high_fidelity_harmful"])
            for item in decisions
        ),
        "selected_harmful_flip_count": sum(
            bool(item["harmful_flip"]) for item in decisions
        ),
    }


def _case_analysis(
    decisions: list[dict[str, Any]], scored: list[ScoredCandidate]
) -> dict[str, Any]:
    candidate_lookup = {(item.task_id, item.action_id): item for item in scored}

    def grouped_counts(key: str) -> dict[str, Any]:
        values = sorted({str(item[key]) for item in decisions})
        return {
            value: _decision_counts(
                [item for item in decisions if str(item[key]) == value]
            )
            for value in values
        }

    qualifying_cases = []
    for decision in decisions:
        if not decision["simulated_top_contains_high_fidelity_harmful"]:
            continue
        members = []
        for action_id in decision["simulated_top_high_fidelity_harmful_action_ids"]:
            candidate = candidate_lookup[(decision["task_id"], action_id)]
            members.append(
                {
                    "action_id": action_id,
                    "action_role": candidate.action_role,
                    "real_value": candidate.real_value,
                    "simulated_value": candidate.simulated_value,
                    "real_regret": max(
                        0.0, float(decision["best_real_value"]) - candidate.real_value
                    ),
                    "token_f1": candidate.token_f1,
                    "structured_f1": candidate.structured_f1,
                    "critical_field_accuracy": candidate.critical_field_accuracy,
                }
            )
        qualifying_cases.append(
            {
                "task_id": decision["task_id"],
                "domain": decision["domain"],
                "template_id": decision["template_id"],
                "world_id": decision["world_id"],
                "group_id": decision["group_id"],
                "selected_action_id": decision["selected_action_id"],
                "selected_action_role": decision["selected_action_role"],
                "selected_regret": decision["regret"],
                "simulated_top_action_ids": decision["simulated_top_action_ids"],
                "simulated_top_expected_regret": decision[
                    "simulated_top_expected_regret"
                ],
                "simulated_top_worst_regret": decision["simulated_top_worst_regret"],
                "harmful_members": members,
            }
        )
    return {
        "overall": _decision_counts(decisions),
        "by_domain": grouped_counts("domain"),
        "by_template": grouped_counts("template_id"),
        "by_group": grouped_counts("group_id"),
        "high_text_fidelity_unsafe_cases": qualifying_cases,
        "selected_harmful_task_ids": [
            item["task_id"] for item in decisions if item["harmful_flip"]
        ],
    }


def _case_analysis_markdown(analysis: Mapping[str, Any]) -> str:
    overall = analysis["overall"]
    lines = [
        "# Collection case analysis",
        "",
        f"- Tasks: {overall['task_count']}",
        f"- Unsafe top sets: {overall['unsafe_top_set_count']}",
        "- High-token-F1 unsafe top sets: "
        f"{overall['high_text_fidelity_unsafe_top_set_count']}",
        f"- Selected harmful flips: {overall['selected_harmful_flip_count']}",
        "",
        "## High-token-F1 unsafe cases",
        "",
    ]
    cases = analysis["high_text_fidelity_unsafe_cases"]
    if not cases:
        lines.append("None.")
    else:
        lines.extend(
            [
                "| Task | Domain | Template | World | Harmful member(s) | Max token F1 | Expected / worst top-set regret | Selected |",
                "|---|---|---|---|---|---:|---:|---|",
            ]
        )
        for case in cases:
            members = case["harmful_members"]
            member_ids = ", ".join(item["action_id"] for item in members)
            max_token_f1 = max(item["token_f1"] for item in members)
            lines.append(
                f"| `{case['task_id']}` | `{case['domain']}` | "
                f"`{case['template_id']}` | `{case['world_id']}` | "
                f"`{member_ids}` | {max_token_f1:.4f} | "
                f"{case['simulated_top_expected_regret']:.4f} / "
                f"{case['simulated_top_worst_regret']:.4f} | "
                f"`{case['selected_action_id']}` (`{case['selected_action_role']}`) |"
            )
    lines.append("")
    return "\n".join(lines)


def analyze_collection(
    source_runs: list[str | Path],
    output_dir: str | Path,
    *,
    model_revision: str,
) -> dict[str, Any]:
    if len(source_runs) < 2:
        raise ValueError("A collection requires at least two source runs")
    if not model_revision.strip():
        raise ValueError("model_revision is required for collection provenance")
    sources = [Path(path).resolve() for path in source_runs]
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite collection directory: {output}")
    for source in sources:
        missing = [name for name in _REQUIRED if not (source / name).is_file()]
        if missing:
            raise FileNotFoundError(f"Source run {source} is missing artifacts: {missing}")

    configs = [read_json(source / "config.snapshot.json") for source in sources]
    contracts = [_semantic_contract(config) for config in configs]
    if any(contract != contracts[0] for contract in contracts[1:]):
        raise ValueError("Source runs have different semantic experiment contracts")

    manifests = [read_json(source / "manifest.json") for source in sources]
    for source, manifest in zip(sources, manifests):
        _verify_source_artifacts(source, manifest)
    provider_names = {
        str(manifest.get("provider", {}).get("name")) for manifest in manifests
    }
    if len(provider_names) != 1:
        raise ValueError(f"Source provider mismatch: {sorted(provider_names)}")
    recorded_revisions = {
        str(revision)
        for manifest in manifests
        if (
            revision := manifest.get("provider", {})
            .get("model_provenance", {})
            .get("revision")
        )
    }
    if recorded_revisions and recorded_revisions != {model_revision}:
        raise ValueError(
            "Explicit model revision does not match source manifests: "
            f"explicit={model_revision}, recorded={sorted(recorded_revisions)}"
        )

    tasks: list[TaskSpec] = []
    branches: list[BranchRecord] = []
    predictions: list[PredictionRecord] = []
    for source in sources:
        tasks.extend(TaskSpec.from_dict(item) for item in iter_jsonl(source / "tasks.jsonl"))
        branches.extend(
            BranchRecord.from_dict(item)
            for item in iter_jsonl(source / "real_branches.jsonl")
        )
        predictions.extend(
            PredictionRecord.from_dict(item)
            for item in iter_jsonl(source / "predictions.jsonl")
        )
    task_ids = [task.task_id for task in tasks]
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("Collection source runs contain duplicate task IDs")

    config = dict(configs[0])
    config["collection_source_count"] = len(sources)
    config["collection_model_revision"] = model_revision
    config["bootstrap_resamples"] = max(
        int(item.get("bootstrap_resamples", 2000)) for item in configs
    )
    config["collection_bootstrap_resamples_rule"] = "maximum_source_resamples"
    config["gate"] = dict(config["gate"])
    config["gate"]["scientific"] = False
    value_config = ValueConfig(
        max_oracle_steps=int(config.get("max_oracle_steps", 2)),
        action_cost=float(config.get("action_cost", 0.08)),
        violation_penalty=float(config.get("violation_penalty", 0.40)),
        invalid_penalty=float(config.get("invalid_penalty", 0.35)),
    )
    scored = score_candidates(tasks, branches, predictions, value_config)
    task_lookup = {task.task_id: task for task in tasks}
    prompt_style = str(
        config.get("provider", {}).get("prompt_style", "direct")
    )
    trace_audits = [
        audit_rule_trace(task_lookup[prediction.task_id], prediction)
        for prediction in predictions
    ]
    trace_metrics = aggregate_trace_audits(trace_audits, prompt_style)
    expected = sum(len(task.candidates) for task in tasks)
    if not (len(branches) == len(predictions) == len(scored) == expected):
        raise RuntimeError("Collection artifact count mismatch")
    decisions = score_task_decisions(
        scored,
        tasks,
        harmful_regret_threshold=float(config["harmful_regret_threshold"]),
        high_text_fidelity_threshold=float(config["high_text_fidelity_threshold"]),
    )
    case_analysis = _case_analysis(decisions, scored)
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
    source_providers = [manifest.get("provider", {}) for manifest in manifests]
    metrics = {
        "schema_version": config.get("schema_version", "unknown"),
        "run_type": "collection_rescore",
        "source_runs": [str(source) for source in sources],
        "source_providers": source_providers,
        "model_revision": model_revision,
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

    output.mkdir(parents=True)
    atomic_write_json(output / "config.snapshot.json", config)
    write_jsonl(output / "tasks.jsonl", (item.to_dict() for item in tasks))
    write_jsonl(output / "real_branches.jsonl", (item.to_dict() for item in branches))
    write_jsonl(output / "predictions.jsonl", (item.to_dict() for item in predictions))
    write_jsonl(output / "scored_candidates.jsonl", (item.to_dict() for item in scored))
    write_jsonl(output / "task_decisions.jsonl", decisions)
    write_jsonl(output / "trace_audits.jsonl", trace_audits)
    write_jsonl(output / "risk_scores.jsonl", risk_scores)
    write_jsonl(output / "repairs.jsonl", repairs)
    write_jsonl(output / "failures.jsonl", failures)
    atomic_write_json(output / "metrics.json", metrics)
    atomic_write_json(output / "gate_decision.json", gate_decision)
    atomic_write_text(output / "gate_decision.md", gate_markdown(gate_decision))
    atomic_write_json(output / "case_analysis.json", case_analysis)
    atomic_write_text(
        output / "case_analysis.md", _case_analysis_markdown(case_analysis)
    )

    project_root = Path(__file__).resolve().parents[2]
    source_records = []
    for source, manifest in zip(sources, manifests):
        source_records.append(
            {
                "path": str(source),
                "manifest_sha256": digest_file(source / "manifest.json"),
                "provider": manifest.get("provider"),
                "artifacts": {
                    name: {
                        "sha256": digest_file(source / name),
                        "bytes": (source / name).stat().st_size,
                    }
                    for name in _REQUIRED[:-1]
                },
            }
        )
    manifest = {
        "schema_version": config.get("schema_version", "unknown"),
        "package_version": __version__,
        "run_status": "complete",
        "run_type": "collection_rescore",
        "created_at_utc": _utc_now(),
        "output_dir": str(output),
        "model_revision": model_revision,
        "semantic_contract_sha256": digest_json(contracts[0]),
        "source_runs": source_records,
        "source_providers": source_providers,
        "gate_status": gate_decision["status"],
        "counts": {
            "tasks": len(tasks),
            "real_branches": len(branches),
            "predictions": len(predictions),
            "trace_audits": len(trace_audits),
            "scored_candidates": len(scored),
            "repairs": len(repairs),
            "failures": len(failures),
            "high_text_fidelity_unsafe_cases": len(
                case_analysis["high_text_fidelity_unsafe_cases"]
            ),
        },
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "hostname": socket.gethostname(),
        },
        "git_commit": _git_commit(project_root),
        "rescorer_source_hashes": _source_hashes(project_root),
        "artifacts": _artifact_metadata(output),
    }
    atomic_write_json(output / "manifest.json", manifest)
    return {"manifest": manifest, "metrics": metrics, "gate_decision": gate_decision}
