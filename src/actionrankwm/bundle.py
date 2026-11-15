from __future__ import annotations

import os
import platform
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from actionrankwm import __version__
from actionrankwm.io import (
    atomic_write_json,
    canonical_json,
    digest_file,
    digest_json,
    iter_jsonl,
    read_json,
    write_jsonl,
)
from actionrankwm.providers.prompts import build_messages
from actionrankwm.types import TaskSpec


BUNDLE_SCHEMA = "actionrankwm.model_inputs.v1"
_JOIN_NAMESPACE = "actionrankwm-v091-label-blind-join"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def opaque_join_id(task_id: str, action_id: str) -> str:
    """Return a stable opaque key without exposing semantic task identifiers."""

    return digest_json(
        {"namespace": _JOIN_NAMESPACE, "task_id": task_id, "action_id": action_id}
    )


def _prompt_audit(task: TaskSpec, action_index: int, messages: list[dict[str, Any]]) -> None:
    action = task.candidates[action_index]
    if len(messages) != 2 or [item.get("role") for item in messages] != ["system", "user"]:
        raise ValueError(f"Unexpected message envelope for {task.task_id}/{action.action_id}")
    import json

    user = json.loads(str(messages[1]["content"]))
    expected_user = {
        "domain": task.domain,
        "current_state": task.initial_state,
        "action": {"tool": action.tool, "arguments": action.arguments},
        "tool_docs": task.metadata.get("tool_docs", {}),
        "benchmark_context": {
            "current_timestamp": task.metadata.get("frozen_timestamp")
        },
    }
    if user != expected_user:
        raise ValueError(f"Model-visible projection mismatch for {task.task_id}/{action.action_id}")
    if set(user) != {
        "domain",
        "current_state",
        "action",
        "tool_docs",
        "benchmark_context",
    }:
        raise ValueError(f"Unexpected model-visible keys for {task.task_id}/{action.action_id}")
    if set(user["action"]) != {"tool", "arguments"}:
        raise ValueError(f"Unexpected model-visible action keys for {task.task_id}/{action.action_id}")
    serialized = canonical_json(messages)
    if task.task_id in serialized:
        raise ValueError(f"Semantic task identifier leaked for {task.task_id}/{action.action_id}")
    forbidden_keys = (
        '"action_id":',
        '"candidate",',
        '"constraints":',
        '"oracle_plan":',
        '"real_observation":',
        '"real_value":',
        '"target":',
    )
    leaked = [key for key in forbidden_keys if key in serialized]
    if leaked:
        raise ValueError(
            f"Hidden supervision keys leaked for {task.task_id}/{action.action_id}: {leaked}"
        )


def build_model_inputs(
    tasks: Sequence[TaskSpec], prompt_style: str = "direct"
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for task in tasks:
        if task.domain != "toolsandbox":
            raise ValueError(f"Label-blind external bundle only supports toolsandbox, got {task.domain}")
        for action_index, action in enumerate(task.candidates):
            messages = build_messages(task, action, prompt_style)
            _prompt_audit(task, action_index, messages)
            join_id = opaque_join_id(task.task_id, action.action_id)
            if join_id in seen:
                raise ValueError(f"Duplicate opaque join ID: {join_id}")
            seen.add(join_id)
            records.append(
                {
                    "schema": BUNDLE_SCHEMA,
                    "join_id": join_id,
                    "messages": messages,
                    "prompt_sha256": digest_json(messages),
                }
            )
    return records


def _verified_preflight(preflight_dir: Path) -> dict[str, Any]:
    manifest_path = preflight_dir / "manifest.json"
    manifest = read_json(manifest_path)
    if not manifest.get("passed"):
        raise ValueError(f"Preflight did not pass: {manifest_path}")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError("Preflight manifest has no artifact inventory")
    for name, metadata in artifacts.items():
        path = preflight_dir / str(name)
        if not path.is_file():
            raise FileNotFoundError(path)
        expected = str(metadata["sha256"])
        actual = digest_file(path)
        if actual != expected:
            raise ValueError(f"Preflight artifact hash mismatch for {name}: {actual} != {expected}")
    branch_a = str(artifacts["branches_pass_a.jsonl"]["sha256"])
    branch_b = str(artifacts["branches_pass_b.jsonl"]["sha256"])
    if branch_a != branch_b:
        raise ValueError("Preflight branch passes are not byte-identical")
    return manifest


def export_model_bundle(
    preflight_dir: str | Path,
    config_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    preflight_dir = Path(preflight_dir).resolve()
    config_path = Path(config_path).resolve()
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite model bundle: {output_dir}")
    output_dir.mkdir(parents=True)

    preflight = _verified_preflight(preflight_dir)
    config = read_json(config_path)
    prompt_style = str(config.get("provider", {}).get("prompt_style", "direct"))
    tasks_path = preflight_dir / "tasks.jsonl"
    tasks = [TaskSpec.from_dict(row) for row in iter_jsonl(tasks_path)]
    if len(tasks) != int(preflight["task_count"]):
        raise ValueError("Task count does not match frozen preflight manifest")
    inputs = build_model_inputs(tasks, prompt_style)
    expected = sum(len(task.candidates) for task in tasks)
    if len(inputs) != expected or expected != int(preflight["candidate_count"]):
        raise ValueError("Candidate count does not match frozen preflight manifest")

    provider_config = dict(config.get("provider", {}))
    provider_config.pop("prompt_style", None)
    provider_config.setdefault("concurrency", 1)
    provider_config.setdefault("max_tokens", 1536)
    provider_config.setdefault("timeout_seconds", 180)
    provider_config.setdefault("max_retries", 1)
    atomic_write_json(output_dir / "provider_config.json", provider_config)
    write_jsonl(output_dir / "model_inputs.jsonl", inputs)

    manifest = {
        "schema": BUNDLE_SCHEMA,
        "package_version": __version__,
        "created_at_utc": utc_now(),
        "purpose": "label-blind model inference only",
        "prompt_style": prompt_style,
        "counts": {"tasks": len(tasks), "requests": len(inputs)},
        "source": {
            "preflight_dir": str(preflight_dir),
            "preflight_manifest_sha256": digest_file(preflight_dir / "manifest.json"),
            "tasks_sha256": digest_file(tasks_path),
            "branches_sha256": preflight["artifacts"]["branches_pass_a.jsonl"]["sha256"],
            "config_sha256": digest_file(config_path),
        },
        "isolation": {
            "model_receives_real_branches": False,
            "model_receives_task_or_action_ids": False,
            "model_receives_targets": False,
            "model_receives_constraints": False,
            "model_receives_oracle_plans": False,
            "model_receives_candidate_roles_or_risks": False,
            "model_receives_real_values": False,
            "opaque_join_ids_are_sent_to_model": False,
            "visible_user_keys": [
                "domain",
                "current_state",
                "action",
                "tool_docs",
                "benchmark_context",
            ],
        },
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "hostname": socket.gethostname(),
            "pid": os.getpid(),
        },
        "artifacts": {
            name: {"sha256": digest_file(output_dir / name), "bytes": (output_dir / name).stat().st_size}
            for name in ("model_inputs.jsonl", "provider_config.json")
        },
    }
    atomic_write_json(output_dir / "manifest.json", manifest)
    return manifest
