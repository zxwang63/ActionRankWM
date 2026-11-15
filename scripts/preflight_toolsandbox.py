#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any

from actionrankwm import __version__
from actionrankwm.branches import ValueConfig, evaluate_real_branch, evaluate_simulated_branch
from actionrankwm.io import (
    atomic_write_json,
    atomic_write_text,
    canonical_json,
    digest_file,
    write_jsonl,
)
from actionrankwm.providers.prompts import build_messages
from actionrankwm.tasks import generate_tasks
from actionrankwm.toolsandbox_tasks import TEMPLATES, UPSTREAM_COMMIT
from actionrankwm.types import BranchRecord, TaskSpec


TEMPLATE_IDS = [template_id for template_id, _ in TEMPLATES]
NETWORK_TOOL_FRAGMENTS = ("search_stock", "search_weather", "search_holiday", "search_location")
SPLIT_INDICES = {
    "e4a": [0, 1],
    "e4b": list(range(10, 30)),
    "e4c": list(range(100, 150)),
}


def _git_revision(path: Path) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _branch_pass(tasks: list[TaskSpec], value_config: ValueConfig) -> list[BranchRecord]:
    return [
        evaluate_real_branch(task, action, value_config)
        for task in tasks
        for action in task.candidates
    ]


def _dependency_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package in (
        "tool-sandbox",
        "polars",
        "dill",
        "numpy",
        "scipy",
        "ccy",
        "phonenumbers",
    ):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _prompt_contract(tasks: list[TaskSpec]) -> dict[str, Any]:
    failures: list[dict[str, Any]] = []
    allowed_user_keys = {
        "domain",
        "current_state",
        "action",
        "tool_docs",
        "benchmark_context",
    }
    allowed_action_keys = {"tool", "arguments"}
    checked = 0
    for task in tasks:
        for action in task.candidates:
            messages = build_messages(task, action)
            user = json.loads(messages[-1]["content"])
            reasons: list[str] = []
            if set(user) != allowed_user_keys:
                reasons.append(f"user_keys:{sorted(user)}")
            if set(user.get("action", {})) != allowed_action_keys:
                reasons.append(f"action_keys:{sorted(user.get('action', {}))}")
            serialized = canonical_json(user)
            forbidden_literals = (
                '"real_value"',
                '"terminal_success"',
                '"constraints"',
                '"oracle_plan"',
                '"role"',
                '"risk"',
                "toolsandbox_low_battery_cascade_",
                "toolsandbox_low_battery_permission_",
                "toolsandbox_cellular_message_dependency_",
                "toolsandbox_self_identity_cardinality_",
                "toolsandbox_duplicate_contact_semantics_",
                "toolsandbox_idempotent_setting_error_",
            )
            if any(literal in serialized for literal in forbidden_literals):
                reasons.append("forbidden_supervision_field")
            if reasons:
                failures.append(
                    {
                        "task_id": task.task_id,
                        "action_id": action.action_id,
                        "reasons": reasons,
                    }
                )
            checked += 1
    return {"checked": checked, "failure_count": len(failures), "failures": failures}


def _evaluate_controls(
    tasks: list[TaskSpec],
    first: list[BranchRecord],
    second: list[BranchRecord],
    value_config: ValueConfig,
) -> dict[str, Any]:
    first_lines = [canonical_json(branch.to_dict()) for branch in first]
    second_lines = [canonical_json(branch.to_dict()) for branch in second]
    nondeterministic = [
        {
            "index": index,
            "task_id": first[index].task_id,
            "action_id": first[index].action.action_id,
        }
        for index, (left, right) in enumerate(zip(first_lines, second_lines))
        if left != right
    ]
    if len(first_lines) != len(second_lines):
        nondeterministic.append(
            {"reason": "branch_count_mismatch", "first": len(first_lines), "second": len(second_lines)}
        )

    task_lookup = {task.task_id: task for task in tasks}
    roundtrip_failures: list[dict[str, Any]] = []
    for branch in first:
        simulated, _, _, error = evaluate_simulated_branch(
            task_lookup[branch.task_id], branch.action, branch.real_observation, value_config
        )
        if error is not None or abs(simulated - branch.real_value) > 1e-9:
            roundtrip_failures.append(
                {
                    "task_id": branch.task_id,
                    "action_id": branch.action.action_id,
                    "real_value": branch.real_value,
                    "simulated_value": simulated,
                    "error": error,
                }
            )

    by_task: dict[str, list[BranchRecord]] = {}
    for branch in first:
        by_task.setdefault(branch.task_id, []).append(branch)
    value_failures: list[dict[str, Any]] = []
    value_rows: list[dict[str, Any]] = []
    for task in tasks:
        branches = by_task[task.task_id]
        best = max(branch.real_value for branch in branches)
        oracle = next(branch for branch in branches if branch.action.role == "oracle")
        critical = next(
            branch for branch in branches if branch.action.role == "mechanism_critical"
        )
        gap = oracle.real_value - critical.real_value
        row = {
            "task_id": task.task_id,
            "mechanism": task.metadata["mechanism"],
            "oracle_value": oracle.real_value,
            "critical_value": critical.real_value,
            "gap": gap,
            "best_value": best,
            "oracle_in_top_set": abs(oracle.real_value - best) <= 1e-12,
        }
        value_rows.append(row)
        if not row["oracle_in_top_set"] or gap < 0.20:
            value_failures.append(row)

    network_tools = sorted(
        {
            action.tool
            for task in tasks
            for action in [*task.candidates, *task.oracle_plan]
            if any(fragment in action.tool for fragment in NETWORK_TOOL_FRAGMENTS)
        }
    )
    prompt = _prompt_contract(tasks)
    return {
        "determinism": {
            "passed": not nondeterministic,
            "compared": min(len(first_lines), len(second_lines)),
            "failure_count": len(nondeterministic),
            "failures": nondeterministic,
        },
        "roundtrip": {
            "passed": not roundtrip_failures,
            "rate": 1.0 - len(roundtrip_failures) / len(first) if first else 0.0,
            "failure_count": len(roundtrip_failures),
            "failures": roundtrip_failures,
        },
        "value_gaps": {
            "passed": not value_failures,
            "minimum_required": 0.20,
            "failure_count": len(value_failures),
            "failures": value_failures,
            "rows": value_rows,
        },
        "prompt_contract": {"passed": prompt["failure_count"] == 0, **prompt},
        "offline_tools": {
            "passed": not network_tools,
            "network_tools": network_tools,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--source-root", default="third_party/ToolSandbox")
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument("--split", choices=tuple(SPLIT_INDICES), default="e4a")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    if run_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {run_dir}")
    run_dir.mkdir(parents=True)
    try:
        source_root = Path(args.source_root).resolve()
        revision = _git_revision(source_root)
        if revision != UPSTREAM_COMMIT:
            raise RuntimeError(
                f"ToolSandbox revision mismatch: expected {UPSTREAM_COMMIT}, found {revision}"
            )
        task_indices = SPLIT_INDICES[args.split]
        tasks = generate_tasks(
            domains=["toolsandbox"],
            tasks_per_template=len(task_indices),
            seed=args.seed,
            template_ids=TEMPLATE_IDS,
            task_indices=task_indices,
        )
        value_config = ValueConfig()
        first = _branch_pass(tasks, value_config)
        second = _branch_pass(tasks, value_config)
        controls = _evaluate_controls(tasks, first, second, value_config)

        write_jsonl(run_dir / "tasks.jsonl", (task.to_dict() for task in tasks))
        write_jsonl(run_dir / "branches_pass_a.jsonl", (branch.to_dict() for branch in first))
        write_jsonl(run_dir / "branches_pass_b.jsonl", (branch.to_dict() for branch in second))
        atomic_write_json(run_dir / "controls.json", controls)
        manifest = {
            "protocol": "external_validity_v091",
            "package_version": __version__,
            "parent_protocol": "external_validity_v09",
            "pre_model_amendment": "remove model-visible task_id and action_id",
            "upstream": {
                "repository": "https://github.com/apple/ToolSandbox",
                "revision": revision,
            },
            "python": {"version": platform.python_version(), "executable": sys.executable},
            "dependencies": _dependency_versions(),
            "task_count": len(tasks),
            "candidate_count": len(first),
            "split": args.split.upper(),
            "task_indices": task_indices,
            "seed": args.seed,
            "artifacts": {
                path.name: {"sha256": digest_file(path), "bytes": path.stat().st_size}
                for path in sorted(run_dir.iterdir())
                if path.is_file()
            },
        }
        passed = all(
            section.get("passed", False)
            for section in controls.values()
            if isinstance(section, dict)
        )
        manifest["passed"] = passed
        atomic_write_json(run_dir / "manifest.json", manifest)
        if not passed:
            atomic_write_text(
                run_dir / "BLOCKER.md",
                "# BLOCKER\n\nToolSandbox v0.9.1 deterministic preflight failed. "
                "See `controls.json`. No learned-model call was made.\n",
            )
            print("FAILED")
            return 1
        print(f"PASSED: {len(tasks)} tasks, {len(first)} branches")
        return 0
    except Exception as exc:
        atomic_write_text(
            run_dir / "BLOCKER.md",
            "# BLOCKER\n\n"
            f"- Error: `{type(exc).__name__}: {exc}`\n\n"
            "No learned-model call was made.\n\n"
            "```text\n"
            f"{traceback.format_exc()}"
            "```\n",
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
