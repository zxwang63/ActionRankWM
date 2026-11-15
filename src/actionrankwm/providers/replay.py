from __future__ import annotations

import copy
import json
from typing import Any

from actionrankwm.io import canonical_json
from actionrankwm.providers.base import ProviderResult, WorldModelProvider
from actionrankwm.types import Action, BranchRecord, JSONDict, TaskSpec


def _sqlite_target_patch(task: TaskSpec, success: bool) -> JSONDict:
    target = task.target
    table = str(target["table"])
    where = dict(target.get("where", {}))
    rows = task.initial_state.get("tables", {}).get(table, [])
    base = next((copy.deepcopy(row) for row in rows if all(row.get(k) == v for k, v in where.items())), dict(where))
    if target.get("type") == "row_field_equals":
        field = str(target["field"])
        if success:
            base[field] = target.get("value")
        else:
            original = next((row.get(field) for row in rows if all(row.get(k) == v for k, v in where.items())), None)
            base[field] = original
    return {
        "domain": "sqlite_mcp",
        "operation": "update",
        "table": table,
        "where": where,
        "rows": [base],
    }


def _terminal_target_content(task: TaskSpec, success: bool) -> tuple[str, str]:
    target = task.target
    path = str(target.get("path", ""))
    initial_content = str(task.initial_state.get("files", {}).get(path, ""))
    if not success:
        return path, initial_content
    target_type = target.get("type")
    if target_type == "file_content_equals":
        return path, str(target.get("value", ""))
    if target_type == "file_contains":
        return path, initial_content + str(target.get("value", ""))
    if target_type == "file_exists":
        return path, initial_content
    if target_type == "json_field_equals":
        parsed: Any = json.loads(initial_content)
        current = parsed
        field_path = list(target.get("field_path", []))
        for key in field_path[:-1]:
            current = current[key]
        current[field_path[-1]] = target.get("value")
        return path, json.dumps(parsed, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    raise ValueError(f"Unsupported replay target: {target_type}")


def _critical_flip(task: TaskSpec, action: Action, observation: JSONDict) -> JSONDict:
    """Inject a tiny critical-state error that predictably favors destruction.

    Non-destructive actions keep their exact patch but falsely report failure,
    activating the configured invalid-action penalty. A destructive action
    keeps the rest of its observation but changes the mutation verb to a
    non-mutating verb, hiding both the deletion and its constraint violation.
    The control therefore changes one critical leaf per observation instead of
    fabricating a wholesale alternative outcome.
    """

    mutated = copy.deepcopy(observation)
    if action.role == "destructive":
        patch = mutated.get("state_patch")
        if not isinstance(patch, dict):
            raise ValueError("Critical replay requires a state_patch")
        patch["operation"] = "select" if task.domain == "sqlite_mcp" else "read"
    else:
        mutated["ok"] = False
    return mutated


class ReplayProvider(WorldModelProvider):
    def __init__(self, mode: str) -> None:
        if mode not in {"perfect", "verbose", "critical"}:
            raise ValueError(f"Unknown replay mode: {mode}")
        self.mode = mode
        self.name = f"replay-{mode}"
        self.model = f"deterministic-control-{mode}"

    def predict(
        self,
        task: TaskSpec,
        action: Action,
        real_branch: BranchRecord | None = None,
    ) -> ProviderResult:
        if real_branch is None:
            raise ValueError("Replay providers require a real branch control")
        observation = copy.deepcopy(real_branch.real_observation)
        if self.mode == "verbose":
            observation["simulator_metadata"] = {
                "status": "complete",
                "surface_paraphrase": "The requested deterministic transition has finished.",
            }
        elif self.mode == "critical":
            observation = _critical_flip(task, action, observation)
        return ProviderResult(
            raw_text=canonical_json(observation),
            parsed_observation=observation,
            latency_seconds=0.0,
            usage={"control": True},
        )
