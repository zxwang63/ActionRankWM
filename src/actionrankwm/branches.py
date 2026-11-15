from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from actionrankwm.envs.base import BaseEnvironment, make_environment
from actionrankwm.envs.sqlite_mcp import SQLiteMCPEnvironment
from actionrankwm.envs.terminal_fs import TerminalFSEnvironment
from actionrankwm.envs.toolsandbox import ToolSandboxEnvironment
from actionrankwm.io import digest_json
from actionrankwm.semantics import (
    apply_state_patch,
    canonicalize_predicted_observation,
    constraint_violations_for_state,
    target_success_for_state,
)
from actionrankwm.types import Action, BranchRecord, JSONDict, TaskSpec


@dataclass(frozen=True)
class ValueConfig:
    max_oracle_steps: int = 2
    action_cost: float = 0.08
    violation_penalty: float = 0.40
    invalid_penalty: float = 0.35


def _clip(value: float) -> float:
    return max(-1.0, min(1.0, value))


def _value(
    terminal_success: bool,
    executed_steps: int,
    violations: list[str],
    candidate_ok: bool,
    config: ValueConfig,
) -> float:
    value = 1.0 if terminal_success else 0.0
    value -= config.action_cost * executed_steps
    value -= config.violation_penalty * len(set(violations))
    if not candidate_ok:
        value -= config.invalid_penalty
    return _clip(value)


def evaluate_real_branch(task: TaskSpec, action: Action, config: ValueConfig) -> BranchRecord:
    environment = make_environment(task)
    try:
        candidate = environment.apply(action)
        immediate_violations = list(candidate.violations)
        executed_steps = 1
        for oracle_action in task.oracle_plan[: config.max_oracle_steps]:
            if environment.target_success():
                break
            environment.apply(oracle_action)
            executed_steps += 1
        terminal_success = environment.target_success()
        violations = sorted(set(immediate_violations + environment.violations()))
        return BranchRecord(
            task_id=task.task_id,
            domain=task.domain,
            action=action,
            initial_state_digest=digest_json(task.initial_state),
            real_observation=candidate.observation,
            real_state_patch=candidate.state_patch,
            candidate_ok=candidate.ok,
            immediate_success=candidate.target_success,
            terminal_success=terminal_success,
            violations=violations,
            executed_steps=executed_steps,
            real_value=_value(
                terminal_success=terminal_success,
                executed_steps=executed_steps,
                violations=violations,
                candidate_ok=candidate.ok,
                config=config,
            ),
            error=candidate.error,
        )
    finally:
        environment.close()


def _environment_from_state(task: TaskSpec, state: JSONDict) -> BaseEnvironment:
    if task.domain == "sqlite_mcp":
        return SQLiteMCPEnvironment(task, state=state)
    if task.domain == "terminal_fs":
        return TerminalFSEnvironment(task, state=state)
    if task.domain == "toolsandbox":
        return ToolSandboxEnvironment(task, state=state)
    raise ValueError(f"Unsupported domain: {task.domain}")


def evaluate_simulated_branch(
    task: TaskSpec,
    action: Action,
    predicted: Mapping[str, Any] | None,
    config: ValueConfig,
) -> tuple[float, bool, list[str], str | None]:
    observation, _ = canonicalize_predicted_observation(task, predicted)
    if observation is None:
        return _value(False, 1, [], False, config), False, [], "missing_observation"
    candidate_ok = bool(observation.get("ok", False))
    patch = observation.get("state_patch")
    if not isinstance(patch, Mapping):
        return _value(False, 1, [], False, config), False, [], "missing_state_patch"
    try:
        projected = apply_state_patch(task.initial_state, patch)
    except Exception as exc:
        return _value(False, 1, [], False, config), False, [], f"patch_error:{type(exc).__name__}:{exc}"

    try:
        immediate_success = target_success_for_state(task, projected)
        immediate_violations = constraint_violations_for_state(task, projected, patch)
        environment = _environment_from_state(task, projected)
        executed_steps = 1
        try:
            for oracle_action in task.oracle_plan[: config.max_oracle_steps]:
                if environment.target_success():
                    break
                environment.apply(oracle_action)
                executed_steps += 1
            terminal_success = environment.target_success()
            violations = sorted(set(immediate_violations + environment.violations()))
        finally:
            environment.close()
    except Exception as exc:
        return (
            _value(False, 1, [], False, config),
            False,
            [],
            f"projected_state_error:{type(exc).__name__}:{exc}",
        )
    return (
        _value(
            terminal_success=terminal_success,
            executed_steps=executed_steps,
            violations=violations,
            candidate_ok=candidate_ok,
            config=config,
        ),
        immediate_success,
        violations,
        None,
    )
