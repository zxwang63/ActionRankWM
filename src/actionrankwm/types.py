from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


JSONDict = dict[str, Any]


@dataclass(frozen=True)
class Action:
    action_id: str
    tool: str
    arguments: JSONDict
    description: str = ""
    role: str = "candidate"
    risk: str = "low"

    def to_dict(self) -> JSONDict:
        return {
            "action_id": self.action_id,
            "tool": self.tool,
            "arguments": self.arguments,
            "description": self.description,
            "role": self.role,
            "risk": self.risk,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "Action":
        return cls(
            action_id=str(value["action_id"]),
            tool=str(value["tool"]),
            arguments=dict(value.get("arguments", {})),
            description=str(value.get("description", "")),
            role=str(value.get("role", "candidate")),
            risk=str(value.get("risk", "low")),
        )


@dataclass
class TaskSpec:
    task_id: str
    domain: str
    instruction: str
    initial_state: JSONDict
    target: JSONDict
    constraints: list[JSONDict]
    oracle_plan: list[Action]
    candidates: list[Action]
    metadata: JSONDict = field(default_factory=dict)

    def to_dict(self) -> JSONDict:
        return {
            "task_id": self.task_id,
            "domain": self.domain,
            "instruction": self.instruction,
            "initial_state": self.initial_state,
            "target": self.target,
            "constraints": self.constraints,
            "oracle_plan": [action.to_dict() for action in self.oracle_plan],
            "candidates": [action.to_dict() for action in self.candidates],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TaskSpec":
        return cls(
            task_id=str(value["task_id"]),
            domain=str(value["domain"]),
            instruction=str(value["instruction"]),
            initial_state=dict(value["initial_state"]),
            target=dict(value["target"]),
            constraints=[dict(item) for item in value.get("constraints", [])],
            oracle_plan=[Action.from_dict(item) for item in value.get("oracle_plan", [])],
            candidates=[Action.from_dict(item) for item in value.get("candidates", [])],
            metadata=dict(value.get("metadata", {})),
        )


@dataclass
class StepResult:
    ok: bool
    observation: JSONDict
    state_before: JSONDict
    state_after: JSONDict
    state_patch: JSONDict
    target_success: bool
    violations: list[str]
    error: str | None = None

    def to_dict(self) -> JSONDict:
        return {
            "ok": self.ok,
            "observation": self.observation,
            "state_before": self.state_before,
            "state_after": self.state_after,
            "state_patch": self.state_patch,
            "target_success": self.target_success,
            "violations": self.violations,
            "error": self.error,
        }


@dataclass
class BranchRecord:
    task_id: str
    domain: str
    action: Action
    initial_state_digest: str
    real_observation: JSONDict
    real_state_patch: JSONDict
    candidate_ok: bool
    immediate_success: bool
    terminal_success: bool
    violations: list[str]
    executed_steps: int
    real_value: float
    error: str | None = None

    def to_dict(self) -> JSONDict:
        return {
            "task_id": self.task_id,
            "domain": self.domain,
            "action": self.action.to_dict(),
            "initial_state_digest": self.initial_state_digest,
            "real_observation": self.real_observation,
            "real_state_patch": self.real_state_patch,
            "candidate_ok": self.candidate_ok,
            "immediate_success": self.immediate_success,
            "terminal_success": self.terminal_success,
            "violations": self.violations,
            "executed_steps": self.executed_steps,
            "real_value": self.real_value,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "BranchRecord":
        return cls(
            task_id=str(value["task_id"]),
            domain=str(value["domain"]),
            action=Action.from_dict(value["action"]),
            initial_state_digest=str(value["initial_state_digest"]),
            real_observation=dict(value.get("real_observation", {})),
            real_state_patch=dict(value.get("real_state_patch", {})),
            candidate_ok=bool(value.get("candidate_ok", False)),
            immediate_success=bool(value.get("immediate_success", False)),
            terminal_success=bool(value.get("terminal_success", False)),
            violations=[str(item) for item in value.get("violations", [])],
            executed_steps=int(value.get("executed_steps", 0)),
            real_value=float(value.get("real_value", 0.0)),
            error=value.get("error"),
        )


@dataclass
class PredictionRecord:
    task_id: str
    domain: str
    action_id: str
    provider: str
    model: str
    raw_text: str
    parsed_observation: JSONDict | None
    parse_ok: bool
    latency_seconds: float
    usage: JSONDict = field(default_factory=dict)
    error: str | None = None

    def to_dict(self) -> JSONDict:
        return {
            "task_id": self.task_id,
            "domain": self.domain,
            "action_id": self.action_id,
            "provider": self.provider,
            "model": self.model,
            "raw_text": self.raw_text,
            "parsed_observation": self.parsed_observation,
            "parse_ok": self.parse_ok,
            "latency_seconds": self.latency_seconds,
            "usage": self.usage,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PredictionRecord":
        parsed = value.get("parsed_observation")
        return cls(
            task_id=str(value["task_id"]),
            domain=str(value["domain"]),
            action_id=str(value["action_id"]),
            provider=str(value["provider"]),
            model=str(value.get("model", "unknown")),
            raw_text=str(value.get("raw_text", "")),
            parsed_observation=dict(parsed) if isinstance(parsed, Mapping) else None,
            parse_ok=bool(value.get("parse_ok", False)),
            latency_seconds=float(value.get("latency_seconds", 0.0)),
            usage=dict(value.get("usage", {})),
            error=value.get("error"),
        )


@dataclass
class ScoredCandidate:
    task_id: str
    domain: str
    action_id: str
    action_role: str
    action_risk: str
    real_value: float
    simulated_value: float
    token_f1: float
    structured_f1: float
    critical_field_accuracy: float
    predicted_success: bool
    parse_ok: bool
    real_terminal_success: bool
    real_violations: list[str]
    patch_repair: str | None = None
    prediction_error: str | None = None

    def to_dict(self) -> JSONDict:
        return {
            "task_id": self.task_id,
            "domain": self.domain,
            "action_id": self.action_id,
            "action_role": self.action_role,
            "action_risk": self.action_risk,
            "real_value": self.real_value,
            "simulated_value": self.simulated_value,
            "token_f1": self.token_f1,
            "structured_f1": self.structured_f1,
            "critical_field_accuracy": self.critical_field_accuracy,
            "predicted_success": self.predicted_success,
            "parse_ok": self.parse_ok,
            "real_terminal_success": self.real_terminal_success,
            "real_violations": self.real_violations,
            "patch_repair": self.patch_repair,
            "prediction_error": self.prediction_error,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ScoredCandidate":
        return cls(
            task_id=str(value["task_id"]),
            domain=str(value["domain"]),
            action_id=str(value["action_id"]),
            action_role=str(value.get("action_role", "candidate")),
            action_risk=str(value.get("action_risk", "low")),
            real_value=float(value["real_value"]),
            simulated_value=float(value["simulated_value"]),
            token_f1=float(value.get("token_f1", 0.0)),
            structured_f1=float(value.get("structured_f1", 0.0)),
            critical_field_accuracy=float(value.get("critical_field_accuracy", 0.0)),
            predicted_success=bool(value.get("predicted_success", False)),
            parse_ok=bool(value.get("parse_ok", False)),
            real_terminal_success=bool(value.get("real_terminal_success", False)),
            real_violations=[str(item) for item in value.get("real_violations", [])],
            patch_repair=value.get("patch_repair"),
            prediction_error=value.get("prediction_error"),
        )
