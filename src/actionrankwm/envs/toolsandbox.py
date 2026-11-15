from __future__ import annotations

import copy
import datetime as datetime_module
import importlib
import json
import uuid
from contextlib import ExitStack
from types import ModuleType
from typing import Any, Mapping
from unittest.mock import patch

from actionrankwm.envs.base import BaseEnvironment
from actionrankwm.semantics import constraint_violations_for_state, target_success_for_state
from actionrankwm.types import Action, JSONDict, StepResult, TaskSpec


_NAMESPACES = ("setting", "contact", "messaging", "reminder")


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return str(value)


def _replace_generated(value: Any, generated_id: str, placeholder: str) -> Any:
    if isinstance(value, str):
        return placeholder if value == generated_id else value
    if isinstance(value, list):
        return [
            _replace_generated(item, generated_id, placeholder) for item in value
        ]
    if isinstance(value, Mapping):
        return {
            str(key): _replace_generated(item, generated_id, placeholder)
            for key, item in value.items()
        }
    return value


def _sorted_rows(rows: Any) -> list[JSONDict]:
    if not isinstance(rows, list):
        raise ValueError("ToolSandbox namespace rows must be a list")
    normalized = [dict(row) for row in rows if isinstance(row, Mapping)]
    if len(normalized) != len(rows):
        raise ValueError("ToolSandbox namespace rows must all be objects")
    return sorted(normalized, key=lambda row: json.dumps(row, sort_keys=True, default=str))


def _canonical_state(state: Mapping[str, Any]) -> JSONDict:
    raw_namespaces = state.get("namespaces")
    if not isinstance(raw_namespaces, Mapping):
        raise ValueError("ToolSandbox state requires a namespaces object")
    unknown = set(raw_namespaces) - set(_NAMESPACES)
    if unknown:
        raise ValueError(f"Unknown ToolSandbox namespaces: {sorted(unknown)}")
    return {
        "namespaces": {
            namespace: _sorted_rows(raw_namespaces.get(namespace, []))
            for namespace in _NAMESPACES
        }
    }


def _state_patch(before: JSONDict, after: JSONDict) -> JSONDict:
    changed = {
        namespace: copy.deepcopy(after["namespaces"][namespace])
        for namespace in _NAMESPACES
        if before["namespaces"][namespace] != after["namespaces"][namespace]
    }
    return {
        "domain": "toolsandbox",
        "operation": "replace_namespaces" if changed else "none",
        "namespaces": changed,
    }


class ToolSandboxEnvironment(BaseEnvironment):
    """Thin optional adapter over Apple's unmodified ToolSandbox tool functions."""

    def __init__(self, task: TaskSpec, state: JSONDict | None = None) -> None:
        super().__init__(task)
        self._state = _canonical_state(state or task.initial_state)
        self._last_patch: JSONDict = {
            "domain": "toolsandbox",
            "operation": "none",
            "namespaces": {},
        }
        self._step_index = 0

    @staticmethod
    def _modules() -> dict[str, Any]:
        try:
            execution_context = importlib.import_module(
                "tool_sandbox.common.execution_context"
            )
            polars = importlib.import_module("polars")
            tool_modules: dict[str, ModuleType] = {
                name: importlib.import_module(f"tool_sandbox.tools.{name}")
                for name in ("setting", "contact", "messaging", "reminder")
            }
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "The toolsandbox domain requires the optional Apple ToolSandbox "
                "runtime; see README.md (Rerun with your own models)"
            ) from exc
        return {
            "ExecutionContext": execution_context.ExecutionContext,
            "DatabaseNamespace": execution_context.DatabaseNamespace,
            "new_context": execution_context.new_context,
            "polars": polars,
            "tool_modules": tool_modules,
        }

    def _context_from_state(self, modules: Mapping[str, Any]) -> Any:
        execution_context = modules["ExecutionContext"](
            tool_allow_list=sorted(self.task.metadata.get("tool_docs", {}))
        )
        database_namespace = modules["DatabaseNamespace"]
        polars = modules["polars"]
        for namespace in _NAMESPACES:
            enum_value = database_namespace[namespace.upper()]
            rows = self._state["namespaces"][namespace]
            dataframe = polars.DataFrame(
                rows,
                schema=execution_context.dbs_schemas[enum_value],
            )
            execution_context.update_database(enum_value, dataframe)
        return execution_context

    @staticmethod
    def _state_from_context(context: Any, database_namespace: Any) -> JSONDict:
        return _canonical_state(
            {
                "namespaces": {
                    namespace: context.get_database(
                        database_namespace[namespace.upper()]
                    ).to_dicts()
                    for namespace in _NAMESPACES
                }
            }
        )

    def _deterministic_uuid(self, action: Action) -> uuid.UUID:
        payload = (
            f"{self.task.task_id}:{action.action_id}:{self._step_index}:"
            f"{action.tool}:{json.dumps(action.arguments, sort_keys=True)}"
        )
        return uuid.uuid5(uuid.NAMESPACE_URL, payload)

    def clone(self) -> "ToolSandboxEnvironment":
        clone = ToolSandboxEnvironment(self.task, state=self._state)
        clone._step_index = self._step_index
        clone._last_patch = copy.deepcopy(self._last_patch)
        return clone

    def state_dict(self) -> JSONDict:
        return copy.deepcopy(self._state)

    def apply(self, action: Action) -> StepResult:
        modules = self._modules()
        before = self.state_dict()
        context = self._context_from_state(modules)
        tool = context.name_to_tool.get(action.tool)
        ok = False
        result: Any = None
        error: str | None = None
        error_type: str | None = None
        deterministic_uuid = self._deterministic_uuid(action)
        frozen_timestamp = float(
            self.task.metadata.get("frozen_timestamp", 1_784_380_800.0)
        )

        class FrozenDateTime(datetime_module.datetime):
            @classmethod
            def now(cls, tz: datetime_module.tzinfo | None = None) -> "FrozenDateTime":
                return cls.fromtimestamp(frozen_timestamp, tz=tz)

        if tool is None:
            error_type = "UnknownTool"
            error = f"Unknown or unavailable ToolSandbox tool: {action.tool}"
        else:
            with ExitStack() as stack:
                for module in modules["tool_modules"].values():
                    if hasattr(module, "uuid4"):
                        stack.enter_context(
                            patch.object(module, "uuid4", lambda: deterministic_uuid)
                        )
                stack.enter_context(
                    patch.object(datetime_module, "datetime", FrozenDateTime)
                )
                try:
                    with modules["new_context"](context):
                        result = tool(**copy.deepcopy(action.arguments))
                    ok = True
                except Exception as exc:
                    error_type = type(exc).__name__
                    error = str(exc)

        generated_id = str(deterministic_uuid)
        generated_placeholder = f"<generated_id_{self._step_index}>"
        after = _canonical_state(
            _replace_generated(
                self._state_from_context(context, modules["DatabaseNamespace"]),
                generated_id,
                generated_placeholder,
            )
        )
        state_patch = _state_patch(before, after)
        self._state = after
        self._last_patch = state_patch
        self._step_index += 1
        observation: JSONDict = {
            "ok": ok,
            "result": _replace_generated(
                _jsonable(result), generated_id, generated_placeholder
            ),
            "error_type": error_type,
            "error": error,
            "state_patch": copy.deepcopy(state_patch),
        }
        violations = constraint_violations_for_state(
            self.task, self._state, self._last_patch
        )
        return StepResult(
            ok=ok,
            observation=observation,
            state_before=before,
            state_after=self.state_dict(),
            state_patch=copy.deepcopy(state_patch),
            target_success=self.target_success(),
            violations=violations,
            error=error,
        )

    def target_success(self) -> bool:
        return target_success_for_state(self.task, self._state)

    def violations(self) -> list[str]:
        return constraint_violations_for_state(
            self.task, self._state, self._last_patch
        )
