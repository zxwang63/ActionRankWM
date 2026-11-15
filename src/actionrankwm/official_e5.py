from __future__ import annotations

import copy
import datetime as datetime_module
import importlib
import json
import uuid
from contextlib import ExitStack
from dataclasses import dataclass
from typing import Any, Mapping, Sequence
from unittest.mock import patch

from actionrankwm import official_scenarios as v1
from actionrankwm import official_scenarios_v11 as v11
from actionrankwm.bundle import BUNDLE_SCHEMA, opaque_join_id
from actionrankwm.io import canonical_json, digest_json
from actionrankwm.providers.prompts import build_messages
from actionrankwm.toolsandbox_tasks import FROZEN_TIMESTAMP, _TOOL_DOCS
from actionrankwm.types import Action, TaskSpec


E5_PROTOCOL = "official_scenario_e5_bundle_roundtrip_v11"
E5_TASK_NAMESPACE = "actionrankwm-official-e5-task-v1"
E5_ACTION_NAMESPACE = "actionrankwm-official-e5-action-v1"
NAMESPACES = ("setting", "contact", "messaging", "reminder")
GENERATED_ID_TOOLS = frozenset({"add_contact", "send_message_with_phone_number"})
PLACEHOLDER = "<generated_id_0>"


class OfficialE5PredictionError(ValueError):
    """A model observation cannot be injected into the official trajectory."""


@dataclass(frozen=True)
class OfficialE5Branch:
    scenario_name: str
    semantic_group: str
    task_id: str
    action_id: str
    join_id: str
    action_key: str
    action: v1.OfficialAction
    initial_state: dict[str, Any]
    messages: list[dict[str, Any]]
    real_value: float

    def private_dict(self) -> dict[str, Any]:
        return {
            "scenario_name": self.scenario_name,
            "semantic_group": self.semantic_group,
            "task_id": self.task_id,
            "action_id": self.action_id,
            "join_id": self.join_id,
            "action_key": self.action_key,
            "action": self.action.to_dict(),
            "initial_state": copy.deepcopy(self.initial_state),
            "prompt_sha256": digest_json(self.messages),
            "real_value": self.real_value,
        }

    def model_input_dict(self) -> dict[str, Any]:
        return {
            "schema": BUNDLE_SCHEMA,
            "join_id": self.join_id,
            "messages": copy.deepcopy(self.messages),
            "prompt_sha256": digest_json(self.messages),
        }


def _replace_exact(value: Any, source: str, replacement: str) -> Any:
    if isinstance(value, str):
        return replacement if value == source else value
    if isinstance(value, list):
        return [_replace_exact(item, source, replacement) for item in value]
    if isinstance(value, Mapping):
        return {
            str(key): _replace_exact(item, source, replacement)
            for key, item in value.items()
        }
    return value


def _contains_generated_marker(value: Any) -> bool:
    if isinstance(value, str):
        return "<generated_id_" in value
    if isinstance(value, list):
        return any(_contains_generated_marker(item) for item in value)
    if isinstance(value, Mapping):
        return any(_contains_generated_marker(item) for item in value.values())
    return False


def _json_rows(context: Any, namespace_name: str) -> list[dict[str, Any]]:
    execution = importlib.import_module("tool_sandbox.common.execution_context")
    namespace = execution.DatabaseNamespace[namespace_name.upper()]
    return [
        {
            str(key): v1._jsonable(value)
            for key, value in row.items()
            if key != "sandbox_message_index"
        }
        for row in context.get_database(namespace).to_dicts()
    ]


def state_from_context(context: Any, *, canonical_rows: bool = False) -> dict[str, Any]:
    namespaces: dict[str, list[dict[str, Any]]] = {}
    for namespace in NAMESPACES:
        rows = _json_rows(context, namespace)
        if canonical_rows:
            rows = sorted(rows, key=canonical_json)
        namespaces[namespace] = rows
    return {"namespaces": namespaces}


def official_initial_state(scenario: Any) -> dict[str, Any]:
    return state_from_context(scenario.starting_context, canonical_rows=True)


def _state_patch(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, Any]:
    changed = {
        namespace: copy.deepcopy(after["namespaces"][namespace])
        for namespace in NAMESPACES
        if before["namespaces"][namespace] != after["namespaces"][namespace]
    }
    return {
        "domain": "toolsandbox",
        "operation": "replace_namespaces" if changed else "none",
        "namespaces": changed,
    }


def _deterministic_id(
    plan: v1.OfficialPlan,
    first_action_key: str,
    step_index: int,
    action: v1.OfficialAction,
) -> uuid.UUID:
    return uuid.uuid5(
        uuid.NAMESPACE_URL,
        canonical_json(
            {
                "scenario": plan.scenario_name,
                "first_action_key": first_action_key,
                "step_index": step_index,
                "action": action.to_dict(),
                "replay": v11.REPLAY_PROTOCOL,
            }
        ),
    )


def _tool_modules() -> list[Any]:
    return [
        importlib.import_module(f"tool_sandbox.tools.{name}")
        for name in NAMESPACES
    ]


def _append_message(context: Any, message: Mapping[str, Any]) -> dict[str, Any]:
    execution = importlib.import_module("tool_sandbox.common.execution_context")
    context.add_to_database(
        execution.DatabaseNamespace.SANDBOX,
        [
            {
                "sender": execution.RoleType[str(message["sender"])],
                "recipient": execution.RoleType[str(message["recipient"])],
                "content": str(message["content"]),
            }
        ],
    )
    return {
        "kind": "message",
        "sandbox_message_index": context.max_sandbox_message_index,
        "ok": True,
    }


def _append_request(context: Any, action: v1.OfficialAction) -> int:
    execution = importlib.import_module("tool_sandbox.common.execution_context")
    context.add_to_database(
        execution.DatabaseNamespace.SANDBOX,
        [
            {
                "sender": execution.RoleType.AGENT,
                "recipient": execution.RoleType.EXECUTION_ENVIRONMENT,
                "content": v11._request_content(action),
            }
        ],
    )
    return int(context.max_sandbox_message_index)


def _append_response(
    context: Any,
    *,
    error_text: str | None,
    traces: Sequence[str],
) -> int:
    execution = importlib.import_module("tool_sandbox.common.execution_context")
    response: dict[str, Any] = {
        "sender": execution.RoleType.EXECUTION_ENVIRONMENT,
        "recipient": execution.RoleType.AGENT,
        "content": "" if error_text is None else error_text,
    }
    if traces:
        response["tool_trace"] = list(traces)
    if error_text is not None:
        response["tool_call_exception"] = error_text
    context.add_to_database(execution.DatabaseNamespace.SANDBOX, [response])
    return int(context.max_sandbox_message_index)


def _verify_transaction(
    context: Any,
    request_index: int,
    response_index: int,
    *,
    succeeded: bool,
) -> None:
    execution = importlib.import_module("tool_sandbox.common.execution_context")
    if response_index != request_index + 1:
        raise RuntimeError(
            f"Non-adjacent request/response indices: {request_index}, {response_index}"
        )
    request_frame = context.get_database(
        execution.DatabaseNamespace.SANDBOX,
        sandbox_message_index=request_index,
    )
    response_frame = context.get_database(
        execution.DatabaseNamespace.SANDBOX,
        sandbox_message_index=response_index,
    )
    if request_frame["tool_trace"][0] is not None:
        raise RuntimeError("Trace remained on agent request")
    response_trace = response_frame["tool_trace"][0]
    if succeeded != (response_trace is not None):
        raise RuntimeError("Response trace/error mismatch")


def _execute_actual_action(
    context: Any,
    plan: v1.OfficialPlan,
    first_action_key: str,
    step_index: int,
    event: v1.OfficialEvent,
) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    assert event.action is not None
    action = event.action
    before = state_from_context(context)
    request_index = _append_request(context, action)
    execution = importlib.import_module("tool_sandbox.common.execution_context")
    function = context.name_to_tool.get(action.tool)
    result: Any = None
    error_type: str | None = None
    error_message: str | None = None
    deterministic_id = _deterministic_id(plan, first_action_key, step_index, action)
    if function is None:
        error_type = "UnknownTool"
        error_message = action.tool
    else:
        with ExitStack() as stack:
            stack.enter_context(
                patch.object(datetime_module, "datetime", v1.FrozenDateTime)
            )
            for module in _tool_modules():
                if hasattr(module, "uuid4"):
                    stack.enter_context(
                        patch.object(module, "uuid4", lambda: deterministic_id)
                    )
            try:
                with execution.new_context(context):
                    result = function(**copy.deepcopy(action.arguments))
            except Exception as exc:
                error_type = type(exc).__name__
                error_message = str(exc)

    traces = v11._trace_list(context, execution)
    if traces:
        v11._clear_request_trace(context, execution)
    succeeded = error_type is None
    if succeeded and len(traces) != 1:
        raise RuntimeError(
            f"Expected exactly one successful trace for {action.key}, found {len(traces)}"
        )
    if not succeeded and traces:
        raise RuntimeError(f"Failed tool unexpectedly emitted a trace: {action.key}")
    error_text = (
        None if succeeded else f"{error_type}:{error_message}"
    )
    response_index = _append_response(
        context, error_text=error_text, traces=traces
    )
    _verify_transaction(
        context,
        request_index,
        response_index,
        succeeded=succeeded,
    )
    after = state_from_context(context)
    generated_id = str(deterministic_id)
    observation = {
        "ok": succeeded,
        "result": _replace_exact(v1._jsonable(result), generated_id, PLACEHOLDER),
        "error_type": error_type,
        "error": error_message,
        "state_patch": _replace_exact(
            _state_patch(before, after), generated_id, PLACEHOLDER
        ),
    }
    record = {
        "event_key": event.key,
        "kind": "action",
        "action": action.to_dict(),
        "request_index": request_index,
        "response_index": response_index,
        "ok": succeeded,
        "result": v1._jsonable(result),
        "error": error_text,
        "trace_count": len(traces),
    }
    return record, observation, traces


def _validate_observation(
    observation: Mapping[str, Any],
    action: v1.OfficialAction,
    deterministic_id: uuid.UUID,
) -> tuple[dict[str, Any], Any, str | None, list[str]]:
    required = {"ok", "result", "error_type", "error", "state_patch"}
    if set(observation) != required:
        raise OfficialE5PredictionError(
            f"observation_keys:{sorted(set(observation))}"
        )
    if type(observation["ok"]) is not bool:
        raise OfficialE5PredictionError("ok_must_be_boolean")
    succeeded = bool(observation["ok"])
    error_type = observation["error_type"]
    error_message = observation["error"]
    if succeeded:
        if error_type is not None or error_message is not None:
            raise OfficialE5PredictionError("successful_observation_has_error")
    else:
        if not isinstance(error_type, str) or not error_type:
            raise OfficialE5PredictionError("failed_observation_missing_error_type")
        if not isinstance(error_message, str):
            raise OfficialE5PredictionError("failed_observation_missing_error_message")

    raw_result = copy.deepcopy(observation["result"])
    raw_patch = observation["state_patch"]
    if not isinstance(raw_patch, Mapping):
        raise OfficialE5PredictionError("state_patch_must_be_object")
    if set(raw_patch) != {"domain", "operation", "namespaces"}:
        raise OfficialE5PredictionError("state_patch_keys")
    if raw_patch.get("domain") != "toolsandbox":
        raise OfficialE5PredictionError("state_patch_domain")
    operation = raw_patch.get("operation")
    replacements = raw_patch.get("namespaces")
    if not isinstance(replacements, Mapping):
        raise OfficialE5PredictionError("state_patch_namespaces_must_be_object")
    if operation == "none":
        if replacements:
            raise OfficialE5PredictionError("none_patch_must_have_empty_namespaces")
    elif operation == "replace_namespaces":
        if not replacements:
            raise OfficialE5PredictionError("replacement_patch_must_be_nonempty")
        unknown = set(replacements) - set(NAMESPACES)
        if unknown:
            raise OfficialE5PredictionError(
                f"unknown_namespaces:{sorted(unknown)}"
            )
    else:
        raise OfficialE5PredictionError(f"unsupported_patch_operation:{operation}")

    uses_marker = _contains_generated_marker(
        {"result": raw_result, "state_patch": raw_patch}
    )
    if action.tool in GENERATED_ID_TOOLS and succeeded:
        if raw_result != PLACEHOLDER:
            raise OfficialE5PredictionError("generated_result_placeholder_required")
    elif uses_marker:
        raise OfficialE5PredictionError("generated_placeholder_not_allowed")
    expanded_result = _replace_exact(raw_result, PLACEHOLDER, str(deterministic_id))
    expanded_patch = _replace_exact(raw_patch, PLACEHOLDER, str(deterministic_id))
    if _contains_generated_marker(
        {"result": expanded_result, "state_patch": expanded_patch}
    ):
        raise OfficialE5PredictionError("unknown_generated_placeholder")

    trace_payload = {
        "tool_name": action.tool,
        "arguments": copy.deepcopy(action.arguments),
        "result": copy.deepcopy(expanded_result),
    }
    traces = [json.dumps(trace_payload)] if succeeded else []
    error_text = None if succeeded else f"{error_type}:{error_message}"
    return dict(expanded_patch), expanded_result, error_text, traces


def _apply_patch_to_context(
    context: Any,
    patch_value: Mapping[str, Any],
    action: v1.OfficialAction,
) -> None:
    execution = importlib.import_module("tool_sandbox.common.execution_context")
    operation = patch_value["operation"]
    if operation == "none":
        return
    replacements = patch_value["namespaces"]
    for namespace_name, raw_rows in replacements.items():
        if not isinstance(raw_rows, list) or any(
            not isinstance(row, Mapping) for row in raw_rows
        ):
            raise OfficialE5PredictionError(
                f"namespace_rows_must_be_objects:{namespace_name}"
            )
        namespace = execution.DatabaseNamespace[str(namespace_name).upper()]
        schema = execution.ExecutionContext.dbs_schemas[namespace]
        expected_fields = set(schema) - {"sandbox_message_index"}
        for row in raw_rows:
            if set(row) != expected_fields:
                raise OfficialE5PredictionError(
                    f"namespace_row_fields:{namespace_name}:"
                    f"{sorted(set(row))}!={sorted(expected_fields)}"
                )
        try:
            dataframe = execution.pl.DataFrame(
                [dict(row) for row in raw_rows],
                schema={key: value for key, value in schema.items() if key != "sandbox_message_index"},
            )
            context.update_database(namespace, dataframe)
            if action.tool not in {
                "set_low_battery_mode_status",
                "set_location_service_status",
                "set_cellular_service_status",
                "set_wifi_status",
            }:
                snapshot_index = context.max_sandbox_message_index + 1
                full_frame = context._dbs[namespace]
                remove_headguard = (
                    (execution.pl.col("sandbox_message_index") == snapshot_index)
                    & context.headguard_predicate(set(full_frame.columns))
                )
                context._dbs[namespace] = full_frame.filter(~remove_headguard)
        except Exception as exc:
            raise OfficialE5PredictionError(
                f"namespace_schema:{namespace_name}:{type(exc).__name__}:{exc}"
            ) from exc


def _inject_predicted_action(
    context: Any,
    plan: v1.OfficialPlan,
    first_action_key: str,
    step_index: int,
    event: v1.OfficialEvent,
    observation: Mapping[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    assert event.action is not None
    action = event.action
    request_index = _append_request(context, action)
    deterministic_id = _deterministic_id(
        plan, first_action_key, step_index, action
    )
    expanded_patch, expanded_result, error_text, traces = _validate_observation(
        observation, action, deterministic_id
    )
    _apply_patch_to_context(context, expanded_patch, action)
    if traces:
        execution = importlib.import_module(
            "tool_sandbox.common.execution_context"
        )
        v11._clear_request_trace(context, execution)
    response_index = _append_response(
        context, error_text=error_text, traces=traces
    )
    succeeded = error_text is None
    _verify_transaction(
        context,
        request_index,
        response_index,
        succeeded=succeeded,
    )
    record = {
        "event_key": event.key,
        "kind": "action",
        "action": action.to_dict(),
        "request_index": request_index,
        "response_index": response_index,
        "ok": succeeded,
        "result": v1._jsonable(expanded_result),
        "error": error_text,
        "trace_count": len(traces),
    }
    return record, traces


def execute_official_e5_branch(
    scenario: Any,
    plan: v1.OfficialPlan,
    first_action_key: str,
    *,
    injected_observation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    context = copy.deepcopy(scenario.starting_context)
    records: list[dict[str, Any]] = []
    first_observation: dict[str, Any] | None = None
    first_trace: list[str] = []
    events = v1._ordered_events(plan, first_action_key)
    for step_index, event in enumerate(events):
        if event.action is None:
            assert event.message is not None
            record = _append_message(context, event.message)
            record["event_key"] = event.key
            records.append(record)
            continue
        if step_index == 0 and injected_observation is not None:
            record, traces = _inject_predicted_action(
                context,
                plan,
                first_action_key,
                step_index,
                event,
                injected_observation,
            )
            first_trace = traces
        else:
            record, observation, traces = _execute_actual_action(
                context, plan, first_action_key, step_index, event
            )
            if step_index == 0:
                first_observation = observation
                first_trace = traces
        records.append(record)

    result = scenario.evaluation.evaluate(
        execution_context=context,
        max_turn_count=scenario.max_messages,
    )
    return {
        "replay_protocol": E5_PROTOCOL,
        "first_action_key": first_action_key,
        "similarity": float(result.similarity),
        "milestone_similarity": float(result.milestone_similarity),
        "minefield_similarity": float(result.minefield_similarity),
        "milestone_mapping": v1._jsonable(dict(result.milestone_mapping)),
        "minefield_mapping": v1._jsonable(dict(result.minefield_mapping)),
        "turn_count": int(result.turn_count),
        "execution_records": records,
        "context_digest": v1._context_digest(context),
        "first_observation": first_observation,
        "first_trace": first_trace,
    }


def _hidden_task_id(scenario_name: str) -> str:
    return "e5t_" + digest_json(
        {"namespace": E5_TASK_NAMESPACE, "scenario_name": scenario_name}
    )[:24]


def _hidden_action_id(scenario_name: str, action_key: str) -> str:
    return "e5a_" + digest_json(
        {
            "namespace": E5_ACTION_NAMESPACE,
            "scenario_name": scenario_name,
            "action_key": action_key,
        }
    )[:24]


def _prompt_task(
    task_id: str,
    action_id: str,
    action: v1.OfficialAction,
    initial_state: dict[str, Any],
) -> tuple[TaskSpec, Action]:
    candidate = Action(
        action_id=action_id,
        tool=action.tool,
        arguments=copy.deepcopy(action.arguments),
    )
    task = TaskSpec(
        task_id=task_id,
        domain="toolsandbox",
        instruction="",
        initial_state=copy.deepcopy(initial_state),
        target={},
        constraints=[],
        oracle_plan=[],
        candidates=[candidate],
        metadata={
            "frozen_timestamp": FROZEN_TIMESTAMP,
            "tool_docs": {action.tool: copy.deepcopy(_TOOL_DOCS[action.tool])},
        },
    )
    return task, candidate


def audit_label_blind_messages(
    messages: Sequence[Mapping[str, Any]],
    *,
    task_id: str,
    action_id: str,
    scenario_names: Sequence[str],
    semantic_groups: Sequence[str],
    action_keys: Sequence[str],
) -> dict[str, Any]:
    if len(messages) != 2 or [item.get("role") for item in messages] != [
        "system",
        "user",
    ]:
        raise ValueError("Unexpected message envelope")
    user = json.loads(str(messages[1]["content"]))
    expected_keys = {
        "domain",
        "current_state",
        "action",
        "tool_docs",
        "benchmark_context",
    }
    if set(user) != expected_keys:
        raise ValueError(f"Unexpected model-visible keys: {sorted(user)}")
    if set(user["action"]) != {"tool", "arguments"}:
        raise ValueError("Unexpected model-visible action keys")
    if set(user["benchmark_context"]) != {"current_timestamp"}:
        raise ValueError("Unexpected benchmark-context keys")
    if set(user["tool_docs"]) != {user["action"]["tool"]}:
        raise ValueError("Tool docs reveal tools other than the current action")
    serialized = canonical_json(list(messages))
    serialized_user = canonical_json(user)
    hidden_values = {
        task_id,
        action_id,
        *[str(item) for item in scenario_names],
        *[str(item) for item in semantic_groups],
        *[str(item) for item in action_keys],
    }
    leaked_values = sorted(value for value in hidden_values if value and value in serialized)
    if leaked_values:
        raise ValueError(f"Hidden identifiers leaked: {leaked_values[:5]}")
    forbidden_keys = (
        '"task_id":',
        '"action_id":',
        '"candidate',
        '"constraints":',
        '"instruction":',
        '"milestone',
        '"minefield',
        '"oracle_plan":',
        '"real_',
        '"role":',
        '"risk":',
        '"scenario',
        '"semantic_group":',
        '"target":',
        '"utility',
    )
    leaked_keys = [key for key in forbidden_keys if key in serialized_user]
    if leaked_keys:
        raise ValueError(f"Hidden supervision keys leaked: {leaked_keys}")
    return {
        "passed": True,
        "visible_user_keys": sorted(expected_keys),
        "visible_action_keys": sorted(user["action"]),
        "visible_tool_count": len(user["tool_docs"]),
    }


def build_official_e5_branches(
    eligible_rows: Sequence[Mapping[str, Any]],
) -> list[OfficialE5Branch]:
    scenarios = v1.load_unaugmented_scenarios()
    scenario_names = [str(row["scenario_name"]) for row in eligible_rows]
    semantic_groups = [str(row["semantic_group"]) for row in eligible_rows]
    action_keys = [
        str(event["action"]["key"])
        for row in eligible_rows
        for event in row["plan"]["events"]
        if event.get("action") is not None
    ]
    branches: list[OfficialE5Branch] = []
    for row in eligible_rows:
        scenario_name = str(row["scenario_name"])
        semantic_group = str(row["semantic_group"])
        family, scenario = scenarios[scenario_name]
        plan = v1.extract_official_plan(scenario_name, family, scenario)
        if plan.to_dict() != row["plan"]:
            raise ValueError(f"Frozen plan mismatch: {scenario_name}")
        values = [float(value) for value in row["utility_vector"]]
        if len(values) != len(plan.actions):
            raise ValueError(f"Utility/action mismatch: {scenario_name}")
        initial_state = official_initial_state(scenario)
        task_id = _hidden_task_id(scenario_name)
        for action, real_value in zip(plan.actions, values):
            action_id = _hidden_action_id(scenario_name, action.key)
            task, candidate = _prompt_task(
                task_id, action_id, action, initial_state
            )
            messages = build_messages(task, candidate, "direct")
            audit_label_blind_messages(
                messages,
                task_id=task_id,
                action_id=action_id,
                scenario_names=scenario_names,
                semantic_groups=semantic_groups,
                action_keys=action_keys,
            )
            join_id = opaque_join_id(task_id, action_id)
            branches.append(
                OfficialE5Branch(
                    scenario_name=scenario_name,
                    semantic_group=semantic_group,
                    task_id=task_id,
                    action_id=action_id,
                    join_id=join_id,
                    action_key=action.key,
                    action=action,
                    initial_state=copy.deepcopy(initial_state),
                    messages=messages,
                    real_value=real_value,
                )
            )
    if len(branches) != 32 or len({item.join_id for item in branches}) != 32:
        raise ValueError(
            f"Expected 32 unique official E5 requests, found {len(branches)}"
        )
    return branches


def comparable_result(result: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: copy.deepcopy(result[key])
        for key in (
            "first_action_key",
            "similarity",
            "milestone_similarity",
            "minefield_similarity",
            "milestone_mapping",
            "minefield_mapping",
            "turn_count",
            "execution_records",
            "context_digest",
        )
    }
